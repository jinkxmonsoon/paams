"""Rate-aware, one-shot executor for frozen functional micro-pilot v1.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .real_llm_clients import GroqClient
from .run_frozen_micro_pilot import (
    CONNECTIVITY_PROMPT,
    REQUIRED_OUTPUT_FILES,
    _git_sha,
    _safe_environment,
)
from .runner_llm_real_smoke import POLICY_MAP, run_episode


MANIFEST_PATH = Path(__file__).with_name("real_llm_micro_pilot_manifest_v1_1.json")
V1_MANIFEST_PATH = Path(__file__).with_name("real_llm_micro_pilot_manifest.json")
OUTPUT_ROOT = Path(__file__).with_name("outputs")
PROTOCOL_VERSION = "btom-v2-functional-micro-pilot-1.1-rate-aware"
V1_MANIFEST_SHA256 = "d0495c6f5eca130ef9cbaf153b860ffc6fe257adf215e525c156e7425a073bf9"
EXPECTED_MATRIX_SIZE = 12
EXTRA_OUTPUT_FILES = (
    "api_call_schedule.json",
    "api_failure_audit.json",
    "corrected_metrics.json",
    "partial_episode_records.jsonl",
)
SCIENTIFIC_FIELDS = (
    "scenarios",
    "policies",
    "seeds",
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_steps",
    "max_llm_calls_per_episode",
    "primary_functional_criteria",
    "primary_behavioral_metrics",
    "exclusion_criteria",
)


class RateLimitAbort(BaseException):
    """Escapes policy-level Exception fallbacks and aborts the complete run."""

    def __init__(self, audit_record):
        super().__init__("rate limit abort")
        self.audit_record = audit_record


@dataclass
class GlobalCallScheduler:
    minimum_interval: float
    clock: object = time.monotonic
    sleep: object = time.sleep

    def __post_init__(self):
        self.last_start = None
        self.records = []

    def before_call(self, context):
        observed = self.clock()
        requested = 0.0 if self.last_start is None else max(
            0.0, self.minimum_interval - (observed - self.last_start)
        )
        sleep_started = self.clock()
        if requested:
            self.sleep(requested)
        actual = self.clock() - sleep_started
        started = self.clock()
        record = {
            "call_order": len(self.records) + 1,
            "requested_sleep_seconds": requested,
            "actual_sleep_seconds": actual,
            "call_start_monotonic": started,
            "context": dict(context),
        }
        self.records.append(record)
        self.last_start = started
        return record


def _redact(value, secret=None):
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, secret) for item in value]
    if not isinstance(value, str):
        return value
    text = value.replace(secret, "[REDACTED]") if secret else value
    text = re.sub(r"org_[A-Za-z0-9_-]+", "[REDACTED_ORG]", text)
    text = re.sub(
        r"(?i)authorization\s*[:=]\s*(?:bearer\s+)?[^,;\s]+",
        "Authorization=[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", text)
    return text


def _write_json(path, value, secret=None):
    path.write_text(json.dumps(_redact(value, secret), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path, rows, secret=None):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_redact(row, secret), sort_keys=True) + "\n")


def load_manifest(path=MANIFEST_PATH):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    parent = json.loads(V1_MANIFEST_PATH.read_bytes())
    assert manifest["protocol_version"] == PROTOCOL_VERSION
    assert manifest["parent_protocol"] == "btom-v2-functional-micro-pilot-1.0"
    assert manifest["global_minimum_inter_api_call_seconds"] == 10.0
    assert manifest["rate_limit_behavior"] == "abort_entire_run"
    assert manifest["retry_on_rate_limit"] is False
    assert manifest["retry_on_other_api_failure"] is False
    assert hashlib.sha256(V1_MANIFEST_PATH.read_bytes()).hexdigest() == V1_MANIFEST_SHA256
    for field in SCIENTIFIC_FIELDS:
        if manifest[field] != parent[field]:
            raise AssertionError(f"v1.1 scientific field differs from immutable v1.0 parent: {field}")
    return manifest, raw


def execution_matrix(manifest):
    matrix = [
        (scenario, policy, seed)
        for scenario in manifest["scenarios"]
        for policy in manifest["policies"]
        for seed in manifest["seeds"]
    ]
    if len(matrix) != EXPECTED_MATRIX_SIZE or len(matrix) != len(set(matrix)):
        raise ValueError("v1.1 requires exactly 12 unique episode combinations")
    return matrix


def _number(message, label):
    match = re.search(rf"(?i)\b{label}\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", message)
    return float(match.group(1)) if match else None


def _rate_limit_details(message, error):
    lowered = message.lower()
    if "token" in lowered or "tpm" in lowered:
        dimension = "tokens_per_minute"
    elif "request" in lowered or "rpm" in lowered:
        dimension = "requests_per_minute"
    else:
        dimension = None
    retry = _number(message, "retry-after") or _number(message, "try again in")
    return {
        "http_status": getattr(error, "http_status", None),
        "error_type": getattr(error, "error_type", type(error).__name__),
        "rate_limit_dimension": dimension,
        "limit": _number(message, "limit"),
        "used": _number(message, "used"),
        "requested": _number(message, "requested"),
        "retry_after_seconds": retry,
    }


class RateAwareClient:
    def __init__(self, client, scheduler, secret):
        self.client = client
        self.scheduler = scheduler
        self.secret = secret
        self.context = {"phase": "connectivity"}
        self.api_audit = []

    @property
    def transport(self):
        return getattr(self.client, "transport", None)

    @property
    def last_usage(self):
        return getattr(self.client, "last_usage", None)

    @property
    def last_latency_sec(self):
        return getattr(self.client, "last_latency_sec", None)

    @property
    def last_error(self):
        return getattr(self.client, "last_error", None)

    def set_context(self, **context):
        self.context = context

    def generate(self, prompt, **kwargs):
        schedule = self.scheduler.before_call(self.context)
        try:
            output = self.client.generate(prompt, **kwargs)
        except Exception as error:
            message = _redact(
                getattr(error, "sanitized_error_message", str(error)), self.secret
            )[:500]
            details = _rate_limit_details(message, error)
            is_rate_limit = details["http_status"] == 429 or details["error_type"] == "rate_limit"
            record = {
                **schedule,
                **self.context,
                **details,
                "success": False,
                "sanitized_error_message": message,
                "run_aborted": is_rate_limit,
            }
            self.api_audit.append(record)
            if is_rate_limit:
                raise RateLimitAbort(record)
            raise
        usage = self.last_usage or {}
        self.api_audit.append({
            **schedule,
            **self.context,
            "success": True,
            "http_status": 200,
            "error_type": None,
            "rate_limit_dimension": None,
            "limit": None,
            "used": None,
            "requested": None,
            "retry_after_seconds": None,
            "run_aborted": False,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })
        return output


def _milestones(trace_rows, episode_id):
    events = [row["event"] for row in trace_rows if row["episode_id"] == episode_id]
    valid = [
        event for event in events
        if event["event"] == "action_result" and event["details"].get("action_valid")
    ]
    red = any(e["agent"] == "A" and e["details"].get("action") == "open_box" for e in valid)
    blue = any(e["agent"] == "B" and e["details"].get("action") == "open_box" for e in valid)
    acquired = any(
        e["agent"] == "C" and e["details"].get("action") == "pickup"
        and e["details"].get("target") == "medical_kit" for e in valid
    )
    rescued = any(e["details"].get("action") == "rescue" for e in valid)
    return {
        "red_key_applied": red,
        "blue_key_applied": blue,
        "locked_box_open": red and blue,
        "medical_kit_acquired": acquired,
        "victim_rescued": rescued,
    }


def corrected_metrics(row, api_records, trace_rows):
    attempts = len(api_records)
    responses = sum(record["success"] for record in api_records)
    parsed = sum(call.get("parse_success", False) for call in row.get("call_audit", []))
    direct_results = [
        call["environment_result"] for call in row.get("call_audit", [])
        if call.get("parse_success") and call.get("environment_result")
    ]
    valid_non_stay = 0
    for call in row.get("call_audit", []):
        first_target = call.get("valid_actions", [{}])[0].get("target")
        non_stay = call.get("parsed_action") != "move" or call.get("parsed_target") != first_target
        if call.get("parse_success") and non_stay and call.get("environment_result", {}).get("valid"):
            valid_non_stay += 1
    return {
        "api_response_success_rate": None if attempts == 0 else responses / attempts,
        "parser_success_given_api_response": None if responses == 0 else parsed / responses,
        "end_to_end_parseable_action_rate": None if attempts == 0 else parsed / attempts,
        "direct_model_action_environment_valid_rate": (
            None if not direct_results else sum(result["valid"] for result in direct_results) / len(direct_results)
        ),
        "fallback_action_count": row.get("fallback_actions", 0),
        "valid_non_stay_model_actions": valid_non_stay,
        "objective_task_milestones_reached": _milestones(trace_rows, row["episode_id"]),
        "task_success": bool(row.get("success")),
        "deprecated_metric": {"parse_success_rate": "removed from v1.1 report"},
    }


def _exclusion(row, api_records):
    failures = [record for record in api_records if not record["success"]]
    if failures:
        return {
            "episode_id": row["episode_id"],
            "excluded": None,
            "exclusion_status": "requires_manual_audit",
            "criterion": "episode has an API failure that prevents the planned action opportunity",
            "evidence": {"api_failure_call_orders": [record["call_order"] for record in failures]},
            "reason": "The available trace does not mechanically establish that the API failure prevented the planned action opportunity.",
        }
    parser_failures = row.get("parse_failures", 0)
    invalid_actions = row.get("environment_invalid_action_count", 0)
    no_correction = row.get("correction_opportunities", 0) == 0
    if row["scenario_id"].startswith("C7") and no_correction and (parser_failures or invalid_actions):
        relevant_calls = [
            call.get("call_idx")
            for call in row.get("call_audit", [])
            if (
                not call.get("parse_success", False)
                or call.get("environment_result", {}).get("valid") is False
            )
        ]
        return {
            "episode_id": row["episode_id"],
            "excluded": None,
            "exclusion_status": "requires_manual_audit",
            "criterion": "episode never reaches the correction opportunity because of parser or environment-invalid failures",
            "evidence": {
                "correction_opportunities": row.get("correction_opportunities", 0),
                "parse_failures": parser_failures,
                "parser_error_type_counts": row.get("parser_error_type_counts", {}),
                "environment_invalid_action_count": invalid_actions,
                "relevant_call_orders": [call for call in relevant_calls if call is not None],
            },
            "reason": "The technical blockers are observable, but their causal role in preventing the correction opportunity requires manual audit.",
        }
    return {
        "episode_id": row["episode_id"],
        "excluded": False,
        "exclusion_status": "not_excluded",
        "criterion": None,
        "evidence": {"task_success": bool(row.get("success"))},
    }


def _initialize(output_dir, manifest, raw, metadata, secret):
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "manifest_snapshot.json", manifest, secret)
    _write_json(output_dir / "execution_metadata.json", metadata, secret)
    for name in REQUIRED_OUTPUT_FILES + EXTRA_OUTPUT_FILES:
        path = output_dir / name
        if name.endswith(".jsonl") or name.endswith(".csv"):
            path.write_text("")
        elif not path.exists():
            _write_json(path, [] if name in {"grouped_metrics.json", "exclusion_decisions.json"} else {}, secret)
    assert hashlib.sha256(raw).hexdigest() == metadata["manifest_sha256"]


def _persist(
    output_dir, rows, traces, partial_records, client, exclusions, metrics, metadata, secret
):
    audits = []
    for row in rows:
        identity = {key: row[key] for key in ("episode_id", "scenario_id", "policy", "seed")}
        audits.extend({**identity, "call": call} for call in row.get("call_audit", []))
    for partial in partial_records:
        identity = {
            "episode_id": partial["episode_id"],
            "scenario_id": partial["scenario"],
            "policy": partial["policy"],
            "seed": partial["seed"],
            "completed": False,
        }
        audits.extend({**identity, "call": call} for call in partial.get("call_audit", []))
    _write_jsonl(output_dir / "episode_summaries.jsonl", rows, secret)
    _write_jsonl(output_dir / "complete_environment_traces.jsonl", traces, secret)
    _write_jsonl(output_dir / "complete_call_audit.jsonl", audits, secret)
    _write_jsonl(output_dir / "partial_episode_records.jsonl", partial_records, secret)
    _write_json(output_dir / "api_call_schedule.json", client.scheduler.records, secret)
    _write_json(output_dir / "api_failure_audit.json", [r for r in client.api_audit if not r["success"]], secret)
    _write_json(output_dir / "corrected_metrics.json", metrics, secret)
    grouped = [
        {
            "scenario": row["scenario_id"],
            "policy": row["policy"],
            "seed": row["seed"],
            **metrics[row["episode_id"]],
        }
        for row in rows
    ]
    _write_json(output_dir / "grouped_metrics.json", grouped, secret)
    if grouped:
        flat = [
            {**item, "objective_task_milestones_reached": json.dumps(item["objective_task_milestones_reached"], sort_keys=True),
             "deprecated_metric": json.dumps(item["deprecated_metric"], sort_keys=True)}
            for item in grouped
        ]
        with (output_dir / "grouped_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    parser_errors = Counter()
    parsed_actions = Counter()
    for row in rows:
        parser_errors.update(row.get("parser_error_type_counts", {}))
        parsed_actions.update(row.get("parsed_action_counts", {}))
    _write_json(output_dir / "parser_error_distribution.json", dict(parser_errors), secret)
    _write_json(output_dir / "parsed_action_distribution.json", dict(parsed_actions), secret)
    _write_json(output_dir / "fallback_summary.json", {
        row["episode_id"]: {
            "fallback_action_count": row.get("fallback_actions", 0),
            "budget_cap_events": row.get("budget_cap_hits", 0),
        } for row in rows
    }, secret)
    _write_json(output_dir / "token_usage_summary.json", {
        "input_tokens": sum(row.get("input_tokens", 0) for row in rows),
        "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
        "total_tokens": sum(row.get("total_tokens", 0) for row in rows),
    }, secret)
    latencies = [value for row in rows for value in row.get("latency_per_call_sec", [])]
    _write_json(output_dir / "latency_summary.json", {
        "count": len(latencies),
        "minimum_seconds": min(latencies) if latencies else None,
        "maximum_seconds": max(latencies) if latencies else None,
        "mean_seconds": sum(latencies) / len(latencies) if latencies else None,
        "values_seconds": latencies,
    }, secret)
    _write_json(output_dir / "exclusion_decisions.json", exclusions, secret)
    _write_json(output_dir / "api_failure_summary.json", {
        "attempts": len(client.api_audit),
        "successful_responses": sum(r["success"] for r in client.api_audit),
        "failures": sum(not r["success"] for r in client.api_audit),
    }, secret)
    _write_json(output_dir / "final_report.json", {
        "run_status": metadata["run_status"],
        "episodes_completed": len(rows),
        "corrected_metrics": metrics,
        "exclusions": exclusions,
        "rate_limit_abort": metadata.get("rate_limit_abort"),
        "parse_success_rate": {"deprecated": True, "replacement": "parser_success_given_api_response"},
    }, secret)
    _write_json(output_dir / "execution_metadata.json", metadata, secret)


def execute(
    output_root=OUTPUT_ROOT,
    client_factory=GroqClient,
    episode_runner=run_episode,
    clock=time.monotonic,
    sleep=time.sleep,
    timestamp=None,
    manifest_path=MANIFEST_PATH,
):
    manifest, raw = load_manifest(manifest_path)
    matrix = execution_matrix(manifest)
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_root) / f'{manifest["protocol_version"]}_{stamp}'
    secret = os.getenv("GROQ_API_KEY")
    metadata = {
        "protocol_version": manifest["protocol_version"],
        "git_sha": _git_sha(),
        "timestamp_utc": stamp,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "v1_manifest_sha256": hashlib.sha256(V1_MANIFEST_PATH.read_bytes()).hexdigest(),
        "credential_present": bool(secret),
        "runner_environment": _safe_environment(),
        "run_status": "initialized",
    }
    _initialize(output_dir, manifest, raw, metadata, secret)
    if not secret:
        metadata["run_status"] = "missing_credential"
        _write_json(output_dir / "execution_metadata.json", metadata)
        return 2, output_dir

    scheduler = GlobalCallScheduler(
        manifest["global_minimum_inter_api_call_seconds"], clock=clock, sleep=sleep
    )
    client = RateAwareClient(client_factory(model=manifest["model"]), scheduler, secret)
    try:
        connectivity = client.generate(
            CONNECTIVITY_PROMPT,
            model=manifest["model"],
            temperature=manifest["temperature"],
            top_p=manifest["top_p"],
            max_tokens=manifest["max_tokens"],
        )
    except RateLimitAbort as abort:
        metadata.update(run_status="rate_limit_abort", rate_limit_abort=abort.audit_record)
        _write_json(output_dir / "connectivity_result.json", {"success": False, **abort.audit_record}, secret)
        _persist(output_dir, [], [], [], client, [], {}, metadata, secret)
        return 3, output_dir
    except Exception as error:
        metadata["run_status"] = "connectivity_failed"
        _write_json(output_dir / "connectivity_result.json", {
            "success": False, "error_type": type(error).__name__,
            "sanitized_error_message": _redact(str(error), secret)[:500],
        }, secret)
        _persist(output_dir, [], [], [], client, [], {}, metadata, secret)
        return 4, output_dir
    _write_json(output_dir / "connectivity_result.json", {
        "success": True, "transport": client.transport,
        "raw_response_excerpt": _redact(str(connectivity)[:500], secret),
        "call_order": client.api_audit[-1]["call_order"],
    }, secret)

    args = SimpleNamespace(
        model=manifest["model"], temperature=manifest["temperature"],
        top_p=manifest["top_p"], max_tokens=manifest["max_tokens"],
        max_steps=manifest["max_steps"], max_llm_calls=manifest["max_llm_calls_per_episode"],
    )
    rows, traces, partial_records, exclusions, metrics = [], [], [], [], {}
    aborted = None
    for index, (scenario, policy_name, seed) in enumerate(matrix, start=1):
        episode_id = f"episode-{index:02d}-{scenario}-{policy_name}-seed-{seed}"
        client.set_context(
            phase="episode", episode_id=episode_id, scenario=scenario,
            policy=policy_name, seed=seed,
        )
        try:
            row = episode_runner(
                scenario, seed, POLICY_MAP[policy_name], client, args,
                trace_sink=traces, episode_id=episode_id, partial_sink=partial_records,
            )
        except RateLimitAbort as abort:
            aborted = abort.audit_record
            break
        except Exception as error:
            exclusions.append({
                "episode_id": episode_id, "excluded": True,
                "exclusion_status": "excluded",
                "criterion": "world-state invariant failure" if isinstance(error, AssertionError) else "missing call-audit or episode-summary artifact",
                "evidence": {"error_type": type(error).__name__, "message": _redact(str(error), secret)[:500]},
            })
            continue
        row["episode_id"] = episode_id
        row.pop("parse_success_rate", None)
        rows.append(row)
        episode_api = [record for record in client.api_audit if record.get("episode_id") == episode_id]
        metrics[episode_id] = corrected_metrics(row, episode_api, traces)
        exclusions.append(_exclusion(row, episode_api))

    if aborted:
        metadata.update(run_status="rate_limit_abort", rate_limit_abort=aborted)
    else:
        metadata["run_status"] = "completed" if len(rows) == len(matrix) else "episode_failure"
    metadata.update(episodes_planned=len(matrix), episodes_completed=len(rows))
    _persist(
        output_dir, rows, traces, partial_records, client, exclusions, metrics, metadata, secret
    )
    return (5 if aborted else 0 if len(rows) == len(matrix) else 6), output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    status, output_dir = execute(output_root=args.output_root)
    print(json.dumps({"status_code": status, "output_directory": str(output_dir)}))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
