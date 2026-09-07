"""One-shot executor for the version-controlled real-LLM micro-pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .real_llm_clients import GroqClient, RealLLMClientError, RealLLMSetupError
from .runner_llm_real_smoke import POLICY_MAP, group, run_episode


MANIFEST_PATH = Path(__file__).with_name("real_llm_micro_pilot_manifest.json")
OUTPUT_ROOT = Path(__file__).with_name("outputs")
CONNECTIVITY_PROMPT = (
    'Return only:\n{"action":"move","target":"staging","message":"",'
    '"reason":"connectivity test"}'
)
EXPECTED_PROTOCOL = "btom-v2-functional-micro-pilot-1.0"
EXPECTED_SCENARIOS = (
    "C5b_costly_false_belief",
    "C7a_partner_belief_stale",
    "C7b_partner_belief_current",
)
EXPECTED_POLICIES = (
    "DeterministicBaseline",
    "LLMReactiveReal",
    "LLMBeliefStateReal",
    "LLMBToMReal",
)
EXPECTED_SEEDS = (0,)
REQUIRED_OUTPUT_FILES = (
    "manifest_snapshot.json",
    "execution_metadata.json",
    "connectivity_result.json",
    "episode_summaries.jsonl",
    "grouped_metrics.json",
    "grouped_metrics.csv",
    "complete_environment_traces.jsonl",
    "complete_call_audit.jsonl",
    "parser_error_distribution.json",
    "parsed_action_distribution.json",
    "api_failure_summary.json",
    "fallback_summary.json",
    "token_usage_summary.json",
    "latency_summary.json",
    "exclusion_decisions.json",
    "final_report.json",
)


def load_manifest(path=MANIFEST_PATH):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    assert manifest["protocol_version"] == EXPECTED_PROTOCOL
    assert manifest["frozen_before_real_execution"] is True
    assert tuple(manifest["scenarios"]) == EXPECTED_SCENARIOS
    assert tuple(manifest["policies"]) == EXPECTED_POLICIES
    assert tuple(manifest["seeds"]) == EXPECTED_SEEDS
    assert manifest["temperature"] == 0
    assert manifest["top_p"] == 1
    assert manifest["max_tokens"] == 96
    assert manifest["max_steps"] == 40
    assert manifest["max_llm_calls_per_episode"] == 40
    return manifest, raw


def execution_matrix(manifest):
    matrix = [
        (scenario, policy, seed)
        for scenario in manifest["scenarios"]
        for policy in manifest["policies"]
        for seed in manifest["seeds"]
    ]
    if len(matrix) != len(set(matrix)):
        raise ValueError("duplicate episode combination in frozen manifest")
    return matrix


def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _safe_environment():
    names = ("GITHUB_ACTIONS", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "RUNNER_OS", "RUNNER_ARCH")
    return {name: os.getenv(name) for name in names}


def _sanitize(value, secret):
    if isinstance(value, dict):
        return {key: _sanitize(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[REDACTED]")
    return value


def _write_json(path, value, secret=None):
    path.write_text(json.dumps(_sanitize(value, secret), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path, rows, secret=None):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_sanitize(row, secret), sort_keys=True) + "\n")


def _initialize_outputs(output_dir, manifest, raw_manifest, metadata, secret):
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "manifest_snapshot.json", manifest, secret)
    _write_json(output_dir / "execution_metadata.json", metadata, secret)
    placeholders = {
        "connectivity_result.json": {},
        "grouped_metrics.json": [],
        "parser_error_distribution.json": {},
        "parsed_action_distribution.json": {},
        "api_failure_summary.json": {},
        "fallback_summary.json": {},
        "token_usage_summary.json": {},
        "latency_summary.json": {},
        "exclusion_decisions.json": [],
        "final_report.json": {},
    }
    for filename, value in placeholders.items():
        _write_json(output_dir / filename, value, secret)
    for filename in ("episode_summaries.jsonl", "complete_environment_traces.jsonl", "complete_call_audit.jsonl"):
        (output_dir / filename).write_text("")
    (output_dir / "grouped_metrics.csv").write_text("")
    if hashlib.sha256(raw_manifest).hexdigest() != metadata["manifest_sha256"]:
        raise RuntimeError("manifest hash changed while initializing outputs")


def _connect(client_factory, manifest, secret):
    client = None
    try:
        client = client_factory(model=manifest["model"])
        raw = client.generate(
            CONNECTIVITY_PROMPT,
            model=manifest["model"],
            temperature=manifest["temperature"],
            top_p=manifest["top_p"],
            max_tokens=manifest["max_tokens"],
        )
    except (RealLLMClientError, RealLLMSetupError) as error:
        usage = getattr(client, "last_usage", None) or {}
        return client, {
            "transport": getattr(client, "transport", None),
            "success": False,
            "latency_sec": getattr(client, "last_latency_sec", None),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "error_type": getattr(error, "error_type", type(error).__name__),
            "error_message": _sanitize(
                getattr(error, "sanitized_error_message", str(error)), secret
            ),
            "raw_response_excerpt": None,
        }
    except Exception as error:  # Persist an unexpected one-shot connectivity failure.
        return client, {
            "transport": getattr(client, "transport", None),
            "success": False,
            "latency_sec": getattr(client, "last_latency_sec", None),
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "error_type": type(error).__name__,
            "error_message": _sanitize(str(error), secret)[:500],
            "raw_response_excerpt": None,
        }
    usage = getattr(client, "last_usage", None) or {}
    return client, {
        "transport": getattr(client, "transport", None),
        "success": True,
        "latency_sec": getattr(client, "last_latency_sec", None),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "error_type": None,
        "error_message": None,
        "raw_response_excerpt": _sanitize(str(raw)[:500], secret),
    }


def _exclusion(summary, runner_error=None):
    opportunity = bool(summary and summary.get("correction_opportunities"))
    timing = "after_relevant_action_opportunity" if opportunity else "before_relevant_action_opportunity"
    if runner_error:
        return {
            "excluded": True,
            "exclusion_status": "excluded",
            "criterion": "world-state invariant failure" if runner_error["type"] == "AssertionError" else "missing call-audit or episode-summary artifact",
            "evidence": runner_error,
            "failure_timing": timing,
        }
    if summary.get("api_failure_count", 0):
        return {
            "excluded": True,
            "exclusion_status": "excluded",
            "criterion": "episode has an API failure that prevents the planned action opportunity",
            "evidence": {"api_failure_count": summary["api_failure_count"]},
            "failure_timing": timing,
        }
    technical_block = summary.get("parse_failures", 0) or summary.get("environment_invalid_action_count", 0)
    if summary["scenario_id"].startswith("C7") and not opportunity and technical_block:
        return {
            "excluded": True,
            "exclusion_status": "excluded",
            "criterion": "episode never reaches the correction opportunity because of parser or environment-invalid failures",
            "evidence": {
                "parse_failures": summary.get("parse_failures", 0),
                "environment_invalid_action_count": summary.get("environment_invalid_action_count", 0),
            },
            "failure_timing": "before_relevant_action_opportunity",
        }
    return {
        "excluded": False,
        "exclusion_status": "not_excluded",
        "criterion": None,
        "evidence": {"task_success": summary.get("success")},
        "failure_timing": None,
    }


def _distribution(rows, key):
    total = Counter()
    for row in rows:
        total.update(row.get(key, {}))
    return dict(sorted(total.items()))


def _persist_results(output_dir, manifest, rows, traces, exclusions, secret):
    audits = []
    for row in rows:
        identity = {
            "episode_id": row["episode_id"],
            "scenario": row["scenario_id"],
            "policy": row["policy"],
            "seed": row["seed"],
        }
        audits.extend({**identity, "call": call} for call in row.get("call_audit", []))
    grouped = [group(rows, scenario, policy) for scenario in manifest["scenarios"] for policy in manifest["policies"]]
    parser_errors = _distribution(rows, "parser_error_type_counts")
    parsed_actions = _distribution(rows, "parsed_action_counts")
    latencies = [latency for row in rows for latency in row.get("latency_per_call_sec", [])]
    api_failures = {row["episode_id"]: row.get("api_failure_count", 0) for row in rows}
    fallback = {
        row["episode_id"]: {
            "fallback_actions": row.get("fallback_actions", 0),
            "budget_cap_hits": row.get("budget_cap_hits", 0),
        }
        for row in rows
    }
    tokens = {
        "input_tokens": sum(row.get("input_tokens", 0) for row in rows),
        "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
        "total_tokens": sum(row.get("total_tokens", 0) for row in rows),
        "by_episode": {
            row["episode_id"]: {
                "input_tokens": row.get("input_tokens", 0),
                "output_tokens": row.get("output_tokens", 0),
                "total_tokens": row.get("total_tokens", 0),
            }
            for row in rows
        },
    }
    latency_summary = {
        "count": len(latencies),
        "minimum_sec": min(latencies) if latencies else None,
        "maximum_sec": max(latencies) if latencies else None,
        "mean_sec": sum(latencies) / len(latencies) if latencies else None,
        "values_sec": latencies,
    }
    _write_jsonl(output_dir / "episode_summaries.jsonl", rows, secret)
    _write_jsonl(output_dir / "complete_environment_traces.jsonl", traces, secret)
    _write_jsonl(output_dir / "complete_call_audit.jsonl", audits, secret)
    _write_json(output_dir / "grouped_metrics.json", grouped, secret)
    with (output_dir / "grouped_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    _write_json(output_dir / "parser_error_distribution.json", parser_errors, secret)
    _write_json(output_dir / "parsed_action_distribution.json", parsed_actions, secret)
    _write_json(output_dir / "api_failure_summary.json", api_failures, secret)
    _write_json(output_dir / "fallback_summary.json", fallback, secret)
    _write_json(output_dir / "token_usage_summary.json", tokens, secret)
    _write_json(output_dir / "latency_summary.json", latency_summary, secret)
    _write_json(output_dir / "exclusion_decisions.json", exclusions, secret)
    final = {
        "episode_count": len(rows),
        "api_failures": sum(api_failures.values()),
        "parser_failures": sum(row.get("parse_failures", 0) for row in rows),
        "invalid_environment_actions": sum(row.get("environment_invalid_action_count", 0) for row in rows),
        "fallback_actions": sum(item["fallback_actions"] for item in fallback.values()),
        "budget_cap_events": sum(item["budget_cap_hits"] for item in fallback.values()),
        "valid_but_strategically_poor_actions": sum(row.get("valid_but_strategically_poor_actions") or 0 for row in rows),
        "successful_progression_actions": sum(row.get("progress_actions") or 0 for row in rows),
        "task_successes": sum(bool(row.get("success")) for row in rows),
        "exclusions": exclusions,
        "missing_times_preserved_as_null": True,
    }
    _write_json(output_dir / "final_report.json", final, secret)


def execute(output_root=OUTPUT_ROOT, client_factory=GroqClient, episode_runner=run_episode, now=None):
    manifest, raw_manifest = load_manifest()
    matrix = execution_matrix(manifest)
    timestamp = now or _timestamp()
    output_dir = Path(output_root) / f'{manifest["protocol_version"]}_{timestamp}'
    secret = os.getenv("GROQ_API_KEY")
    metadata = {
        "protocol_version": manifest["protocol_version"],
        "git_sha": _git_sha(),
        "timestamp_utc": timestamp,
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "credential_present": bool(secret),
        "python_version": platform.python_version(),
        "runner_environment": _safe_environment(),
        "execution_status": "initialized",
    }
    _initialize_outputs(output_dir, manifest, raw_manifest, metadata, secret)
    if not secret:
        metadata["execution_status"] = "missing_credential"
        _write_json(output_dir / "execution_metadata.json", metadata)
        _write_json(output_dir / "final_report.json", {"execution_status": "missing_credential"})
        return 2, output_dir

    client, connectivity = _connect(client_factory, manifest, secret)
    _write_json(output_dir / "connectivity_result.json", connectivity, secret)
    if not connectivity["success"]:
        metadata["execution_status"] = "connectivity_failed"
        _write_json(output_dir / "execution_metadata.json", metadata, secret)
        _write_json(output_dir / "final_report.json", {"execution_status": "connectivity_failed", "episodes_executed": 0}, secret)
        return 3, output_dir

    args = SimpleNamespace(
        model=manifest["model"],
        temperature=manifest["temperature"],
        top_p=manifest["top_p"],
        max_tokens=manifest["max_tokens"],
        max_steps=manifest["max_steps"],
        max_llm_calls=manifest["max_llm_calls_per_episode"],
    )
    rows, traces, exclusions = [], [], []
    for index, (scenario, policy_name, seed) in enumerate(matrix, start=1):
        episode_id = f"episode-{index:02d}-{scenario}-{policy_name}-seed-{seed}"
        try:
            row = episode_runner(
                scenario,
                seed,
                POLICY_MAP[policy_name],
                client,
                args,
                trace_sink=traces,
                episode_id=episode_id,
            )
        except Exception as error:
            runner_error = {"type": type(error).__name__, "message": _sanitize(str(error), secret)[:500]}
            exclusions.append({"episode_id": episode_id, **_exclusion(None, runner_error)})
            continue
        row["episode_id"] = episode_id
        rows.append(row)
        exclusions.append({"episode_id": episode_id, **_exclusion(row)})

    _persist_results(output_dir, manifest, rows, traces, exclusions, secret)
    metadata["execution_status"] = "completed" if len(rows) == len(matrix) else "episode_failure"
    metadata["episodes_planned"] = len(matrix)
    metadata["episodes_completed"] = len(rows)
    _write_json(output_dir / "execution_metadata.json", metadata, secret)
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"required output artifacts missing: {missing}")
    return (0 if len(rows) == len(matrix) else 4), output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    status, output_dir = execute(output_root=args.output_root)
    print(json.dumps({"execution_status_code": status, "output_directory": str(output_dir)}))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
