"""Transport-only repair for the frozen Journal J1 v0.5.0 pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import run_journal_j1_development_pilot_v0_5_0 as frozen

MANIFEST_PATH = frozen.MANIFEST_PATH
USER_AGENT = "Mozilla/5.0 (compatible; BToM-MAS/1.0.0; +https://github.com/jinkxmonsoon/paams)"
PREDECESSOR = {
    "predecessor_version": "v0.5.0",
    "predecessor_remote_sha": "07bbcce3ada7a50e22e33613969f6653061bbdf1",
    "predecessor_actions_run": 34664024173,
    "predecessor_artifact_id": 10288762441,
    "predecessor_failure": "144/144 HTTP 403 Cloudflare 1010 browser_signature_banned",
    "predecessor_behavioral_observations": 0,
    "transport_change": "add established explicit User-Agent and fail-fast preflight",
    "scientific_protocol_changed": False,
}
OUTPUTS = (
    "workflow_status.json",
    "journal_j1_development_pilot_manifest_snapshot_v0_5_1.json",
    "journal_j1_development_pilot_request_order_v0_5_1.json",
    "journal_j1_development_pilot_prompt_freeze_audit_v0_5_1.json",
    "journal_j1_development_pilot_call_records_v0_5_1.jsonl",
    "journal_j1_development_pilot_raw_api_responses_v0_5_1.jsonl",
    "journal_j1_development_pilot_behavioral_results_v0_5_1.jsonl",
    "journal_j1_development_pilot_summary_v0_5_1.json",
    "journal_j1_development_pilot_usage_latency_v0_5_1.json",
)


def request_body(prompt_text: str, seed: int) -> dict[str, Any]:
    """Return the authoritative v0.5.0 model-facing request unchanged."""
    return frozen.request_body(prompt_text, seed)


def classify_transport(http_status: int | None, payload: Any = None, error: Any = None) -> str | None:
    combined = " ".join((json.dumps(payload, sort_keys=True) if payload is not None else "", str(error or ""))).lower()
    if http_status == 403 and ("1010" in combined or "browser_signature_banned" in combined):
        return "cloudflare_1010_client_signature"
    if http_status is None:
        return "transport_error"
    if not 200 <= http_status < 300:
        return "http_error"
    return None


class Client:
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("missing GROQ_API_KEY")
        self.api_key = api_key

    def headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}

    def call(self, prompt_text: str, seed: int) -> tuple[bool, int | None, dict[str, Any] | None, str | None, float]:
        request = urllib.request.Request(self.endpoint, data=json.dumps(request_body(prompt_text, seed)).encode(), headers=self.headers(), method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=frozen.TIMEOUT_SECONDS) as response:
                return True, response.status, json.loads(response.read()), None, time.time() - started
        except urllib.error.HTTPError as exc:
            return False, exc.code, None, frozen.sanitize(exc.read().decode(errors="replace"), self.api_key), time.time() - started
        except Exception as exc:
            return False, None, None, frozen.sanitize(exc, self.api_key), time.time() - started


def parse_call(row: dict[str, Any], result: tuple[Any, ...], secret: str) -> dict[str, Any]:
    ok, status, payload, error, latency = result
    choice = (payload.get("choices") or [{}])[0] if payload else {}
    content = (choice.get("message") or {}).get("content")
    action = frozen.strict_parse(content) if ok else None
    returned_seed = ((payload or {}).get("x_groq") or {}).get("seed")
    complete = bool(ok and choice.get("finish_reason") == "stop" and content and action in frozen.ALLOWED_ACTIONS)
    return {**row, "returned_seed": returned_seed, "seed_matches": returned_seed == row["requested_seed"] if returned_seed is not None else None, "system_fingerprint": (payload or {}).get("system_fingerprint"), "service_tier": (payload or {}).get("service_tier"), "http_status": status, "http_success": bool(ok), "finish_reason": choice.get("finish_reason"), "nonempty_content": bool(content), "strict_parse_success": action is not None, "legal_action": action in frozen.ALLOWED_ACTIONS if action else False, "action": action, "latency": latency, "usage": (payload or {}).get("usage"), "raw_api_response": payload, "sanitized_api_error": frozen.sanitize(error, secret), "complete": complete, "fallback_action": False, "transport_classification": classify_transport(status, payload, error)}


def status_payload(records: list[dict[str, Any]], analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    technical_pass = len(records) == frozen.MAX_REQUESTS and all(row["complete"] for row in records)
    classifications = [row["transport_classification"] for row in records if row["transport_classification"]]
    base = {**PREDECESSOR, "model": frozen.MODEL, "requested_calls": frozen.MAX_REQUESTS, "attempted_calls": len(records), "complete_calls": sum(row["complete"] for row in records), "api_failures": sum(not row["http_success"] for row in records), "parse_failures": sum(row["http_success"] and not row["strict_parse_success"] for row in records), "behavioral_retries": 0, "fallback_actions": 0, "transport_classification": classifications[0] if classifications else None, "transport_failure": bool(classifications), "technical_pass": technical_pass, "scenario_discrimination_pass": False, "ready_for_confirmatory_rendering": False, "behavioral_observations": sum(row["complete"] for row in records), "confirmatory_prompts_generated": False, "icaart_reopened": False}
    if technical_pass and analysis is not None:
        base.update({"framing_metrics": analysis["framing_metrics"], **analysis["gate"], **analysis["deltas"], "ready_for_confirmatory_rendering": analysis["gate"]["scenario_discrimination_pass"]})
    return base


def execute(output_dir: Path, prompt_freeze_dir: Path, sleep: Callable[[float], None] = time.sleep, client_factory: Callable[[str], Any] = Client, environ: dict[str, str] | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in OUTPUTS:
        (output_dir / name).write_text("" if name.endswith(".jsonl") else "{}\n")
    manifest = json.loads(MANIFEST_PATH.read_text())
    frozen.dump(output_dir / "journal_j1_development_pilot_manifest_snapshot_v0_5_1.json", manifest)
    records: list[dict[str, Any]] = []
    try:
        hashes, prompts, audit = frozen.validate_prompt_freeze(prompt_freeze_dir)
        order = frozen.request_order(prompts)
        if order != manifest["frozen_request_order"]:
            raise RuntimeError("request order differs from authoritative v0.5.0 manifest")
        frozen.dump(output_dir / "journal_j1_development_pilot_prompt_freeze_audit_v0_5_1.json", {"passed": True, "hashes": hashes, "audit": audit, **PREDECESSOR, "client_constructed": False})
        frozen.dump(output_dir / "journal_j1_development_pilot_request_order_v0_5_1.json", order)
        prompt_by_hash = {hashlib.sha256(p["prompt_text"].encode()).hexdigest(): p["prompt_text"] for p in prompts}
        environment = os.environ if environ is None else environ
        secret = environment.get("GROQ_API_KEY", "")
        client = client_factory(secret)
        for row in order:
            if records:
                sleep(frozen.DELAY_SECONDS)
            record = parse_call(row, client.call(prompt_by_hash[row["prompt_sha256"]], row["requested_seed"]), secret)
            records.append(record)
            # The first frozen call is the preflight and is retained if complete.
            # Any incomplete preflight stops the batch without a second attempt.
            if row["ordinal"] == 1 and not record["complete"]:
                break
        technical_pass = len(records) == frozen.MAX_REQUESTS and all(row["complete"] for row in records)
        analysis = frozen.analyze(records, technical_pass)
        frozen.jsonl(output_dir / "journal_j1_development_pilot_call_records_v0_5_1.jsonl", records)
        frozen.jsonl(output_dir / "journal_j1_development_pilot_raw_api_responses_v0_5_1.jsonl", [{"ordinal": row["ordinal"], "raw_api_response": row["raw_api_response"]} for row in records])
        frozen.jsonl(output_dir / "journal_j1_development_pilot_behavioral_results_v0_5_1.jsonl", [{key: row[key] for key in ("ordinal", "variant_id", "archetype", "difficulty", "F", "M_R", "M_I", "complete", "action", "fallback_action", "transport_classification")} for row in records])
        latencies = [row["latency"] for row in records]
        usage = {key: sum((row["usage"] or {}).get(key, 0) for row in records) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        frozen.dump(output_dir / "journal_j1_development_pilot_usage_latency_v0_5_1.json", {"usage_totals": usage, "latency": {"mean": statistics.mean(latencies), "minimum": min(latencies), "maximum": max(latencies)}})
        status = status_payload(records, analysis)
        frozen.dump(output_dir / "journal_j1_development_pilot_summary_v0_5_1.json", status); frozen.dump(output_dir / "workflow_status.json", status)
        return 0 if technical_pass else 1
    except Exception as exc:
        status = {**status_payload(records), "error": frozen.sanitize(exc)}
        frozen.dump(output_dir / "journal_j1_development_pilot_summary_v0_5_1.json", status); frozen.dump(output_dir / "workflow_status.json", status)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--prompt-freeze-dir", required=True, type=Path)
    args = parser.parse_args()
    return execute(args.output_dir, args.prompt_freeze_dir)


if __name__ == "__main__":
    raise SystemExit(main())
