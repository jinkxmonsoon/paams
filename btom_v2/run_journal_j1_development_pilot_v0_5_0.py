"""Execute and descriptively analyze the frozen Journal J1 development pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("journal_j1_development_pilot_manifest_v0_5_0.json")
MODEL = "openai/gpt-oss-20b"
PAIR_SEED_LITERAL = "journal-j1-development-pilot-pair-seed-v0.5.0"
ORDER_LITERAL = "journal-j1-development-pilot-request-order-v0.5.0"
MAX_REQUESTS = 144
DELAY_SECONDS = 20
BEHAVIORAL_RETRIES = 0
TIMEOUT_SECONDS = 120
ALLOWED_ACTIONS = ("CONTINUE_TASK", "SEND_CORRECTION")
PROMPT_FREEZE_HASHES = {
    "journal_j1_candidate_realizations_v0_4_2.json": "44d5ee7a50db0b4a85c992329a5aa1cc742f3566dbf0d95bd1e7e3a6d1771664",
    "journal_j1_development_realizations_v0_4_2.json": "d45d3c49d063ded04210b9cdb4762bd5da16814a4792a2c5ba9767eb8b5da457",
    "journal_j1_development_prompts_v0_4_2.json": "5dfc758825588afcc2cab1e576e24e3160df5237ed9863f5b4cb1e2511e1ac19",
    "journal_j1_prompt_audit_v0_4_2.json": "8b55111eebcb4ae618ea5f84389796269fd0d56082320ef48c4424fb9f64879f",
    "journal_j1_prompt_audit_v0_4_2.csv": "9e9b7b5ef66fd1aa44d187c91246658092b25ec897caefb61493f8db26f20be8",
    "tokenizer_candidate_table_v0_4_2.json": "989eac08057322c89216a7aff3464e926315cffd79a2d54261b834a44de32167",
}
OUTPUTS = (
    "workflow_status.json", "journal_j1_development_pilot_manifest_snapshot_v0_5_0.json",
    "journal_j1_development_pilot_request_order_v0_5_0.json", "journal_j1_development_pilot_prompt_freeze_audit_v0_5_0.json",
    "journal_j1_development_pilot_call_records_v0_5_0.jsonl", "journal_j1_development_pilot_raw_api_responses_v0_5_0.jsonl",
    "journal_j1_development_pilot_behavioral_results_v0_5_0.jsonl", "journal_j1_development_pilot_cell_rates_v0_5_0.csv",
    "journal_j1_development_pilot_variant_estimands_v0_5_0.csv", "journal_j1_development_pilot_subgroup_diagnostics_v0_5_0.json",
    "journal_j1_development_pilot_technical_coverage_v0_5_0.json", "journal_j1_development_pilot_seed_fingerprint_diagnostics_v0_5_0.json",
    "journal_j1_development_pilot_usage_latency_v0_5_0.json", "journal_j1_development_pilot_summary_v0_5_0.json",
)


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def pair_seed(variant_id: str, mr: int, mi: int) -> int:
    digest = hashlib.sha256(f"{PAIR_SEED_LITERAL}|{variant_id}|{mr}|{mi}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") % 2147483647
    return value or 1


def request_order(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(p["variant_id"], p["F"], p["M_R"], p["M_I"]): p for p in prompts}
    variants = sorted({p["variant_id"] for p in prompts}, key=lambda v: hashlib.sha256(f"{ORDER_LITERAL}|{v}".encode()).digest())
    rows = []
    for variant in variants:
        pairs = sorted(((mr, mi) for mr in (0, 1) for mi in (0, 1)), key=lambda pair: hashlib.sha256(f"{ORDER_LITERAL}|{variant}|{pair[0]}|{pair[1]}".encode()).digest())
        for position, (mr, mi) in enumerate(pairs, 1):
            framing_order = ("R", "B") if position in (1, 3) else ("B", "R")
            seed = pair_seed(variant, mr, mi)
            for framing in framing_order:
                prompt = by_key[variant, framing, mr, mi]
                rows.append({"ordinal": len(rows) + 1, "variant_id": variant, "archetype": prompt["archetype"], "difficulty": prompt["difficulty"], "F": framing, "M_R": mr, "M_I": mi, "prompt_sha256": hashlib.sha256(prompt["prompt_text"].encode()).hexdigest(), "requested_seed": seed})
    return rows


def response_schema() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "j1_development_action", "strict": True, "schema": {"type": "object", "properties": {"action": {"type": "string", "enum": list(ALLOWED_ACTIONS)}}, "required": ["action"], "additionalProperties": False}}}


def request_body(prompt_text: str, seed: int) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt_text}], "temperature": 0, "top_p": 1, "max_completion_tokens": 1024, "reasoning_effort": "low", "include_reasoning": False, "response_format": response_schema(), "stream": False, "seed": seed}


def strict_parse(content: str | None) -> str | None:
    if not content:
        return None
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {"action"} or value["action"] not in ALLOWED_ACTIONS:
        return None
    return value["action"]


def sanitize(message: Any, secret: str = "") -> str:
    text = str(message or "")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    for marker in ("GROQ_API_KEY", "Authorization", "Bearer "):
        text = text.replace(marker, "[REDACTED]")
    return text[:500]


class Client:
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("missing GROQ_API_KEY")
        self.api_key = api_key

    def call(self, prompt_text: str, seed: int) -> tuple[bool, int | None, dict[str, Any] | None, str | None, float]:
        body = request_body(prompt_text, seed)
        request = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return True, response.status, json.loads(response.read()), None, time.time() - started
        except urllib.error.HTTPError as exc:
            return False, exc.code, None, sanitize(exc.read().decode(errors="replace"), self.api_key), time.time() - started
        except Exception as exc:
            return False, None, None, sanitize(exc, self.api_key), time.time() - started


def validate_prompt_freeze(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    hashes = {name: {"expected": expected, "actual": hashlib.sha256((directory / name).read_bytes()).hexdigest()} for name, expected in PROMPT_FREEZE_HASHES.items()}
    if any(row["expected"] != row["actual"] for row in hashes.values()):
        raise RuntimeError("v0.4.2 prompt freeze hash mismatch")
    audit = json.loads((directory / "journal_j1_prompt_audit_v0_4_2.json").read_text())
    required = {"pilot_ready": True, "selected_candidate_id": 1, "selected_R_label": "Current partner record:", "selected_B_label": "Current partner belief:", "matched_pair_count": 72, "normalized_pair_equality": 72, "equal_harmony_token_count_pairs": 72, "max_harmony_difference": 0, "surface_cue_count": 0}
    if any(audit.get(key) != value for key, value in required.items()):
        raise RuntimeError("v0.4.2 prompt audit mismatch")
    prompt_artifact = json.loads((directory / "journal_j1_development_prompts_v0_4_2.json").read_text())
    prompts = prompt_artifact["prompts"]
    if len(prompts) != 144:
        raise RuntimeError("v0.4.2 prompt count mismatch")
    return hashes, prompts, audit


def _rate(values: list[int]) -> float:
    return sum(values) / len(values)


def discrimination(cell_rates: dict[tuple[str, int, int], float], technical_pass: bool) -> dict[str, Any]:
    if not technical_pass:
        return {"record_relevance_pooled": None, "record_relevance_MI0": None, "record_relevance_MI1": None, "record_irrelevant_MR0": None, "record_irrelevant_MR1": None, "criteria": {}, "scenario_discrimination_pass": False}
    pooled = ((cell_rates["R", 1, 0] + cell_rates["R", 1, 1]) / 2) - ((cell_rates["R", 0, 0] + cell_rates["R", 0, 1]) / 2)
    values = {"record_relevance_pooled": pooled, "record_relevance_MI0": cell_rates["R", 1, 0] - cell_rates["R", 0, 0], "record_relevance_MI1": cell_rates["R", 1, 1] - cell_rates["R", 0, 1], "record_irrelevant_MR0": cell_rates["R", 0, 1] - cell_rates["R", 0, 0], "record_irrelevant_MR1": cell_rates["R", 1, 1] - cell_rates["R", 1, 0]}
    criteria = {"record_relevance_pooled_gte_0_25": values["record_relevance_pooled"] >= 0.25, "record_relevance_MI0_gt_0": values["record_relevance_MI0"] > 0, "record_relevance_MI1_gt_0": values["record_relevance_MI1"] > 0, "record_irrelevant_MR0_abs_lte_0_25": abs(values["record_irrelevant_MR0"]) <= 0.25, "record_irrelevant_MR1_abs_lte_0_25": abs(values["record_irrelevant_MR1"]) <= 0.25}
    return {**values, "criteria": criteria, "scenario_discrimination_pass": all(criteria.values())}


def analyze(records: list[dict[str, Any]], technical_pass: bool) -> dict[str, Any]:
    complete = [r for r in records if r["complete"]]
    cell_values = defaultdict(list)
    for row in complete:
        cell_values[row["F"], row["M_R"], row["M_I"]].append(1 if row["action"] == "SEND_CORRECTION" else 0)
    cell_rates = {key: _rate(values) for key, values in cell_values.items()} if complete else {}
    gate = discrimination(cell_rates, technical_pass)
    framing_metrics = {}
    for framing in "RB":
        rows = [r for r in complete if r["F"] == framing]
        necessary = [1 if r["action"] == "SEND_CORRECTION" else 0 for r in rows if r["M_R"] == 1]
        unnecessary = [1 if r["action"] == "SEND_CORRECTION" else 0 for r in rows if r["M_R"] == 0]
        framing_metrics[framing] = {
            "necessary_correction_rate": _rate(necessary) if necessary else None,
            "unnecessary_correction_rate": _rate(unnecessary) if unnecessary else None,
            "missed_necessary_rate": 1 - _rate(necessary) if necessary else None,
            "gold_action_accuracy": _rate([int((r["action"] == "SEND_CORRECTION") == bool(r["M_R"])) for r in rows]) if rows else None,
            "communication_rate": _rate([int(r["action"] == "SEND_CORRECTION") for r in rows]) if rows else None,
        }
    variant_estimands = []
    if technical_pass:
        by_variant = defaultdict(dict)
        for row in complete:
            by_variant[row["variant_id"]][row["F"], row["M_R"], row["M_I"]] = 1 if row["action"] == "SEND_CORRECTION" else 0
        for variant, c in sorted(by_variant.items()):
            a = {(f, mr): (c[f, mr, 0] + c[f, mr, 1]) / 2 for f in "RB" for mr in (0, 1)}
            q = {(f, mi): (c[f, 0, mi] + c[f, 1, mi]) / 2 for f in "RB" for mi in (0, 1)}
            d = (a["B", 1] - a["B", 0]) - (a["R", 1] - a["R", 0])
            e = (q["B", 1] - q["B", 0]) - (q["R", 1] - q["R", 0])
            m = statistics.mean(c["B", mr, mi] for mr in (0, 1) for mi in (0, 1)) - statistics.mean(c["R", mr, mi] for mr in (0, 1) for mi in (0, 1))
            variant_estimands.append({"variant_id": variant, "D_i": d, "E_i": e, "S_i": d - e, "M_i": m})
    deltas = {name: statistics.mean(row[field] for row in variant_estimands) if variant_estimands else None for name, field in (("Delta_role", "D_i"), ("Delta_irrelevant", "E_i"), ("Delta_specificity", "S_i"), ("Delta_communication_prior", "M_i"))}
    return {"cell_rates": cell_rates, "framing_metrics": framing_metrics, "gate": gate, "variant_estimands": variant_estimands, "deltas": deltas}


def execute(output_dir: Path, prompt_freeze_dir: Path, sleep: Callable[[float], None] = time.sleep, client_factory: Callable[[str], Any] = Client, environ: dict[str, str] | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in OUTPUTS:
        (output_dir / name).write_text("" if name.endswith(".jsonl") or name.endswith(".csv") else "{}\n")
    manifest = json.loads(MANIFEST_PATH.read_text())
    dump(output_dir / "journal_j1_development_pilot_manifest_snapshot_v0_5_0.json", manifest)
    records = []
    try:
        hashes, prompts, audit = validate_prompt_freeze(prompt_freeze_dir)
        order = request_order(prompts)
        if order != manifest["frozen_request_order"]:
            raise RuntimeError("request order differs from manifest")
        dump(output_dir / "journal_j1_development_pilot_prompt_freeze_audit_v0_5_0.json", {"passed": True, "hashes": hashes, "audit": audit, "client_constructed": False})
        dump(output_dir / "journal_j1_development_pilot_request_order_v0_5_0.json", order)
        prompt_by_hash = {hashlib.sha256(p["prompt_text"].encode()).hexdigest(): p["prompt_text"] for p in prompts}
        env = os.environ if environ is None else environ
        api_key = env.get("GROQ_API_KEY", "")
        client = client_factory(api_key)  # constructed only after every pre-API gate
        for row in order:
            if records:
                sleep(DELAY_SECONDS)
            ok, status, payload, error, latency = client.call(prompt_by_hash[row["prompt_sha256"]], row["requested_seed"])
            choice = (payload.get("choices") or [{}])[0] if payload else {}
            content = (choice.get("message") or {}).get("content")
            action = strict_parse(content) if ok else None
            returned_seed = ((payload or {}).get("x_groq") or {}).get("seed")
            complete = bool(ok and choice.get("finish_reason") == "stop" and content and action in ALLOWED_ACTIONS)
            records.append({**row, "returned_seed": returned_seed, "seed_matches": returned_seed == row["requested_seed"] if returned_seed is not None else None, "system_fingerprint": (payload or {}).get("system_fingerprint"), "service_tier": (payload or {}).get("service_tier"), "http_status": status, "http_success": ok, "finish_reason": choice.get("finish_reason"), "nonempty_content": bool(content), "strict_parse_success": action is not None, "legal_action": action in ALLOWED_ACTIONS if action else False, "action": action, "latency": latency, "usage": (payload or {}).get("usage"), "raw_api_response": payload, "sanitized_api_error": sanitize(error, api_key), "complete": complete, "fallback_action": False})
        technical_pass = len(records) == MAX_REQUESTS and all(r["complete"] for r in records)
        analysis = analyze(records, technical_pass)
        jsonl(output_dir / "journal_j1_development_pilot_call_records_v0_5_0.jsonl", records)
        jsonl(output_dir / "journal_j1_development_pilot_raw_api_responses_v0_5_0.jsonl", [{"ordinal": r["ordinal"], "raw_api_response": r["raw_api_response"]} for r in records])
        jsonl(output_dir / "journal_j1_development_pilot_behavioral_results_v0_5_0.jsonl", [{k: r[k] for k in ("ordinal", "variant_id", "archetype", "difficulty", "F", "M_R", "M_I", "complete", "action", "fallback_action")} for r in records])
        with (output_dir / "journal_j1_development_pilot_cell_rates_v0_5_0.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n"); writer.writerow(("F", "M_R", "M_I", "communication_rate")); writer.writerows((*key, value) for key, value in sorted(analysis["cell_rates"].items()))
        with (output_dir / "journal_j1_development_pilot_variant_estimands_v0_5_0.csv").open("w", newline="") as handle:
            fields = ("variant_id", "D_i", "E_i", "S_i", "M_i"); writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(analysis["variant_estimands"])
        subgroup = {dimension: {value: dict(Counter(r["action"] for r in records if r[dimension] == value and r["complete"])) for value in sorted({r[dimension] for r in records})} for dimension in ("difficulty", "archetype")}
        dump(output_dir / "journal_j1_development_pilot_subgroup_diagnostics_v0_5_0.json", subgroup)
        dump(output_dir / "journal_j1_development_pilot_technical_coverage_v0_5_0.json", {"attempted_calls": len(records), "complete_calls": sum(r["complete"] for r in records), "api_failures": sum(not r["http_success"] for r in records), "parse_failures": sum(r["http_success"] and not r["strict_parse_success"] for r in records), "technical_pass": technical_pass})
        dump(output_dir / "journal_j1_development_pilot_seed_fingerprint_diagnostics_v0_5_0.json", {"records": [{k: r[k] for k in ("ordinal", "requested_seed", "returned_seed", "seed_matches", "system_fingerprint", "service_tier")} for r in records]})
        latencies = [r["latency"] for r in records]; usage = {key: sum((r["usage"] or {}).get(key, 0) for r in records) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        dump(output_dir / "journal_j1_development_pilot_usage_latency_v0_5_0.json", {"usage_totals": usage, "latency": {"mean": statistics.mean(latencies), "minimum": min(latencies), "maximum": max(latencies)}})
        summary = {"model": MODEL, "requested_calls": MAX_REQUESTS, "attempted_calls": len(records), "complete_calls": sum(r["complete"] for r in records), "api_failures": sum(not r["http_success"] for r in records), "parse_failures": sum(r["http_success"] and not r["strict_parse_success"] for r in records), "behavioral_retries": 0, "fallback_actions": 0, "technical_pass": technical_pass, "framing_metrics": analysis["framing_metrics"], **analysis["gate"], **analysis["deltas"], "ready_for_confirmatory_rendering": technical_pass and analysis["gate"]["scenario_discrimination_pass"], "behavioral_observations": sum(r["complete"] for r in records), "confirmatory_prompts_generated": False, "icaart_reopened": False}
        dump(output_dir / "journal_j1_development_pilot_summary_v0_5_0.json", summary); dump(output_dir / "workflow_status.json", summary)
        return 0 if technical_pass else 1
    except Exception as exc:
        status = {"model": MODEL, "requested_calls": MAX_REQUESTS, "attempted_calls": len(records), "complete_calls": sum(r.get("complete", False) for r in records), "api_failures": sum(not r.get("http_success", False) for r in records), "parse_failures": sum(r.get("http_success", False) and not r.get("strict_parse_success", False) for r in records), "behavioral_retries": 0, "fallback_actions": 0, "technical_pass": False, "scenario_discrimination_pass": False, "ready_for_confirmatory_rendering": False, "behavioral_observations": sum(r.get("complete", False) for r in records), "confirmatory_prompts_generated": False, "icaart_reopened": False, "error": sanitize(exc)}
        dump(output_dir / "workflow_status.json", status); dump(output_dir / "journal_j1_development_pilot_summary_v0_5_0.json", status)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--prompt-freeze-dir", required=True, type=Path)
    args = parser.parse_args()
    return execute(args.output_dir, args.prompt_freeze_dir)


if __name__ == "__main__":
    raise SystemExit(main())
