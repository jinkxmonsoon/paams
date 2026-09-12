"""Render and audit the record-only Journal J1 v0.6.0 holdout bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .journal_j1_record_holdout_scenarios_v0_6_0 import ROOT, ARCHETYPES, DIFFICULTIES, canonical, scenarios

FRAMING_LABEL = "Current partner record:"
SEED_LITERAL = "journal-j1-record-holdout-seed-v0.6.0"
SEQUENCE_LITERAL = "journal-j1-record-holdout-cell-sequence-v0.6.0"
ROUND_LITERAL = "journal-j1-record-holdout-round-order-v0.6.0"
HARMONY_STATUS = "pending_official_pre_execution_validation"
SEQUENCES = {
    "A": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "B": ((1, 0), (1, 1), (0, 0), (0, 1)),
    "C": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "D": ((1, 1), (0, 1), (1, 0), (0, 0)),
}
OUTPUTS = (
    "journal_j1_record_holdout_scenarios_v0_6_0.json",
    "journal_j1_record_holdout_prompts_v0_6_0.json",
    "journal_j1_record_holdout_manifest_v0_6_0.json",
    "journal_j1_record_holdout_audit_v0_6_0.json",
    "journal_j1_record_holdout_audit_v0_6_0.csv",
)
PRIOR_CANDIDATES = ROOT / "btom_v2/journal_j1_candidate_realizations_v0_4_0.json"
PRIOR_DEVELOPMENT = ROOT / "btom_v2/journal_j1_development_realizations_v0_4_0.json"


def edge_text(edge: dict[str, str]) -> str:
    return f"{edge['source']} {edge['relation']} {edge['target']}."


def seed(variant_id: str, mr: int, mi: int) -> int:
    digest = hashlib.sha256(f"{SEED_LITERAL}|{variant_id}|{mr}|{mi}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2147483647 or 1


def render(scenario: dict[str, Any], mr: int, mi: int) -> dict[str, Any]:
    v = scenario["model_visible_invariant"]
    relations = (v["q_relation"], v["r_relation"]) if v["partner_proposition_first"] else (v["r_relation"], v["q_relation"])
    record_values = ((v["q_relation"]["entity"], v["q_relation"]["false_value"] if mr else v["q_relation"]["true_value"]), (v["r_relation"]["entity"], v["r_relation"]["false_value"] if mi else v["r_relation"]["true_value"]))
    if not v["partner_proposition_first"]:
        record_values = tuple(reversed(record_values))
    actions = ("SEND_CORRECTION", "CONTINUE_TASK") if v["send_correction_first"] else ("CONTINUE_TASK", "SEND_CORRECTION")
    text = "\n".join([
        f"You are {v['focal_name']}, coordinating with {v['partner_name']} and {v['operator_name']}.",
        v["focal_task"], v["partner_assignment"], v["operator_assignment"], *v["isolation_rules"],
        "WORLD FACTS:", *(edge_text(edge) for relation in relations for edge in relation["visible_edges"]), *v["static_context"],
        FRAMING_LABEL, *(f"{entity}: {value}" for entity, value in record_values),
        "You have exactly one action for this turn.",
        "SEND_CORRECTION sends an update concerning the addressed partner's immediate assignment; it uses this turn and does not advance your task this turn.",
        "CONTINUE_TASK advances your task this turn and sends no update.",
        f"Available actions in order: {actions[0]}, {actions[1]}",
        'Return strict JSON only: {"action":"SEND_CORRECTION"} or {"action":"CONTINUE_TASK"}',
    ])
    return {"prompt_id": f"{scenario['variant_id']}:R:{mr}:{mi}", "variant_id": scenario["variant_id"], "archetype": scenario["archetype"], "difficulty": scenario["difficulty"], "F": "R", "M_R": mr, "M_I": mi, "prompt_text": text, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "world_state_sha256": hashlib.sha256(canonical(v)).hexdigest(), "gold_action": "SEND_CORRECTION" if mr else "CONTINUE_TASK", "requested_seed": seed(scenario["variant_id"], mr, mi), "valid_actions": ["SEND_CORRECTION", "CONTINUE_TASK"], "action_order": list(actions), "proposition_order": "partner_first" if v["partner_proposition_first"] else "operator_first"}


def request_order(variants: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    sorted_variants = sorted((v["variant_id"] for v in variants), key=lambda variant: hashlib.sha256(f"{SEQUENCE_LITERAL}|{variant}".encode()).digest())
    assignments = {variant: ("A" if index < 5 else "B" if index < 10 else "C" if index < 14 else "D") for index, variant in enumerate(sorted_variants)}
    by_key = {(p["variant_id"], p["M_R"], p["M_I"]): p for p in prompts}
    rows = []
    for round_number in range(1, 5):
        ordered = sorted(sorted_variants, key=lambda variant: hashlib.sha256(f"{ROUND_LITERAL}|{round_number}|{variant}".encode()).digest())
        for variant in ordered:
            mr, mi = SEQUENCES[assignments[variant]][round_number - 1]
            p = by_key[variant, mr, mi]
            rows.append({"ordinal": len(rows) + 1, "round": round_number, "cell_sequence": assignments[variant], "variant_id": variant, "archetype": p["archetype"], "difficulty": p["difficulty"], "F": "R", "M_R": mr, "M_I": mi, "prompt_id": p["prompt_id"], "prompt_sha256": p["prompt_sha256"], "requested_seed": p["requested_seed"]})
    return rows, assignments


def strings(value: Any) -> set[str]:
    if isinstance(value, str): return {value}
    if isinstance(value, dict): return set().union(*(strings(v) for v in value.values())) if value else set()
    if isinstance(value, list): return set().union(*(strings(v) for v in value)) if value else set()
    return set()


def visible_values(variant: dict[str, Any]) -> set[str]:
    v = variant["model_visible_invariant"]
    values = {v["focal_name"], v["partner_name"], v["operator_name"]}
    for relation in (v["q_relation"], v["r_relation"]):
        values |= {relation["entity"], relation["true_value"], relation["false_value"]}
        if relation["intermediate"]: values.add(relation["intermediate"])
    values |= set(v["static_context"])
    return values


def build() -> tuple[Any, ...]:
    variants = scenarios()
    prompts = [render(variant, mr, mi) for variant in variants for mr in (0, 1) for mi in (0, 1)]
    order, assignments = request_order(variants, prompts)
    prior_candidates = json.loads(PRIOR_CANDIDATES.read_text())["realizations"]
    prior_development = json.loads(PRIOR_DEVELOPMENT.read_text())["realizations"]
    candidate_values = set().union(*(strings(row["model_visible"]) for row in prior_candidates))
    development_values = set().union(*(strings(row["model_visible"]) for row in prior_development))
    audit_rows = []
    for variant in variants:
        v = variant["model_visible_invariant"]; values = visible_values(variant); variant_prompts = [p for p in prompts if p["variant_id"] == variant["variant_id"]]
        audit_rows.append({"variant_id": variant["variant_id"], "archetype": variant["archetype"], "difficulty": variant["difficulty"], "partner_name": v["partner_name"], "separate_operator_name": v["operator_name"], "q_entity": v["q_relation"]["entity"], "q_true_value": v["q_relation"]["true_value"], "q_false_value": v["q_relation"]["false_value"], "r_entity": v["r_relation"]["entity"], "r_true_value": v["r_relation"]["true_value"], "r_false_value": v["r_relation"]["false_value"], "q_path_depth": v["q_relation"]["path_depth"], "r_path_depth": v["r_relation"]["path_depth"], "q_visible_edges": v["q_relation"]["visible_edges"], "r_visible_edges": v["r_relation"]["visible_edges"], "static_distractor_count": len(v["static_context"]), "partner_proposition_first": v["partner_proposition_first"], "send_correction_first": v["send_correction_first"], "semantic_signature_sha256": variant["semantic_signature_sha256"], "prior_candidate_entity_overlap_count": len(values & candidate_values), "prior_development_entity_overlap_count": len(values & development_values), "model_visible_internal_id_leaks": sum(variant["variant_id"].lower() in p["prompt_text"].lower() for p in variant_prompts), "model_visible_experimental_term_leaks": 0, "prompt_count_per_variant": len(variant_prompts), "four_cells_complete": {(p["M_R"], p["M_I"]) for p in variant_prompts} == {(0,0),(0,1),(1,0),(1,1)}, "cell_sequence": assignments[variant["variant_id"]], "harmony_validation_status": HARMONY_STATUS})
    return variants, prompts, order, assignments, audit_rows


def manifest(variants: list[dict[str, Any]], prompts: list[dict[str, Any]], order: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    return {"experiment": "journal_j1_record_holdout", "version": "journal-j1-v0.6.0", "scientific_question": "Can a current partner-record condition distinguish misinformation on the partner's immediate-action path from comparable misinformation on an independent operator branch?", "instrument_validation_status": "frozen_not_executed", "record_only": True, "belief_condition_present": False, "variant_count": 18, "prompt_count": 72, "archetypes": list(ARCHETYPES), "difficulties": list(DIFFICULTIES), "framing_label": FRAMING_LABEL, "five_scenario_control_gates": {"record_relevance_pooled": ">= 0.25", "record_relevance_MI0": "> 0", "record_relevance_MI1": "> 0", "abs_record_irrelevant_MR0": "<= 0.25", "abs_record_irrelevant_MR1": "<= 0.25"}, "headroom_diagnostic": {"record_send_MR1_near_boundary": ">= 0.95 or <= 0.05", "record_send_MR0_near_boundary": ">= 0.95 or <= 0.05", "measurement_headroom_warning": "near_boundary_MR1 OR near_boundary_MR0", "is_sixth_gate": False, "threshold_kind": "heuristic development measurement adequacy"}, "post_execution_classifications": {"A": "record_holdout_validated", "B": "generic_mismatch_salience_replication", "C": "record_holdout_relevance_failure", "D": "causal_control_valid_but_measurement_saturated"}, "ready_for_confirmatory_realization": "scenario_control_pass AND NOT measurement_headroom_warning", "seed_literal": SEED_LITERAL, "seed_algorithm": "SHA-256 UTF-8 literal|variant_id|M_R|M_I; first 8 bytes unsigned big-endian modulo 2147483647; zero becomes one", "frozen_seeds": {f"{p['variant_id']}|{p['M_R']}|{p['M_I']}": p["requested_seed"] for p in prompts}, "cell_sequence_literal": SEQUENCE_LITERAL, "round_order_literal": ROUND_LITERAL, "cell_sequences": {key: [list(cell) for cell in value] for key, value in SEQUENCES.items()}, "sequence_assignments": assignments, "frozen_request_order": order, "v0_5_1_provenance": {"remote_sha": "46c6cf230bcdb2ca231e4ba2c76ad8edb7e037c0", "actions_run": 34667774646, "artifact_id": 10290557533, "artifact_digest": "sha256:0c364af013fdcfc2731c9569161051fd0adfc2e43273503aa3c22d582b7385ec", "aggregate_gate_failure_used_as_motivation": "abs(record_irrelevant_MR0) <= 0.25", "subgroup_or_variant_outcomes_used": False}, "behavioral_observations_used_to_select_holdout_variants": 0, "confirmatory_selection_file_read": False, "harmony_validation_status": HARMONY_STATUS, "model_or_api_execution": False, "confirmatory_prompts_generated": False, "icaart_reopened": False}


def aggregate_audit(variants: list[dict[str, Any]], prompts: list[dict[str, Any]], order: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    proposition = Counter("partner_first" if row["partner_proposition_first"] else "operator_first" for row in rows); actions = Counter("send_first" if row["send_correction_first"] else "continue_first" for row in rows); joint = Counter(("partner_first" if row["partner_proposition_first"] else "operator_first", "send_first" if row["send_correction_first"] else "continue_first") for row in rows)
    prohibited = ("m_r", "m_i", "q_true", "q_false", "r_true", "r_false", "relevant", "irrelevant", "negative control", "causal control", "causal branch", "mismatch", "experimental condition", "gold action", "ground truth condition", "wrong", "incorrect", "false", "stale", "outdated")
    checks = {"variant_count": len(variants) == 18, "prompt_count": len(prompts) == 72, "one_per_cell": len({(v["archetype"], v["difficulty"]) for v in variants}) == 18, "record_only": all(p["F"] == "R" and FRAMING_LABEL in p["prompt_text"] and "partner belief" not in p["prompt_text"].lower() for p in prompts), "four_cells": all(row["four_cells_complete"] for row in rows), "depths_matched": all(row["q_path_depth"] == row["r_path_depth"] == (2 if row["difficulty"] == "compositional" else 1) for row in rows), "distractor_static_fact": all(row["static_distractor_count"] == (1 if row["difficulty"] == "irrelevant_distractor" else 0) for row in rows), "proposition_order": proposition == {"partner_first": 9, "operator_first": 9}, "action_order": actions == {"send_first": 9, "continue_first": 9}, "joint_order": joint == {("partner_first", "send_first"):5, ("partner_first", "continue_first"):4, ("operator_first", "send_first"):4, ("operator_first", "continue_first"):5}, "semantic_signatures_unique": len({v["semantic_signature_sha256"] for v in variants}) == 18, "candidate_overlap_zero": all(row["prior_candidate_entity_overlap_count"] == 0 for row in rows), "development_overlap_zero": all(row["prior_development_entity_overlap_count"] == 0 for row in rows), "no_visible_leakage": not any(term in p["prompt_text"].lower() for p in prompts for term in prohibited) and all(v["variant_id"] not in p["prompt_text"] for v in variants for p in prompts), "unique_seeds": len({p["requested_seed"] for p in prompts}) == 72, "four_rounds": Counter(row["round"] for row in order) == {1:18,2:18,3:18,4:18}, "variant_once_per_round": all(len({row["variant_id"] for row in order if row["round"] == rnd}) == 18 for rnd in range(1,5)), "global_cells_balanced": Counter((row["M_R"], row["M_I"]) for row in order) == {(0,0):18,(0,1):18,(1,0):18,(1,1):18}}
    return {"passed": all(checks.values()), "checks": checks, "variant_audits": rows, "proposition_order_counts": dict(proposition), "action_order_counts": dict(actions), "joint_order_counts": {f"{a}|{b}": value for (a,b), value in joint.items()}, "seed_uniqueness": len({p["requested_seed"] for p in prompts}), "round_assignments": {str(rnd): [row["variant_id"] for row in order if row["round"] == rnd] for rnd in range(1,5)}, "harmony_validation_status": HARMONY_STATUS}


def generate(output_dir: Path = ROOT / "btom_v2") -> None:
    variants, prompts, order, assignments, rows = build(); audit = aggregate_audit(variants, prompts, order, rows)
    if not audit["passed"]: raise RuntimeError("record holdout audit failed")
    payloads = ({"variant_count": 18, "scenarios": variants}, {"prompt_count": 72, "prompts": prompts}, manifest(variants, prompts, order, assignments), audit)
    for name, payload in zip(OUTPUTS[:4], payloads, strict=True): (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output_dir / OUTPUTS[4]).open("w", newline="") as handle:
        fields = tuple(rows[0]); writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict,list)) else value for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=ROOT / "btom_v2"); args = parser.parse_args(); generate(args.output_dir)


if __name__ == "__main__": main()
