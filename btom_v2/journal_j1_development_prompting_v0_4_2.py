"""Prospective v0.4.2 framing audit using official GPT-OSS tokenization.

The finite candidate set is frozen in source.  Candidate diagnostics are
persisted before selection failure is raised.  This module has no model or API
functionality and renders development prompts only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .journal_j1_development_prompting_v0_4_1 import (
    edge_text,
    render_prompt,
)
from .journal_j1_model_safe_realization_v0_4_1 import (
    ROOT,
    canonical,
    load_and_realize,
)

VERSION = "journal-j1-v0.4.2"
PLACEHOLDER = "PARTNER REPRESENTATION:"
FRAMING_CANDIDATES = (
    {"candidate_id": 1, "R_label": "Current partner record:", "B_label": "Current partner belief:"},
    {"candidate_id": 2, "R_label": "PARTNER RECORD:", "B_label": "PARTNER BELIEF:"},
    {"candidate_id": 3, "R_label": "Partner record:", "B_label": "Partner belief:"},
    {"candidate_id": 4, "R_label": "CURRENT PARTNER RECORD:", "B_label": "CURRENT PARTNER BELIEF:"},
    {"candidate_id": 5, "R_label": "Current partner state record:", "B_label": "Current partner belief state:"},
    {"candidate_id": 6, "R_label": "PARTNER STATE RECORD:", "B_label": "PARTNER BELIEF STATE:"},
    {"candidate_id": 7, "R_label": "Partner state record:", "B_label": "Partner belief state:"},
    {"candidate_id": 8, "R_label": "CURRENT PARTNER STATE RECORD:", "B_label": "CURRENT PARTNER BELIEF STATE:"},
)
TABLE_JSON = "tokenizer_candidate_table_v0_4_2.json"
TABLE_CSV = "tokenizer_candidate_table_v0_4_2.csv"
FINAL_OUTPUTS = (
    "journal_j1_candidate_realizations_v0_4_2.json",
    "journal_j1_development_realizations_v0_4_2.json",
    "journal_j1_development_prompts_v0_4_2.json",
    "journal_j1_prompt_audit_v0_4_2.json",
    "journal_j1_prompt_audit_v0_4_2.csv",
)
PREDECESSOR = {
    "predecessor_version": "v0.4.1",
    "predecessor_remote_sha": "b87b67d91e6aff93ec51750c6af7b7cbce90cf89",
    "predecessor_actions_run": 34661682287,
    "predecessor_failure": "no prospective v0.4.1 framing candidate achieved complete Harmony parity",
    "behavioral_observations_before_selection": 0,
}


def framing(candidate: dict[str, Any]) -> dict[str, Any]:
    return {"candidate": candidate["candidate_id"], "R": candidate["R_label"], "B": candidate["B_label"]}


def normalize_prompt(text: str, candidate: dict[str, Any]) -> str:
    return text.replace(candidate["R_label"], PLACEHOLDER).replace(candidate["B_label"], PLACEHOLDER)


def build_prospective() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, developments, source = load_and_realize()
    prompts = []
    for candidate in FRAMING_CANDIDATES:
        for realization, structure in zip(developments, source["scenarios"], strict=True):
            for cell in structure["counterfactual_cells"]:
                text = render_prompt(realization, cell, framing(candidate))
                prompts.append({
                    "candidate_id": candidate["candidate_id"],
                    "variant_id": realization["variant_id"],
                    "archetype": realization["archetype"],
                    "difficulty": realization["difficulty"],
                    "F": cell["F"], "M_R": cell["M_R"], "M_I": cell["M_I"],
                    "gold_action": "SEND_CORRECTION" if cell["M_R"] else "CONTINUE_TASK",
                    "prompt_text": text,
                    "normalized_prompt_sha256": hashlib.sha256(normalize_prompt(text, candidate).encode()).hexdigest(),
                    "action_order": realization["model_visible"]["action_order"],
                    "proposition_order": realization["model_visible"]["proposition_order"],
                    "realization_sha256": realization["realization_sha256"],
                })
    audit_local(candidates, developments, prompts)
    return candidates, developments, prompts


def audit_local(candidates: list[dict[str, Any]], developments: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> None:
    assert len(candidates) == 360 and len(developments) == 18 and len(prompts) == 1152
    assert [row["candidate_id"] for row in FRAMING_CANDIDATES] == list(range(1, 9))
    assert all("record" in row["R_label"].lower() and "belief" in row["B_label"].lower() for row in FRAMING_CANDIDATES)
    groups: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        groups[(prompt["candidate_id"], prompt["variant_id"], prompt["M_R"], prompt["M_I"])].append(prompt)
    assert len(groups) == 576
    assert all(len(rows) == 2 and len({row["normalized_prompt_sha256"] for row in rows}) == 1 for rows in groups.values())
    assert Counter((r["model_visible"]["proposition_order"], r["model_visible"]["action_order"][0]) for r in developments) == {
        ("causal_first", "SEND_CORRECTION"): 5,
        ("causal_first", "CONTINUE_TASK"): 4,
        ("noncausal_first", "SEND_CORRECTION"): 4,
        ("noncausal_first", "CONTINUE_TASK"): 5,
    }
    forbidden = ("task item", "context item", "relevant", "irrelevant", "important fact", "unimportant fact", "active fact", "other fact", "padding", "filler")
    assert not any(term in prompt["prompt_text"].lower() for prompt in prompts for term in forbidden)
    for realization in candidates + developments:
        depth = 2 if realization["difficulty"] == "compositional" else 1
        for role in ("causal_relation", "noncausal_relation"):
            relation = realization["model_visible"][role]
            assert relation["path_depth"] == len(relation["visible_edges"]) == depth
            if realization["namespace"] == "development":
                rendered = [p["prompt_text"] for p in prompts if p["variant_id"] == realization["variant_id"]]
                assert all(all(edge_text(edge) in text for text in rendered) for edge in relation["visible_edges"])


def official_tokenizers() -> tuple[Callable[[str], list[int]], Callable[[str], list[int]], dict[str, str]]:
    import tiktoken
    from openai_harmony import Conversation, HarmonyEncodingName, Message, Role, load_harmony_encoding
    versions = {name: importlib.metadata.version(name) for name in ("tiktoken", "openai-harmony", "pytest")}
    expected = {"tiktoken": "0.13.0", "openai-harmony": "0.0.8", "pytest": "8.4.2"}
    if versions != expected:
        raise RuntimeError(f"dependency mismatch: {versions!r}")
    raw_encoding = tiktoken.get_encoding("o200k_harmony")
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    def harmony(text: str) -> list[int]:
        conversation = Conversation.from_messages([Message.from_role_and_content(Role.USER, text)])
        return list(harmony_encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))
    return raw_encoding.encode, harmony, versions


def evaluate_candidates(prompts: list[dict[str, Any]], raw: Callable[[str], list[int]], harmony: Callable[[str], list[int]]) -> list[dict[str, Any]]:
    table = []
    for candidate in FRAMING_CANDIDATES:
        rows = [p for p in prompts if p["candidate_id"] == candidate["candidate_id"]]
        groups = defaultdict(list)
        for row in rows:
            raw_ids, harmony_ids = list(raw(row["prompt_text"])), list(harmony(row["prompt_text"]))
            row.update({
                "raw_token_count": len(raw_ids), "raw_token_id_sha256": hashlib.sha256(canonical(raw_ids)).hexdigest(),
                "harmony_token_count": len(harmony_ids), "harmony_token_id_sha256": hashlib.sha256(canonical(harmony_ids)).hexdigest(),
            })
            groups[(row["variant_id"], row["M_R"], row["M_I"])].append(row)
        signed = []
        normalized_equal = 0
        raw_equal = 0
        for pair in groups.values():
            by_f = {row["F"]: row for row in pair}
            signed.append(by_f["R"]["harmony_token_count"] - by_f["B"]["harmony_token_count"])
            normalized_equal += by_f["R"]["normalized_prompt_sha256"] == by_f["B"]["normalized_prompt_sha256"]
            raw_equal += by_f["R"]["raw_token_count"] == by_f["B"]["raw_token_count"]
        equal_harmony = sum(value == 0 for value in signed)
        table.append({
            **candidate,
            "matched_pair_count": len(groups), "normalized_equal_pair_count": normalized_equal,
            "equal_harmony_token_count_pairs": equal_harmony, "harmony_parity_rate": equal_harmony / len(groups),
            "min_signed_token_difference": min(signed), "max_signed_token_difference": max(signed),
            "max_abs_token_difference": max(abs(value) for value in signed), "mean_signed_token_difference": mean(signed),
            "equal_raw_token_count_pairs": raw_equal, "raw_parity_rate": raw_equal / len(groups),
        })
    return table


def passing_candidate(table: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in table if row["matched_pair_count"] == row["normalized_equal_pair_count"] == row["equal_harmony_token_count_pairs"] == 72 and row["max_abs_token_difference"] == 0), None)


def write_tables(output_dir: Path, table: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TABLE_JSON).write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    with (output_dir / TABLE_CSV).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(table[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(table)


def run_tokenizer_audit(
    output_dir: Path,
    fail_if_none: bool = True,
    tokenizers: tuple[Callable[[str], list[int]], Callable[[str], list[int]], dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, tuple[Any, ...]]:
    candidates, developments, prompts = build_prospective()
    raw, harmony, versions = tokenizers or official_tokenizers()
    table = evaluate_candidates(prompts, raw, harmony)
    write_tables(output_dir, table)  # diagnostics must precede selection failure
    selected = passing_candidate(table)
    status = {**PREDECESSOR, "dependency_versions": versions, "selected_candidate_id": selected["candidate_id"] if selected else None, "model_api_execution": False, "behavioral_observations": 0, "confirmatory_prompts_generated": False, "icaart_reopened": False, "pilot_ready": False}
    (output_dir / "workflow_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if selected is None and fail_if_none:
        raise RuntimeError("no frozen v0.4.2 framing candidate achieved complete Harmony parity")
    return table, selected, (candidates, developments, prompts)


def generate_final(output_dir: Path) -> None:
    table, selected, built = run_tokenizer_audit(output_dir)
    assert selected is not None
    candidates, developments, prompts = built
    final_prompts = [p for p in prompts if p["candidate_id"] == selected["candidate_id"]]
    joint = Counter((r["model_visible"]["proposition_order"], r["model_visible"]["action_order"][0]) for r in developments)
    audit = {
        **PREDECESSOR, "selected_candidate_id": selected["candidate_id"], "selected_R_label": selected["R_label"], "selected_B_label": selected["B_label"],
        "selection_rule": "first_complete_harmony_parity", "matched_pair_count": 72, "normalized_pair_equality": 72,
        "equal_harmony_token_count_pairs": 72, "max_harmony_difference": 0,
        "proposition_order_counts": dict(Counter(r["model_visible"]["proposition_order"] for r in developments)),
        "action_order_counts": dict(Counter(r["model_visible"]["action_order"][0] for r in developments)),
        "joint_nuisance_order": {f"{a}|{b}": count for (a, b), count in joint.items()},
        "path_depth_audit": {"direct": 1, "irrelevant_distractor": 1, "compositional": 2},
        "gold_invariant_to_f_and_mi": all(p["gold_action"] == ("SEND_CORRECTION" if p["M_R"] else "CONTINUE_TASK") for p in final_prompts),
        "surface_cue_count": 0, "model_api_execution": False, "confirmatory_prompts_generated": False, "icaart_reopened": False, "pilot_ready": True,
    }
    payloads = ({"count": 360, "realizations": candidates}, {"count": 18, "realizations": developments}, {"count": 144, "prompts": final_prompts}, audit)
    for name, payload in zip(FINAL_OUTPUTS[:4], payloads, strict=True):
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output_dir / FINAL_OUTPUTS[4]).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(("metric", "value")); writer.writerows((("pilot_ready", True), ("selected_candidate_id", selected["candidate_id"]), ("matched_pair_count", 72)))
    status = {**PREDECESSOR, "selected_candidate_id": selected["candidate_id"], "model_api_execution": False, "behavioral_observations": 0, "confirmatory_prompts_generated": False, "icaart_reopened": False, "pilot_ready": True}
    (output_dir / "workflow_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        run_tokenizer_audit(args.output_dir)
    else:
        generate_final(args.output_dir)


if __name__ == "__main__":
    main()
