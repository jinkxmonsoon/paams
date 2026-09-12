"""Build and validate J1 development prompts; tokenize only when requested."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .journal_j1_model_safe_realization_v0_4_1 import ROOT, canonical, load_and_realize

FRAMING_CANDIDATES = (
    {"candidate": 1, "R": "CURRENT PARTNER STATE:", "B": "CURRENT PARTNER BELIEF:"},
    {"candidate": 2, "R": "PARTNER STATE:", "B": "PARTNER BELIEF:"},
    {"candidate": 3, "R": "CURRENT PARTNER STATUS:", "B": "CURRENT PARTNER BELIEF:"},
)
PLACEHOLDER = "PARTNER REPRESENTATION:"
ACTIONS = ("SEND_CORRECTION", "CONTINUE_TASK")
OUTPUT_NAMES = (
    "journal_j1_candidate_realizations_v0_4_1.json",
    "journal_j1_development_realizations_v0_4_1.json",
    "journal_j1_development_prompts_v0_4_1.json",
    "journal_j1_prompt_audit_v0_4_1.json",
    "journal_j1_prompt_audit_v0_4_1.csv",
)


def edge_text(edge: dict[str, str]) -> str:
    return f"{edge['source']} {edge['relation']} {edge['target']}."


def render_prompt(realization: dict[str, Any], cell: dict[str, Any], framing: dict[str, Any]) -> str:
    visible = realization["model_visible"]
    relations = [visible["causal_relation"], visible["noncausal_relation"]]
    if visible["proposition_order"] == "noncausal_first":
        relations.reverse()
    facts = [edge_text(edge) for relation in relations for edge in relation["visible_edges"]]
    facts.extend(visible["static_context"])
    values = visible["represented_values"]
    represented = [
        (visible["causal_relation"]["entity"], values["causal_false"] if cell["M_R"] else values["causal_true"]),
        (visible["noncausal_relation"]["entity"], values["noncausal_false"] if cell["M_I"] else values["noncausal_true"]),
    ]
    if visible["proposition_order"] == "noncausal_first":
        represented.reverse()
    representation_lines = [f"{entity}: {value}" for entity, value in represented]
    first, second = visible["action_order"]
    return "\n".join([
        f"You are {visible['focal_agent']}, coordinating with {visible['partner']}.",
        visible["immediate_assignment"],
        "WORLD FACTS:", *facts,
        framing[cell["F"]], *representation_lines,
        "You have exactly one action for this turn.",
        "SEND_CORRECTION sends an update concerning the partner's immediate assignment; it uses this turn and does not advance your task this turn.",
        "CONTINUE_TASK advances your task this turn and sends no update.",
        f"Available actions in order: {first}, {second}",
        'Return strict JSON only: {"action":"SEND_CORRECTION"} or {"action":"CONTINUE_TASK"}',
    ])


def normalize_prompt(text: str, framing: dict[str, Any]) -> str:
    return text.replace(framing["R"], PLACEHOLDER).replace(framing["B"], PLACEHOLDER)


def build_prospective() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, developments, source = load_and_realize()
    prompts: list[dict[str, Any]] = []
    for framing in FRAMING_CANDIDATES:
        for realization, structure in zip(developments, source["scenarios"], strict=True):
            for cell in structure["counterfactual_cells"]:
                text = render_prompt(realization, cell, framing)
                prompts.append({"framing_candidate": framing["candidate"], "variant_id": realization["variant_id"], "archetype": realization["archetype"], "difficulty": realization["difficulty"], "F": cell["F"], "M_R": cell["M_R"], "M_I": cell["M_I"], "gold_action": "SEND_CORRECTION" if cell["M_R"] else "CONTINUE_TASK", "prompt_text": text, "normalized_prompt_sha256": hashlib.sha256(normalize_prompt(text, framing).encode()).hexdigest(), "action_order": realization["model_visible"]["action_order"], "proposition_order": realization["model_visible"]["proposition_order"], "realization_sha256": realization["realization_sha256"]})
    audit_local(candidates, developments, prompts)
    return candidates, developments, prompts


def audit_local(candidates: list[dict[str, Any]], developments: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> None:
    assert len(candidates) == 360 and len(developments) == 18 and len(prompts) == 432
    forbidden = ("task item", "context item", "relevant", "irrelevant", "important fact", "unimportant fact", "active fact", "other fact", "m_r", "m_i", "q_true", "q_false", "r_true", "r_false")
    assert not any(term in p["prompt_text"].lower() for p in prompts for term in forbidden)
    assert Counter(r["model_visible"]["proposition_order"] for r in developments) == {"causal_first": 9, "noncausal_first": 9}
    assert Counter(r["model_visible"]["action_order"][0] for r in developments) == {"SEND_CORRECTION": 9, "CONTINUE_TASK": 9}
    for realization in candidates + developments:
        expected = 2 if realization["difficulty"] == "compositional" else 1
        for key in ("causal_relation", "noncausal_relation"):
            relation = realization["model_visible"][key]
            assert relation["path_depth"] == len(relation["visible_edges"]) == expected
    grouped: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        grouped[(prompt["framing_candidate"], prompt["variant_id"], prompt["M_R"], prompt["M_I"])].append(prompt)
    assert len(grouped) == 216
    for pair in grouped.values():
        assert len(pair) == 2
        assert len({p["normalized_prompt_sha256"] for p in pair}) == 1
        assert len({tuple(p["action_order"]) for p in pair}) == 1


def _tokenizers():
    import tiktoken
    from openai_harmony import Conversation, HarmonyEncodingName, Message, Role, load_harmony_encoding
    if importlib.metadata.version("tiktoken") != "0.13.0" or importlib.metadata.version("openai-harmony") != "0.0.8":
        raise RuntimeError("frozen tokenizer dependency mismatch")
    raw = tiktoken.get_encoding("o200k_harmony")
    harmony = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    def harmony_tokens(text: str) -> list[int]:
        conversation = Conversation.from_messages([Message.from_role_and_content(Role.USER, text)])
        return list(harmony.render_conversation_for_completion(conversation, Role.ASSISTANT))
    return raw, harmony_tokens


def generate_actions_outputs(output_dir: Path) -> None:
    candidates, developments, prompts = build_prospective()
    raw, harmony = _tokenizers()
    table = []
    selected = None
    for framing in FRAMING_CANDIDATES:
        rows = [p for p in prompts if p["framing_candidate"] == framing["candidate"]]
        grouped = defaultdict(list)
        for row in rows:
            raw_ids = raw.encode(row["prompt_text"]); harmony_ids = harmony(row["prompt_text"])
            row.update({"raw_token_count": len(raw_ids), "raw_token_id_sha256": hashlib.sha256(canonical(raw_ids)).hexdigest(), "harmony_token_count": len(harmony_ids), "harmony_token_id_sha256": hashlib.sha256(canonical(harmony_ids)).hexdigest()})
            grouped[(row["variant_id"], row["M_R"], row["M_I"])].append(row)
        differences = [abs(pair[0]["harmony_token_count"] - pair[1]["harmony_token_count"]) for pair in grouped.values()]
        entry = {"candidate": framing["candidate"], "R": framing["R"], "B": framing["B"], "matched_pair_count": len(grouped), "equal_harmony_token_count_pairs": sum(d == 0 for d in differences), "parity_rate": sum(d == 0 for d in differences) / len(differences), "max_abs_token_difference": max(differences)}
        table.append(entry)
        if selected is None and entry["equal_harmony_token_count_pairs"] == 72 and entry["max_abs_token_difference"] == 0:
            selected = framing["candidate"]
    if selected is None:
        raise RuntimeError("no prospective framing candidate has complete Harmony parity")
    final_prompts = [p for p in prompts if p["framing_candidate"] == selected]
    proposition_counts = Counter(r["model_visible"]["proposition_order"] for r in developments)
    action_counts = Counter(r["model_visible"]["action_order"][0] for r in developments)
    depth_pass = all(
        relation["path_depth"] == (2 if row["difficulty"] == "compositional" else 1)
        for row in developments
        for relation in (row["model_visible"]["causal_relation"], row["model_visible"]["noncausal_relation"])
    )
    selected_table = next(row for row in table if row["candidate"] == selected)
    gates = {
        "development_variant_count_is_18": len(developments) == 18,
        "development_prompt_count_is_144": len(final_prompts) == 144,
        "matched_rb_pair_count_is_72": selected_table["matched_pair_count"] == 72,
        "normalized_rb_byte_equality_is_72_of_72": len({p["normalized_prompt_sha256"] for p in final_prompts}) <= 72,
        "harmony_parity_is_72_of_72": selected_table["equal_harmony_token_count_pairs"] == 72,
        "max_harmony_token_difference_is_zero": selected_table["max_abs_token_difference"] == 0,
        "proposition_order_is_9_and_9": proposition_counts == {"causal_first": 9, "noncausal_first": 9},
        "action_order_is_9_and_9": action_counts == {"SEND_CORRECTION": 9, "CONTINUE_TASK": 9},
        "path_depths_match_difficulty": depth_pass,
        "gold_depends_only_on_mr": all(p["gold_action"] == ("SEND_CORRECTION" if p["M_R"] else "CONTINUE_TASK") for p in final_prompts),
        "behavioral_data_used": False,
        "model_api_execution": False,
        "confirmatory_prompts_generated": False,
        "icaart_reopened": False,
    }
    audit = {"pilot_ready": all(value is True for key, value in gates.items() if key not in {"behavioral_data_used", "model_api_execution", "confirmatory_prompts_generated", "icaart_reopened"}) and not any(gates[key] for key in ("behavioral_data_used", "model_api_execution", "confirmatory_prompts_generated", "icaart_reopened")), "local_status_before_tokenization": "pending_actions_tokenizer_validation", "development_prompt_count": 144, "matched_rb_pair_count": 72, "selected_framing_candidate": selected, "tokenizer_candidate_table": table, "proposition_order_counts": dict(proposition_counts), "action_order_counts": dict(action_counts), "gates": gates, "model_api_execution": False, "behavioral_observations": 0, "confirmatory_prompts_generated": False, "icaart_reopened": False}
    if not audit["pilot_ready"]:
        raise RuntimeError("post-tokenization pilot-readiness audit failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = ({"count": 360, "realizations": candidates}, {"count": 18, "realizations": developments}, {"count": 144, "prompts": final_prompts}, audit)
    for name, payload in zip(OUTPUT_NAMES[:4], payloads, strict=True):
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output_dir / OUTPUT_NAMES[4]).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(("metric", "value")); writer.writerows((("pilot_ready", True), ("development_prompt_count", 144), ("matched_rb_pair_count", 72)))
    (output_dir / "tokenizer_candidate_table.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    (output_dir / "workflow_status.json").write_text(json.dumps({"model_api_execution": False, "behavioral_observations": 0, "confirmatory_prompts_generated": False, "icaart_reopened": False, "pilot_ready": True}, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True)
    generate_actions_outputs(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
