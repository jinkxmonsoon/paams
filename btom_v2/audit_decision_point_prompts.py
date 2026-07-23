"""Audit matched decision-point prompt rendering without any model client."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .decision_point_prompting import (
    CONDITIONS_BY_FAMILY,
    EXPLICIT_FIRST,
    EXPLICIT_SECOND,
    NEUTRAL_FIRST,
    NEUTRAL_SECOND,
    PROTOCOL_VERSION,
    audit_record,
    render_all_prompts,
)
from .decision_point_scenarios import CASE_BY_ID, second_order_is_message_derived


PROTOCOL_PATH = Path(__file__).with_name("decision_point_prompt_protocol_v0_2.json")
CORE_SECTIONS = ("TASK", "LOCAL OBSERVATION", "RAW DELIVERED MESSAGES", "VALID ACTIONS", "OUTPUT FORMAT")


def _by_case(rendered):
    grouped = {}
    for item in rendered:
        grouped.setdefault(item.case_id, []).append(item)
    return grouped


def _section(item, name):
    return dict(item.sections).get(name, "")


def _block_shape(block):
    lines = block.splitlines()
    keys = tuple(line.split("=", 1)[0] for line in lines[1:])
    punctuation = tuple(
        '="..."' if line.split("=", 1)[1].startswith('"') else "=..."
        for line in lines[1:]
    )
    return len(block), len(lines), keys, punctuation, len(block.split())


def run_audit():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    rendered = render_all_prompts()
    records = tuple(audit_record(item) for item in rendered)
    grouped = _by_case(rendered)
    raw_invariant = all(
        len({_section(item, "RAW DELIVERED MESSAGES") for item in items}) == 1
        for items in grouped.values()
    )
    actions_invariant = all(
        len({_section(item, "VALID ACTIONS") for item in items}) == 1
        for items in grouped.values()
    )
    core_invariant = all(
        all(len({_section(item, section) for item in items}) == 1 for section in CORE_SECTIONS)
        for items in grouped.values()
    )
    first_match = all(
        _block_shape(next(item for item in items if item.condition == NEUTRAL_FIRST).first_order_block)
        == _block_shape(next(item for item in items if item.condition == EXPLICIT_FIRST).first_order_block)
        for items in grouped.values()
    )
    second_match = all(
        _block_shape(next(item for item in items if item.condition == NEUTRAL_SECOND).second_order_block)
        == _block_shape(next(item for item in items if item.condition == EXPLICIT_SECOND).second_order_block)
        for case_id, items in grouped.items()
        if CASE_BY_ID[case_id].family == "DP7"
    )
    dp5a, dp5b = grouped["DP5a_self_belief_false"], grouped["DP5b_self_belief_current"]
    dp7a, dp7b = grouped["DP7a_partner_belief_stale"], grouped["DP7b_partner_belief_current"]
    length_difference_match = (
        len(next(i for i in dp5a if i.condition == EXPLICIT_FIRST).prompt)
        - len(next(i for i in dp5b if i.condition == EXPLICIT_FIRST).prompt)
        == len(next(i for i in dp5a if i.condition == NEUTRAL_FIRST).prompt)
        - len(next(i for i in dp5b if i.condition == NEUTRAL_FIRST).prompt)
        and len(next(i for i in dp7a if i.condition == EXPLICIT_SECOND).prompt)
        - len(next(i for i in dp7b if i.condition == EXPLICIT_SECOND).prompt)
        == len(next(i for i in dp7a if i.condition == NEUTRAL_SECOND).prompt)
        - len(next(i for i in dp7b if i.condition == NEUTRAL_SECOND).prompt)
    )
    invalid_action_exposure = sum(
        any(
            json.dumps({"action": action.action, "target": action.target}, separators=(",", ":"))
            not in _section(item, "VALID ACTIONS")
            for action in CASE_BY_ID[item.case_id].valid_actions
        )
        for item in rendered
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_count": len(rendered),
        "DP5_prompt_count": sum(item.family == "DP5" for item in rendered),
        "DP7_prompt_count": sum(item.family == "DP7" for item in rendered),
        "unique_prompt_hash_count": len({record.prompt_sha256 for record in records}),
        "hidden_truth_leak_count": sum(record.hidden_field_inclusion_count for record in records),
        "forbidden_label_leak_count": sum(record.forbidden_label_inclusion_count for record in records),
        "invalid_action_exposure_count": invalid_action_exposure,
        "raw_evidence_invariance_passed": raw_invariant,
        "valid_action_invariance_passed": actions_invariant,
        "core_section_invariance_passed": core_invariant,
        "first_order_matching_passed": first_match,
        "second_order_matching_passed": second_match,
        "matched_pair_length_difference_passed": length_difference_match,
        "second_order_message_derivation_passed": all(second_order_is_message_derived(case) for case in CASE_BY_ID.values()),
        "primary_contrasts_frozen": set(protocol["primary_contrasts"]) == {"H1", "H2"},
        "model_tokenizer_audit_required": protocol["model_tokenizer_audit_required"],
        "real_execution_authorized": protocol["real_execution_authorized"],
        "condition_counts": {family: len(conditions) for family, conditions in CONDITIONS_BY_FAMILY.items()},
        "prompt_records": [asdict(record) for record in records],
    }


def main():
    print(json.dumps(run_audit(), sort_keys=True))


if __name__ == "__main__":
    main()
