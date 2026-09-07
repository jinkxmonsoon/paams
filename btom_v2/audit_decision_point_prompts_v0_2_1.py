"""Audit v0.2.1 provenance separation without a model or tokenizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .audit_decision_point_prompts import CORE_SECTIONS, _block_shape
from .decision_point_prompting import (
    EXPLICIT_FIRST,
    EXPLICIT_SECOND,
    NEUTRAL_FIRST,
    NEUTRAL_SECOND,
    render_all_prompts as render_v02_prompts,
)
from .decision_point_prompting_v0_2_1 import (
    EVIDENCE_REF,
    PROTOCOL_VERSION,
    audit_record,
    audit_record_as_dict,
    render_all_prompts,
)
from .decision_point_scenarios import CASE_BY_ID, second_order_is_message_derived


PROTOCOL_PATH = Path(__file__).with_name("decision_point_prompt_protocol_v0_2_1.json")
PARENT_PROTOCOL_PATH = Path(__file__).with_name("decision_point_prompt_protocol_v0_2.json")


def _group(rendered):
    grouped = {}
    for item in rendered:
        grouped.setdefault(item.case_id, []).append(item)
    return grouped


def _section(item, name):
    return dict(item.sections).get(name, "")


def run_audit():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    parent_protocol = json.loads(PARENT_PROTOCOL_PATH.read_text())
    rendered = render_all_prompts()
    parent_rendered = {item.prompt_id: item for item in render_v02_prompts()}
    records = tuple(audit_record(item) for item in rendered)
    grouped = _group(rendered)
    changed = tuple(
        item.prompt_id for item in rendered if item.prompt != parent_rendered[item.prompt_id].prompt
    )
    raw_invariance = all(
        len({_section(item, "RAW DELIVERED MESSAGES") for item in items}) == 1
        for items in grouped.values()
    )
    action_invariance = all(
        len({_section(item, "VALID ACTIONS") for item in items}) == 1
        for items in grouped.values()
    )
    core_invariance = all(
        all(len({_section(item, section) for item in items}) == 1 for section in CORE_SECTIONS)
        for items in grouped.values()
    )
    first_matching = all(
        _block_shape(next(i for i in items if i.condition == NEUTRAL_FIRST).first_order_block)
        == _block_shape(next(i for i in items if i.condition == EXPLICIT_FIRST).first_order_block)
        for items in grouped.values()
    )
    second_matching = all(
        _block_shape(next(i for i in items if i.condition == NEUTRAL_SECOND).second_order_block)
        == _block_shape(next(i for i in items if i.condition == EXPLICIT_SECOND).second_order_block)
        for case_id, items in grouped.items() if CASE_BY_ID[case_id].family == "DP7"
    )
    dp7a = grouped["DP7a_partner_belief_stale"]
    dp7b = grouped["DP7b_partner_belief_current"]
    pair_length_matching = (
        len(next(i for i in dp7a if i.condition == EXPLICIT_SECOND).prompt)
        - len(next(i for i in dp7b if i.condition == EXPLICIT_SECOND).prompt)
        == len(next(i for i in dp7a if i.condition == NEUTRAL_SECOND).prompt)
        - len(next(i for i in dp7b if i.condition == NEUTRAL_SECOND).prompt)
    )
    dp7_records = [
        record for record in records if CASE_BY_ID[record.base.prompt_id.split(":", 1)[0]].family == "DP7"
    ]
    second_records = [record for record in dp7_records if record.base.second_order_block_character_count]
    resolution = all(
        record.provenance.evidence_ref == EVIDENCE_REF
        and record.provenance.resolved_raw_message_index == 1
        for record in second_records
    )
    hash_match = all(
        record.provenance.raw_message_sha256
        == hashlib.sha256(CASE_BY_ID[record.base.prompt_id.split(":", 1)[0]].raw_delivered_messages[0].encode()).hexdigest()
        for record in dp7_records
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
        "unique_prompt_hash_count": len({record.base.prompt_sha256 for record in records}),
        "hidden_truth_leak_count": sum(record.base.hidden_field_inclusion_count for record in records),
        "forbidden_label_leak_count": sum(record.base.forbidden_label_inclusion_count for record in records),
        "invalid_action_exposure_count": invalid_action_exposure,
        "raw_evidence_invariance_passed": raw_invariance,
        "valid_action_invariance_passed": action_invariance,
        "core_section_invariance_passed": core_invariance,
        "first_order_matching_passed": first_matching,
        "second_order_matching_passed": second_matching,
        "matched_pair_length_difference_passed": pair_length_matching,
        "second_order_message_derivation_passed": all(second_order_is_message_derived(case) for case in CASE_BY_ID.values()),
        "duplicated_raw_evidence_count": sum(max(0, r.raw_message_occurrences_in_prompt - 1) for r in dp7_records),
        "model_visible_source_message_field_count": sum(r.model_visible_source_message_field_count for r in records),
        "raw_message_exactly_once_passed": all(r.raw_message_occurrences_in_prompt == 1 for r in dp7_records),
        "evidence_reference_invariance_passed": all(
            'evidence_ref="raw_message_1"' in next(i for i in rendered if i.prompt_id == r.base.prompt_id).second_order_block
            for r in second_records
        ),
        "evidence_reference_resolution_passed": resolution,
        "provenance_hash_match_passed": hash_match,
        "v0_2_unchanged_prompt_count": len(rendered) - len(changed),
        "v0_2_changed_prompt_count": len(changed),
        "changed_prompt_ids": changed,
        "primary_contrasts_frozen": protocol["primary_contrasts"] == parent_protocol["primary_contrasts"],
        "model_tokenizer_audit_required": protocol["model_tokenizer_audit_required"],
        "real_execution_authorized": protocol["real_execution_authorized"],
        "prompt_records": [audit_record_as_dict(record) for record in records],
    }


def main():
    print(json.dumps(run_audit(), sort_keys=True))


if __name__ == "__main__":
    main()
