import hashlib
import inspect
import json
import re
from pathlib import Path

from btom_v2.audit_decision_point_prompts import _block_shape
from btom_v2.audit_decision_point_prompts_v0_2_1 import run_audit
from btom_v2.decision_point_prompting import (
    CONDITIONS_BY_FAMILY,
    EXPLICIT_SECOND,
    FILLER_FORBIDDEN,
    NEUTRAL_SECOND,
    render_all_prompts as render_v02_prompts,
)
from btom_v2.decision_point_prompting_v0_2_1 import (
    EVIDENCE_REF,
    audit_record,
    render_all_prompts,
)
from btom_v2.decision_point_scenarios import CASE_BY_ID


PROTOCOL = json.loads(Path("btom_v2/decision_point_prompt_protocol_v0_2_1.json").read_text())
PARENT_PROTOCOL = json.loads(Path("btom_v2/decision_point_prompt_protocol_v0_2.json").read_text())
RENDERED = render_all_prompts()
PARENT = {item.prompt_id: item for item in render_v02_prompts()}


def grouped():
    result = {}
    for item in RENDERED:
        result.setdefault(item.case_id, []).append(item)
    return result


def test_matrix_and_condition_applicability_are_unchanged():
    assert len(RENDERED) == 16
    assert sum(item.family == "DP5" for item in RENDERED) == 6
    assert sum(item.family == "DP7" for item in RENDERED) == 10
    assert PROTOCOL["applicable_conditions_by_family"] == PARENT_PROTOCOL["applicable_conditions_by_family"]
    assert tuple(PROTOCOL["applicable_conditions_by_family"]["DP5"]) == CONDITIONS_BY_FAMILY["DP5"]
    assert tuple(PROTOCOL["applicable_conditions_by_family"]["DP7"]) == CONDITIONS_BY_FAMILY["DP7"]


def test_exactly_four_second_order_prompts_change_from_v02():
    changed = [item.prompt_id for item in RENDERED if item.prompt != PARENT[item.prompt_id].prompt]
    expected = [
        item.prompt_id
        for item in RENDERED
        if item.family == "DP7" and item.condition in {NEUTRAL_SECOND, EXPLICIT_SECOND}
    ]
    assert changed == expected
    assert len(changed) == 4
    assert all(
        item.prompt == PARENT[item.prompt_id].prompt
        for item in RENDERED
        if item.prompt_id not in changed
    )


def test_all_dp5_and_nonsecond_dp7_prompts_are_byte_identical_to_v02():
    unchanged = [
        item for item in RENDERED
        if item.family == "DP5" or item.condition not in {NEUTRAL_SECOND, EXPLICIT_SECOND}
    ]
    assert len(unchanged) == 12
    assert all(item.prompt.encode() == PARENT[item.prompt_id].prompt.encode() for item in unchanged)


def test_raw_message_appears_exactly_once_and_source_message_is_not_visible():
    for item in RENDERED:
        case = CASE_BY_ID[item.case_id]
        if case.family != "DP7":
            continue
        message = case.raw_delivered_messages[0]
        assert item.prompt.count(message) == 1
        assert "source_message=" not in item.prompt


def test_evidence_reference_is_invariant_and_resolves_to_raw_message():
    for case_id, items in grouped().items():
        case = CASE_BY_ID[case_id]
        if case.family != "DP7":
            continue
        blocks = [
            item for item in items if item.condition in {NEUTRAL_SECOND, EXPLICIT_SECOND}
        ]
        assert all(f'evidence_ref="{EVIDENCE_REF}"' in item.second_order_block for item in blocks)
        assert len({
            next(line for line in item.second_order_block.splitlines() if line.startswith("evidence_ref="))
            for item in blocks
        }) == 1
        for item in blocks:
            provenance = audit_record(item).provenance
            assert provenance.evidence_ref == EVIDENCE_REF
            assert provenance.resolved_raw_message_index == 1
            message = case.raw_delivered_messages[0]
            assert provenance.raw_message_sha256 == hashlib.sha256(message.encode()).hexdigest()


def test_explicit_value_is_derived_and_neutral_value_is_opaque_length_matched():
    for case_id, items in grouped().items():
        case = CASE_BY_ID[case_id]
        if case.family != "DP7":
            continue
        explicit = next(item for item in items if item.condition == EXPLICIT_SECOND)
        neutral = next(item for item in items if item.condition == NEUTRAL_SECOND)
        belief = case.second_order_representation[0]
        explicit_record = audit_record(explicit)
        assert explicit_record.provenance.extracted_value == belief.believed_value
        assert explicit_record.provenance.explicit_nested_value_matches_extracted_value is True
        assert f'believed_value="{belief.believed_value}"' in explicit.second_order_block
        neutral_value = dict(neutral.fillers)["second_order.0.believed_value"]
        assert len(neutral_value) == len(belief.believed_value)
        assert re.fullmatch(r"[A-Z0-9_]+", neutral_value)
        assert not any(term in neutral_value for term in FILLER_FORBIDDEN)


def test_second_order_matching_and_pairwise_length_difference():
    items = grouped()
    for case_id in ("DP7a_partner_belief_stale", "DP7b_partner_belief_current"):
        neutral = next(i for i in items[case_id] if i.condition == NEUTRAL_SECOND)
        explicit = next(i for i in items[case_id] if i.condition == EXPLICIT_SECOND)
        assert neutral.first_order_block == explicit.first_order_block
        assert _block_shape(neutral.second_order_block) == _block_shape(explicit.second_order_block)
        assert len(neutral.second_order_block.encode()) == len(explicit.second_order_block.encode())
    def length(case_id, condition):
        return len(next(i.prompt for i in items[case_id] if i.condition == condition))
    assert (
        length("DP7a_partner_belief_stale", EXPLICIT_SECOND)
        - length("DP7b_partner_belief_current", EXPLICIT_SECOND)
        == length("DP7a_partner_belief_stale", NEUTRAL_SECOND)
        - length("DP7b_partner_belief_current", NEUTRAL_SECOND)
    )


def test_first_order_core_actions_output_and_leakage_controls_remain_unchanged():
    forbidden = (
        "hidden_world_state", "behavioral_scoring", "productive_branch",
        "necessary_correction", "unnecessary_correction",
    )
    for item in RENDERED:
        parent = PARENT[item.prompt_id]
        assert item.first_order_block == parent.first_order_block
        for name in ("TASK", "LOCAL OBSERVATION", "RAW DELIVERED MESSAGES", "VALID ACTIONS", "OUTPUT FORMAT"):
            assert dict(item.sections)[name] == dict(parent.sections)[name]
        assert item.case_id not in item.prompt
        assert item.condition not in item.prompt
        assert all(term not in item.prompt for term in forbidden)
        action_lines = dict(item.sections)["VALID ACTIONS"].splitlines()[1:]
        assert [json.loads(line) for line in action_lines] == [
            {"action": action.action, "target": action.target}
            for action in CASE_BY_ID[item.case_id].valid_actions
        ]
        output = dict(item.sections)["OUTPUT FORMAT"]
        assert output == dict(parent.sections)["OUTPUT FORMAT"]
        assert "never evidence of cognition" in output


def test_protocol_freeze_tokenizer_blocker_and_no_real_execution():
    assert PROTOCOL["primary_contrasts"] == PARENT_PROTOCOL["primary_contrasts"]
    assert PROTOCOL["secondary_descriptive_references"] == PARENT_PROTOCOL["secondary_descriptive_references"]
    assert PROTOCOL["raw_evidence_max_occurrences_per_prompt"] == 1
    assert PROTOCOL["model_visible_source_message_field_allowed"] is False
    assert PROTOCOL["invariant_evidence_reference"] == EVIDENCE_REF
    assert PROTOCOL["provenance_resolution_required_in_audit"] is True
    assert PROTOCOL["model_tokenizer_status"] == "unresolved_and_blocks_real_execution"
    assert PROTOCOL["real_execution_authorized"] is False


def test_no_api_http_tokenizer_or_real_model_dependency():
    sources = (
        Path("btom_v2/decision_point_prompting_v0_2_1.py").read_text().lower(),
        Path("btom_v2/audit_decision_point_prompts_v0_2_1.py").read_text().lower(),
    )
    forbidden = ("groq", "openai", "real_llm", "urllib", "requests", "httpx", "transformers", "tiktoken")
    assert all(term not in source for source in sources for term in forbidden)


def test_complete_v021_audit():
    audit = run_audit()
    assert audit["prompt_count"] == 16
    assert audit["DP5_prompt_count"] == 6
    assert audit["DP7_prompt_count"] == 10
    assert audit["duplicated_raw_evidence_count"] == 0
    assert audit["model_visible_source_message_field_count"] == 0
    assert audit["v0_2_unchanged_prompt_count"] == 12
    assert audit["v0_2_changed_prompt_count"] == 4
    assert len(audit["changed_prompt_ids"]) == 4
    for field in (
        "raw_evidence_invariance_passed", "valid_action_invariance_passed",
        "core_section_invariance_passed", "first_order_matching_passed",
        "second_order_matching_passed", "matched_pair_length_difference_passed",
        "second_order_message_derivation_passed", "raw_message_exactly_once_passed",
        "evidence_reference_invariance_passed", "evidence_reference_resolution_passed",
        "provenance_hash_match_passed", "primary_contrasts_frozen",
        "model_tokenizer_audit_required",
    ):
        assert audit[field] is True
    assert audit["hidden_truth_leak_count"] == 0
    assert audit["forbidden_label_leak_count"] == 0
    assert audit["invalid_action_exposure_count"] == 0
    assert audit["real_execution_authorized"] is False
