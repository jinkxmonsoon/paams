import inspect
import json
import re
from pathlib import Path

from btom_v2.audit_decision_point_prompts import CORE_SECTIONS, _block_shape, run_audit
from btom_v2.decision_point_prompting import (
    CONDITIONS_BY_FAMILY,
    EXPLICIT_FIRST,
    EXPLICIT_SECOND,
    EXPERIMENTAL_WORDS,
    FILLER_FORBIDDEN,
    FORBIDDEN_LABELS,
    NEUTRAL_FIRST,
    NEUTRAL_SECOND,
    SECTION_ORDER,
    audit_record,
    opaque_filler,
    render_all_prompts,
)
from btom_v2.decision_point_scenarios import CASE_BY_ID, second_order_is_message_derived


PROTOCOL = json.loads(Path("btom_v2/decision_point_prompt_protocol_v0_2.json").read_text())
RENDERED = render_all_prompts()


def grouped():
    result = {}
    for item in RENDERED:
        result.setdefault(item.case_id, []).append(item)
    return result


def section(item, name):
    return dict(item.sections).get(name, "")


def test_exact_prompt_and_condition_counts():
    assert len(RENDERED) == 16
    assert sum(item.family == "DP5" for item in RENDERED) == 6
    assert sum(item.family == "DP7" for item in RENDERED) == 10
    assert tuple(CONDITIONS_BY_FAMILY["DP5"]) == tuple(PROTOCOL["applicable_conditions_by_family"]["DP5"])
    assert tuple(CONDITIONS_BY_FAMILY["DP7"]) == tuple(PROTOCOL["applicable_conditions_by_family"]["DP7"])


def test_section_order_and_condition_names_are_not_visible():
    for item in RENDERED:
        names = tuple(name for name, _ in item.sections)
        assert names == tuple(name for name in SECTION_ORDER if name in names)
        assert item.case_id not in item.prompt
        assert CASE_BY_ID[item.case_id].purpose not in item.prompt
        assert item.condition not in item.prompt


def test_scoring_hidden_fields_and_experimental_labels_do_not_leak():
    for item in RENDERED:
        lowered = item.prompt.lower()
        assert all(label.lower() not in lowered for label in FORBIDDEN_LABELS)
        words = set(re.findall(r"[a-z]+", lowered))
        assert not words.intersection(EXPERIMENTAL_WORDS)
        record = audit_record(item)
        assert record.hidden_field_inclusion_count == 0
        assert record.forbidden_label_inclusion_count == 0


def test_prompt_constructor_never_reads_hidden_or_scoring_fields():
    source = inspect.getsource(__import__("btom_v2.decision_point_prompting", fromlist=["*"]))
    assert ".hidden_world_state" not in source
    assert ".behavioral_scoring" not in source
    assert ".allowed_pairwise_differences" not in source


def test_raw_messages_actions_and_core_sections_are_invariant_within_case():
    for case_id, items in grouped().items():
        assert len({section(item, "RAW DELIVERED MESSAGES") for item in items}) == 1
        assert len({section(item, "VALID ACTIONS") for item in items}) == 1
        for core in CORE_SECTIONS:
            assert len({section(item, core) for item in items}) == 1
        frozen = CASE_BY_ID[case_id].valid_actions
        action_lines = section(items[0], "VALID ACTIONS").splitlines()[1:]
        assert [json.loads(line) for line in action_lines] == [
            {"action": action.action, "target": action.target} for action in frozen
        ]


def test_first_order_blocks_are_exactly_matched_and_authorized():
    for case_id, items in grouped().items():
        neutral = next(item for item in items if item.condition == NEUTRAL_FIRST)
        explicit = next(item for item in items if item.condition == EXPLICIT_FIRST)
        assert _block_shape(neutral.first_order_block) == _block_shape(explicit.first_order_block)
        for key, value in CASE_BY_ID[case_id].first_order_representation:
            assert f'{key}="{value}"' in explicit.first_order_block
        assert "observer=" not in explicit.first_order_block


def test_second_order_blocks_are_exactly_matched_and_message_derived():
    for case_id, items in grouped().items():
        case = CASE_BY_ID[case_id]
        if case.family != "DP7":
            continue
        neutral = next(item for item in items if item.condition == NEUTRAL_SECOND)
        explicit = next(item for item in items if item.condition == EXPLICIT_SECOND)
        assert neutral.first_order_block == explicit.first_order_block
        assert _block_shape(neutral.second_order_block) == _block_shape(explicit.second_order_block)
        assert second_order_is_message_derived(case)
        for belief in case.second_order_representation:
            assert f'believed_value="{belief.believed_value}"' in explicit.second_order_block
            assert f'source_message="{belief.source_message}"' in explicit.second_order_block
        assert "medical_kit_actual_location" not in explicit.second_order_block
        assert "actual_expected_location_after_box_open" not in explicit.second_order_block


def test_pairwise_prompt_length_differences_are_reproduced():
    items = grouped()
    def prompt_length(case_id, condition):
        return len(next(item.prompt for item in items[case_id] if item.condition == condition))
    assert (
        prompt_length("DP5a_self_belief_false", EXPLICIT_FIRST)
        - prompt_length("DP5b_self_belief_current", EXPLICIT_FIRST)
        == prompt_length("DP5a_self_belief_false", NEUTRAL_FIRST)
        - prompt_length("DP5b_self_belief_current", NEUTRAL_FIRST)
    )
    assert (
        prompt_length("DP7a_partner_belief_stale", EXPLICIT_SECOND)
        - prompt_length("DP7b_partner_belief_current", EXPLICIT_SECOND)
        == prompt_length("DP7a_partner_belief_stale", NEUTRAL_SECOND)
        - prompt_length("DP7b_partner_belief_current", NEUTRAL_SECOND)
    )


def test_opaque_fillers_are_exact_deterministic_and_nonsemantic():
    for length in (0, 1, 8, 10, 67):
        first = opaque_filler("audit_namespace", length)
        assert first == opaque_filler("audit_namespace", length)
        assert len(first) == length
        assert re.fullmatch(r"[A-Z0-9_]*", first)
        assert first not in {"A", "B", "C"}
        assert not any(term in first for term in FILLER_FORBIDDEN)
    assert any(item.fillers for item in RENDERED)


def test_output_schema_is_identical_and_excludes_chain_of_thought():
    outputs = {section(item, "OUTPUT FORMAT") for item in RENDERED}
    assert len(outputs) == 1
    output = outputs.pop()
    assert '"action":"move|send_message"' in output
    assert '"target":"one exact listed target"' in output
    assert "never evidence of cognition" in output
    for item in RENDERED:
        assert "chain of thought" not in item.prompt.lower()
        assert "chain-of-thought" not in item.prompt.lower()


def test_protocol_freezes_primary_contrasts_and_blocks_execution():
    assert set(PROTOCOL["primary_contrasts"]) == {"H1", "H2"}
    assert PROTOCOL["primary_contrasts"]["H1"]["matched_neutral_condition"] == NEUTRAL_FIRST
    assert PROTOCOL["primary_contrasts"]["H2"]["matched_neutral_condition"] == NEUTRAL_SECOND
    assert PROTOCOL["model_tokenizer_audit_required"] is True
    assert PROTOCOL["model_tokenizer_status"] == "unresolved_and_blocks_real_execution"
    assert PROTOCOL["status"] == "mock_only_not_authorized_for_real_execution"
    assert PROTOCOL["real_execution_authorized"] is False


def test_no_api_http_or_real_model_dependency():
    modules = (
        Path("btom_v2/decision_point_prompting.py").read_text().lower(),
        Path("btom_v2/audit_decision_point_prompts.py").read_text().lower(),
    )
    forbidden_imports = ("groq", "real_llm", "urllib", "requests", "httpx", "openai")
    assert all(term not in source for source in modules for term in forbidden_imports)


def test_complete_audit_passes():
    audit = run_audit()
    assert audit["prompt_count"] == 16
    assert audit["DP5_prompt_count"] == 6
    assert audit["DP7_prompt_count"] == 10
    # The no-explicit-belief DP5 pair is intentionally byte-identical.
    assert audit["unique_prompt_hash_count"] == 15
    assert audit["hidden_truth_leak_count"] == 0
    assert audit["forbidden_label_leak_count"] == 0
    assert audit["invalid_action_exposure_count"] == 0
    for field in (
        "raw_evidence_invariance_passed",
        "valid_action_invariance_passed",
        "core_section_invariance_passed",
        "first_order_matching_passed",
        "second_order_matching_passed",
        "matched_pair_length_difference_passed",
        "second_order_message_derivation_passed",
        "primary_contrasts_frozen",
        "model_tokenizer_audit_required",
    ):
        assert audit[field] is True
    assert audit["real_execution_authorized"] is False
    assert len(audit["prompt_records"]) == 16
