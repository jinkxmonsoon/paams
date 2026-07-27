import hashlib
import json
from pathlib import Path

from btom_v2.decision_point_natural_control_prompting_v0_3_0 import (
    CONDITIONS_BY_FAMILY, EXPLICIT_FIRST, EXPLICIT_SECOND, MESSAGE_PROVENANCE,
    OPERATIONAL_METADATA, REACTIVE, audit_record, first_order_block,
    message_provenance_block, operational_metadata_block, render_all_prompts,
    run_audit, second_order_block,
)
from btom_v2.decision_point_prompting_v0_2_1 import render_prompt as render_parent
from btom_v2.decision_point_scenarios import CASES


ROOT = Path(__file__).parents[1]
MANIFEST = json.loads((ROOT / "btom_v2/decision_point_natural_control_manifest_v0_3_0.json").read_text())


def test_exact_conditions_and_prompt_matrix():
    assert CONDITIONS_BY_FAMILY == {
        "DP5": (REACTIVE, OPERATIONAL_METADATA, EXPLICIT_FIRST),
        "DP7": (REACTIVE, OPERATIONAL_METADATA, EXPLICIT_FIRST, MESSAGE_PROVENANCE, EXPLICIT_SECOND),
    }
    prompts = render_all_prompts()
    assert len(prompts) == 16
    assert sum(prompt.family == "DP5" for prompt in prompts) == 6
    assert sum(prompt.family == "DP7" for prompt in prompts) == 10


def test_exact_model_visible_blocks():
    dp5 = next(case for case in CASES if case.family == "DP5")
    dp7 = next(case for case in CASES if case.family == "DP7")
    assert first_order_block(dp5).splitlines() == [
        "FIRST-ORDER REPRESENTATION", 'agent="C"',
        'proposition="medical_kit_location"', 'represented_value="decoy_room"',
        'representation_scope="current_decision"',
    ]
    assert operational_metadata_block(dp5).splitlines() == [
        "DECISION METADATA", 'agent="C"', 'decision_scope="single_action"',
        'action_source="listed_actions"', 'output_mode="json_object"',
    ]
    assert second_order_block(dp7).splitlines() == [
        "SECOND-ORDER REPRESENTATION", 'observer="A"', 'target_agent="C"',
        'proposition="expected_medical_kit_location_after_box_open"',
        'believed_value="decoy_room"', 'epistemic_status="communicated"',
        'evidence_ref="raw_message_1"',
    ]
    assert message_provenance_block(dp7).splitlines() == [
        "MESSAGE PROVENANCE METADATA", 'observer="A"', 'message_sender="C"',
        'message_type="structured_statement"', 'content_reference="raw_message_1"',
        'delivery_status="available"', 'evidence_ref="raw_message_1"',
    ]


def test_common_sections_are_parent_identical_and_actions_frozen():
    for prompt in render_all_prompts():
        case = next(case for case in CASES if case.case_id == prompt.case_id)
        parent = render_parent(case, "reactive_no_explicit_belief")
        for section in ("TASK", "LOCAL OBSERVATION", "RAW DELIVERED MESSAGES", "VALID ACTIONS", "OUTPUT FORMAT"):
            assert dict(prompt.sections)[section].encode() == dict(parent.sections)[section].encode()


def test_primary_blocks_have_equal_field_and_line_counts():
    audit = run_audit()
    assert audit["H1_equal_field_and_line_counts"] is True
    assert audit["H2_equal_field_and_line_counts"] is True
    assert MANIFEST["model_visible_blocks"] == {
        "first_order_field_count": 4, "operational_metadata_field_count": 4,
        "second_order_field_count": 6, "message_provenance_field_count": 6,
    }


def test_raw_evidence_leakage_and_filler_audit():
    audit = run_audit()
    assert audit["DP7_raw_message_exactly_once"] is True
    assert audit["hidden_field_leak_count"] == 0
    assert audit["scoring_label_leak_count"] == 0
    assert audit["condition_name_leak_count"] == 0
    assert audit["source_message_field_count"] == 0
    assert audit["legacy_filler_count"] == 0
    for prompt in render_all_prompts():
        assert prompt.case_id not in prompt.prompt
        assert prompt.condition not in prompt.prompt
        assert not any(value in prompt.prompt for value in ("loc_", "BBBBBBBB", "989_____"))


def test_operational_and_provenance_controls_withhold_epistemic_content():
    forbidden_operational = ("belief", "treatment", "control", "neutral", "false", "current", "stale", "oracle", "theory of mind")
    forbidden_provenance = ("belief", "believes", "expected location", "correction", "stale", "current", "true", "false", "source_message")
    for case in CASES:
        assert not any(term in operational_metadata_block(case).lower() for term in forbidden_operational)
        if case.family == "DP7":
            block = message_provenance_block(case).lower()
            assert not any(term in block for term in forbidden_provenance)
            assert case.raw_delivered_messages[0] not in block
    for family in ("DP5", "DP7"):
        paired = [operational_metadata_block(case) for case in CASES if case.family == family]
        assert len(set(paired)) == 1


def test_audit_records_counts_without_tokenizer_claims():
    records = [audit_record(prompt) for prompt in render_all_prompts()]
    assert all(record.character_count > 0 and record.utf8_byte_count > 0 for record in records)
    assert all(record.line_count > 0 and record.approximate_whitespace_token_count > 0 for record in records)
    audit = run_audit()
    assert audit["prompt_count"] == 16
    assert audit["common_section_immutability_passed"] is True
    assert audit["valid_actions_frozen"] is True
    assert audit["raw_messages_frozen"] is True
    assert audit["tokenizer_parity_claimed"] is False
    assert audit["model_or_API_execution"] is False
    assert audit["real_execution_authorized"] is False


def test_manifest_and_parent_files_are_frozen():
    assert MANIFEST["conditions_by_family"] == {key: list(value) for key, value in CONDITIONS_BY_FAMILY.items()}
    assert MANIFEST["expected_prompt_counts"] == {"DP5": 6, "DP7": 10, "total": 16}
    assert MANIFEST["tokenizer_parity_claimed"] is False
    assert MANIFEST["later_tokenizer_only_calibration_required"] is True
    assert MANIFEST["real_execution_authorized"] is False
    for relative, expected in MANIFEST["immutable_parent_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_no_tokenizer_api_or_model_client_imports():
    source = (ROOT / "btom_v2/decision_point_natural_control_prompting_v0_3_0.py").read_text().lower()
    for forbidden in (
        "tiktoken", "openai_harmony", "from groq", "import groq", "openai import",
        "requests", "httpx", "urllib", "api.groq.com", "api.openai.com",
    ):
        assert forbidden not in source
