import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from btom_v2.decision_point_content_matched_control_prompting_v0_4_0 import (
    CONDITIONS_BY_FAMILY,
    EXPLICIT_SELF_BELIEF,
    FIRST_ORDER_FIELDS,
    FIRST_ORDER_TITLE,
    MATCHED_DECISION_RECORD,
    MESSAGE_RECORD,
    PARTNER_BELIEF,
    REACTIVE,
    SECOND_ORDER_FIELDS,
    SECOND_ORDER_TITLE,
    audit_primary_pair,
    nested_partner_value,
    parse_communicated_value,
    render_all_prompts,
    render_prompt,
    run_audit,
)
from btom_v2.decision_point_natural_control_prompting_v0_3_0 import (
    COMMON_SECTIONS,
    render_prompt as render_parent_prompt,
)
from btom_v2.decision_point_scenarios import CASES


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "btom_v2/decision_point_content_matched_control_prompting_v0_4_0.py"
MANIFEST_PATH = ROOT / "btom_v2/decision_point_content_matched_control_manifest_v0_4_0.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def _case(family):
    return next(case for case in CASES if case.family == family)


def _changed_section(prompt, title, transform):
    sections = tuple(
        (name, transform(text) if name == title else text)
        for name, text in prompt.sections
    )
    return replace(prompt, sections=sections, prompt="\n\n".join(text for _, text in sections))


def _blocks(case, reference_condition, treatment_condition, title):
    reference = render_prompt(case, reference_condition)
    treatment = render_prompt(case, treatment_condition)
    return reference, treatment, dict(reference.sections)[title], dict(treatment.sections)[title]


def test_exact_conditions_and_prompt_population():
    assert CONDITIONS_BY_FAMILY == {
        "DP5": (REACTIVE, MATCHED_DECISION_RECORD, EXPLICIT_SELF_BELIEF),
        "DP7": (REACTIVE, MATCHED_DECISION_RECORD, EXPLICIT_SELF_BELIEF, MESSAGE_RECORD, PARTNER_BELIEF),
    }
    prompts = render_all_prompts()
    assert len(prompts) == 16
    assert sum(prompt.family == "DP5" for prompt in prompts) == 6
    assert sum(prompt.family == "DP7" for prompt in prompts) == 10
    assert MANIFEST["conditions_by_family"] == {
        family: list(conditions) for family, conditions in CONDITIONS_BY_FAMILY.items()
    }


def test_exact_H1_blocks_differ_only_in_role():
    for case in CASES:
        reference, treatment, reference_block, treatment_block = _blocks(
            case, MATCHED_DECISION_RECORD, EXPLICIT_SELF_BELIEF, FIRST_ORDER_TITLE
        )
        reference_lines = reference_block.splitlines()
        treatment_lines = treatment_block.splitlines()
        assert reference_lines[0] == treatment_lines[0] == "DECISION-STATE REPRESENTATION"
        assert tuple(line.split("=", 1)[0] for line in reference_lines[1:]) == FIRST_ORDER_FIELDS
        assert reference_lines[:-1] == treatment_lines[:-1]
        assert reference_lines[-1] == 'representation_role="decision_record"'
        assert treatment_lines[-1] == 'representation_role="self_belief"'
        assert audit_primary_pair(reference, treatment, case, FIRST_ORDER_TITLE)["passed"] is True


def test_exact_H2_blocks_differ_only_in_role():
    for case in (case for case in CASES if case.family == "DP7"):
        reference, treatment, reference_block, treatment_block = _blocks(
            case, MESSAGE_RECORD, PARTNER_BELIEF, SECOND_ORDER_TITLE
        )
        reference_lines = reference_block.splitlines()
        treatment_lines = treatment_block.splitlines()
        assert reference_lines[0] == treatment_lines[0] == "PARTNER-CONTEXT REPRESENTATION"
        assert tuple(line.split("=", 1)[0] for line in reference_lines[1:]) == SECOND_ORDER_FIELDS
        differences = [
            index for index, pair in enumerate(zip(reference_lines, treatment_lines))
            if pair[0] != pair[1]
        ]
        assert differences == [5]
        assert reference_lines[5] == 'representation_role="message_record"'
        assert treatment_lines[5] == 'representation_role="partner_belief"'
        assert reference_lines[6] == treatment_lines[6] == 'evidence_ref="raw_message_1"'
        assert audit_primary_pair(reference, treatment, case, SECOND_ORDER_TITLE)["passed"] is True


def test_H2_independent_provenance_sources_are_equal_and_recorded():
    for case in (case for case in CASES if case.family == "DP7"):
        assert parse_communicated_value(case) == nested_partner_value(case)
        reference = render_prompt(case, MESSAGE_RECORD)
        treatment = render_prompt(case, PARTNER_BELIEF)
        assert dict(reference.source_fields_by_layer)["partner_context_representation"] == (
            "raw_delivered_messages",
        )
        assert dict(treatment.source_fields_by_layer)["partner_context_representation"] == (
            "second_order_representation",
        )


def test_global_audit_passes_and_common_sections_are_immutable():
    audit = run_audit()
    assert audit["prompt_count"] == 16
    assert audit["DP5_prompt_count"] == 6
    assert audit["DP7_prompt_count"] == 10
    assert audit["common_sections_byte_identical"] is True
    assert audit["valid_actions_frozen"] is True
    assert audit["observations_frozen"] is True
    assert audit["raw_messages_frozen"] is True
    assert audit["DP7_raw_evidence_exactly_once"] is True
    assert audit["hidden_state_leak_count"] == 0
    assert audit["scoring_label_leak_count"] == 0
    assert audit["condition_name_leak_count"] == 0
    assert audit["source_message_field_count"] == 0
    assert audit["legacy_marker_count"] == 0
    assert audit["primary_pair_audits_passed"] is True
    assert len(audit["pairwise_audits"]) == 6
    assert audit["tokenizer_parity_claimed"] is False
    assert audit["model_or_api_execution"] is False
    assert audit["real_execution_authorized"] is False
    for prompt in render_all_prompts():
        case = next(case for case in CASES if case.case_id == prompt.case_id)
        parent = render_parent_prompt(case, "reactive_no_epistemic_representation")
        for section in COMMON_SECTIONS:
            assert dict(prompt.sections)[section].encode() == dict(parent.sections)[section].encode()


def test_negative_differing_value_and_proposition_fail():
    case = _case("DP5")
    reference = render_prompt(case, MATCHED_DECISION_RECORD)
    treatment = render_prompt(case, EXPLICIT_SELF_BELIEF)
    wrong_value = _changed_section(
        treatment, FIRST_ORDER_TITLE,
        lambda text: text.replace('represented_value="decoy room"', 'represented_value="box room"'),
    )
    wrong_proposition = _changed_section(
        treatment, FIRST_ORDER_TITLE,
        lambda text: text.replace('proposition="medical_kit_location"', 'proposition="other_location"'),
    )
    assert audit_primary_pair(reference, wrong_value, case, FIRST_ORDER_TITLE)["passed"] is False
    assert audit_primary_pair(reference, wrong_proposition, case, FIRST_ORDER_TITLE)["passed"] is False


def test_negative_another_field_change_fails():
    case = _case("DP5")
    reference = render_prompt(case, MATCHED_DECISION_RECORD)
    treatment = render_prompt(case, EXPLICIT_SELF_BELIEF)
    altered = _changed_section(
        treatment, FIRST_ORDER_TITLE,
        lambda text: text.replace('agent="C"', 'agent="A"'),
    )
    result = audit_primary_pair(reference, altered, case, FIRST_ORDER_TITLE)
    assert result["exactly_one_differing_field"] is False
    assert result["passed"] is False


def test_negative_control_nested_source_and_treatment_hidden_source_fail():
    case = _case("DP7")
    reference = render_prompt(case, MESSAGE_RECORD)
    treatment = render_prompt(case, PARTNER_BELIEF)
    nested_control = replace(
        reference,
        source_fields_by_layer=tuple(
            (layer, ("second_order_representation",) if layer == "partner_context_representation" else sources)
            for layer, sources in reference.source_fields_by_layer
        ),
    )
    hidden_treatment = replace(
        treatment,
        source_fields_by_layer=tuple(
            (layer, ("hidden_world_state",) if layer == "partner_context_representation" else sources)
            for layer, sources in treatment.source_fields_by_layer
        ),
    )
    assert audit_primary_pair(nested_control, treatment, case, SECOND_ORDER_TITLE)["passed"] is False
    assert audit_primary_pair(reference, hidden_treatment, case, SECOND_ORDER_TITLE)["passed"] is False


def test_negative_duplicated_raw_evidence_fails():
    case = _case("DP7")
    reference = render_prompt(case, MESSAGE_RECORD)
    treatment = render_prompt(case, PARTNER_BELIEF)
    duplicated = replace(treatment, prompt=treatment.prompt + "\n" + case.raw_delivered_messages[0])
    result = audit_primary_pair(reference, duplicated, case, SECOND_ORDER_TITLE)
    assert result["raw_evidence_not_duplicated"] is False
    assert result["passed"] is False


def test_manifest_hashes_and_no_execution_or_dependencies():
    for relative, expected in MANIFEST["immutable_parent_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert MANIFEST["tokenizer_parity_claimed"] is False
    assert MANIFEST["tokenizer_execution_authorized"] is False
    assert MANIFEST["model_or_api_execution"] is False
    assert MANIFEST["real_execution_authorized"] is False
    source = MODULE_PATH.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "tiktoken" not in imports
    assert "openai_harmony" not in imports
    lowered = source.lower()
    for forbidden in (
        "import groq", "from groq", "openai import", "requests", "httpx",
        "urllib", "api.groq.com", "api.openai.com",
    ):
        assert forbidden not in lowered
    # Later-version workflows are allowed; this assertion is scoped to the v0.4.0 design artifact.
    assert not (
        ROOT
        / ".github/workflows"
        / "decision_point_content_matched_control_v0_4_0.yml"
    ).exists()
