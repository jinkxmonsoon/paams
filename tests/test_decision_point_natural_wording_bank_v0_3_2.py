import hashlib
import json
from pathlib import Path

from btom_v2.decision_point_natural_control_prompting_v0_3_0 import (
    COMMON_SECTIONS, CONDITIONS_BY_FAMILY, render_prompt as render_parent_prompt,
)
from btom_v2.decision_point_natural_wording_bank_v0_3_2 import (
    ACTION_SOURCES, DECISION_SCOPES, DELIVERY_STATUSES, FUTURE_SELECTION_RULE,
    H1_OPERATIONAL_VARIANTS, H2_PROVENANCE_VARIANTS, LOCATION_SURFACES,
    MESSAGE_TYPES, OUTPUT_MODES, audit_wording_bank, candidate_id,
    candidate_specifications, render_candidate_set, render_prompt_variant,
)
from btom_v2.decision_point_scenarios import CASES


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "btom_v2/decision_point_natural_wording_bank_manifest_v0_3_2.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def test_exact_location_surface_bank():
    assert LOCATION_SURFACES == (
        ("location_surface_01", ("box room", "decoy room")),
        ("location_surface_02", ("the box room", "the decoy room")),
        ("location_surface_03", ("box-room location", "decoy-room location")),
        ("location_surface_04", ("location: box room", "location: decoy room")),
    )
    assert MANIFEST["location_surface_bank"] == {
        identifier: {"box_room": pair[0], "decoy_room": pair[1]}
        for identifier, pair in LOCATION_SURFACES
    }


def test_exact_H1_and_H2_banks():
    assert DECISION_SCOPES == ("single_action", "exactly_one_action", "one_action_only", "one_listed_action")
    assert ACTION_SOURCES == ("listed_actions", "provided_action_list", "available_listed_actions")
    assert OUTPUT_MODES == ("json_object", "single_json_object", "one_json_object")
    assert len(H1_OPERATIONAL_VARIANTS) == 36
    assert len({value for _, value in H1_OPERATIONAL_VARIANTS}) == 36
    assert MESSAGE_TYPES == ("structured_statement", "one_structured_statement", "structured_message")
    assert DELIVERY_STATUSES == ("available", "available_to_observer", "delivered_and_available")
    assert len(H2_PROVENANCE_VARIANTS) == 9
    assert len({value for _, value in H2_PROVENANCE_VARIANTS}) == 9
    assert len(MANIFEST["H1_operational_variants"]) == 36
    assert len(MANIFEST["H2_provenance_variants"]) == 9


def test_candidate_specs_are_complete_unique_stable_and_hash_identified():
    specifications = candidate_specifications()
    assert len(specifications) == 1296
    assert len(set(specifications)) == 1296
    assert len({item.candidate_id for item in specifications}) == 1296
    assert specifications == tuple(sorted(
        specifications,
        key=lambda item: (
            item.location_surface_id,
            item.H1_operational_variant_id,
            item.H2_provenance_variant_id,
        ),
    ))
    first = specifications[0]
    assert first.candidate_id == candidate_id(
        first.location_surface_id,
        first.H1_operational_variant_id,
        first.H2_provenance_variant_id,
    )
    assert MANIFEST["expected_candidate_triples"] == 1296


def test_rendering_changes_only_permitted_fields_and_preserves_common_sections():
    specification = candidate_specifications()[0]
    prompts = render_candidate_set(specification)
    assert len(prompts) == 16
    for prompt in prompts:
        case = next(case for case in CASES if case.case_id == prompt.case_id)
        parent = render_parent_prompt(case, prompt.condition)
        for name in COMMON_SECTIONS:
            assert dict(prompt.sections)[name].encode() == dict(parent.sections)[name].encode()
        assert case.raw_delivered_messages == next(
            original.raw_delivered_messages for original in CASES if original.case_id == case.case_id
        )
        assert tuple((action.action, action.target) for action in case.valid_actions) == tuple(
            (action.action, action.target)
            for action in next(original for original in CASES if original.case_id == case.case_id).valid_actions
        )


def test_all_representative_audits_pass_without_execution():
    audit = audit_wording_bank()
    assert audit["location_surface_count"] == 4
    assert audit["H1_operational_variant_count"] == 36
    assert audit["H2_provenance_variant_count"] == 9
    assert audit["candidate_specification_count"] == 1296
    assert audit["unique_candidate_id_count"] == 1296
    assert audit["all_bank_values_natural_ascii"] is True
    assert audit["metadata_controls_non_epistemic"] is True
    assert audit["field_and_line_counts_frozen"] is True
    assert audit["token_counts_available"] is False
    assert audit["final_wording_selected"] is False
    assert audit["real_execution_authorized"] is False
    for record in audit["representative_candidate_audits"]:
        assert record["prompt_count"] == 16
        assert record["common_sections_byte_identical"] is True
        assert record["actions_and_raw_messages_frozen"] is True
        assert record["DP7_raw_evidence_exactly_once"] is True
        assert record["no_hidden_scoring_condition_or_source_message_leaks"] is True
        assert record["representation_sources_valid"] is True
        assert record["real_execution_authorized"] is False


def test_selection_rule_is_exact_prospective_and_nonrelaxing():
    assert FUTURE_SELECTION_RULE["tolerance_threshold"] is None
    assert set(FUTURE_SELECTION_RULE["required_exact_parity"].values()) == {True, 0}
    assert FUTURE_SELECTION_RULE["ranking"] == (
        "fewer_characters_across_four_compared_blocks",
        "fewer_changed_literal_values_relative_to_v0.3.0",
        "lexicographic_candidate_specification",
    )
    assert FUTURE_SELECTION_RULE["if_none_satisfy"] == "no_candidate"
    assert FUTURE_SELECTION_RULE["automatic_relaxation_authorized"] is False


def test_manifest_hashes_and_no_execution_authorization():
    for relative, expected in MANIFEST["immutable_parent_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert MANIFEST["tokenizer_execution_authorized"] is False
    assert MANIFEST["model_execution_authorized"] is False
    assert MANIFEST["real_execution_authorized"] is False
    assert MANIFEST["final_wording_selected"] is False


def test_no_filler_padding_tokenizer_api_or_workflow():
    source = (ROOT / "btom_v2/decision_point_natural_wording_bank_v0_3_2.py").read_text().lower()
    for forbidden in (
        "tiktoken", "openai_harmony", "from groq", "import groq", "openai import",
        "requests", "httpx", "urllib", "api.groq.com", "api.openai.com",
    ):
        assert forbidden not in source
    assert not list((ROOT / ".github/workflows").glob("*natural_wording_bank*"))
    visible_values = [value for _, pair in LOCATION_SURFACES for value in pair]
    visible_values += [value for _, variant in H1_OPERATIONAL_VARIANTS for value in variant]
    visible_values += [value for _, variant in H2_PROVENANCE_VARIANTS for value in variant]
    for value in visible_values:
        assert all(marker not in value for marker in ("loc_", "BBBBBBBB", "989_____", "filler", "padding"))


def test_render_function_has_exact_signature_and_rejects_unknown_ids():
    case = CASES[0]
    try:
        render_prompt_variant(case, CONDITIONS_BY_FAMILY[case.family][0], "bad", "bad", "bad")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown frozen identifier accepted")
