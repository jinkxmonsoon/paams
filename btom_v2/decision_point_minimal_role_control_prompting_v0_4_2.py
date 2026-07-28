"""Mock-only v0.4.2 wrapper normalizing representation roles to record/belief."""

from __future__ import annotations

from dataclasses import asdict, replace

from .decision_point_content_matched_control_prompting_v0_4_0 import (
    CONDITIONS_BY_FAMILY,
    EXPLICIT_SELF_BELIEF,
    FIRST_ORDER_TITLE,
    MATCHED_DECISION_RECORD,
    MESSAGE_RECORD,
    PARTNER_BELIEF,
    SECOND_ORDER_TITLE,
    audit_primary_pair as audit_parent_pair,
    audit_record,
    parse_communicated_value,
    nested_partner_value,
    render_prompt as render_parent_prompt,
)
from .decision_point_natural_control_prompting_v0_3_0 import COMMON_SECTIONS
from .decision_point_scenarios import CASES, DecisionPointCase


PROTOCOL_VERSION = "btom-v2-decision-point-minimal-role-control-0.4.2-mock-only"
REFERENCE_ROLE = "record"
TREATMENT_ROLE = "belief"
ROLE_NORMALIZATION = {
    "decision_record": REFERENCE_ROLE,
    "self_belief": TREATMENT_ROLE,
    "message_record": REFERENCE_ROLE,
    "partner_belief": TREATMENT_ROLE,
}
LEGACY_ROLE_VALUES = tuple(ROLE_NORMALIZATION)


def _normalize_role(block: str) -> str:
    lines = block.splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith('representation_role="')]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one representation_role field")
    index = matches[0]
    old = lines[index].split('"', 2)[1]
    try:
        new = ROLE_NORMALIZATION[old]
    except KeyError as error:
        raise RuntimeError("unexpected frozen v0.4.0 representation role") from error
    lines[index] = f'representation_role="{new}"'
    return "\n".join(lines)


def render_prompt(case: DecisionPointCase, condition: str):
    """Delegate to v0.4.0 and replace only model-visible representation_role values."""
    parent = render_parent_prompt(case, condition)
    sections = tuple(
        (name, _normalize_role(text) if name in {FIRST_ORDER_TITLE, SECOND_ORDER_TITLE} else text)
        for name, text in parent.sections
    )
    return replace(
        parent,
        prompt="\n\n".join(text for _, text in sections),
        sections=sections,
        first_order_block=dict(sections).get(FIRST_ORDER_TITLE, ""),
        second_order_block=dict(sections).get(SECOND_ORDER_TITLE, ""),
    )


def render_all_prompts():
    return tuple(
        render_prompt(case, condition)
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    )


def _field_map(block: str) -> dict[str, str]:
    return {
        line.split("=", 1)[0]: line.split('"', 2)[1]
        for line in block.splitlines()[1:]
    }


def _only_role_changed(parent, rendered) -> bool:
    if tuple(name for name, _ in parent.sections) != tuple(name for name, _ in rendered.sections):
        return False
    for (parent_name, parent_text), (name, text) in zip(parent.sections, rendered.sections):
        if parent_name != name:
            return False
        parent_lines = parent_text.splitlines()
        lines = text.splitlines()
        if len(parent_lines) != len(lines):
            return False
        for parent_line, line in zip(parent_lines, lines):
            if parent_line != line and not (
                parent_line.startswith('representation_role="')
                and line.startswith('representation_role="')
                and ROLE_NORMALIZATION.get(parent_line.split('"', 2)[1]) == line.split('"', 2)[1]
            ):
                return False
    return True


def audit_pair(reference, treatment, case: DecisionPointCase, block_title: str) -> dict:
    base = audit_parent_pair(reference, treatment, case, block_title)
    reference_block = dict(reference.sections)[block_title]
    treatment_block = dict(treatment.sections)[block_title]
    reference_fields = _field_map(reference_block)
    treatment_fields = _field_map(treatment_block)
    result = {
        **base,
        "identical_character_count": len(reference.prompt) == len(treatment.prompt),
        "identical_utf8_byte_count": len(reference.prompt.encode()) == len(treatment.prompt.encode()),
        "identical_block_character_count": len(reference_block) == len(treatment_block),
        "identical_block_utf8_byte_count": len(reference_block.encode()) == len(treatment_block.encode()),
        "reference_role_is_record": reference_fields.get("representation_role") == REFERENCE_ROLE,
        "treatment_role_is_belief": treatment_fields.get("representation_role") == TREATMENT_ROLE,
        "evidence_reference_identical": (
            reference_fields.get("evidence_ref") == treatment_fields.get("evidence_ref")
        ),
    }
    # The parent pair audit's exact role-difference and provenance checks remain applicable.
    result["passed"] = all(value for key, value in result.items() if key != "passed")
    return result


def run_audit() -> dict:
    prompts = render_all_prompts()
    by_id = {prompt.prompt_id: prompt for prompt in prompts}
    parent_by_id = {
        f"{case.case_id}:{condition}": render_parent_prompt(case, condition)
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    }
    pairwise = []
    for case in CASES:
        pairwise.append({
            "case_id": case.case_id,
            "contrast": "H1",
            **audit_pair(
                by_id[f"{case.case_id}:{MATCHED_DECISION_RECORD}"],
                by_id[f"{case.case_id}:{EXPLICIT_SELF_BELIEF}"],
                case,
                FIRST_ORDER_TITLE,
            ),
        })
        if case.family == "DP7":
            pairwise.append({
                "case_id": case.case_id,
                "contrast": "H2",
                **audit_pair(
                    by_id[f"{case.case_id}:{MESSAGE_RECORD}"],
                    by_id[f"{case.case_id}:{PARTNER_BELIEF}"],
                    case,
                    SECOND_ORDER_TITLE,
                ),
            })
    records = tuple(audit_record(prompt) for prompt in prompts)
    common_frozen = all(
        all(dict(prompt.sections)[name] == dict(parent_by_id[prompt.prompt_id].sections)[name]
            for name in COMMON_SECTIONS)
        for prompt in prompts
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_count": len(prompts),
        "DP5_prompt_count": sum(prompt.family == "DP5" for prompt in prompts),
        "DP7_prompt_count": sum(prompt.family == "DP7" for prompt in prompts),
        "conditions_unchanged": CONDITIONS_BY_FAMILY,
        "only_representation_role_changed_from_v0_4_0": all(
            _only_role_changed(parent_by_id[prompt.prompt_id], prompt) for prompt in prompts
        ),
        "common_sections_byte_identical_to_v0_4_0": common_frozen,
        "observations_actions_messages_output_frozen": common_frozen,
        "pairwise_audit_count": len(pairwise),
        "pairwise_audits_passed": all(pair["passed"] for pair in pairwise),
        "pairwise_audits": pairwise,
        "independent_H2_provenance_preserved": all(
            parse_communicated_value(case) == nested_partner_value(case)
            for case in CASES if case.family == "DP7"
        ),
        "DP7_raw_evidence_exactly_once": all(
            record.raw_message_occurrence_count == 1 for record in records if record.family == "DP7"
        ),
        "hidden_state_leak_count": sum(record.hidden_field_leak_count for record in records),
        "scoring_label_leak_count": sum(record.scoring_label_leak_count for record in records),
        "condition_name_leak_count": sum(record.condition_name_leak_count for record in records),
        "source_message_field_count": sum(record.source_message_field_count for record in records),
        "legacy_marker_count": sum(record.legacy_marker_count for record in records),
        "role_values_ascii_six_characters": all(
            role.isascii() and len(role) == 6 and len(role.split()) == 1
            for role in (REFERENCE_ROLE, TREATMENT_ROLE)
        ),
        "tokenizer_counts_produced": False,
        "tokenizer_parity_claimed": False,
        "model_or_api_execution": False,
        "real_execution_authorized": False,
        "prompt_records": tuple(asdict(record) for record in records),
    }
