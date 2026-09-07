"""Mock-only content-matched epistemic framing controls, protocol v0.4.0."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, replace

from .decision_point_natural_control_prompting_v0_3_0 import (
    COMMON_SECTIONS,
    SCORING_LABELS,
    render_prompt as render_v0_3_0_prompt,
)
from .decision_point_prompting import RenderedPrompt
from .decision_point_scenarios import CASES, DecisionPointCase


PROTOCOL_VERSION = "btom-v2-decision-point-content-matched-control-0.4.0-mock-only"
REACTIVE = "reactive_no_representation"
MATCHED_DECISION_RECORD = "matched_decision_record"
EXPLICIT_SELF_BELIEF = "explicit_self_belief"
MESSAGE_RECORD = "self_belief_plus_message_record"
PARTNER_BELIEF = "self_belief_plus_partner_belief"
CONDITIONS_BY_FAMILY = {
    "DP5": (REACTIVE, MATCHED_DECISION_RECORD, EXPLICIT_SELF_BELIEF),
    "DP7": (
        REACTIVE,
        MATCHED_DECISION_RECORD,
        EXPLICIT_SELF_BELIEF,
        MESSAGE_RECORD,
        PARTNER_BELIEF,
    ),
}
LOCATION_SURFACES = {"box_room": "box room", "decoy_room": "decoy room"}
FIRST_ORDER_TITLE = "DECISION-STATE REPRESENTATION"
SECOND_ORDER_TITLE = "PARTNER-CONTEXT REPRESENTATION"
FIRST_ORDER_FIELDS = ("agent", "proposition", "represented_value", "representation_role")
SECOND_ORDER_FIELDS = (
    "observer", "related_agent", "proposition", "represented_value",
    "representation_role", "evidence_ref",
)
LEGACY_MARKERS = ("loc_", "BBBBBBBB", "989_____", "padding", "filler")


@dataclass(frozen=True)
class ContentMatchedAuditRecord:
    prompt_id: str
    family: str
    condition: str
    prompt_sha256: str
    character_count: int
    utf8_byte_count: int
    line_count: int
    approximate_whitespace_token_count: int
    raw_message_occurrence_count: int
    hidden_field_leak_count: int
    scoring_label_leak_count: int
    condition_name_leak_count: int
    source_message_field_count: int
    legacy_marker_count: int


def _section(title: str, fields: tuple[tuple[str, str], ...]) -> str:
    return "\n".join((title, *(f'{name}="{value}"' for name, value in fields)))


def _natural_location(symbolic_value: str) -> str:
    try:
        return LOCATION_SURFACES[symbolic_value]
    except KeyError as error:
        raise ValueError("unsupported frozen representation value") from error


def first_order_symbolic_value(case: DecisionPointCase) -> str:
    values = dict(case.first_order_representation)
    if set(values) != {"medical_kit_location"}:
        raise ValueError("one frozen first-order location representation required")
    return values["medical_kit_location"]


def parse_communicated_value(case: DecisionPointCase) -> str:
    """Parse the H2 reference value only from the frozen raw delivered message."""
    if case.family != "DP7" or len(case.raw_delivered_messages) != 1:
        raise ValueError("one DP7 raw message required")
    match = re.fullmatch(
        r"belief:(expected_medical_kit_location_after_box_open)=(box_room|decoy_room)",
        case.raw_delivered_messages[0],
    )
    if match is None:
        raise ValueError("raw message does not match the frozen belief statement grammar")
    return match.group(2)


def nested_partner_value(case: DecisionPointCase) -> str:
    """Read the H2 treatment value only from the frozen nested representation."""
    if case.family != "DP7" or len(case.second_order_representation) != 1:
        raise ValueError("one frozen partner representation required")
    return case.second_order_representation[0].believed_value


def decision_state_block(case: DecisionPointCase, role: str) -> str:
    if role not in {"decision_record", "self_belief"}:
        raise ValueError("unsupported decision-state representation role")
    return _section(FIRST_ORDER_TITLE, (
        ("agent", case.acting_agent),
        ("proposition", "medical_kit_location"),
        ("represented_value", _natural_location(first_order_symbolic_value(case))),
        ("representation_role", role),
    ))


def partner_context_block(case: DecisionPointCase, role: str) -> str:
    if role == "message_record":
        symbolic_value = parse_communicated_value(case)
    elif role == "partner_belief":
        symbolic_value = nested_partner_value(case)
    else:
        raise ValueError("unsupported partner-context representation role")
    return _section(SECOND_ORDER_TITLE, (
        ("observer", "A"),
        ("related_agent", "C"),
        ("proposition", "expected_medical_kit_location_after_box_open"),
        ("represented_value", _natural_location(symbolic_value)),
        ("representation_role", role),
        ("evidence_ref", "raw_message_1"),
    ))


def render_prompt(case: DecisionPointCase, condition: str) -> RenderedPrompt:
    if condition not in CONDITIONS_BY_FAMILY[case.family]:
        raise ValueError(f"condition is not applicable to {case.family}")
    parent = render_v0_3_0_prompt(case, "reactive_no_epistemic_representation")
    core = [(name, text) for name, text in parent.sections if name != "OUTPUT FORMAT"]
    blocks: list[tuple[str, str]] = []
    sources: list[tuple[str, tuple[str, ...]]] = []
    layers: list[str] = []
    first_block = ""
    second_block = ""

    if condition in {MATCHED_DECISION_RECORD, EXPLICIT_SELF_BELIEF, MESSAGE_RECORD, PARTNER_BELIEF}:
        role = "decision_record" if condition == MATCHED_DECISION_RECORD else "self_belief"
        first_block = decision_state_block(case, role)
        blocks.append((FIRST_ORDER_TITLE, first_block))
        layers.append("decision_state_representation")
        sources.append(("decision_state_representation", (
            "first_order_representation.medical_kit_location",
        )))
    if condition in {MESSAGE_RECORD, PARTNER_BELIEF}:
        role = "message_record" if condition == MESSAGE_RECORD else "partner_belief"
        second_block = partner_context_block(case, role)
        blocks.append((SECOND_ORDER_TITLE, second_block))
        layers.append("partner_context_representation")
        source = "raw_delivered_messages" if condition == MESSAGE_RECORD else "second_order_representation"
        sources.append(("partner_context_representation", (source,)))

    sections = tuple(core + blocks + [("OUTPUT FORMAT", dict(parent.sections)["OUTPUT FORMAT"])])
    return RenderedPrompt(
        prompt_id=f"{case.case_id}:{condition}",
        case_id=case.case_id,
        family=case.family,
        condition=condition,
        prompt="\n\n".join(text for _, text in sections),
        sections=sections,
        included_layers=tuple(layers),
        source_fields_by_layer=tuple(sources),
        first_order_block=first_block,
        second_order_block=second_block,
        fillers=(),
    )


def render_all_prompts() -> tuple[RenderedPrompt, ...]:
    return tuple(
        render_prompt(case, condition)
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    )


def _parse_block(block: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    lines = block.splitlines()
    if not lines:
        raise ValueError("empty representation block")
    fields = []
    for line in lines[1:]:
        match = re.fullmatch(r'([^=]+)="([^"]*)"', line)
        if match is None:
            raise ValueError("malformed representation field")
        fields.append((match.group(1), match.group(2)))
    return lines[0], tuple(fields)


def audit_primary_pair(
    reference: RenderedPrompt,
    treatment: RenderedPrompt,
    case: DecisionPointCase,
    block_title: str,
) -> dict:
    """Audit that a primary pair differs only in framing role and source provenance."""
    result = {
        "same_section_order": False,
        "other_sections_identical": False,
        "same_section_title": False,
        "same_field_names_and_order": False,
        "same_field_and_line_counts": False,
        "same_proposition": False,
        "same_represented_value": False,
        "exactly_one_differing_field": False,
        "only_representation_role_differs": False,
        "provenance_sources_valid": False,
        "raw_evidence_not_duplicated": False,
    }
    try:
        reference_block = dict(reference.sections)[block_title]
        treatment_block = dict(treatment.sections)[block_title]
        reference_section_names = tuple(name for name, _ in reference.sections)
        treatment_section_names = tuple(name for name, _ in treatment.sections)
        result["same_section_order"] = reference_section_names == treatment_section_names
        result["other_sections_identical"] = all(
            reference_text == dict(treatment.sections)[name]
            for name, reference_text in reference.sections
            if name != block_title
        )
        reference_title, reference_fields = _parse_block(reference_block)
        treatment_title, treatment_fields = _parse_block(treatment_block)
        reference_values = dict(reference_fields)
        treatment_values = dict(treatment_fields)
        reference_names = tuple(name for name, _ in reference_fields)
        treatment_names = tuple(name for name, _ in treatment_fields)
        expected_names = FIRST_ORDER_FIELDS if block_title == FIRST_ORDER_TITLE else SECOND_ORDER_FIELDS
        differences = tuple(
            name for name in reference_names
            if reference_values[name] != treatment_values[name]
        ) if reference_names == treatment_names else ()
        result.update({
            "same_section_title": reference_title == treatment_title == block_title,
            "same_field_names_and_order": reference_names == treatment_names == expected_names,
            "same_field_and_line_counts": (
                len(reference_fields) == len(treatment_fields) == len(expected_names)
                and len(reference_block.splitlines()) == len(treatment_block.splitlines())
                == len(expected_names) + 1
            ),
            "same_proposition": reference_values.get("proposition") == treatment_values.get("proposition"),
            "same_represented_value": reference_values.get("represented_value") == treatment_values.get("represented_value"),
            "exactly_one_differing_field": len(differences) == 1,
            "only_representation_role_differs": differences == ("representation_role",),
            "raw_evidence_not_duplicated": all(
                reference.prompt.count(message) == treatment.prompt.count(message) == 1
                for message in case.raw_delivered_messages
            ) if case.family == "DP7" else True,
        })
        reference_sources = dict(reference.source_fields_by_layer)
        treatment_sources = dict(treatment.source_fields_by_layer)
        if block_title == FIRST_ORDER_TITLE:
            expected = ("first_order_representation.medical_kit_location",)
            result["provenance_sources_valid"] = (
                reference_sources.get("decision_state_representation") == expected
                and treatment_sources.get("decision_state_representation") == expected
            )
        else:
            result["provenance_sources_valid"] = (
                reference_sources.get("partner_context_representation") == ("raw_delivered_messages",)
                and treatment_sources.get("partner_context_representation") == ("second_order_representation",)
                and parse_communicated_value(case) == nested_partner_value(case)
            )
    except (KeyError, ValueError):
        pass
    result["passed"] = all(result.values())
    return result


def audit_record(prompt: RenderedPrompt) -> ContentMatchedAuditRecord:
    case = next(case for case in CASES if case.case_id == prompt.case_id)
    return ContentMatchedAuditRecord(
        prompt_id=prompt.prompt_id,
        family=prompt.family,
        condition=prompt.condition,
        prompt_sha256=hashlib.sha256(prompt.prompt.encode("utf-8")).hexdigest(),
        character_count=len(prompt.prompt),
        utf8_byte_count=len(prompt.prompt.encode("utf-8")),
        line_count=len(prompt.prompt.splitlines()),
        approximate_whitespace_token_count=len(prompt.prompt.split()),
        raw_message_occurrence_count=sum(prompt.prompt.count(message) for message in case.raw_delivered_messages),
        hidden_field_leak_count=sum(key in prompt.prompt for key, _ in case.hidden_world_state),
        scoring_label_leak_count=sum(label in prompt.prompt for label in SCORING_LABELS),
        condition_name_leak_count=sum(
            condition in prompt.prompt
            for conditions in CONDITIONS_BY_FAMILY.values()
            for condition in conditions
        ),
        source_message_field_count=prompt.prompt.count("source_message="),
        legacy_marker_count=sum(marker in prompt.prompt for marker in LEGACY_MARKERS),
    )


def run_audit() -> dict:
    prompts = render_all_prompts()
    records = tuple(audit_record(prompt) for prompt in prompts)
    by_id = {prompt.prompt_id: prompt for prompt in prompts}
    parents = {
        case.case_id: render_v0_3_0_prompt(case, "reactive_no_epistemic_representation")
        for case in CASES
    }
    pairs = []
    for case in CASES:
        h1_reference = by_id[f"{case.case_id}:{MATCHED_DECISION_RECORD}"]
        h1_treatment = by_id[f"{case.case_id}:{EXPLICIT_SELF_BELIEF}"]
        pairs.append({
            "case_id": case.case_id,
            "contrast": "H1",
            **audit_primary_pair(h1_reference, h1_treatment, case, FIRST_ORDER_TITLE),
        })
        if case.family == "DP7":
            h2_reference = by_id[f"{case.case_id}:{MESSAGE_RECORD}"]
            h2_treatment = by_id[f"{case.case_id}:{PARTNER_BELIEF}"]
            pairs.append({
                "case_id": case.case_id,
                "contrast": "H2",
                **audit_primary_pair(h2_reference, h2_treatment, case, SECOND_ORDER_TITLE),
            })
    common_immutable = all(
        all(dict(prompt.sections)[name].encode("utf-8") == dict(parents[prompt.case_id].sections)[name].encode("utf-8")
            for name in COMMON_SECTIONS)
        for prompt in prompts
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_count": len(prompts),
        "DP5_prompt_count": sum(prompt.family == "DP5" for prompt in prompts),
        "DP7_prompt_count": sum(prompt.family == "DP7" for prompt in prompts),
        "common_sections_byte_identical": common_immutable,
        "valid_actions_frozen": common_immutable,
        "observations_frozen": common_immutable,
        "raw_messages_frozen": common_immutable,
        "DP7_raw_evidence_exactly_once": all(
            record.raw_message_occurrence_count == 1 for record in records if record.family == "DP7"
        ),
        "hidden_state_leak_count": sum(record.hidden_field_leak_count for record in records),
        "scoring_label_leak_count": sum(record.scoring_label_leak_count for record in records),
        "condition_name_leak_count": sum(record.condition_name_leak_count for record in records),
        "source_message_field_count": sum(record.source_message_field_count for record in records),
        "legacy_marker_count": sum(record.legacy_marker_count for record in records),
        "primary_pair_audits_passed": all(pair["passed"] for pair in pairs),
        "pairwise_audits": pairs,
        "tokenizer_parity_claimed": False,
        "model_or_api_execution": False,
        "real_execution_authorized": False,
        "prompt_records": tuple(asdict(record) for record in records),
    }
