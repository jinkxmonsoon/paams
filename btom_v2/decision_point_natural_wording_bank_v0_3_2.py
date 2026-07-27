"""Finite, mock-only natural wording bank for future tokenizer selection."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, replace

from .decision_point_natural_control_prompting_v0_3_0 import (
    COMMON_SECTIONS,
    CONDITIONS_BY_FAMILY,
    EXPLICIT_FIRST,
    EXPLICIT_SECOND,
    MESSAGE_PROVENANCE,
    OPERATIONAL_METADATA,
    SCORING_LABELS,
    render_prompt as render_parent_prompt,
)
from .decision_point_scenarios import CASES, DecisionPointCase


DESIGN_VERSION = "btom-v2-decision-point-natural-wording-bank-0.3.2-mock-only"
LOCATION_SURFACES = (
    ("location_surface_01", ("box room", "decoy room")),
    ("location_surface_02", ("the box room", "the decoy room")),
    ("location_surface_03", ("box-room location", "decoy-room location")),
    ("location_surface_04", ("location: box room", "location: decoy room")),
)
DECISION_SCOPES = (
    "single_action", "exactly_one_action", "one_action_only", "one_listed_action",
)
ACTION_SOURCES = (
    "listed_actions", "provided_action_list", "available_listed_actions",
)
OUTPUT_MODES = ("json_object", "single_json_object", "one_json_object")
MESSAGE_TYPES = (
    "structured_statement", "one_structured_statement", "structured_message",
)
DELIVERY_STATUSES = ("available", "available_to_observer", "delivered_and_available")
LEGACY_REFERENCES = (
    {
        "symbolic_values": ("box_room", "decoy_room"),
        "reference_only": True,
        "rejected_for_final_selection": True,
        "rejection_reason": "one-token interaction differential in v0.3.1",
    },
)


def _numbered_variants(prefix: str, values):
    return tuple(
        (f"{prefix}_{index:02d}", value)
        for index, value in enumerate(values, 1)
    )


H1_OPERATIONAL_VARIANTS = _numbered_variants(
    "h1_operational_variant",
    itertools.product(DECISION_SCOPES, ACTION_SOURCES, OUTPUT_MODES),
)
H2_PROVENANCE_VARIANTS = _numbered_variants(
    "h2_provenance_variant",
    itertools.product(MESSAGE_TYPES, DELIVERY_STATUSES),
)


@dataclass(frozen=True)
class CandidateSpecification:
    location_surface_id: str
    h1_operational_variant_id: str
    h2_provenance_variant_id: str
    candidate_id: str


def _canonical_specification(
    location_surface_id: str,
    h1_variant_id: str,
    h2_variant_id: str,
) -> str:
    return json.dumps(
        {
            "h1_operational_variant_id": h1_variant_id,
            "h2_provenance_variant_id": h2_variant_id,
            "location_surface_id": location_surface_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def candidate_id(location_surface_id: str, h1_variant_id: str, h2_variant_id: str) -> str:
    return hashlib.sha256(
        _canonical_specification(location_surface_id, h1_variant_id, h2_variant_id).encode()
    ).hexdigest()


def candidate_specifications() -> tuple[CandidateSpecification, ...]:
    return tuple(
        CandidateSpecification(
            location_id,
            h1_id,
            h2_id,
            candidate_id(location_id, h1_id, h2_id),
        )
        for location_id, _ in LOCATION_SURFACES
        for h1_id, _ in H1_OPERATIONAL_VARIANTS
        for h2_id, _ in H2_PROVENANCE_VARIANTS
    )


def _lookup(identifier: str, bank):
    try:
        return dict(bank)[identifier]
    except KeyError as error:
        raise ValueError(f"unknown frozen wording-bank identifier: {identifier}") from error


def location_surface(symbolic_value: str, surface_id: str) -> str:
    box_surface, decoy_surface = _lookup(surface_id, LOCATION_SURFACES)
    mapping = {"box_room": box_surface, "decoy_room": decoy_surface}
    try:
        return mapping[symbolic_value]
    except KeyError as error:
        raise ValueError("surface mapping only accepts frozen representation values") from error


def _replace_quoted_field(block: str, field: str, value: str) -> str:
    lines = block.splitlines()
    prefix = f'{field}="'
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one model-visible field: {field}")
    lines[matches[0]] = f'{field}="{value}"'
    return "\n".join(lines)


def render_prompt_variant(
    case: DecisionPointCase,
    condition: str,
    location_surface_id: str,
    h1_operational_variant_id: str,
    h2_provenance_variant_id: str,
):
    """Render by changing only the seven prospectively permitted field values."""
    _lookup(location_surface_id, LOCATION_SURFACES)
    _lookup(h1_operational_variant_id, H1_OPERATIONAL_VARIANTS)
    _lookup(h2_provenance_variant_id, H2_PROVENANCE_VARIANTS)
    parent = render_parent_prompt(case, condition)
    sections = list(parent.sections)
    first_order = parent.first_order_block
    second_order = parent.second_order_block

    if condition in {EXPLICIT_FIRST, MESSAGE_PROVENANCE, EXPLICIT_SECOND}:
        frozen_value = dict(case.first_order_representation)["medical_kit_location"]
        first_order = _replace_quoted_field(
            first_order,
            "represented_value",
            location_surface(frozen_value, location_surface_id),
        )
    if condition == EXPLICIT_SECOND:
        frozen_value = case.second_order_representation[0].believed_value
        second_order = _replace_quoted_field(
            second_order,
            "believed_value",
            location_surface(frozen_value, location_surface_id),
        )
    if condition == OPERATIONAL_METADATA:
        decision_scope, action_source, output_mode = _lookup(
            h1_operational_variant_id, H1_OPERATIONAL_VARIANTS
        )
        metadata = dict(sections)["DECISION METADATA"]
        metadata = _replace_quoted_field(metadata, "decision_scope", decision_scope)
        metadata = _replace_quoted_field(metadata, "action_source", action_source)
        metadata = _replace_quoted_field(metadata, "output_mode", output_mode)
        sections = [
            (name, metadata if name == "DECISION METADATA" else text)
            for name, text in sections
        ]
    if condition == MESSAGE_PROVENANCE:
        message_type, delivery_status = _lookup(
            h2_provenance_variant_id, H2_PROVENANCE_VARIANTS
        )
        provenance = dict(sections)["MESSAGE PROVENANCE METADATA"]
        provenance = _replace_quoted_field(provenance, "message_type", message_type)
        provenance = _replace_quoted_field(provenance, "delivery_status", delivery_status)
        sections = [
            (name, provenance if name == "MESSAGE PROVENANCE METADATA" else text)
            for name, text in sections
        ]
    sections = [
        (
            name,
            first_order if name == "FIRST-ORDER REPRESENTATION"
            else second_order if name == "SECOND-ORDER REPRESENTATION"
            else text,
        )
        for name, text in sections
    ]
    sections = tuple(sections)
    return replace(
        parent,
        prompt="\n\n".join(text for _, text in sections),
        sections=sections,
        first_order_block=first_order,
        second_order_block=second_order,
        fillers=(),
    )


def render_candidate_set(specification: CandidateSpecification):
    return tuple(
        render_prompt_variant(
            case,
            condition,
            specification.location_surface_id,
            specification.h1_operational_variant_id,
            specification.h2_provenance_variant_id,
        )
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    )


def _natural_ascii(value: str) -> bool:
    return value.isascii() and bool(value) and not any(
        marker in value for marker in ("loc_", "BBBBBBBB", "989_____", "padding", "filler")
    )


PERMITTED_REPLACEMENT_FIELDS = frozenset({
    "represented_value",
    "believed_value",
    "decision_scope",
    "action_source",
    "output_mode",
    "message_type",
    "delivery_status",
})
PERMITTED_FIELDS_BY_SECTION = {
    "FIRST-ORDER REPRESENTATION": frozenset({"represented_value"}),
    "SECOND-ORDER REPRESENTATION": frozenset({"believed_value"}),
    "DECISION METADATA": frozenset({"decision_scope", "action_source", "output_mode"}),
    "MESSAGE PROVENANCE METADATA": frozenset({"message_type", "delivery_status"}),
}
EXPECTED_FIELDS_BY_SECTION = {
    "FIRST-ORDER REPRESENTATION": (
        "agent", "proposition", "represented_value", "representation_scope",
    ),
    "SECOND-ORDER REPRESENTATION": (
        "observer", "target_agent", "proposition", "believed_value",
        "epistemic_status", "evidence_ref",
    ),
    "DECISION METADATA": ("agent", "decision_scope", "action_source", "output_mode"),
    "MESSAGE PROVENANCE METADATA": (
        "observer", "message_sender", "message_type", "content_reference",
        "delivery_status", "evidence_ref",
    ),
}


def _parse_block(block: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Parse one rendered section, rejecting malformed or duplicate fields."""
    lines = block.splitlines()
    if not lines:
        raise ValueError("empty model-visible section")
    fields = []
    for line in lines[1:]:
        if "=" not in line:
            raise ValueError("model-visible field line lacks an equals sign")
        name, quoted = line.split("=", 1)
        if not name or len(quoted) < 2 or not quoted.startswith('"') or not quoted.endswith('"'):
            raise ValueError("malformed model-visible field line")
        fields.append((name, quoted[1:-1]))
    names = [name for name, _ in fields]
    if len(names) != len(set(names)):
        raise ValueError("duplicate model-visible field")
    return lines[0], tuple(fields)


def audit_prompt_structure(prompt, parent, case: DecisionPointCase) -> dict:
    """Calculate structural invariants for one candidate prompt against v0.3.0."""
    result = {
        "section_order_frozen": False,
        "field_names_frozen": False,
        "field_order_frozen": False,
        "field_counts_frozen": False,
        "line_counts_frozen": False,
        "permitted_replacement_boundary_passed": False,
        "common_sections_byte_identical": False,
        "raw_messages_byte_identical": False,
        "valid_actions_byte_identical": False,
        "DP7_raw_evidence_exactly_once": case.family != "DP7",
        "no_leaks": False,
    }
    try:
        candidate_names = tuple(name for name, _ in prompt.sections)
        parent_names = tuple(name for name, _ in parent.sections)
        result["section_order_frozen"] = candidate_names == parent_names
        candidate_sections = dict(prompt.sections)
        parent_sections = dict(parent.sections)
        result["common_sections_byte_identical"] = all(
            candidate_sections.get(name, "").encode() == parent_sections[name].encode()
            for name in COMMON_SECTIONS
        )
        result["raw_messages_byte_identical"] = (
            candidate_sections.get("RAW DELIVERED MESSAGES")
            == parent_sections["RAW DELIVERED MESSAGES"]
        )
        result["valid_actions_byte_identical"] = (
            candidate_sections.get("VALID ACTIONS") == parent_sections["VALID ACTIONS"]
        )

        names_frozen = True
        order_frozen = True
        counts_frozen = True
        lines_frozen = True
        boundary_passed = True
        boundary_passed &= prompt.prompt == "\n\n".join(text for _, text in prompt.sections)
        for section_name, parent_text in parent.sections:
            if section_name in COMMON_SECTIONS:
                continue
            candidate_text = candidate_sections[section_name]
            parent_title, parent_fields = _parse_block(parent_text)
            candidate_title, candidate_fields = _parse_block(candidate_text)
            parent_field_names = tuple(name for name, _ in parent_fields)
            candidate_field_names = tuple(name for name, _ in candidate_fields)
            expected_names = EXPECTED_FIELDS_BY_SECTION[section_name]
            names_frozen &= set(candidate_field_names) == set(parent_field_names) == set(expected_names)
            order_frozen &= candidate_field_names == parent_field_names == expected_names
            counts_frozen &= len(candidate_fields) == len(parent_fields)
            lines_frozen &= len(candidate_text.splitlines()) == len(parent_text.splitlines())
            boundary_passed &= candidate_title == parent_title == section_name
            if candidate_field_names == parent_field_names:
                for (field, candidate_value), (_, parent_value) in zip(
                    candidate_fields, parent_fields
                ):
                    if (
                        candidate_value != parent_value
                        and field not in PERMITTED_FIELDS_BY_SECTION.get(section_name, frozenset())
                    ):
                        boundary_passed = False
            else:
                boundary_passed = False
        result.update({
            "field_names_frozen": names_frozen,
            "field_order_frozen": order_frozen,
            "field_counts_frozen": counts_frozen,
            "line_counts_frozen": lines_frozen,
            "permitted_replacement_boundary_passed": boundary_passed,
        })
        if case.family == "DP7":
            result["DP7_raw_evidence_exactly_once"] = all(
                prompt.prompt.count(message) == 1 for message in case.raw_delivered_messages
            )
        result["no_leaks"] = (
            "source_message=" not in prompt.prompt
            and not any(key in prompt.prompt for key, _ in case.hidden_world_state)
            and not any(label in prompt.prompt for label in SCORING_LABELS)
            and prompt.condition not in prompt.prompt
            and not any(marker in prompt.prompt for marker in ("loc_", "BBBBBBBB", "989_____"))
        )
    except (KeyError, ValueError):
        pass
    result["passed"] = all(result.values())
    return result


def _candidate_values_match(prompt, case: DecisionPointCase, specification) -> bool:
    """Verify that permitted replacements come from the selected frozen bank entries."""
    sections = dict(prompt.sections)
    if "FIRST-ORDER REPRESENTATION" in sections:
        _, fields = _parse_block(sections["FIRST-ORDER REPRESENTATION"])
        expected = location_surface(
            dict(case.first_order_representation)["medical_kit_location"],
            specification.location_surface_id,
        )
        if dict(fields)["represented_value"] != expected:
            return False
    if "SECOND-ORDER REPRESENTATION" in sections:
        _, fields = _parse_block(sections["SECOND-ORDER REPRESENTATION"])
        expected = location_surface(
            case.second_order_representation[0].believed_value,
            specification.location_surface_id,
        )
        if dict(fields)["believed_value"] != expected:
            return False
    if "DECISION METADATA" in sections:
        _, fields = _parse_block(sections["DECISION METADATA"])
        values = dict(fields)
        expected = _lookup(
            specification.h1_operational_variant_id, H1_OPERATIONAL_VARIANTS
        )
        if (values["decision_scope"], values["action_source"], values["output_mode"]) != expected:
            return False
        if values["agent"] != case.acting_agent:
            return False
    if "MESSAGE PROVENANCE METADATA" in sections:
        _, fields = _parse_block(sections["MESSAGE PROVENANCE METADATA"])
        values = dict(fields)
        expected = _lookup(
            specification.h2_provenance_variant_id, H2_PROVENANCE_VARIANTS
        )
        if (values["message_type"], values["delivery_status"]) != expected:
            return False
        frozen = dict(_parse_block(
            dict(render_parent_prompt(case, MESSAGE_PROVENANCE).sections)[
                "MESSAGE PROVENANCE METADATA"
            ]
        )[1])
        for field in ("observer", "message_sender", "content_reference", "evidence_ref"):
            if values[field] != frozen[field]:
                return False
    return True


def audit_wording_bank() -> dict:
    location_values = [value for _, pair in LOCATION_SURFACES for value in pair]
    h1_values = [value for _, variant in H1_OPERATIONAL_VARIANTS for value in variant]
    h2_values = [value for _, variant in H2_PROVENANCE_VARIANTS for value in variant]
    metadata_forbidden = (
        "belief", "believes", "false", "stale", "outcome", "recommend",
        "productive", "decoy", "score",
    )
    bank_values = location_values + h1_values + h2_values
    specifications = candidate_specifications()
    parent_by_prompt_id = {
        f"{case.case_id}:{condition}": render_parent_prompt(case, condition)
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    }
    cases_by_id = {case.case_id: case for case in CASES}
    structural_results = []
    failed_candidate_ids = []
    prompts_audited = 0
    for specification in specifications:
        prompts = render_candidate_set(specification)
        candidate_passed = len(prompts) == 16
        candidate_passed &= sum(prompt.family == "DP5" for prompt in prompts) == 6
        candidate_passed &= sum(prompt.family == "DP7" for prompt in prompts) == 10
        for prompt in prompts:
            prompts_audited += 1
            result = audit_prompt_structure(
                prompt, parent_by_prompt_id[prompt.prompt_id], cases_by_id[prompt.case_id]
            )
            structural_results.append(result)
            candidate_passed &= result["passed"]
            candidate_passed &= _candidate_values_match(prompt, cases_by_id[prompt.case_id], specification)
        if not candidate_passed:
            failed_candidate_ids.append(specification.candidate_id)
    structural_keys = (
        "section_order_frozen", "field_names_frozen", "field_order_frozen",
        "field_counts_frozen", "line_counts_frozen",
        "permitted_replacement_boundary_passed",
    )
    aggregates = {
        key: all(result[key] for result in structural_results) for key in structural_keys
    }
    return {
        "design_version": DESIGN_VERSION,
        "location_surface_count": len(LOCATION_SURFACES),
        "H1_operational_variant_count": len(H1_OPERATIONAL_VARIANTS),
        "H2_provenance_variant_count": len(H2_PROVENANCE_VARIANTS),
        "candidate_specification_count": len(specifications),
        "unique_candidate_id_count": len({item.candidate_id for item in specifications}),
        "all_bank_values_natural_ascii": all(_natural_ascii(value) for value in bank_values),
        "metadata_controls_non_epistemic": not any(
            term in value.lower() for value in h1_values + h2_values for term in metadata_forbidden
        ),
        "candidate_sets_audited": len(specifications),
        "prompts_audited": prompts_audited,
        "candidate_sets_failed": len(failed_candidate_ids),
        "failed_candidate_ids": failed_candidate_ids,
        **aggregates,
        "field_and_line_counts_frozen": all(
            aggregates[key]
            for key in ("field_names_frozen", "field_order_frozen", "field_counts_frozen", "line_counts_frozen")
        ),
        "token_counts_available": False,
        "final_wording_selected": False,
        "real_execution_authorized": False,
    }


FUTURE_SELECTION_RULE = {
    "required_exact_parity": {
        "all_16_structural_audits": True,
        "every_H1_primary_raw_and_Harmony_delta": 0,
        "every_H2_primary_raw_and_Harmony_delta": 0,
        "H1_raw_and_Harmony_interaction_differentials": 0,
        "H2_raw_and_Harmony_interaction_differentials": 0,
    },
    "tolerance_threshold": None,
    "ranking": (
        "fewer_characters_across_four_compared_blocks",
        "fewer_changed_literal_values_relative_to_v0.3.0",
        "lexicographic_candidate_specification",
    ),
    "if_none_satisfy": "no_candidate",
    "automatic_relaxation_authorized": False,
}
