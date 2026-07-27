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
    H1_operational_variant_id: str
    H2_provenance_variant_id: str
    candidate_id: str


def _canonical_specification(
    location_surface_id: str,
    h1_variant_id: str,
    h2_variant_id: str,
) -> str:
    return json.dumps(
        {
            "H1_operational_variant_id": h1_variant_id,
            "H2_provenance_variant_id": h2_variant_id,
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
    """Render by changing only the five prospectively permitted literal fields."""
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
            specification.H1_operational_variant_id,
            specification.H2_provenance_variant_id,
        )
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    )


def _natural_ascii(value: str) -> bool:
    return value.isascii() and bool(value) and not any(
        marker in value for marker in ("loc_", "BBBBBBBB", "989_____", "padding", "filler")
    )


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
    representative = (specifications[0], specifications[len(specifications) // 2], specifications[-1])
    representative_results = []
    for specification in representative:
        prompts = render_candidate_set(specification)
        common_immutable = True
        evidence_once = True
        no_leaks = True
        representation_sources_valid = True
        for prompt in prompts:
            case = next(case for case in CASES if case.case_id == prompt.case_id)
            parent = render_parent_prompt(case, prompt.condition)
            common_immutable &= all(
                dict(prompt.sections)[name].encode() == dict(parent.sections)[name].encode()
                for name in COMMON_SECTIONS
            )
            if case.family == "DP7":
                evidence_once &= prompt.prompt.count(case.raw_delivered_messages[0]) == 1
            no_leaks &= (
                "source_message=" not in prompt.prompt
                and not any(key in prompt.prompt for key, _ in case.hidden_world_state)
                and not any(label in prompt.prompt for label in SCORING_LABELS)
                and prompt.condition not in prompt.prompt
            )
            allowed_surfaces = {
                location_surface(value, specification.location_surface_id)
                for _, value in case.first_order_representation
            } | {
                location_surface(nested.believed_value, specification.location_surface_id)
                for nested in case.second_order_representation
            }
            visible_representation_lines = [
                line for name, text in prompt.sections
                if name in {"FIRST-ORDER REPRESENTATION", "SECOND-ORDER REPRESENTATION"}
                for line in text.splitlines()
                if line.startswith(("represented_value=", "believed_value="))
            ]
            representation_sources_valid &= all(
                line.split('"', 2)[1] in allowed_surfaces
                for line in visible_representation_lines
            )
        representative_results.append({
            "candidate_id": specification.candidate_id,
            "prompt_count": len(prompts),
            "common_sections_byte_identical": common_immutable,
            "actions_and_raw_messages_frozen": common_immutable,
            "DP7_raw_evidence_exactly_once": evidence_once,
            "no_hidden_scoring_condition_or_source_message_leaks": no_leaks,
            "representation_sources_valid": representation_sources_valid,
            "real_execution_authorized": False,
        })
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
        "field_and_line_counts_frozen": True,
        "token_counts_available": False,
        "final_wording_selected": False,
        "real_execution_authorized": False,
        "representative_candidate_audits": representative_results,
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
