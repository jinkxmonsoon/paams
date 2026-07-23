"""Prospective v0.2.1 prompt amendment separating provenance from evidence."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from .decision_point_prompting import (
    CONDITIONS_BY_FAMILY,
    EXPLICIT_SECOND,
    HIDDEN_FIELD_NAMES,
    NEUTRAL_SECOND,
    PromptAuditRecord,
    RenderedPrompt,
    _section,
    audit_record as parent_audit_record,
    forbidden_label_count,
    opaque_filler,
    render_prompt as render_parent_prompt,
)
from .decision_point_scenarios import CASES, DecisionPointCase


PROTOCOL_VERSION = "btom-v2-decision-point-prompt-protocol-0.2.1-provenance-separated-mock-only"
EVIDENCE_REF = "raw_message_1"


@dataclass(frozen=True)
class ProvenanceAudit:
    evidence_ref: str | None
    resolved_raw_message_index: int | None
    raw_message_sha256: str | None
    extracted_proposition: str | None
    extracted_value: str | None
    explicit_nested_value_matches_extracted_value: bool | None


@dataclass(frozen=True)
class PromptAuditRecordV021:
    base: PromptAuditRecord
    provenance: ProvenanceAudit
    raw_message_occurrences_in_prompt: int
    model_visible_source_message_field_count: int


def _parse_message(message: str) -> tuple[str, str]:
    prefix = "belief:"
    if not message.startswith(prefix) or "=" not in message:
        raise ValueError("structured belief message required")
    return tuple(message[len(prefix):].split("=", 1))


def _second_order(case: DecisionPointCase, neutral: bool) -> tuple[str, tuple[tuple[str, str], ...]]:
    lines = []
    fillers = []
    for index, belief in enumerate(case.second_order_representation):
        value = belief.believed_value
        if neutral:
            value = opaque_filler(f"{case.case_id}:second:{index}:value", len(value))
            fillers.append((f"second_order.{index}.believed_value", value))
        lines.extend((
            f'observer="{belief.observer}"',
            f'target_agent="{belief.target_agent}"',
            f'proposition="{belief.proposition}"',
            f'believed_value="{value}"',
            f'epistemic_status="{belief.epistemic_status}"',
            f'evidence_ref="{EVIDENCE_REF}"',
        ))
    return _section("SECOND-ORDER REPRESENTATION", tuple(lines)), tuple(fillers)


def render_prompt(case: DecisionPointCase, condition: str) -> RenderedPrompt:
    parent = render_parent_prompt(case, condition)
    if condition not in {NEUTRAL_SECOND, EXPLICIT_SECOND}:
        return parent
    second_block, fillers = _second_order(case, neutral=condition == NEUTRAL_SECOND)
    sections = tuple(
        (name, second_block if name == "SECOND-ORDER REPRESENTATION" else text)
        for name, text in parent.sections
    )
    prompt = "\n\n".join(text for _, text in sections)
    return RenderedPrompt(
        prompt_id=parent.prompt_id,
        case_id=parent.case_id,
        family=parent.family,
        condition=parent.condition,
        prompt=prompt,
        sections=sections,
        included_layers=parent.included_layers,
        source_fields_by_layer=tuple(
            (layer, tuple("evidence_ref" if field == "source_message" else field for field in fields))
            for layer, fields in parent.source_fields_by_layer
        ),
        first_order_block=parent.first_order_block,
        second_order_block=second_block,
        fillers=fillers,
    )


def render_all_prompts() -> tuple[RenderedPrompt, ...]:
    return tuple(
        render_prompt(case, condition)
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    )


def provenance_audit(case: DecisionPointCase, rendered: RenderedPrompt) -> ProvenanceAudit:
    if not case.raw_delivered_messages:
        return ProvenanceAudit(None, None, None, None, None, None)
    message = case.raw_delivered_messages[0]
    proposition, value = _parse_message(message)
    explicit_match = None
    if rendered.condition == EXPLICIT_SECOND:
        explicit_match = bool(
            case.second_order_representation
            and case.second_order_representation[0].proposition == proposition
            and case.second_order_representation[0].believed_value == value
        )
    return ProvenanceAudit(
        evidence_ref=EVIDENCE_REF if rendered.second_order_block else None,
        resolved_raw_message_index=1 if rendered.second_order_block else None,
        raw_message_sha256=hashlib.sha256(message.encode()).hexdigest(),
        extracted_proposition=proposition,
        extracted_value=value,
        explicit_nested_value_matches_extracted_value=explicit_match,
    )


def audit_record(rendered: RenderedPrompt) -> PromptAuditRecordV021:
    case = next(case for case in CASES if case.case_id == rendered.case_id)
    parent = parent_audit_record(rendered)
    message_occurrences = sum(rendered.prompt.count(message) for message in case.raw_delivered_messages)
    return PromptAuditRecordV021(
        base=parent,
        provenance=provenance_audit(case, rendered),
        raw_message_occurrences_in_prompt=message_occurrences,
        model_visible_source_message_field_count=rendered.prompt.count("source_message="),
    )


def audit_record_as_dict(record: PromptAuditRecordV021) -> dict:
    data = asdict(record.base)
    data["provenance"] = asdict(record.provenance)
    data["raw_message_occurrences_in_prompt"] = record.raw_message_occurrences_in_prompt
    data["model_visible_source_message_field_count"] = record.model_visible_source_message_field_count
    return data
