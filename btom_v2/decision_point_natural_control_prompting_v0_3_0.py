"""Mock-only natural metadata controls for immutable decision-point cases."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .decision_point_prompting import RenderedPrompt, _section
from .decision_point_prompting_v0_2_1 import render_prompt as render_parent_prompt
from .decision_point_scenarios import CASES, DecisionPointCase


PROTOCOL_VERSION = "btom-v2-decision-point-natural-control-design-0.3.0-mock-only"
REACTIVE = "reactive_no_epistemic_representation"
OPERATIONAL_METADATA = "operational_metadata_reference"
EXPLICIT_FIRST = "explicit_first_order"
MESSAGE_PROVENANCE = "first_order_plus_message_provenance"
EXPLICIT_SECOND = "explicit_second_order"
CONDITIONS_BY_FAMILY = {
    "DP5": (REACTIVE, OPERATIONAL_METADATA, EXPLICIT_FIRST),
    "DP7": (
        REACTIVE,
        OPERATIONAL_METADATA,
        EXPLICIT_FIRST,
        MESSAGE_PROVENANCE,
        EXPLICIT_SECOND,
    ),
}
COMMON_SECTIONS = (
    "TASK", "LOCAL OBSERVATION", "RAW DELIVERED MESSAGES",
    "VALID ACTIONS", "OUTPUT FORMAT",
)
SCORING_LABELS = (
    "productive_branch", "decoy_branch", "neutral_branch",
    "necessary_correction", "missed_necessary_correction",
    "unnecessary_correction", "appropriate_no_correction",
)
LEGACY_FILLERS = ("loc_", "BBBBBBBB", "989_____")


@dataclass(frozen=True)
class NaturalControlAuditRecord:
    prompt_id: str
    case_family: str
    condition: str
    prompt_sha256: str
    character_count: int
    utf8_byte_count: int
    line_count: int
    approximate_whitespace_token_count: int
    included_blocks: tuple[str, ...]
    block_field_counts: tuple[tuple[str, int], ...]
    raw_message_occurrence_count: int
    hidden_field_leak_count: int
    scoring_label_leak_count: int
    condition_name_leak_count: int
    source_message_field_count: int
    legacy_filler_count: int
    valid_action_count: int


def first_order_block(case: DecisionPointCase) -> str:
    values = dict(case.first_order_representation)
    return _section("FIRST-ORDER REPRESENTATION", (
        f'agent="{case.acting_agent}"',
        'proposition="medical_kit_location"',
        f'represented_value="{values["medical_kit_location"]}"',
        'representation_scope="current_decision"',
    ))


def operational_metadata_block(case: DecisionPointCase) -> str:
    return _section("DECISION METADATA", (
        f'agent="{case.acting_agent}"',
        'decision_scope="single_action"',
        'action_source="listed_actions"',
        'output_mode="json_object"',
    ))


def second_order_block(case: DecisionPointCase) -> str:
    if len(case.second_order_representation) != 1:
        raise ValueError("exactly one frozen second-order representation required")
    nested = case.second_order_representation[0]
    return _section("SECOND-ORDER REPRESENTATION", (
        f'observer="{nested.observer}"',
        f'target_agent="{nested.target_agent}"',
        f'proposition="{nested.proposition}"',
        f'believed_value="{nested.believed_value}"',
        f'epistemic_status="{nested.epistemic_status}"',
        'evidence_ref="raw_message_1"',
    ))


def message_provenance_block(case: DecisionPointCase) -> str:
    if case.family != "DP7" or len(case.raw_delivered_messages) != 1:
        raise ValueError("message provenance requires one DP7 raw message")
    return _section("MESSAGE PROVENANCE METADATA", (
        'observer="A"',
        'message_sender="C"',
        'message_type="structured_statement"',
        'content_reference="raw_message_1"',
        'delivery_status="available"',
        'evidence_ref="raw_message_1"',
    ))


def _block_fields(block: str) -> int:
    return max(0, len(block.splitlines()) - 1)


def render_prompt(case: DecisionPointCase, condition: str) -> RenderedPrompt:
    if condition not in CONDITIONS_BY_FAMILY[case.family]:
        raise ValueError(f"condition is not applicable to {case.family}")
    parent = render_parent_prompt(case, "reactive_no_explicit_belief")
    core = [(name, text) for name, text in parent.sections if name != "OUTPUT FORMAT"]
    blocks: list[tuple[str, str]] = []
    sources = list(parent.source_fields_by_layer)
    layers = []
    if condition == OPERATIONAL_METADATA:
        blocks.append(("DECISION METADATA", operational_metadata_block(case)))
        layers.append("operational_metadata")
        sources.append(("operational_metadata", (
            "acting_agent", "single_decision_scope", "listed_actions", "json_output_mode",
        )))
    elif condition == EXPLICIT_FIRST:
        blocks.append(("FIRST-ORDER REPRESENTATION", first_order_block(case)))
        layers.append("first_order")
        sources.append(("first_order", ("acting_agent", "first_order_representation")))
    elif condition == MESSAGE_PROVENANCE:
        blocks.extend((
            ("FIRST-ORDER REPRESENTATION", first_order_block(case)),
            ("MESSAGE PROVENANCE METADATA", message_provenance_block(case)),
        ))
        layers.extend(("first_order", "message_provenance"))
        sources.extend((
            ("first_order", ("acting_agent", "first_order_representation")),
            ("message_provenance", ("acting_agent", "raw_message_presence")),
        ))
    elif condition == EXPLICIT_SECOND:
        blocks.extend((
            ("FIRST-ORDER REPRESENTATION", first_order_block(case)),
            ("SECOND-ORDER REPRESENTATION", second_order_block(case)),
        ))
        layers.extend(("first_order", "second_order"))
        sources.extend((
            ("first_order", ("acting_agent", "first_order_representation")),
            ("second_order", ("second_order_representation", "raw_message_1")),
        ))
    sections = tuple(core + blocks + [("OUTPUT FORMAT", dict(parent.sections)["OUTPUT FORMAT"])])
    prompt = "\n\n".join(text for _, text in sections)
    return RenderedPrompt(
        prompt_id=f"{case.case_id}:{condition}",
        case_id=case.case_id,
        family=case.family,
        condition=condition,
        prompt=prompt,
        sections=sections,
        included_layers=tuple(layers),
        source_fields_by_layer=tuple(sources),
        first_order_block=dict(blocks).get("FIRST-ORDER REPRESENTATION", ""),
        second_order_block=dict(blocks).get("SECOND-ORDER REPRESENTATION", ""),
        fillers=(),
    )


def render_all_prompts() -> tuple[RenderedPrompt, ...]:
    return tuple(
        render_prompt(case, condition)
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    )


def audit_record(rendered: RenderedPrompt) -> NaturalControlAuditRecord:
    case = next(case for case in CASES if case.case_id == rendered.case_id)
    blocks = tuple(
        (name, _block_fields(text))
        for name, text in rendered.sections if name not in COMMON_SECTIONS
    )
    return NaturalControlAuditRecord(
        prompt_id=rendered.prompt_id,
        case_family=rendered.family,
        condition=rendered.condition,
        prompt_sha256=hashlib.sha256(rendered.prompt.encode()).hexdigest(),
        character_count=len(rendered.prompt),
        utf8_byte_count=len(rendered.prompt.encode()),
        line_count=len(rendered.prompt.splitlines()),
        approximate_whitespace_token_count=len(rendered.prompt.split()),
        included_blocks=tuple(name for name, _ in blocks),
        block_field_counts=blocks,
        raw_message_occurrence_count=sum(
            rendered.prompt.count(message) for message in case.raw_delivered_messages
        ),
        hidden_field_leak_count=sum(
            key in rendered.prompt for key, _ in case.hidden_world_state
        ),
        scoring_label_leak_count=sum(label in rendered.prompt for label in SCORING_LABELS),
        condition_name_leak_count=sum(
            condition in rendered.prompt
            for conditions in CONDITIONS_BY_FAMILY.values()
            for condition in conditions
        ),
        source_message_field_count=rendered.prompt.count("source_message="),
        legacy_filler_count=sum(value in rendered.prompt for value in LEGACY_FILLERS),
        valid_action_count=len(case.valid_actions),
    )


def run_audit() -> dict:
    rendered = render_all_prompts()
    records = tuple(audit_record(prompt) for prompt in rendered)
    parents = {
        case.case_id: render_parent_prompt(case, "reactive_no_explicit_belief")
        for case in CASES
    }
    common_immutable = all(
        all(dict(prompt.sections)[name] == dict(parents[prompt.case_id].sections)[name]
            for name in COMMON_SECTIONS)
        for prompt in rendered
    )
    by_id = {prompt.prompt_id: prompt for prompt in rendered}
    h1_equal = all(
        _block_fields(by_id[f"{case.case_id}:{EXPLICIT_FIRST}"].first_order_block)
        == _block_fields(dict(by_id[f"{case.case_id}:{OPERATIONAL_METADATA}"].sections)["DECISION METADATA"])
        == 4
        and len(by_id[f"{case.case_id}:{EXPLICIT_FIRST}"].first_order_block.splitlines())
        == len(dict(by_id[f"{case.case_id}:{OPERATIONAL_METADATA}"].sections)["DECISION METADATA"].splitlines())
        for case in CASES if case.family == "DP5"
    )
    h2_equal = all(
        _block_fields(by_id[f"{case.case_id}:{EXPLICIT_SECOND}"].second_order_block)
        == _block_fields(dict(by_id[f"{case.case_id}:{MESSAGE_PROVENANCE}"].sections)["MESSAGE PROVENANCE METADATA"])
        == 6
        and len(by_id[f"{case.case_id}:{EXPLICIT_SECOND}"].second_order_block.splitlines())
        == len(dict(by_id[f"{case.case_id}:{MESSAGE_PROVENANCE}"].sections)["MESSAGE PROVENANCE METADATA"].splitlines())
        for case in CASES if case.family == "DP7"
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_count": len(rendered),
        "DP5_prompt_count": sum(prompt.family == "DP5" for prompt in rendered),
        "DP7_prompt_count": sum(prompt.family == "DP7" for prompt in rendered),
        "common_section_immutability_passed": common_immutable,
        "valid_actions_frozen": all(
            dict(prompt.sections)["VALID ACTIONS"]
            == dict(parents[prompt.case_id].sections)["VALID ACTIONS"]
            for prompt in rendered
        ),
        "raw_messages_frozen": all(
            dict(prompt.sections)["RAW DELIVERED MESSAGES"]
            == dict(parents[prompt.case_id].sections)["RAW DELIVERED MESSAGES"]
            for prompt in rendered
        ),
        "DP7_raw_message_exactly_once": all(
            record.raw_message_occurrence_count == 1
            for record in records if record.case_family == "DP7"
        ),
        "hidden_field_leak_count": sum(record.hidden_field_leak_count for record in records),
        "scoring_label_leak_count": sum(record.scoring_label_leak_count for record in records),
        "condition_name_leak_count": sum(record.condition_name_leak_count for record in records),
        "source_message_field_count": sum(record.source_message_field_count for record in records),
        "legacy_filler_count": sum(record.legacy_filler_count for record in records),
        "H1_equal_field_and_line_counts": h1_equal,
        "H2_equal_field_and_line_counts": h2_equal,
        "tokenizer_parity_claimed": False,
        "model_or_API_execution": False,
        "real_execution_authorized": False,
        "prompt_records": records,
    }
