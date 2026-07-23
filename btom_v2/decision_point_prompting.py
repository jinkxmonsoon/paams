"""Mock-only, structurally matched prompts for immutable decision-point cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .decision_point_scenarios import Action, CASES, DecisionPointCase


PROTOCOL_VERSION = "btom-v2-decision-point-prompt-protocol-0.2-mock-only"
REACTIVE = "reactive_no_explicit_belief"
NEUTRAL_FIRST = "reactive_neutral_first_order_matched"
EXPLICIT_FIRST = "explicit_first_order"
NEUTRAL_SECOND = "first_order_neutral_second_order_matched"
EXPLICIT_SECOND = "explicit_second_order"
CONDITIONS_BY_FAMILY = {
    "DP5": (REACTIVE, NEUTRAL_FIRST, EXPLICIT_FIRST),
    "DP7": (REACTIVE, NEUTRAL_FIRST, EXPLICIT_FIRST, NEUTRAL_SECOND, EXPLICIT_SECOND),
}
SECTION_ORDER = (
    "TASK",
    "LOCAL OBSERVATION",
    "RAW DELIVERED MESSAGES",
    "VALID ACTIONS",
    "FIRST-ORDER REPRESENTATION",
    "SECOND-ORDER REPRESENTATION",
    "OUTPUT FORMAT",
)
FORBIDDEN_LABELS = (
    "DP5a",
    "DP5b",
    "DP7a",
    "DP7b",
    "hidden_world_state",
    "behavioral_scoring",
    "allowed_pairwise_differences",
    "productive_branch",
    "decoy_branch",
    "necessary_correction",
    "unnecessary_correction",
    "missed_necessary_correction",
    "appropriate_no_correction",
    "truth-oracle",
    "future reward",
    "full-chain outcome",
)
EXPERIMENTAL_WORDS = ("false", "current", "stale", "treatment", "control", "oracle")
HIDDEN_FIELD_NAMES = (
    "medical_kit_actual_location",
    "actual_expected_location_after_box_open",
    "full_chain_prerequisites_required",
    "target_agent_C_at_box_room",
    "decision_history",
    "navigation_history",
)
FILLER_FORBIDDEN = (
    "UNKNOWN",
    "NEUTRAL",
    "FALSE",
    "TRUE",
    "STALE",
    "CURRENT",
    "DECOY",
    "BOX",
    "BELIEF",
    "CORRECTION",
    "KIT",
    "ROOM",
    "CONTROL",
    "MOVE",
    "SEND_MESSAGE",
    "MEDICAL_KIT_LOCATION",
)


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    case_id: str
    family: str
    condition: str
    prompt: str
    sections: tuple[tuple[str, str], ...]
    included_layers: tuple[str, ...]
    source_fields_by_layer: tuple[tuple[str, tuple[str, ...]], ...]
    first_order_block: str
    second_order_block: str
    fillers: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PromptAuditRecord:
    prompt_id: str
    case_family: str
    condition: str
    included_layers: tuple[str, ...]
    source_fields_used_by_layer: tuple[tuple[str, tuple[str, ...]], ...]
    prompt_sha256: str
    character_count: int
    utf8_byte_count: int
    line_count: int
    approximate_whitespace_token_count: int
    valid_action_count: int
    raw_message_count: int
    first_order_block_character_count: int
    second_order_block_character_count: int
    hidden_field_inclusion_count: int
    forbidden_label_inclusion_count: int
    fillers: tuple[tuple[str, str], ...]


def opaque_filler(namespace: str, length: int) -> str:
    """Return deterministic, semantically opaque text with an exact length."""
    if length < 0:
        raise ValueError("length must be non-negative")
    alphabet = "QZX789_"
    output = []
    counter = 0
    while len(output) < length:
        digest = hashlib.sha256(f"{namespace}:{length}:{counter}".encode()).digest()
        output.extend(alphabet[value % len(alphabet)] for value in digest)
        counter += 1
    filler = "".join(output[:length])
    if filler in {"A", "B", "C"} or any(term in filler for term in FILLER_FORBIDDEN):
        return opaque_filler(namespace + "_Q", length)
    return filler


def _json_lines(items: tuple[tuple[str, Any], ...]) -> tuple[str, ...]:
    def visible_value(value):
        if isinstance(value, bool):
            return int(value)
        return value

    return tuple(
        f"{key}={json.dumps(visible_value(value), separators=(',', ':'))}"
        for key, value in items
    )


def _section(title: str, lines: tuple[str, ...]) -> str:
    return "\n".join((title, *lines))


def _task_section() -> str:
    return _section("TASK", (
        "Choose exactly one listed action.",
        "Do not invent actions or targets.",
        "Do not simulate later turns.",
        "Do not claim access to hidden state.",
    ))


def _observation_section(case: DecisionPointCase) -> str:
    return _section("LOCAL OBSERVATION", _json_lines(case.agent_visible_observation))


def _messages_section(case: DecisionPointCase) -> str:
    lines = case.raw_delivered_messages or ("none",)
    return _section("RAW DELIVERED MESSAGES", tuple(lines))


def _actions_section(case: DecisionPointCase) -> str:
    lines = tuple(
        json.dumps({"action": action.action, "target": action.target}, separators=(",", ":"))
        for action in case.valid_actions
    )
    return _section("VALID ACTIONS", lines)


def _first_order(case: DecisionPointCase, neutral: bool) -> tuple[str, tuple[tuple[str, str], ...]]:
    lines = []
    fillers = []
    for key, value in case.first_order_representation:
        rendered = opaque_filler(f"{case.case_id}:first:{key}", len(value)) if neutral else value
        if neutral:
            fillers.append((f"first_order.{key}", rendered))
        lines.append(f'{key}="{rendered}"')
    return _section("FIRST-ORDER REPRESENTATION", tuple(lines)), tuple(fillers)


def _second_order(case: DecisionPointCase, neutral: bool) -> tuple[str, tuple[tuple[str, str], ...]]:
    lines = []
    fillers = []
    for index, belief in enumerate(case.second_order_representation):
        value = belief.believed_value
        source = belief.source_message
        if neutral:
            value = opaque_filler(f"{case.case_id}:second:{index}:value", len(value))
            source = opaque_filler(f"{case.case_id}:second:{index}:source", len(source))
            fillers.extend((
                (f"second_order.{index}.believed_value", value),
                (f"second_order.{index}.source_message", source),
            ))
        lines.extend((
            f'observer="{belief.observer}"',
            f'target_agent="{belief.target_agent}"',
            f'proposition="{belief.proposition}"',
            f'believed_value="{value}"',
            f'epistemic_status="{belief.epistemic_status}"',
            f'source_message="{source}"',
        ))
    return _section("SECOND-ORDER REPRESENTATION", tuple(lines)), tuple(fillers)


def _output_section() -> str:
    return _section("OUTPUT FORMAT", (
        "Return one JSON object only.",
        '{"action":"move|send_message","target":"one exact listed target","message":"","reason":"brief reason"}',
        "The reason may be logged but is never evidence of cognition.",
    ))


def render_prompt(case: DecisionPointCase, condition: str) -> RenderedPrompt:
    if condition not in CONDITIONS_BY_FAMILY[case.family]:
        raise ValueError(f"condition is not applicable to {case.family}")
    sections = [
        ("TASK", _task_section()),
        ("LOCAL OBSERVATION", _observation_section(case)),
        ("RAW DELIVERED MESSAGES", _messages_section(case)),
        ("VALID ACTIONS", _actions_section(case)),
    ]
    layers = []
    sources = [
        ("local_observation", tuple(key for key, _ in case.agent_visible_observation)),
        ("raw_messages", ("raw_delivered_messages",)),
        ("valid_actions", ("valid_actions",)),
    ]
    first_block = ""
    second_block = ""
    fillers = []
    if condition in {NEUTRAL_FIRST, EXPLICIT_FIRST, NEUTRAL_SECOND, EXPLICIT_SECOND}:
        first_block, added = _first_order(case, neutral=condition == NEUTRAL_FIRST)
        sections.append(("FIRST-ORDER REPRESENTATION", first_block))
        layers.append("first_order")
        sources.append(("first_order", tuple(key for key, _ in case.first_order_representation)))
        fillers.extend(added)
    if condition in {NEUTRAL_SECOND, EXPLICIT_SECOND}:
        second_block, added = _second_order(case, neutral=condition == NEUTRAL_SECOND)
        sections.append(("SECOND-ORDER REPRESENTATION", second_block))
        layers.append("second_order")
        sources.append(("second_order", (
            "observer", "target_agent", "proposition", "believed_value",
            "epistemic_status", "source_message",
        )))
        fillers.extend(added)
    sections.append(("OUTPUT FORMAT", _output_section()))
    prompt = "\n\n".join(text for _, text in sections)
    return RenderedPrompt(
        prompt_id=f"{case.case_id}:{condition}",
        case_id=case.case_id,
        family=case.family,
        condition=condition,
        prompt=prompt,
        sections=tuple(sections),
        included_layers=tuple(layers),
        source_fields_by_layer=tuple(sources),
        first_order_block=first_block,
        second_order_block=second_block,
        fillers=tuple(fillers),
    )


def render_all_prompts() -> tuple[RenderedPrompt, ...]:
    return tuple(
        render_prompt(case, condition)
        for case in CASES
        for condition in CONDITIONS_BY_FAMILY[case.family]
    )


def forbidden_label_count(prompt: str) -> int:
    lowered = prompt.lower()
    phrase_count = sum(lowered.count(label.lower()) for label in FORBIDDEN_LABELS)
    words = set(lowered.replace("_", " ").replace("-", " ").split())
    return phrase_count + sum(word in words for word in EXPERIMENTAL_WORDS)


def audit_record(rendered: RenderedPrompt) -> PromptAuditRecord:
    case = next(case for case in CASES if case.case_id == rendered.case_id)
    return PromptAuditRecord(
        prompt_id=rendered.prompt_id,
        case_family=rendered.family,
        condition=rendered.condition,
        included_layers=rendered.included_layers,
        source_fields_used_by_layer=rendered.source_fields_by_layer,
        prompt_sha256=hashlib.sha256(rendered.prompt.encode()).hexdigest(),
        character_count=len(rendered.prompt),
        utf8_byte_count=len(rendered.prompt.encode()),
        line_count=len(rendered.prompt.splitlines()),
        approximate_whitespace_token_count=len(rendered.prompt.split()),
        valid_action_count=len(case.valid_actions),
        raw_message_count=len(case.raw_delivered_messages),
        first_order_block_character_count=len(rendered.first_order_block),
        second_order_block_character_count=len(rendered.second_order_block),
        hidden_field_inclusion_count=sum(field in rendered.prompt for field in HIDDEN_FIELD_NAMES),
        forbidden_label_inclusion_count=forbidden_label_count(rendered.prompt),
        fillers=rendered.fillers,
    )
