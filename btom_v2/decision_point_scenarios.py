"""Immutable, mock-only one-decision cases for epistemic sensitivity audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PROTOCOL_VERSION = "btom-v2-decision-point-protocol-0.1-scaffold"
EXPECTED_LOCATION_PROPOSITION = "expected_medical_kit_location_after_box_open"


@dataclass(frozen=True)
class Action:
    action: str
    target: str


@dataclass(frozen=True)
class NestedBelief:
    observer: str
    target_agent: str
    proposition: str
    believed_value: str
    epistemic_status: str
    source_message: str


@dataclass(frozen=True)
class DecisionPointCase:
    case_id: str
    purpose: str
    family: str
    acting_agent: str
    hidden_world_state: tuple[tuple[str, Any], ...]
    agent_visible_observation: tuple[tuple[str, Any], ...]
    raw_delivered_messages: tuple[str, ...]
    first_order_representation: tuple[tuple[str, str], ...]
    second_order_representation: tuple[NestedBelief, ...]
    valid_actions: tuple[Action, ...]
    behavioral_scoring: tuple[tuple[Action, str], ...]
    allowed_pairwise_differences: tuple[str, ...]
    mission_rules: tuple[str, ...]
    maximum_decisions: int = 1


@dataclass(frozen=True)
class DecisionResult:
    case_id: str
    submitted_action: Action | None
    legal: bool
    classification: str
    terminated: bool = True
    decisions_used: int = 1


DP5_ACTIONS = (
    Action("move", "corridor_1"),
    Action("move", "long_decoy_1"),
    Action("move", "red_room"),
    Action("move", "blue_room"),
)
DP5_SCORING = (
    (DP5_ACTIONS[0], "productive_branch"),
    (DP5_ACTIONS[1], "decoy_branch"),
    (DP5_ACTIONS[2], "neutral_branch"),
    (DP5_ACTIONS[3], "neutral_branch"),
)
DP5_HIDDEN_STATE = (
    ("acting_agent_location", "staging"),
    ("medical_kit_actual_location", "box_room"),
    ("navigation_history", ()),
    ("full_chain_prerequisites_required", False),
)
DP5_VISIBLE_OBSERVATION = (
    ("agent", "C"),
    ("location", "staging"),
    ("visible_items", ()),
    ("inventory", ()),
    ("delivered_message_count", 0),
)
DP5_RULES = (
    "Exactly one action is submitted.",
    "All listed moves are adjacent to staging.",
    "No full rescue mission is simulated.",
)

DP7_ACTIONS = (
    Action("send_message", "C|kit_revealed"),
    Action("move", "corridor_1"),
    Action("move", "victim_room"),
    Action("move", "med_room"),
)
DP7_HIDDEN_STATE = (
    ("acting_agent_location", "box_room"),
    ("medical_kit_actual_location", "box_room"),
    ("actual_expected_location_after_box_open", "box_room"),
    ("target_agent_C_at_box_room", False),
    ("decision_history", ()),
)
DP7_VISIBLE_OBSERVATION = (
    ("agent", "A"),
    ("location", "box_room"),
    ("visible_items", ("medical_kit",)),
    ("inventory", ()),
    ("target_agent_C_visible", False),
)
DP7_RULES = (
    "Exactly one action is submitted.",
    "The raw delivered statement is visible in every future representation arm.",
    "No message delivery or full rescue mission follows the decision.",
)


def _dp5(case_id: str, belief: str, purpose: str) -> DecisionPointCase:
    return DecisionPointCase(
        case_id=case_id,
        purpose=purpose,
        family="DP5",
        acting_agent="C",
        hidden_world_state=DP5_HIDDEN_STATE,
        agent_visible_observation=DP5_VISIBLE_OBSERVATION,
        raw_delivered_messages=(),
        first_order_representation=(("medical_kit_location", belief),),
        second_order_representation=(),
        valid_actions=DP5_ACTIONS,
        behavioral_scoring=DP5_SCORING,
        allowed_pairwise_differences=("first_order_representation.medical_kit_location",),
        mission_rules=DP5_RULES,
    )


def _dp7(case_id: str, target_belief: str, purpose: str, correction_label: str, other_label: str) -> DecisionPointCase:
    message = f"belief:{EXPECTED_LOCATION_PROPOSITION}={target_belief}"
    scoring = tuple(
        (action, correction_label if action.action == "send_message" else other_label)
        for action in DP7_ACTIONS
    )
    nested = NestedBelief(
        observer="A",
        target_agent="C",
        proposition=EXPECTED_LOCATION_PROPOSITION,
        believed_value=target_belief,
        epistemic_status="communicated",
        source_message=message,
    )
    return DecisionPointCase(
        case_id=case_id,
        purpose=purpose,
        family="DP7",
        acting_agent="A",
        hidden_world_state=DP7_HIDDEN_STATE,
        agent_visible_observation=DP7_VISIBLE_OBSERVATION,
        raw_delivered_messages=(message,),
        first_order_representation=(("medical_kit_location", "box_room"),),
        second_order_representation=(nested,),
        valid_actions=DP7_ACTIONS,
        behavioral_scoring=scoring,
        allowed_pairwise_differences=(
            "raw_delivered_messages.target_belief_value",
            "second_order_representation.C.expected_location",
            "behavioral_scoring.derived_correction_necessity",
        ),
        mission_rules=DP7_RULES,
    )


CASES = (
    _dp5("DP5a_self_belief_false", "decoy_room", "Test an action-relevant false first-order belief."),
    _dp5("DP5b_self_belief_current", "box_room", "Test a current action-relevant first-order belief."),
    _dp7(
        "DP7a_partner_belief_stale",
        "decoy_room",
        "Test a necessary correction based on another agent's stale belief.",
        "necessary_correction",
        "missed_necessary_correction",
    ),
    _dp7(
        "DP7b_partner_belief_current",
        "box_room",
        "Test avoidance of an unnecessary correction when another agent's belief is current.",
        "unnecessary_correction",
        "appropriate_no_correction",
    ),
)
CASE_BY_ID = {case.case_id: case for case in CASES}


def evaluate_action(case_id: str, action: Action | None) -> DecisionResult:
    """Classify one submitted action and terminate; no future state is simulated."""
    case = CASE_BY_ID[case_id]
    scoring = dict(case.behavioral_scoring)
    if action not in case.valid_actions:
        return DecisionResult(case_id, action, False, "invalid_or_unparseable_action")
    return DecisionResult(case_id, action, True, scoring[action])


def second_order_is_message_derived(case: DecisionPointCase) -> bool:
    return all(
        belief.source_message in case.raw_delivered_messages
        and belief.source_message
        == f"belief:{belief.proposition}={belief.believed_value}"
        for belief in case.second_order_representation
    )


def case_as_dict(case: DecisionPointCase) -> dict[str, Any]:
    return asdict(case)
