"""Deterministic, mock-only sensitivity and invariance audit."""

from __future__ import annotations

import json
from pathlib import Path

from .decision_point_scenarios import (
    Action,
    CASES,
    CASE_BY_ID,
    PROTOCOL_VERSION,
    evaluate_action,
    second_order_is_message_derived,
)


PROTOCOL_PATH = Path(__file__).with_name("decision_point_protocol_v0_1.json")
PAIRS = (
    ("DP5a_self_belief_false", "DP5b_self_belief_current"),
    ("DP7a_partner_belief_stale", "DP7b_partner_belief_current"),
)


def scripted_actions():
    return {
        "truth_oracle": {
            "DP5a_self_belief_false": Action("move", "corridor_1"),
            "DP5b_self_belief_current": Action("move", "corridor_1"),
            "DP7a_partner_belief_stale": Action("send_message", "C|kit_revealed"),
            "DP7b_partner_belief_current": Action("move", "corridor_1"),
        },
        "belief_following": {
            "DP5a_self_belief_false": Action("move", "long_decoy_1"),
            "DP5b_self_belief_current": Action("move", "corridor_1"),
        },
        "always_correct": {
            "DP7a_partner_belief_stale": Action("send_message", "C|kit_revealed"),
            "DP7b_partner_belief_current": Action("send_message", "C|kit_revealed"),
        },
        "never_correct": {
            "DP7a_partner_belief_stale": Action("move", "corridor_1"),
            "DP7b_partner_belief_current": Action("move", "corridor_1"),
        },
    }


def scripted_outcomes():
    return {
        strategy: {
            case_id: {
                "action": {"action": action.action, "target": action.target},
                "legal": result.legal,
                "classification": result.classification,
                "terminated": result.terminated,
            }
            for case_id, action in actions.items()
            for result in (evaluate_action(case_id, action),)
        }
        for strategy, actions in scripted_actions().items()
    }


def _pairwise_invariance():
    dp5a, dp5b = (CASE_BY_ID[name] for name in PAIRS[0])
    dp7a, dp7b = (CASE_BY_ID[name] for name in PAIRS[1])
    return (
        dp5a.hidden_world_state == dp5b.hidden_world_state
        and dp5a.agent_visible_observation == dp5b.agent_visible_observation
        and dp5a.valid_actions == dp5b.valid_actions
        and dp5a.behavioral_scoring == dp5b.behavioral_scoring
        and dp7a.hidden_world_state == dp7b.hidden_world_state
        and dp7a.agent_visible_observation == dp7b.agent_visible_observation
        and dp7a.valid_actions == dp7b.valid_actions
        and dp7a.first_order_representation == dp7b.first_order_representation
    )


def _hidden_truth_leaks():
    hidden_only_fields = {
        "medical_kit_actual_location",
        "actual_expected_location_after_box_open",
        "full_chain_prerequisites_required",
    }
    return sum(
        any(field in dict(case.agent_visible_observation) for field in hidden_only_fields)
        for case in CASES
    )


def _illegal_actions():
    adjacency = {
        "staging": {"corridor_1", "long_decoy_1", "red_room", "blue_room"},
        "box_room": {"corridor_1", "victim_room", "med_room"},
    }
    illegal = 0
    for case in CASES:
        location = dict(case.agent_visible_observation)["location"]
        for action in case.valid_actions:
            if action.action == "move" and action.target not in adjacency[location]:
                illegal += 1
            elif action.action == "send_message" and action.target != "C|kit_revealed":
                illegal += 1
            elif action.action not in {"move", "send_message"}:
                illegal += 1
    return illegal


def run_audit():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    outcomes = scripted_outcomes()
    discrimination = (
        outcomes["belief_following"]["DP5a_self_belief_false"]["classification"] == "decoy_branch"
        and outcomes["belief_following"]["DP5b_self_belief_current"]["classification"] == "productive_branch"
        and outcomes["always_correct"]["DP7a_partner_belief_stale"]["classification"] == "necessary_correction"
        and outcomes["always_correct"]["DP7b_partner_belief_current"]["classification"] == "unnecessary_correction"
        and outcomes["never_correct"]["DP7a_partner_belief_stale"]["classification"] == "missed_necessary_correction"
        and outcomes["never_correct"]["DP7b_partner_belief_current"]["classification"] == "appropriate_no_correction"
        and all(item["legal"] for strategy in outcomes.values() for item in strategy.values())
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "case_count": len(CASES),
        "pair_count": len(PAIRS),
        "one_decision_cases": all(case.maximum_decisions == 1 for case in CASES),
        "pairwise_invariance_passed": _pairwise_invariance(),
        "hidden_truth_leak_count": _hidden_truth_leaks(),
        "illegal_action_count": _illegal_actions(),
        "second_order_message_derivation_passed": all(second_order_is_message_derived(case) for case in CASES),
        "scripted_strategy_count": len(outcomes),
        "scripted_discrimination_passed": discrimination,
        "parser_fallback_required": False,
        "full_mission_completion_required": False,
        "real_execution_authorized": False,
        "mandatory_controls": protocol["mandatory_controls"],
        "primary_metrics": protocol["primary_metrics"],
        "scripted_outcomes": outcomes,
    }


def main():
    print(json.dumps(run_audit(), sort_keys=True))


if __name__ == "__main__":
    main()
