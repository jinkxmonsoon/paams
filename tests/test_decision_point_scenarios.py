import inspect
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from btom_v2.audit_decision_point_scenarios import PAIRS, run_audit, scripted_outcomes
from btom_v2.decision_point_scenarios import (
    Action,
    CASES,
    CASE_BY_ID,
    DecisionPointCase,
    evaluate_action,
    second_order_is_message_derived,
)


PROTOCOL = json.loads(Path("btom_v2/decision_point_protocol_v0_1.json").read_text())


def test_exactly_four_unique_immutable_one_decision_cases():
    assert len(CASES) == len(CASE_BY_ID) == 4
    assert len({case.case_id for case in CASES}) == 4
    assert all(case.maximum_decisions == 1 for case in CASES)
    with pytest.raises(FrozenInstanceError):
        CASES[0].acting_agent = "A"


def test_dp5_pair_has_only_first_order_manipulation():
    false, current = (CASE_BY_ID[name] for name in PAIRS[0])
    assert false.hidden_world_state == current.hidden_world_state
    assert false.agent_visible_observation == current.agent_visible_observation
    assert false.raw_delivered_messages == current.raw_delivered_messages == ()
    assert false.valid_actions == current.valid_actions
    assert false.behavioral_scoring == current.behavioral_scoring
    assert false.mission_rules == current.mission_rules
    assert false.acting_agent == current.acting_agent == "C"
    assert false.first_order_representation == (("medical_kit_location", "decoy_room"),)
    assert current.first_order_representation == (("medical_kit_location", "box_room"),)
    assert false.allowed_pairwise_differences == current.allowed_pairwise_differences == (
        "first_order_representation.medical_kit_location",
    )


def test_dp7_pair_has_only_communicated_target_belief_manipulation():
    stale, current = (CASE_BY_ID[name] for name in PAIRS[1])
    assert stale.hidden_world_state == current.hidden_world_state
    assert stale.agent_visible_observation == current.agent_visible_observation
    assert stale.valid_actions == current.valid_actions
    assert stale.mission_rules == current.mission_rules
    assert stale.acting_agent == current.acting_agent == "A"
    assert stale.first_order_representation == current.first_order_representation
    assert stale.raw_delivered_messages == (
        "belief:expected_medical_kit_location_after_box_open=decoy_room",
    )
    assert current.raw_delivered_messages == (
        "belief:expected_medical_kit_location_after_box_open=box_room",
    )
    assert stale.second_order_representation[0].believed_value == "decoy_room"
    assert current.second_order_representation[0].believed_value == "box_room"
    assert stale.allowed_pairwise_differences == current.allowed_pairwise_differences


def test_hidden_truth_does_not_leak_and_nested_state_is_message_derived():
    forbidden = {
        "medical_kit_actual_location",
        "actual_expected_location_after_box_open",
        "full_chain_prerequisites_required",
    }
    for case in CASES:
        visible = dict(case.agent_visible_observation)
        assert forbidden.isdisjoint(visible)
        assert second_order_is_message_derived(case)


def test_all_actions_are_legal_adjacent_and_never_pickup_nonportable_objects():
    expected = {
        "staging": ("corridor_1", "long_decoy_1", "red_room", "blue_room"),
        "box_room": ("corridor_1", "victim_room", "med_room"),
    }
    for case in CASES:
        location = dict(case.agent_visible_observation)["location"]
        moves = tuple(action.target for action in case.valid_actions if action.action == "move")
        assert moves == expected[location]
        assert all(action.action != "pickup" for action in case.valid_actions)
        assert all(action.target not in {"locked_box", "victim"} for action in case.valid_actions)
        assert all(evaluate_action(case.case_id, action).legal for action in case.valid_actions)


def test_evaluation_terminates_after_one_decision_without_full_chain_outcome():
    for case in CASES:
        result = evaluate_action(case.case_id, case.valid_actions[0])
        assert result.terminated is True
        assert result.decisions_used == 1
        assert not hasattr(result, "task_success")
        invalid = evaluate_action(case.case_id, Action("pickup", "locked_box"))
        assert invalid.legal is False
        assert invalid.classification == "invalid_or_unparseable_action"


def test_scripted_sensitivity_detects_dp5_harm_and_dp7_correction_costs():
    outcomes = scripted_outcomes()
    assert outcomes["belief_following"]["DP5a_self_belief_false"]["classification"] == "decoy_branch"
    assert outcomes["belief_following"]["DP5b_self_belief_current"]["classification"] == "productive_branch"
    assert outcomes["truth_oracle"]["DP7a_partner_belief_stale"]["classification"] == "necessary_correction"
    assert outcomes["truth_oracle"]["DP7b_partner_belief_current"]["classification"] == "appropriate_no_correction"
    assert outcomes["always_correct"]["DP7b_partner_belief_current"]["classification"] == "unnecessary_correction"
    assert outcomes["never_correct"]["DP7a_partner_belief_stale"]["classification"] == "missed_necessary_correction"
    assert all(record["legal"] for strategy in outcomes.values() for record in strategy.values())


def test_protocol_freezes_metrics_controls_and_forbids_real_execution():
    assert PROTOCOL["protocol_version"] == "btom-v2-decision-point-protocol-0.1-scaffold"
    assert PROTOCOL["status"] == "mock_only_not_authorized_for_real_execution"
    assert PROTOCOL["cases"] == [case.case_id for case in CASES]
    assert set(PROTOCOL["primary_metrics"]) == {"DP5", "DP7"}
    assert PROTOCOL["mandatory_controls"] == {
        "neutral_first_order_matched": True,
        "neutral_second_order_matched": True,
        "required_before_real_execution": True,
        "real_execution_forbidden_until_implemented_and_audited": True,
    }


def test_no_baseline_api_client_or_model_reason_dependency():
    source = inspect.getsource(__import__("btom_v2.decision_point_scenarios", fromlist=["*"]))
    lowered = source.lower()
    assert "deterministicbaseline" not in lowered
    assert "groq" not in lowered
    assert "real_llm" not in lowered
    assert "model_reason" not in lowered
    assert "llm_reason" not in lowered


def test_audit_expected_values():
    audit = run_audit()
    assert audit["case_count"] == 4
    assert audit["pair_count"] == 2
    assert audit["one_decision_cases"] is True
    assert audit["pairwise_invariance_passed"] is True
    assert audit["hidden_truth_leak_count"] == 0
    assert audit["illegal_action_count"] == 0
    assert audit["scripted_strategy_count"] == 4
    assert audit["scripted_discrimination_passed"] is True
    assert audit["parser_fallback_required"] is False
    assert audit["full_mission_completion_required"] is False
    assert audit["real_execution_authorized"] is False
