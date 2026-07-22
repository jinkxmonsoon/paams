import json

import pytest

from btom_v2.env import BTomEnvV2
from btom_v2.epistemic_state import AgentEpistemicState, MEDICAL_KIT_LOCATION
from btom_v2.llm_policy import enumerate_valid_actions
from btom_v2.prompts import PromptBuilder
from test_scenario_discriminativeness import FirstOrderScript, ReactiveScript, SecondOrderScript, action, common_input


SCENARIOS = ("C7a_partner_belief_stale", "C7b_partner_belief_current")
DECOY_ROUTE = {"long_decoy_1", "long_decoy_2", "decoy_room"}


def legal_step(env, agent, action_type, target):
    result = env.step(agent, action_type, target, {})
    assert result.invalid is False, result.reason
    env.assert_invariants()
    return result


def reach_correction_decision(scenario):
    """Reach A's correction decision using only observations and legal actions."""
    env = BTomEnvV2(scenario, 0, max_turns=60)
    env.assert_invariants()

    c_initial = env.get_observation("C")
    stated_belief = c_initial.beliefs[MEDICAL_KIT_LOCATION]
    legal_step(env, "C", "send_message", f"A|belief:{MEDICAL_KIT_LOCATION}={stated_belief}")
    legal_step(env, "A", "move", "red_room")
    legal_step(env, "B", "move", "blue_room")
    legal_step(env, "A", "pickup", "red_key")
    legal_step(env, "B", "pickup", "blue_key")
    legal_step(env, "A", "move", "staging")
    legal_step(env, "B", "move", "staging")
    legal_step(env, "A", "move", "corridor_1")
    legal_step(env, "B", "move", "corridor_1")
    legal_step(env, "A", "move", "box_room")
    legal_step(env, "B", "move", "box_room")
    legal_step(env, "A", "open_box", None)
    legal_step(env, "B", "open_box", None)

    observation_a = env.get_observation("A")
    valid_actions = enumerate_valid_actions(env, "A", observation_a)
    epistemic = AgentEpistemicState()
    epistemic.update_from_observation("A", observation_a)
    first_order = epistemic.first_order_for("A")
    second_order = epistemic.second_order_for("A")
    return env, observation_a, valid_actions, first_order, second_order


def physical_snapshot(env):
    return {
        "locations": dict(env.state.locations),
        "inventories": {agent: list(items) for agent, items in env.state.inventories.items()},
        "room_items": {room: list(items) for room, items in env.state.room_items.items()},
        "task_status": dict(env.state.task_status),
        "turn": env.state.turn,
    }


def prompt_bundle(env, observation, valid_actions, first_order, second_order):
    builder = PromptBuilder()
    prompts = {
        "LLMReactive": builder.build("LLMReactive", "A", env.scenario_id, observation, valid_actions),
        "LLMBeliefState": builder.build(
            "LLMBeliefState", "A", env.scenario_id, observation, valid_actions,
            belief_state=first_order,
        ),
        "LLMBToM": builder.build(
            "LLMBToM", "A", env.scenario_id, observation, valid_actions,
            belief_state=first_order, second_order_state=second_order,
        ),
    }
    return {
        name: {
            "prompt": prompt,
            "characters": len(prompt),
            "tokens_approx": len(prompt.split()),
        }
        for name, prompt in prompts.items()
    }


class TargetRouteScript:
    """Test-only C controller using observation, valid actions and C's first order."""

    received_layers = ("common", "valid_actions", "first_order")

    def choose(self, common, valid_actions, first_order):
        available = {(item["action"], item["target"]) for item in valid_actions}
        belief = first_order["beliefs"][MEDICAL_KIT_LOCATION]["believed_value"]
        location = common["location"]
        inventory = common["inventory"]
        for candidate in (("rescue", None), ("pickup", "medical_kit")):
            if candidate in available:
                return action(*candidate)
        if "medical_kit" in inventory:
            destination = "victim_room"
        elif belief == "decoy_room":
            destination = {"staging": "long_decoy_1", "long_decoy_1": "long_decoy_2", "long_decoy_2": "decoy_room"}.get(location, location)
        else:
            destination = {
                "decoy_room": "long_decoy_2", "long_decoy_2": "long_decoy_1", "long_decoy_1": "staging",
                "staging": "corridor_1", "corridor_1": "box_room", "box_room": "box_room",
            }.get(location, location)
        if ("move", destination) not in available:
            destination = location
        return action("move", destination)


def choose_observer_action(condition, observation, valid_actions, first_order, second_order):
    common = common_input(observation)
    if condition == "ReactiveScript":
        return ReactiveScript().choose(common, valid_actions)
    if condition == "FirstOrderScript":
        return FirstOrderScript().choose(common, valid_actions, first_order)
    return SecondOrderScript().choose(common, valid_actions, first_order, second_order)


def drive_c_to_terminal(env):
    epistemic = AgentEpistemicState()
    route = TargetRouteScript()
    while not env.state.done:
        observation = env.get_observation("C")
        epistemic.update_from_observation("C", observation)
        valid_actions = enumerate_valid_actions(env, "C", observation)
        chosen = route.choose(common_input(observation), valid_actions, epistemic.first_order_for("C"))
        target = chosen["target"]
        legal_step(env, "C", chosen["action"], target)
    return env.summary()


def run_reachable_trajectory(scenario, condition):
    env, observation, valid_actions, first_order, second_order = reach_correction_decision(scenario)
    prompts = prompt_bundle(env, observation, valid_actions, first_order, second_order)
    chosen = choose_observer_action(condition, observation, valid_actions, first_order, second_order)
    target = f'{chosen["target"]}|{chosen["message"]}' if chosen["action"] == "send_message" else chosen["target"]
    legal_step(env, "A", chosen["action"], target)
    legal_step(env, "B", "move", "box_room")
    summary = drive_c_to_terminal(env)
    return {
        "scenario": scenario,
        "condition": condition,
        "decision_turn": 13,
        "raw_evidence": list(observation.delivered_messages),
        "observer_task_status": dict(observation.task_status),
        "first_order": first_order,
        "second_order": second_order,
        "valid_actions": valid_actions,
        "chosen_action": chosen,
        "prompt_lengths": {
            name: {"characters": item["characters"], "tokens_approx": item["tokens_approx"]}
            for name, item in prompts.items()
        },
        "turns": summary.turns,
        "correction_opportunities": summary.correction_opportunities,
        "necessary_correction_messages": summary.necessary_correction_messages,
        "missed_necessary_corrections": summary.missed_necessary_corrections,
        "unnecessary_correction_messages": summary.unnecessary_correction_messages,
        "evidence_to_correction_send_steps": summary.evidence_to_correction_send_steps,
        "correction_delivery_latency": summary.correction_delivery_latency,
        "target_stale_belief_steps": summary.target_stale_belief_steps,
        "target_decoy_branch_steps": summary.target_decoy_branch_steps,
        "post_correction_decoy_steps": summary.post_correction_decoy_steps,
        "total_messages": summary.total_messages,
        "time_to_medical_kit_acquisition": summary.time_to_medical_kit_acquired,
        "time_to_rescue": summary.time_to_rescue,
        "task_success": summary.success,
        "final_task_status": dict(summary.task_status),
        "final_locations": dict(summary.agent_locations),
    }


def test_invalid_reveal_state_is_rejected_by_invariant_checker():
    env = BTomEnvV2("C7a_partner_belief_stale", 0)
    env.state.task_status["medical_kit_revealed"] = True
    with pytest.raises(AssertionError):
        env.assert_invariants()


def test_paired_decision_states_are_legally_reachable_and_physically_matched():
    stale = reach_correction_decision("C7a_partner_belief_stale")
    current = reach_correction_decision("C7b_partner_belief_current")
    stale_env, stale_obs, stale_actions, _, stale_second = stale
    current_env, current_obs, current_actions, _, current_second = current
    assert physical_snapshot(stale_env) == physical_snapshot(current_env)
    assert stale_actions == current_actions
    assert stale_obs.location == current_obs.location == "box_room"
    assert stale_obs.visible_items == current_obs.visible_items == ["medical_kit"]
    assert stale_obs.task_status == current_obs.task_status
    assert stale_second["beliefs_about_others"]["C"][MEDICAL_KIT_LOCATION]["epistemic_status"] == "stale"
    assert current_second["beliefs_about_others"]["C"][MEDICAL_KIT_LOCATION]["believed_value"] == "box_room"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_c_does_not_receive_global_reveal_status(scenario):
    env, observation_a, _, _, _ = reach_correction_decision(scenario)
    observation_c = env.get_observation("C")
    assert observation_a.task_status == {
        "red_key_applied": True, "blue_key_applied": True, "locked_box_open": True,
        "medical_kit_revealed": True, "victim_rescued": False,
    }
    assert observation_c.task_status == {
        "red_key_applied": False, "blue_key_applied": False, "locked_box_open": False,
        "medical_kit_revealed": False, "victim_rescued": False,
    }


def test_key_and_box_status_requires_own_action_or_local_box_observation():
    env = BTomEnvV2("C7a_partner_belief_stale", 0)
    legal_step(env, "A", "move", "red_room")
    legal_step(env, "A", "pickup", "red_key")
    legal_step(env, "A", "move", "staging")
    legal_step(env, "A", "move", "corridor_1")
    legal_step(env, "A", "move", "box_room")
    legal_step(env, "A", "open_box", None)
    assert env.get_observation("A").task_status["red_key_applied"] is True
    assert env.get_observation("B").task_status["red_key_applied"] is False
    assert env.get_observation("C").task_status["red_key_applied"] is False
    assert env.get_observation("A").task_status["locked_box_open"] is False


def test_delivered_correction_is_observable_to_c_and_updates_location_belief():
    env, observation, valid_actions, first_order, second_order = reach_correction_decision("C7a_partner_belief_stale")
    chosen = SecondOrderScript().choose(common_input(observation), valid_actions, first_order, second_order)
    legal_step(env, "A", "send_message", f'{chosen["target"]}|{chosen["message"]}')
    legal_step(env, "B", "move", "box_room")
    observation_c = env.get_observation("C")
    assert observation_c.beliefs[MEDICAL_KIT_LOCATION] == "box_room"
    assert observation_c.task_status["medical_kit_revealed"] is True


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_prompt_conditions_share_raw_evidence_observation_and_actions(scenario):
    env, observation, valid_actions, first_order, second_order = reach_correction_decision(scenario)
    bundle = prompt_bundle(env, observation, valid_actions, first_order, second_order)
    prompts = {name: item["prompt"] for name, item in bundle.items()}
    strip_layers = lambda prompt: "\n".join(
        line for line in prompt.splitlines()
        if not line.startswith(("FIRST-ORDER BELIEFS:", "SECOND-ORDER BELIEFS:"))
    )
    assert len({strip_layers(prompt) for prompt in prompts.values()}) == 1
    assert all(env.scenario_id not in prompt for prompt in prompts.values())
    assert "FIRST-ORDER BELIEFS:" not in prompts["LLMReactive"]
    assert "SECOND-ORDER BELIEFS:" not in prompts["LLMReactive"]
    assert "FIRST-ORDER BELIEFS:" in prompts["LLMBeliefState"]
    assert "SECOND-ORDER BELIEFS:" not in prompts["LLMBeliefState"]
    assert "SECOND-ORDER BELIEFS:" in prompts["LLMBToM"]
    assert all(item["characters"] > 0 and item["tokens_approx"] > 0 for item in bundle.values())


def test_second_order_script_corrects_stale_but_not_current_partner_belief():
    stale = run_reachable_trajectory("C7a_partner_belief_stale", "SecondOrderScript")
    current = run_reachable_trajectory("C7b_partner_belief_current", "SecondOrderScript")
    assert stale["chosen_action"] == action("send_message", "C", "kit_revealed")
    assert stale["necessary_correction_messages"] == 1
    assert stale["missed_necessary_corrections"] == 0
    assert stale["correction_delivery_latency"] == 1
    assert stale["target_decoy_branch_steps"] == 0
    assert current["chosen_action"] == action("move", "box_room")
    assert current["unnecessary_correction_messages"] == 0
    assert stale["task_success"] is current["task_success"] is True


def test_omitted_necessary_correction_has_objective_decoy_cost():
    omitted = run_reachable_trajectory("C7a_partner_belief_stale", "ReactiveScript")
    corrected = run_reachable_trajectory("C7a_partner_belief_stale", "SecondOrderScript")
    assert omitted["missed_necessary_corrections"] == 1
    assert omitted["target_stale_belief_steps"] > corrected["target_stale_belief_steps"]
    assert omitted["target_decoy_branch_steps"] == 5
    assert corrected["target_decoy_branch_steps"] == 0
    assert omitted["time_to_rescue"] > corrected["time_to_rescue"]


def test_unnecessary_correction_has_objective_message_cost():
    env, observation, valid_actions, first_order, second_order = reach_correction_decision("C7b_partner_belief_current")
    assert SecondOrderScript().choose(common_input(observation), valid_actions, first_order, second_order) == action("move", "box_room")
    turn_before = env.state.turn
    legal_step(env, "A", "send_message", "C|kit_revealed")
    metrics = env.correction_metrics()
    assert env.state.turn == turn_before + 1
    assert metrics["unnecessary_correction_messages"] == 1
    assert metrics["total_messages"] == 2


def test_changing_only_nested_target_belief_changes_script_decision():
    env, observation, valid_actions, first_order, stale = reach_correction_decision("C7a_partner_belief_stale")
    current = json.loads(json.dumps(stale))
    current["beliefs_about_others"]["C"][MEDICAL_KIT_LOCATION].update({
        "believed_value": "box_room", "epistemic_status": "communicated",
    })
    script = SecondOrderScript()
    common = common_input(observation)
    assert script.choose(common, valid_actions, first_order, stale) == action("send_message", "C", "kit_revealed")
    assert script.choose(common, valid_actions, first_order, current) == action("move", "box_room")


def test_reachable_scripts_do_not_access_hidden_state_or_baseline():
    for script_class in (ReactiveScript, FirstOrderScript, SecondOrderScript, TargetRouteScript):
        names = set(script_class.choose.__code__.co_names)
        assert "state" not in names
        assert "scenario_id" not in names
        assert "DeterministicBaselinePolicy" not in names


if __name__ == "__main__":
    print(json.dumps({
        scenario: {
            condition: run_reachable_trajectory(scenario, condition)
            for condition in ("ReactiveScript", "FirstOrderScript", "SecondOrderScript")
        }
        for scenario in SCENARIOS
    }, indent=2, sort_keys=True))
