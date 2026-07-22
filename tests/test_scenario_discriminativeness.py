import json
from copy import deepcopy

import pytest

from btom_v2.env import BTomEnvV2
from btom_v2.epistemic_state import AgentEpistemicState, MEDICAL_KIT_LOCATION
from btom_v2.llm_policy import enumerate_valid_actions


SCENARIO_CLASSIFICATIONS = {
    "C1_fully_observable": "non-discriminative",
    "C2_partial_observable": "non-discriminative",
    "C4_communication_delay": "currently ambiguous",
    "C4b_costly_communication_delay": "currently ambiguous",
    "C4c_wrong_branch_communication_delay": "currently ambiguous",
    "C5_false_belief_injection": "discriminative for first-order belief",
    "C5b_costly_false_belief": "discriminative for first-order belief",
    "C6_resource_allocation": "non-discriminative",
}

DECOY_BRANCH = {"long_decoy_1", "long_decoy_2", "decoy_room"}


def common_input(observation):
    return {
        "agent": observation.agent,
        "location": observation.location,
        "visible_items": list(observation.visible_items),
        "inventory": list(observation.inventory),
        "task_status": dict(observation.task_status),
        "delivered_messages": [
            {"from": message.get("from"), "content": message.get("content")}
            for message in observation.delivered_messages
        ],
    }


def action(action_type, target, message=""):
    return {"action": action_type, "target": target, "message": message}


class ReactiveScript:
    """Test fixture restricted to common prompt information."""

    received_layers = ("common", "valid_actions")

    def choose(self, common, valid_actions):
        return action("move", common["location"])


class FirstOrderScript:
    """Test fixture that can inspect first-order, but not nested, beliefs."""

    received_layers = ("common", "valid_actions", "first_order")

    def choose(self, common, valid_actions, first_order):
        belief = first_order["beliefs"][MEDICAL_KIT_LOCATION]["believed_value"]
        move_targets = {item["target"] for item in valid_actions if item["action"] == "move"}
        if belief == "decoy_room" and "long_decoy_1" in move_targets:
            return action("move", "long_decoy_1")
        if belief == "box_room" and "corridor_1" in move_targets:
            return action("move", "corridor_1")
        return action("move", common["location"])


class SecondOrderScript:
    """Test fixture that corrects C only when A models C's belief as stale."""

    received_layers = ("common", "valid_actions", "first_order", "second_order")

    def choose(self, common, valid_actions, first_order, second_order):
        modeled_c = second_order["beliefs_about_others"]["C"][MEDICAL_KIT_LOCATION]
        can_message_c = {item["target"] for item in valid_actions if item["action"] == "send_message"}
        if modeled_c["epistemic_status"] == "stale" and modeled_c["believed_value"] == "decoy_room" and "C" in can_message_c:
            return action("send_message", "C", "kit_revealed")
        return action("move", common["location"])


def apply_action(env, agent, chosen):
    target = chosen["target"]
    if chosen["action"] == "send_message":
        target = f'{target}|{chosen["message"]}'
    return env.step(agent, chosen["action"], target, {})


def sensitivity_record(env, condition, acting_agent, common, first_order, second_order, valid_actions, chosen, correction_turn=None):
    summary = env.summary()
    locations = dict(summary.agent_locations)
    return {
        "scenario": env.scenario_id,
        "seed": env.seed,
        "condition": condition,
        "acting_agent": acting_agent,
        "common_observation": deepcopy(common),
        "first_order_input": deepcopy(first_order),
        "second_order_input": deepcopy(second_order),
        "valid_actions": deepcopy(valid_actions),
        "chosen_action": deepcopy(chosen),
        "messages_sent": summary.messages_sent_count,
        "messages_delivered": summary.delivered_delayed_messages_count,
        "wrong_branch_or_decoy_steps": int(locations["C"] in DECOY_BRANCH),
        "no_op_or_waiting_steps": sum(
            event.event == "move" and event.details.get("target") == "staging"
            for event in env.state.trace
        ),
        "time_to_belief_correction": correction_turn,
        "time_to_medical_kit_acquisition": summary.time_to_medical_kit_acquired,
        "time_to_rescue": summary.time_to_rescue,
        "task_success": summary.success,
        "final_task_status": dict(summary.task_status),
        "final_locations": locations,
    }


def run_first_order_sensitivity():
    base = BTomEnvV2("C5b_costly_false_belief", 0, max_turns=20)
    observation = base.get_observation("C")
    epistemic = AgentEpistemicState()
    epistemic.update_from_observation("C", observation)
    common = common_input(observation)
    first_order = epistemic.first_order_for("C")
    valid_actions = enumerate_valid_actions(base, "C", observation)

    outputs = []
    for condition, script in (("ReactiveScript", ReactiveScript()), ("FirstOrderScript", FirstOrderScript())):
        env = BTomEnvV2("C5b_costly_false_belief", 0, max_turns=20)
        chosen = script.choose(common, valid_actions) if condition == "ReactiveScript" else script.choose(common, valid_actions, first_order)
        apply_action(env, "C", chosen)
        outputs.append(sensitivity_record(
            env, condition, "C", common,
            None if condition == "ReactiveScript" else first_order,
            None, valid_actions, chosen,
        ))
    return outputs


def _second_order_input(stale):
    state = AgentEpistemicState().second_order_for("A")
    if stale:
        state["beliefs_about_others"]["C"][MEDICAL_KIT_LOCATION] = {
            "believed_value": "decoy_room",
            "epistemic_status": "stale",
            "source": "C's delivered structured belief conflicts with A's local observation",
        }
    return state


def run_second_order_sensitivity():
    template = BTomEnvV2("C5b_costly_false_belief", 0, max_turns=20)
    template.state.locations["A"] = "box_room"
    template.state.room_items["box_room"] = ["medical_kit"]
    template.state.task_status["medical_kit_revealed"] = True
    observation_a = template.get_observation("A")
    epistemic_a = AgentEpistemicState()
    epistemic_a.update_from_observation("A", observation_a)
    common_a = common_input(observation_a)
    first_a = epistemic_a.first_order_for("A")
    valid_a = enumerate_valid_actions(template, "A", observation_a)

    configurations = (
        ("ReactiveScript", ReactiveScript(), None),
        ("FirstOrderScript", FirstOrderScript(), None),
        ("SecondOrderScriptUnknown", SecondOrderScript(), _second_order_input(False)),
        ("SecondOrderScriptStale", SecondOrderScript(), _second_order_input(True)),
    )
    outputs = []
    for condition, script, second_order in configurations:
        env = BTomEnvV2("C5b_costly_false_belief", 0, max_turns=20)
        env.state.locations["A"] = "box_room"
        env.state.room_items["box_room"] = ["medical_kit"]
        env.state.task_status["medical_kit_revealed"] = True
        if condition == "ReactiveScript":
            chosen_a = script.choose(common_a, valid_a)
        elif condition == "FirstOrderScript":
            chosen_a = script.choose(common_a, valid_a, first_a)
        else:
            chosen_a = script.choose(common_a, valid_a, first_a, second_order)
        apply_action(env, "A", chosen_a)

        # Advance only the public environment clock so an addressed message can be delivered.
        apply_action(env, "B", action("move", "staging"))
        apply_action(env, "B", action("move", "staging"))
        observation_c = env.get_observation("C")
        epistemic_c = AgentEpistemicState()
        epistemic_c.update_from_observation("C", observation_c)
        first_c = epistemic_c.first_order_for("C")
        valid_c = enumerate_valid_actions(env, "C", observation_c)
        chosen_c = FirstOrderScript().choose(common_input(observation_c), valid_c, first_c)
        apply_action(env, "C", chosen_c)
        correction_turn = 3 if observation_c.delivered_messages else None
        outputs.append(sensitivity_record(
            env, condition, "A", common_a,
            None if condition == "ReactiveScript" else first_a,
            second_order, valid_a, chosen_a, correction_turn,
        ))
    return outputs


def test_every_current_scenario_has_discriminativeness_classification():
    assert set(SCENARIO_CLASSIFICATIONS) == {
        "C1_fully_observable", "C2_partial_observable", "C4_communication_delay",
        "C4b_costly_communication_delay", "C4c_wrong_branch_communication_delay",
        "C5_false_belief_injection", "C5b_costly_false_belief", "C6_resource_allocation",
    }


def test_script_interfaces_enforce_information_boundaries():
    assert ReactiveScript.received_layers == ("common", "valid_actions")
    assert FirstOrderScript.received_layers == ("common", "valid_actions", "first_order")
    assert SecondOrderScript.received_layers == ("common", "valid_actions", "first_order", "second_order")


def test_first_order_input_changes_action_and_has_measurable_decoy_cost():
    reactive, first_order = run_first_order_sensitivity()
    assert reactive["common_observation"] == first_order["common_observation"]
    assert reactive["valid_actions"] == first_order["valid_actions"]
    assert reactive["chosen_action"] == action("move", "staging")
    assert first_order["chosen_action"] == action("move", "long_decoy_1")
    assert reactive["wrong_branch_or_decoy_steps"] == 0
    assert first_order["wrong_branch_or_decoy_steps"] == 1


def test_hidden_world_change_without_observation_change_cannot_change_script_action():
    env = BTomEnvV2("C2_partial_observable", 0)
    observation = env.get_observation("A")
    common = common_input(observation)
    valid_actions = enumerate_valid_actions(env, "A", observation)
    before = ReactiveScript().choose(common, valid_actions)
    env.state.room_items["box_room"].append("medical_kit")
    after = ReactiveScript().choose(common, valid_actions)
    assert before == after


def test_second_order_action_changes_only_with_nested_stale_belief():
    outputs = run_second_order_sensitivity()
    by_condition = {row["condition"]: row for row in outputs}
    common = {json.dumps(row["common_observation"], sort_keys=True) for row in outputs}
    actions = {json.dumps(row["valid_actions"], sort_keys=True) for row in outputs}
    assert len(common) == len(actions) == 1
    assert by_condition["SecondOrderScriptUnknown"]["chosen_action"] == action("move", "box_room")
    assert by_condition["SecondOrderScriptStale"]["chosen_action"] == action("send_message", "C", "kit_revealed")
    assert by_condition["SecondOrderScriptUnknown"]["messages_sent"] == 0
    assert by_condition["SecondOrderScriptStale"]["messages_sent"] == 1
    assert by_condition["SecondOrderScriptUnknown"]["wrong_branch_or_decoy_steps"] == 1
    assert by_condition["SecondOrderScriptStale"]["wrong_branch_or_decoy_steps"] == 0
    assert by_condition["SecondOrderScriptStale"]["time_to_belief_correction"] == 3


def test_scripts_do_not_reference_baseline_or_global_state():
    for script_class in (ReactiveScript, FirstOrderScript, SecondOrderScript):
        names = set(script_class.choose.__code__.co_names)
        assert "DeterministicBaselinePolicy" not in names
        assert "state" not in names
        assert "scenario_id" not in names


if __name__ == "__main__":
    print(json.dumps({
        "first_order_sensitivity": run_first_order_sensitivity(),
        "second_order_sensitivity": run_second_order_sensitivity(),
    }, indent=2, sort_keys=True))
