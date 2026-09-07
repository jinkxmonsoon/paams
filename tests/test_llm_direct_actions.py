import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from btom_v2.action_parser import parse_action
from btom_v2.budget import BudgetTracker
from btom_v2.env import BTomEnvV2
from btom_v2.epistemic_state import AgentEpistemicState, MEDICAL_KIT_LOCATION
from btom_v2.llm_policy import LLMPolicyAdapter, enumerate_valid_actions
from btom_v2.policies import DeterministicBaselinePolicy
from btom_v2.prompts import PromptBuilder
from btom_v2.runner_llm_real_smoke import LLMBeliefStateReal, LLMBToMReal, LLMReactiveReal, run_episode


ALL_ACTIONS = [
    {"action": "move", "target": "red_room"},
    {"action": "pickup", "target": "red_key"},
    {"action": "open_box", "target": None},
    {"action": "rescue", "target": None},
    {"action": "send_message", "target": "B"},
]


@pytest.mark.parametrize("option", ALL_ACTIONS)
def test_every_supported_action_type_parses(option):
    budget = BudgetTracker()
    result = parse_action(
        json.dumps({**option, "message": "status" if option["action"] == "send_message" else "", "reason": "test"}),
        ALL_ACTIONS,
        budget,
        fallback_target="staging",
    )
    assert result["parse_success"] is True
    assert result["action"] == option["action"]
    assert result["target"] == option["target"]


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        ('{"action":"move"', "invalid_json"),
        ('{"action":"teleport","target":"red_room"}', "unsupported_action"),
        ('{"action":"move","target":"victim_room"}', "invalid_target"),
        ('{"action":"move","message":""}', "missing_target"),
        ('{"current_observation":{"location":"staging"}}', "copied_context_no_action"),
    ],
)
def test_parser_rejects_invalid_outputs(response, error_type):
    budget = BudgetTracker()
    result = parse_action(response, ALL_ACTIONS, budget, fallback_target="staging")
    assert result == {
        "action": "move",
        "target": "staging",
        "message": "",
        "reason": error_type,
        "parser_error_type": error_type,
        "parse_success": False,
    }
    assert budget.parse_failures == 1


def test_enumerated_actions_are_local_and_reach_env_step():
    env = BTomEnvV2("C5b_costly_false_belief", 0)
    observation = env.get_observation("A")
    actions = enumerate_valid_actions(env, "A", observation)
    assert {option["target"] for option in actions if option["action"] == "move"} == {
        "staging", "red_room", "blue_room", "corridor_1", "long_decoy_1"
    }
    assert not [option for option in actions if option["action"] == "pickup"]
    result = env.step("A", "move", "red_room", {})
    assert result.invalid is False


def test_each_non_move_action_reaches_env_step_when_enumerated():
    pickup_env = BTomEnvV2("C5b_costly_false_belief", 0)
    pickup_env.state.locations["A"] = "red_room"
    pickup_obs = pickup_env.get_observation("A")
    assert {"action": "pickup", "target": "red_key"} in enumerate_valid_actions(pickup_env, "A", pickup_obs)
    assert pickup_env.step("A", "pickup", "red_key", {}).invalid is False

    open_env = BTomEnvV2("C5b_costly_false_belief", 0)
    open_env.state.locations["A"] = "box_room"
    open_env.state.inventories["A"] = ["red_key"]
    open_obs = open_env.get_observation("A")
    assert {"action": "open_box", "target": None} in enumerate_valid_actions(open_env, "A", open_obs)
    assert open_env.step("A", "open_box", None, {}).invalid is False

    rescue_env = BTomEnvV2("C5b_costly_false_belief", 0)
    rescue_env.state.locations["C"] = "victim_room"
    rescue_env.state.inventories["C"] = ["medical_kit"]
    rescue_obs = rescue_env.get_observation("C")
    assert {"action": "rescue", "target": None} in enumerate_valid_actions(rescue_env, "C", rescue_obs)
    assert rescue_env.step("C", "rescue", None, {}).invalid is False

    message_env = BTomEnvV2("C5b_costly_false_belief", 0)
    message_obs = message_env.get_observation("A")
    assert {"action": "send_message", "target": "B"} in enumerate_valid_actions(message_env, "A", message_obs)
    assert message_env.step("A", "send_message", "B|status", {}).invalid is False


class MalformedClient:
    no_external_api_calls = True

    def generate(self, prompt, **kwargs):
        return "not json"


def test_parser_fallback_never_calls_deterministic_baseline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("deterministic baseline was called")

    monkeypatch.setattr(DeterministicBaselinePolicy, "act", forbidden)
    env = BTomEnvV2("C5b_costly_false_belief", 0)
    policy = LLMPolicyAdapter(client=MalformedClient())
    action, target, meta = policy.act(env, "A", 0)
    assert (action, target) == ("move", "staging")
    assert meta["fallback_type"] == "parser_failure"


def _prompts_for_variants():
    env = BTomEnvV2("C5b_costly_false_belief", 0)
    observation = env.get_observation("C")
    actions = enumerate_valid_actions(env, "C", observation)
    epistemic = AgentEpistemicState()
    epistemic.update_from_observation("C", observation)
    builder = PromptBuilder()
    reactive = builder.build("LLMReactive", "C", env.scenario_id, observation, actions)
    belief = builder.build("LLMBeliefState", "C", env.scenario_id, observation, actions, belief_state=epistemic.first_order_for("C"))
    tom = builder.build(
        "LLMBToM",
        "C",
        env.scenario_id,
        observation,
        actions,
        belief_state=epistemic.first_order_for("C"),
        second_order_state=epistemic.second_order_for("C"),
    )
    return reactive, belief, tom


def test_policy_prompts_differ_only_by_intended_belief_layers():
    reactive, belief, tom = _prompts_for_variants()
    assert "FIRST-ORDER BELIEFS:" not in reactive
    assert "SECOND-ORDER BELIEFS:" not in reactive
    assert "medical_kit_location" not in reactive
    assert "FIRST-ORDER BELIEFS:" in belief
    assert "medical_kit_location" in belief
    assert "SECOND-ORDER BELIEFS:" not in belief
    assert "FIRST-ORDER BELIEFS:" in tom
    assert "SECOND-ORDER BELIEFS:" in tom
    assert '"observer": "C"' in tom
    assert '"beliefs_about_others"' in tom
    assert "responsible_agent_for_medical_kit" not in tom

    remove_belief_layers = lambda prompt: "\n".join(
        line for line in prompt.splitlines()
        if not line.startswith(("FIRST-ORDER BELIEFS:", "SECOND-ORDER BELIEFS:"))
    )
    assert remove_belief_layers(reactive) == remove_belief_layers(belief) == remove_belief_layers(tom)


def test_prompts_do_not_expose_prohibited_global_truth():
    for prompt in _prompts_for_variants():
        for prohibited in ("room_items", "delayed_queue", "trace", "false_belief_injections", "agent_locations"):
            assert prohibited not in prompt
        assert "C5b_costly_false_belief" not in prompt
        assert '"A": "staging"' not in prompt
        assert '"B": "staging"' not in prompt


def test_role_metadata_is_public_task_rule_not_second_order_belief():
    reactive, belief, tom = _prompts_for_variants()
    for prompt in (reactive, belief, tom):
        assert "C may pick up the medical kit and rescue the victim." in prompt
    second_order = AgentEpistemicState().second_order_for("A")
    assert "responsible_agent_for_medical_kit" not in json.dumps(second_order)


def test_epistemic_models_are_partitioned_by_observer():
    env = BTomEnvV2("C2_partial_observable", 0)
    state = AgentEpistemicState()
    message = {
        "from": "B",
        "to": "A",
        "content": "belief:medical_kit_location=decoy_room",
        "sent_step": 1,
        "delivery_step": 2,
    }
    observation_a = replace(env.get_observation("A"), delivered_messages=[message])
    state.update_from_observation("A", observation_a)

    assert state.second_order_for("A")["beliefs_about_others"]["B"][MEDICAL_KIT_LOCATION]["believed_value"] == "decoy_room"
    assert state.second_order_for("B")["beliefs_about_others"]["A"][MEDICAL_KIT_LOCATION]["believed_value"] == "unknown"
    assert state.second_order_for("C")["beliefs_about_others"]["B"][MEDICAL_KIT_LOCATION]["believed_value"] == "unknown"


def test_unstructured_rationalization_is_not_mental_state_evidence():
    env = BTomEnvV2("C2_partial_observable", 0)
    state = AgentEpistemicState()
    message = {"from": "B", "to": "A", "content": "I think the kit may be in decoy_room"}
    state.update_from_observation("A", replace(env.get_observation("A"), delivered_messages=[message]))
    assert state.second_order_for("A")["beliefs_about_others"]["B"][MEDICAL_KIT_LOCATION]["believed_value"] == "unknown"


def test_message_addressed_to_another_agent_is_ignored():
    env = BTomEnvV2("C2_partial_observable", 0)
    state = AgentEpistemicState()
    message = {"from": "A", "to": "B", "content": "belief:medical_kit_location=decoy_room"}
    state.update_from_observation("C", replace(env.get_observation("C"), delivered_messages=[message]))
    assert state.second_order_for("C")["beliefs_about_others"]["A"][MEDICAL_KIT_LOCATION]["believed_value"] == "unknown"


def test_undelivered_messages_do_not_update_any_epistemic_model():
    env = BTomEnvV2("C4_communication_delay", 0)
    state = AgentEpistemicState()
    result = env.step("B", "send_message", "A|belief:medical_kit_location=decoy_room", {})
    assert result.invalid is False
    assert env.state.delayed_queue
    observation_a = env.get_observation("A")
    assert observation_a.delivered_messages == []
    state.update_from_observation("A", observation_a)
    assert state.second_order_for("A")["beliefs_about_others"]["B"][MEDICAL_KIT_LOCATION]["believed_value"] == "unknown"


def test_unknown_and_hidden_room_contents_remain_unknown():
    env = BTomEnvV2("C2_partial_observable", 0)
    env.state.room_items["box_room"].append("medical_kit")
    state = AgentEpistemicState()
    observation = env.get_observation("A")
    state.update_from_observation("A", observation)
    assert state.first_order_for("A")["beliefs"][MEDICAL_KIT_LOCATION]["believed_value"] == "unknown"
    for target in ("B", "C"):
        assert state.second_order_for("A")["beliefs_about_others"][target][MEDICAL_KIT_LOCATION]["believed_value"] == "unknown"
    prompt = PromptBuilder().build("LLMBToM", "A", env.scenario_id, observation, enumerate_valid_actions(env, "A", observation), state.first_order_for("A"), state.second_order_for("A"))
    assert '"believed_value": "box_room"' not in prompt
    assert "Visible items: []" in prompt


def test_second_order_preserves_target_false_belief_when_observer_sees_truth():
    env = BTomEnvV2("C2_partial_observable", 0)
    state = AgentEpistemicState()
    env.state.beliefs["B"][MEDICAL_KIT_LOCATION] = "decoy_room"
    state.update_from_observation("B", env.get_observation("B"))
    communicated_false_belief = {
        "from": "B",
        "to": "A",
        "content": "belief:medical_kit_location=decoy_room",
        "sent_step": 1,
        "delivery_step": 2,
    }
    at_staging = replace(env.get_observation("A"), delivered_messages=[communicated_false_belief])
    state.update_from_observation("A", at_staging)

    env.state.room_items["box_room"].append("medical_kit")
    env.state.locations["A"] = "box_room"
    truth_observation = env.get_observation("A")
    state.update_from_observation("A", truth_observation)

    world_value = "box_room"
    target_first_order_belief = state.first_order_for("B")["beliefs"][MEDICAL_KIT_LOCATION]["believed_value"]
    observer_second_order = state.second_order_for("A")["beliefs_about_others"]["B"][MEDICAL_KIT_LOCATION]
    assert world_value != target_first_order_belief
    assert observer_second_order["believed_value"] == target_first_order_belief
    assert observer_second_order["epistemic_status"] == "stale"
    assert state.first_order_for("A")["beliefs"][MEDICAL_KIT_LOCATION]["believed_value"] == world_value


class FirstValidActionClient:
    no_external_api_calls = True

    def generate(self, prompt, **kwargs):
        self.last_prompt = prompt
        return json.dumps({**kwargs["valid_actions"][0], "message": "", "reason": "audit test"})


@pytest.mark.parametrize(
    ("policy_class", "first_enabled", "second_enabled", "first_count", "second_count"),
    [
        (LLMReactiveReal, False, False, 0, 0),
        (LLMBeliefStateReal, True, False, 1, 0),
        (LLMBToMReal, True, True, 1, 2),
    ],
)
def test_call_audit_records_epistemic_condition_and_prompt_size(
    policy_class, first_enabled, second_enabled, first_count, second_count
):
    env = BTomEnvV2("C5b_costly_false_belief", 0)
    client = FirstValidActionClient()
    policy = policy_class(client=client)
    policy.act(env, "A", 0)
    condition = policy.call_audit[0]["information_condition"]
    assert condition["first_order_enabled"] is first_enabled
    assert condition["second_order_enabled"] is second_enabled
    assert condition["observer_agent"] == "A"
    assert condition["modeled_target_agents"] == (["B", "C"] if second_enabled else [])
    assert condition["first_order_proposition_count"] == first_count
    assert condition["second_order_proposition_count"] == second_count
    assert condition["prompt_character_count"] == len(client.last_prompt)
    assert condition["prompt_token_count_approx"] == len(client.last_prompt.split())


class CoordinatedDirectClient:
    no_external_api_calls = True

    def generate(self, prompt, **kwargs):
        options = kwargs["valid_actions"]
        agent = kwargs["agent"]
        for action in ("rescue", "open_box", "pickup"):
            choice = next((option for option in options if option["action"] == action), None)
            if choice:
                return json.dumps({**choice, "message": "", "reason": "scripted local progress"})

        moves = {option["target"]: option for option in options if option["action"] == "move"}
        if agent == "A":
            route = ["red_room", "staging", "corridor_1", "box_room"]
        elif agent == "B":
            route = ["blue_room", "staging", "corridor_1", "box_room"]
        elif "medical_kit_revealed=true" in prompt:
            route = ["corridor_1", "box_room", "victim_room"]
        else:
            route = []
        current = next(line.split(": ", 1)[1] for line in prompt.splitlines() if line.startswith("Location: "))
        inventory_line = next(line for line in prompt.splitlines() if line.startswith("Inventory: "))
        if agent in {"A", "B"} and "key" not in inventory_line:
            desired = route[0]
        elif current in {"red_room", "blue_room"}:
            desired = "staging"
        elif current == "staging" and agent in {"A", "B"}:
            desired = "corridor_1"
        elif current == "corridor_1":
            desired = "box_room"
        elif agent == "C" and current == "box_room" and "medical_kit" in inventory_line:
            desired = "victim_room"
        else:
            desired = next((location for location in route if location in moves and location != current), current)
        choice = moves.get(desired, moves[current])
        return json.dumps({**choice, "message": "", "reason": "scripted local route"})


def test_mock_direct_action_policy_completes_integration_episode():
    args = SimpleNamespace(max_steps=40, max_llm_calls=80, model="mock", temperature=0, top_p=1, max_tokens=96)
    summary = run_episode("C5b_costly_false_belief", 0, LLMReactiveReal, CoordinatedDirectClient(), args)
    assert summary["success"] is True
    assert summary["parse_failures"] == 0
    assert summary["environment_invalid_action_count"] == 0
    assert summary["parsed_action_counts"]["rescue"] == 1
    assert summary["agents_progressing_beyond_staging"]


def test_unsuccessful_episode_preserves_null_time_to_rescue():
    args = SimpleNamespace(max_steps=1, max_llm_calls=1, model="mock", temperature=0, top_p=1, max_tokens=96)
    summary = run_episode("C5b_costly_false_belief", 0, LLMReactiveReal, MalformedClient(), args)
    assert summary["success"] is False
    assert summary["time_to_rescue"] is None
