import json
from types import SimpleNamespace

import pytest

from btom_v2.action_parser import parse_action
from btom_v2.budget import BudgetTracker
from btom_v2.env import BTomEnvV2
from btom_v2.llm_policy import LLMPolicyAdapter, enumerate_valid_actions
from btom_v2.policies import DeterministicBaselinePolicy
from btom_v2.prompts import PromptBuilder
from btom_v2.runner_llm_real_smoke import LLMReactiveReal, run_episode


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
    builder = PromptBuilder()
    reactive = builder.build("LLMReactive", "C", env.scenario_id, observation, actions)
    belief = builder.build("LLMBeliefState", "C", env.scenario_id, observation, actions, belief_state=observation.beliefs)
    tom = builder.build(
        "LLMBToM",
        "C",
        env.scenario_id,
        observation,
        actions,
        belief_state=observation.beliefs,
        second_order_state={"responsible_agent_for_medical_kit": "C"},
    )
    return reactive, belief, tom


def test_policy_prompts_differ_only_by_intended_belief_layers():
    reactive, belief, tom = _prompts_for_variants()
    assert "First-order beliefs:" not in reactive
    assert "Second-order/responsibility model:" not in reactive
    assert "medical_kit_location" not in reactive
    assert "First-order beliefs:" in belief
    assert "medical_kit_location" in belief
    assert "Second-order/responsibility model:" not in belief
    assert "First-order beliefs:" in tom
    assert "Second-order/responsibility model:" in tom
    assert "responsible_agent_for_medical_kit" in tom


def test_prompts_do_not_expose_prohibited_global_truth():
    for prompt in _prompts_for_variants():
        for prohibited in ("room_items", "delayed_queue", "trace", "false_belief_injections", "agent_locations"):
            assert prohibited not in prompt
        assert "C5b_costly_false_belief" not in prompt
        assert '"A": "staging"' not in prompt
        assert '"B": "staging"' not in prompt


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
