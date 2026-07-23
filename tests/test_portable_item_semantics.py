from copy import deepcopy

import pytest

from btom_v2.env import BTomEnvV2, PORTABLE_ITEMS
from btom_v2.llm_policy import enumerate_valid_actions
from btom_v2.policies import DeterministicBaselinePolicy


def pickup_targets(env, agent):
    observation = env.get_observation(agent)
    return {
        action["target"]
        for action in enumerate_valid_actions(env, agent, observation)
        if action["action"] == "pickup"
    }


def legal_step(env, agent, action, target):
    result = env.step(agent, action, target, {})
    assert result.invalid is False, result.reason
    env.assert_invariants()
    return result


def reveal_medical_kit(env):
    legal_step(env, "A", "move", "red_room")
    legal_step(env, "A", "pickup", "red_key")
    legal_step(env, "A", "move", "staging")
    legal_step(env, "A", "move", "corridor_1")
    legal_step(env, "A", "move", "box_room")
    legal_step(env, "A", "open_box", None)
    legal_step(env, "B", "move", "blue_room")
    legal_step(env, "B", "pickup", "blue_key")
    legal_step(env, "B", "move", "staging")
    legal_step(env, "B", "move", "corridor_1")
    legal_step(env, "B", "move", "box_room")
    legal_step(env, "B", "open_box", None)


def test_authoritative_portable_item_definition():
    assert PORTABLE_ITEMS == frozenset({"red_key", "blue_key", "medical_kit"})
    assert "locked_box" not in PORTABLE_ITEMS
    assert "victim" not in PORTABLE_ITEMS


@pytest.mark.parametrize(
    ("scenario", "agent", "room", "item"),
    [
        ("C5b_costly_false_belief", "C", "box_room", "locked_box"),
        ("C7a_partner_belief_stale", "C", "box_room", "locked_box"),
        ("C7b_partner_belief_current", "C", "victim_room", "victim"),
    ],
)
def test_non_portable_items_are_not_enumerated(scenario, agent, room, item):
    env = BTomEnvV2(scenario, 0)
    for target in ("corridor_1", "box_room") if room == "box_room" else ("corridor_1", "box_room", "victim_room"):
        legal_step(env, agent, "move", target)
    assert item in env.get_observation(agent).visible_items
    assert item not in pickup_targets(env, agent)


@pytest.mark.parametrize(
    ("scenario", "agent", "route", "item"),
    [
        ("C5b_costly_false_belief", "C", ("corridor_1", "box_room"), "locked_box"),
        ("C7a_partner_belief_stale", "A", ("corridor_1", "box_room"), "locked_box"),
        ("C7b_partner_belief_current", "C", ("corridor_1", "box_room", "victim_room"), "victim"),
    ],
)
def test_direct_non_portable_pickup_is_rejected_without_state_mutation(
    scenario, agent, route, item
):
    env = BTomEnvV2(scenario, 0)
    for room in route:
        legal_step(env, agent, "move", room)
    room_items = deepcopy(env.state.room_items)
    inventories = deepcopy(env.state.inventories)
    task_status = deepcopy(env.state.task_status)

    result = env.step(agent, "pickup", item, {})

    assert result.invalid is True
    assert result.reason == "non_portable_item"
    assert env.state.room_items == room_items
    assert env.state.inventories == inventories
    assert env.state.task_status == task_status
    assert item in env.state.room_items[route[-1]]
    env.assert_invariants()


def test_role_aware_portable_pickups_remain_enumerated():
    env = BTomEnvV2("C5b_costly_false_belief", 0, max_turns=100)
    for agent in ("A", "B", "C"):
        legal_step(env, agent, "move", "red_room")
        targets = pickup_targets(env, agent)
        assert ("red_key" in targets) is (agent == "A")
        legal_step(env, agent, "move", "staging")
        legal_step(env, agent, "move", "blue_room")
        targets = pickup_targets(env, agent)
        assert ("blue_key" in targets) is (agent == "B")
        legal_step(env, agent, "move", "staging")

    reveal_medical_kit(env)
    for agent in ("A", "B", "C"):
        legal_step(env, agent, "move", "corridor_1")
        legal_step(env, agent, "move", "box_room")
        targets = pickup_targets(env, agent)
        assert ("medical_kit" in targets) is (agent == "C")
        if agent != "C":
            legal_step(env, agent, "move", "corridor_1")
            legal_step(env, agent, "move", "staging")


def test_locked_box_remains_until_normal_open_transition():
    env = BTomEnvV2("C7b_partner_belief_current", 0)
    assert env.state.room_items["box_room"] == ["locked_box"]
    reveal_medical_kit(env)
    assert "locked_box" not in env.state.room_items["box_room"]
    assert "medical_kit" in env.state.room_items["box_room"]
    assert env.state.task_status["locked_box_open"] is True
    env.assert_invariants()


@pytest.mark.parametrize(
    "scenario",
    ["C5b_costly_false_belief", "C7a_partner_belief_stale", "C7b_partner_belief_current"],
)
def test_deterministic_mission_remains_solvable(scenario):
    env = BTomEnvV2(scenario, 0, max_turns=120)
    policy = DeterministicBaselinePolicy()
    for turn in range(120):
        for agent in ("A", "B", "C"):
            action, target, intent = policy.act(env, agent, turn)
            result = env.step(agent, action, target, intent)
            assert result.invalid is False, (agent, action, target, result.reason)
            env.assert_invariants()
            if env.state.done:
                break
        if env.state.done:
            break

    assert env.state.success is True
    assert env.state.task_status == {
        "red_key_applied": True,
        "blue_key_applied": True,
        "locked_box_open": True,
        "medical_kit_revealed": True,
        "victim_rescued": True,
    }
    assert "medical_kit" in env.state.inventories["C"]
