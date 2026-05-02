"""Minimal deterministic symbolic multi-agent environment for PAAMS B-ToM-MAS.

This first-step implementation intentionally keeps mechanics simple and explicit so
later commits can extend it with richer metrics and second-order beliefs.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional


class BToMEnvironment:
    """Symbolic multi-agent environment with first-order belief state only.

    TODO(next commit): Add second-order belief modeling.
    """

    def __init__(self) -> None:
        self.agents = ["A", "B", "C"]
        self.rooms_graph = {
            "room_0": ["room_1", "room_2"],
            "room_1": ["room_0", "room_3"],
            "room_2": ["room_0", "room_4"],
            "room_3": ["room_1", "room_5"],
            "room_4": ["room_2", "room_5"],
            "room_5": ["room_3", "room_4"],
        }
        self.objects = ["red_key", "blue_key", "locked_box", "medical_kit", "victim"]
        self.skills = {"A": ["red_key"], "B": ["blue_key"], "C": ["medical_kit", "rescue"]}

        self._episode_counter = 0
        self._rng = random.Random(0)

        self.scenario_id = "C1_fully_observable"
        self.seed = 0
        self.step_count = 0
        self.done = False
        self.success = False
        self.false_injection_done = False
        self.false_injection_target: Optional[str] = None
        self.false_injection_step: Optional[int] = None
        self.false_belief_injections = 0

        self.agent_locations: Dict[str, str] = {}
        self.agent_inventory: Dict[str, List[str]] = {}
        self.room_contents: Dict[str, List[str]] = {}
        self.object_locations: Dict[str, str] = {}
        self.task_status: Dict[str, Any] = {}
        self.pending_messages: Dict[str, List[Dict[str, str]]] = {}
        self.beliefs: Dict[str, Dict[str, Any]] = {}
        self.trace_steps: List[Dict[str, Any]] = []

    def reset(self, seed: int, scenario_id: str) -> Dict[str, Any]:
        self._episode_counter += 1
        self.seed = seed
        self.scenario_id = scenario_id
        self._rng = random.Random(seed)

        self.step_count = 0
        self.done = False
        self.success = False
        self.false_injection_done = False
        self.false_belief_injections = 0
        self.false_injection_target = None
        self.false_injection_step = None

        self.agent_locations = {"A": "room_0", "B": "room_1", "C": "room_2"}
        self.agent_inventory = {a: [] for a in self.agents}
        self.room_contents = {room: [] for room in self.rooms_graph}
        # Option 1 mission semantics: locked_box requires both keys; medical_kit is revealed only after box opens.
        self.object_locations = {
            "red_key": "room_1",
            "blue_key": "room_2",
            "locked_box": "room_5",
            "medical_kit": "inside_locked_box",
            "victim": "room_5",
        }
        for obj, room in self.object_locations.items():
            if room in self.room_contents:
                self.room_contents[room].append(obj)

        self.task_status = {
            "red_key_applied": False,
            "blue_key_applied": False,
            "locked_box_open": False,
            "victim_rescued": False,
            "medical_kit_used": False,
        }
        self.pending_messages = {a: [] for a in self.agents}
        self.beliefs = {a: self._init_belief() for a in self.agents}
        self.trace_steps = []

        if scenario_id == "C5_false_belief_injection":
            self.false_injection_target = self._rng.choice(self.agents)
            self.false_injection_step = self._rng.randint(1, 6)

        return {"episode_id": self._episode_counter, "scenario_id": self.scenario_id, "seed": self.seed}

    def _init_belief(self) -> Dict[str, Any]:
        return {"known_object_locations": {}, "known_agent_locations": {}, "known_task_status": {}, "uncertain_facts": []}

    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        delivered = self.pending_messages[agent_id][:]
        self.pending_messages[agent_id] = []
        own_location = self.agent_locations[agent_id]

        obs = {
            "type": "partial" if self.scenario_id != "C1_fully_observable" else "global",
            "current_room": own_location,
            "current_room_contents": list(self.room_contents[own_location]),
            "own_inventory": list(self.agent_inventory[agent_id]),
            "own_location": own_location,
            "delivered_messages": delivered,
            "known_agent_locations": copy.deepcopy(self.agent_locations) if self.scenario_id == "C1_fully_observable" else {agent_id: own_location},
        }

        if self.scenario_id == "C1_fully_observable":
            obs["full_global_state"] = {
                "agent_locations": copy.deepcopy(self.agent_locations),
                "room_contents": copy.deepcopy(self.room_contents),
                "task_status": copy.deepcopy(self.task_status),
                "object_locations": copy.deepcopy(self.object_locations),
            }

        if (
            self.scenario_id == "C5_false_belief_injection"
            and not self.false_injection_done
            and agent_id == self.false_injection_target
            and self.step_count >= (self.false_injection_step or 999)
        ):
            self.false_injection_done = True
            self.false_belief_injections += 1
            obs["injected_false_observation"] = {
                "object": "medical_kit",
                "reported_room": own_location,
                "kind": "stale_or_incorrect",
            }
        return obs

    def _set_belief_fact(
        self,
        conflicts: List[str],
        belief_map: Dict[str, Any],
        key: str,
        new_value: Any,
        source: str,
        allow_conflict: bool = True,
    ) -> None:
        old_value = belief_map.get(key)
        if old_value is None:
            belief_map[key] = new_value
            return
        if old_value != new_value:
            if not allow_conflict:
                belief_map[key] = new_value
                return
            conflicts.append(f"{source}:{key}:{old_value}->{new_value}")
            belief_map[key] = new_value

    def _apply_message_to_belief(self, agent_id: str, msg: str, conflicts: List[str]) -> None:
        belief = self.beliefs[agent_id]
        lower = msg.lower()
        for room in self.rooms_graph:
            if room in lower and "medical_kit" in lower:
                self._set_belief_fact(conflicts, belief["known_object_locations"], "medical_kit", room, "message_object_location_conflict")
            if room in lower and "victim" in lower:
                self._set_belief_fact(conflicts, belief["known_object_locations"], "victim", room, "message_object_location_conflict")
        if "victim rescued" in lower:
            self._set_belief_fact(conflicts, belief["known_task_status"], "victim_rescued", True, "message_task_status_conflict")

    def _update_belief_from_observation(self, agent_id: str, obs: Dict[str, Any]) -> List[str]:
        belief = self.beliefs[agent_id]
        conflicts: List[str] = []

        observed_agent_locations = obs["known_agent_locations"]
        for other, loc in observed_agent_locations.items():
            # We are not yet tracking per-turn movement constraints, so location updates are
            # treated as normal belief refreshes rather than conflicts in this minimal step.
            belief["known_agent_locations"][other] = loc

        if "full_global_state" in obs:
            full = obs["full_global_state"]
            for item, loc in full["object_locations"].items():
                self._set_belief_fact(
                    conflicts, belief["known_object_locations"], item, loc, "object_location_conflict", allow_conflict=False
                )
            for task_key, task_value in full["task_status"].items():
                self._set_belief_fact(
                    conflicts, belief["known_task_status"], task_key, task_value, "task_status_conflict", allow_conflict=False
                )
        else:
            current_room = obs["current_room"]
            for item in obs["current_room_contents"]:
                self._set_belief_fact(conflicts, belief["known_object_locations"], item, current_room, "object_location_conflict")

        for msg in obs.get("delivered_messages", []):
            self._apply_message_to_belief(agent_id, msg["text"], conflicts)

        if "injected_false_observation" in obs:
            f = obs["injected_false_observation"]
            belief["known_object_locations"][f["object"]] = f["reported_room"]
            belief["uncertain_facts"].append(f"possibly_false:{f['object']}@{f['reported_room']}")
            if obs["current_room"] == f["reported_room"] and f["object"] not in obs["current_room_contents"]:
                conflicts.append(f"injected_false_observation_conflict:{f['object']}@{f['reported_room']}")
        return conflicts

    def step(self, agent_id: str, action: str) -> Dict[str, Any]:
        if self.done:
            return {"action_valid": False, "action_result": "episode_already_done", "message_sent": None}

        self.step_count += 1
        action_valid = True
        result = "ok"
        message_sent = None

        if action == "inspect":
            result = f"inspect:{self.agent_locations[agent_id]}"
        elif action.startswith("move:"):
            target = action.split(":", 1)[1]
            cur = self.agent_locations[agent_id]
            if target in self.rooms_graph.get(cur, []):
                self.agent_locations[agent_id] = target
                result = f"moved_to:{target}"
            else:
                action_valid = False
                result = f"cannot_move:{cur}->{target}"
        elif action.startswith("pickup:"):
            obj = action.split(":", 1)[1]
            room = self.agent_locations[agent_id]
            role_owner = {"red_key": "A", "blue_key": "B", "medical_kit": "C"}
            if obj in role_owner and role_owner[obj] != agent_id:
                action_valid = False
                result = f"role_restricted_pickup:{obj}:owner={role_owner[obj]}"
            elif obj == "medical_kit" and not self.task_status["locked_box_open"]:
                action_valid = False
                result = "medical_kit_not_available_before_locked_box_open"
            elif obj in self.room_contents[room]:
                self.room_contents[room].remove(obj)
                self.agent_inventory[agent_id].append(obj)
                self.object_locations[obj] = f"inventory:{agent_id}"
                result = f"picked:{obj}"
            else:
                action_valid = False
                result = f"object_not_in_room:{obj}"
        elif action.startswith("use:"):
            obj = action.split(":", 1)[1]
            role_owner = {"red_key": "A", "blue_key": "B", "medical_kit": "C"}
            if obj in role_owner and role_owner[obj] != agent_id:
                action_valid = False
                result = f"role_restricted_use:{obj}:owner={role_owner[obj]}"
            elif obj not in self.agent_inventory[agent_id]:
                action_valid = False
                result = f"missing_in_inventory:{obj}"
            elif obj in {"red_key", "blue_key"} and self.agent_locations[agent_id] == "room_5":
                if self.task_status["locked_box_open"]:
                    result = "no_effect_already_open"
                else:
                    self.task_status[f"{obj}_applied"] = True
                    if self.task_status["red_key_applied"] and self.task_status["blue_key_applied"]:
                        self.task_status["locked_box_open"] = True
                        if self.object_locations["medical_kit"] == "inside_locked_box":
                            self.object_locations["medical_kit"] = "room_5"
                            self.room_contents["room_5"].append("medical_kit")
                        result = "locked_box_opened"
                    else:
                        result = f"{obj}_applied"
            elif obj == "medical_kit" and self.agent_locations[agent_id] == "room_5":
                self.task_status["medical_kit_used"] = True
                self.agent_inventory[agent_id].remove("medical_kit")
                self.object_locations["medical_kit"] = "used_on_victim_site"
                result = "medical_kit_used_on_site"
            else:
                action_valid = False
                result = f"cannot_use_here:{obj}"
        elif action == "rescue:victim":
            room = self.agent_locations[agent_id]
            has_victim = "victim" in self.room_contents[room]
            has_kit = "medical_kit" in self.agent_inventory[agent_id]
            if agent_id != "C":
                action_valid = False
                result = "role_restricted_rescue:owner=C"
            elif room == "room_5" and has_victim and has_kit:
                self.task_status["victim_rescued"] = True
                self.task_status["medical_kit_used"] = True
                self.room_contents[room].remove("victim")
                self.object_locations["victim"] = "rescued"
                self.agent_inventory[agent_id].remove("medical_kit")
                self.object_locations["medical_kit"] = "used_on_victim_site"
                self.success = True
                self.done = True
                result = "victim_rescued"
            else:
                action_valid = False
                result = "rescue_preconditions_not_met"
        elif action.startswith("send:"):
            txt = action.split(":", 1)[1].strip()
            message_sent = {"from": agent_id, "text": txt}
            for other in self.agents:
                if other != agent_id:
                    self.pending_messages[other].append(message_sent)
            result = "message_sent"
        elif action == "wait":
            result = "waited"
        else:
            action_valid = False
            result = f"unknown_action:{action}"

        if self.step_count >= 30 and not self.done:
            self.done = True
        return {"action_valid": action_valid, "action_result": result, "message_sent": message_sent}

    def run_episode(self, policy_fn: Callable[[str, Dict[str, Any], Dict[str, Any]], str], max_steps: int = 30) -> Dict[str, Any]:
        total_invalid, total_messages, belief_conflicts = 0, 0, 0

        for idx in range(max_steps):
            if self.done:
                break
            agent_id = self.agents[idx % len(self.agents)]
            obs = self.get_observation(agent_id)
            action = policy_fn(agent_id, obs, self)
            step_result = self.step(agent_id, action)

            conflicts = self._update_belief_from_observation(agent_id, obs)
            belief_conflicts += len(conflicts)
            total_invalid += 0 if step_result["action_valid"] else 1
            total_messages += 1 if step_result["message_sent"] is not None else 0

            self.trace_steps.append(
                {
                    "step_idx": idx,
                    "agent_id": agent_id,
                    "observation": obs,
                    "action": action,
                    "action_valid": step_result["action_valid"],
                    "action_result": step_result["action_result"],
                    "message_sent": step_result["message_sent"],
                    "messages_delivered": obs.get("delivered_messages", []),
                    "global_state_snapshot": {
                        "agent_locations": copy.deepcopy(self.agent_locations),
                        "inventories": copy.deepcopy(self.agent_inventory),
                        "room_contents": copy.deepcopy(self.room_contents),
                        "object_locations": copy.deepcopy(self.object_locations),
                        "task_status": copy.deepcopy(self.task_status),
                    },
                    "agent_first_order_belief": copy.deepcopy(self.beliefs[agent_id]),
                    "belief_conflicts_detected": conflicts,
                }
            )

        return {
            "episode_id": self._episode_counter,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "success": self.success,
            "normalized_team_score": 1.0 if self.success else 0.0,
            "turns_to_completion": len(self.trace_steps),
            "total_invalid_actions": total_invalid,
            "total_messages": total_messages,
            "belief_conflict_count": belief_conflicts,
            "false_belief_injections": self.false_belief_injections,
            "steps": self.trace_steps,
        }
