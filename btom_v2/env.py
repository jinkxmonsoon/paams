from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schemas import AGENTS, EpisodeSummary, Event, Observation, StepResult


LOCATIONS = ("staging", "red_room", "blue_room", "box_room", "med_room", "victim_room", "decoy_room")


@dataclass
class WorldState:
    scenario_id: str
    seed: int
    turn: int
    max_turns: int
    done: bool
    success: bool
    invalid_actions: int
    false_belief_injections: int
    belief_conflict_count: int
    false_belief_caused_wasted_action: int

    locations: Dict[str, str]
    inventories: Dict[str, List[str]]
    room_items: Dict[str, List[str]]
    beliefs: Dict[str, Dict[str, str]]

    task_status: Dict[str, bool]
    trace: List[Event]


class BTomEnvV2:
    def __init__(self, scenario_id: str, seed: int, max_turns: int = 30) -> None:
        if scenario_id not in {"C1_fully_observable", "C2_partial_observable", "C5_false_belief_injection"}:
            raise ValueError(f"unsupported scenario: {scenario_id}")
        self.scenario_id = scenario_id
        self.seed = seed
        self.max_turns = max_turns
        self.state = self._init_state()

    def _init_state(self) -> WorldState:
        s = WorldState(
            scenario_id=self.scenario_id,
            seed=self.seed,
            turn=0,
            max_turns=self.max_turns,
            done=False,
            success=False,
            invalid_actions=0,
            false_belief_injections=0,
            belief_conflict_count=0,
            false_belief_caused_wasted_action=0,
            locations={a: "staging" for a in AGENTS},
            inventories={a: [] for a in AGENTS},
            room_items={
                "staging": [], "red_room": ["red_key"], "blue_room": ["blue_key"], "box_room": ["locked_box"],
                "med_room": [], "victim_room": ["victim"], "decoy_room": []
            },
            beliefs={a: {"medical_kit_location": "unknown"} for a in AGENTS},
            task_status={
                "red_key_applied": False,
                "blue_key_applied": False,
                "locked_box_open": False,
                "medical_kit_revealed": False,
                "victim_rescued": False,
            },
            trace=[],
        )
        if self.scenario_id == "C5_false_belief_injection":
            s.beliefs["C"]["medical_kit_location"] = "decoy_room"
            s.false_belief_injections = 1
            s.trace.append(Event(turn=0, event="false_belief_injected", agent="C", details={"medical_kit_location": "decoy_room"}))
        return s

    def get_observation(self, agent: str) -> Observation:
        s = self.state
        loc = s.locations[agent]
        visible = sorted({item for v in s.room_items.values() for item in v}) if self.scenario_id == "C1_fully_observable" else list(s.room_items[loc])
        self._update_belief_from_observation(agent, loc, visible)
        return Observation(agent=agent, location=loc, visible_items=visible, inventory=list(s.inventories[agent]), task_status=dict(s.task_status), beliefs=dict(s.beliefs[agent]))

    def _update_belief_from_observation(self, agent: str, loc: str, visible: List[str]) -> None:
        s = self.state
        if agent == "C" and loc == "decoy_room" and s.beliefs["C"]["medical_kit_location"] == "decoy_room" and "medical_kit" not in visible:
            s.belief_conflict_count += 1
            s.false_belief_caused_wasted_action += 1
            s.beliefs["C"]["medical_kit_location"] = "unknown"
            self._record("belief_conflict_detected", "C", expected="decoy_room", observed_absent="medical_kit")
            self._record("false_belief_wasted_action", "C", room="decoy_room")
        if "medical_kit" in visible:
            s.beliefs[agent]["medical_kit_location"] = loc

    def _record(self, event: str, agent: str | None = None, **details: object) -> None:
        self.state.trace.append(Event(turn=self.state.turn, event=event, agent=agent, details=details))

    def step(self, agent: str, action: str, target: str) -> StepResult:
        s = self.state
        if s.done:
            return StepResult(False, True, True, "episode_done")
        s.turn += 1
        invalid, reason = False, None
        if action == "move":
            if target not in LOCATIONS:
                invalid, reason = True, "unknown_location"
            else:
                s.locations[agent] = target
                self._record("move", agent, target=target)
        elif action == "pickup":
            invalid, reason = self._pickup(agent, target)
        elif action == "open_box":
            invalid, reason = self._open_box(agent)
        elif action == "rescue":
            invalid, reason = self._rescue(agent)
        else:
            invalid, reason = True, "unknown_action"
        if invalid:
            s.invalid_actions += 1
            self._record("invalid_action", agent, action=action, target=target, reason=reason)
        if s.turn >= s.max_turns and not s.done:
            s.done = True
            self._record("episode_timeout")
        return StepResult(success=not invalid, done=s.done, invalid=invalid, reason=reason)

    def _pickup(self, agent: str, item: str) -> Tuple[bool, str | None]:
        s = self.state
        loc = s.locations[agent]
        if item not in s.room_items[loc]:
            return True, "item_not_in_room"
        if item == "red_key" and agent != "A":
            return True, "role_mismatch_pickup"
        if item == "blue_key" and agent != "B":
            return True, "role_mismatch_pickup"
        if item == "medical_kit" and agent != "C":
            return True, "role_mismatch_pickup"
        s.room_items[loc].remove(item)
        s.inventories[agent].append(item)
        self._record("pickup", agent, item=item)
        s.beliefs[agent]["medical_kit_location"] = loc if item == "medical_kit" else s.beliefs[agent]["medical_kit_location"]
        return False, None

    def _open_box(self, agent: str) -> Tuple[bool, str | None]:
        s = self.state
        if s.locations[agent] != "box_room":
            return True, "not_at_box_room"
        if s.task_status["locked_box_open"]:
            return True, "locked_box_already_open"
        if agent == "A":
            if "red_key" not in s.inventories["A"] or s.task_status["red_key_applied"]:
                return True, "red_key_unavailable"
            s.task_status["red_key_applied"] = True
            self._record("red_key_applied", agent)
        elif agent == "B":
            if "blue_key" not in s.inventories["B"] or s.task_status["blue_key_applied"]:
                return True, "blue_key_unavailable"
            s.task_status["blue_key_applied"] = True
            self._record("blue_key_applied", agent)
        else:
            return True, "role_mismatch_open"
        if s.task_status["red_key_applied"] and s.task_status["blue_key_applied"]:
            s.task_status["locked_box_open"] = True
            s.task_status["medical_kit_revealed"] = True
            if "locked_box" in s.room_items["box_room"]:
                s.room_items["box_room"].remove("locked_box")
            s.room_items["box_room"].append("medical_kit")
            self._record("locked_box_open")
            self._record("medical_kit_revealed", item="medical_kit", location="box_room")
        return False, None

    def _rescue(self, agent: str) -> Tuple[bool, str | None]:
        s = self.state
        if agent != "C":
            return True, "role_mismatch_rescue"
        if s.locations[agent] != "victim_room":
            return True, "not_at_victim"
        if "medical_kit" not in s.inventories["C"]:
            return True, "medical_kit_missing"
        s.task_status["victim_rescued"] = True
        s.success = True
        s.done = True
        self._record("victim_rescued", agent)
        return False, None

    def summary(self) -> EpisodeSummary:
        s = self.state
        return EpisodeSummary(
            scenario_id=s.scenario_id, seed=s.seed, success=s.success, turns=s.turn, invalid_actions=s.invalid_actions,
            false_belief_injections=s.false_belief_injections, belief_conflict_count=s.belief_conflict_count,
            false_belief_caused_wasted_action=s.false_belief_caused_wasted_action,
            task_status=dict(s.task_status), agent_locations=dict(s.locations), inventories={k: list(v) for k, v in s.inventories.items()},
        )
