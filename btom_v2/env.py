from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schemas import AGENTS, EpisodeSummary, Event, Observation, StepResult


LOCATIONS = ("staging", "red_room", "blue_room", "box_room", "med_room", "victim_room")


@dataclass
class WorldState:
    scenario_id: str
    seed: int
    turn: int
    max_turns: int
    done: bool
    success: bool
    invalid_actions: int

    locations: Dict[str, str]
    inventories: Dict[str, List[str]]
    room_items: Dict[str, List[str]]

    task_status: Dict[str, bool]
    trace: List[Event]


class BTomEnvV2:
    """Deterministic symbolic environment for initial v2 implementation."""

    def __init__(self, scenario_id: str, seed: int, max_turns: int = 24) -> None:
        if scenario_id not in {"C1_fully_observable", "C2_partial_observable"}:
            raise ValueError(f"unsupported scenario: {scenario_id}")
        self.scenario_id = scenario_id
        self.seed = seed
        self.max_turns = max_turns
        self.state = self._init_state()

    def _init_state(self) -> WorldState:
        room_items = {
            "staging": [],
            "red_room": ["red_key"],
            "blue_room": ["blue_key"],
            "box_room": ["locked_box"],
            "med_room": [],
            "victim_room": ["victim"],
        }
        state = WorldState(
            scenario_id=self.scenario_id,
            seed=self.seed,
            turn=0,
            max_turns=self.max_turns,
            done=False,
            success=False,
            invalid_actions=0,
            locations={a: "staging" for a in AGENTS},
            inventories={a: [] for a in AGENTS},
            room_items=room_items,
            task_status={
                "a_has_red_key": False,
                "b_has_blue_key": False,
                "locked_box_open": False,
                "medical_kit_revealed": False,
                "c_has_medical_kit": False,
                "victim_rescued": False,
            },
            trace=[],
        )
        return state

    def get_observation(self, agent: str) -> Observation:
        s = self.state
        loc = s.locations[agent]
        if self.scenario_id == "C1_fully_observable":
            visible = sorted({item for v in s.room_items.values() for item in v})
        else:
            visible = list(s.room_items[loc])
        return Observation(
            agent=agent,
            location=loc,
            visible_items=visible,
            inventory=list(s.inventories[agent]),
            task_status=dict(s.task_status),
        )

    def _record(self, event: str, agent: str | None = None, **details: object) -> None:
        self.state.trace.append(Event(turn=self.state.turn, event=event, agent=agent, details=details))

    def step(self, agent: str, action: str, target: str) -> StepResult:
        s = self.state
        if s.done:
            return StepResult(False, True, True, "episode_done")

        s.turn += 1
        invalid = False
        reason = None

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

        # Role-gated pickups
        if item == "red_key" and agent != "A":
            return True, "role_mismatch_pickup"
        if item == "blue_key" and agent != "B":
            return True, "role_mismatch_pickup"
        if item == "medical_kit" and agent != "C":
            return True, "role_mismatch_pickup"

        s.room_items[loc].remove(item)
        s.inventories[agent].append(item)
        self._record("pickup", agent, item=item)

        if agent == "A" and item == "red_key":
            s.task_status["a_has_red_key"] = True
        if agent == "B" and item == "blue_key":
            s.task_status["b_has_blue_key"] = True
        if agent == "C" and item == "medical_kit":
            s.task_status["c_has_medical_kit"] = True

        return False, None

    def _open_box(self, agent: str) -> Tuple[bool, str | None]:
        s = self.state
        if s.locations[agent] != "box_room":
            return True, "not_at_box_room"
        if "locked_box" not in s.room_items["box_room"]:
            return True, "locked_box_missing"

        a_ok = "red_key" in s.inventories["A"]
        b_ok = "blue_key" in s.inventories["B"]
        if not (a_ok and b_ok):
            return True, "keys_missing"

        s.room_items["box_room"].remove("locked_box")
        s.task_status["locked_box_open"] = True
        s.task_status["medical_kit_revealed"] = True
        s.room_items["med_room"].append("medical_kit")
        self._record("locked_box_open", agent)
        self._record("medical_kit_revealed", item="medical_kit")
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
            scenario_id=s.scenario_id,
            seed=s.seed,
            success=s.success,
            turns=s.turn,
            invalid_actions=s.invalid_actions,
            task_status=dict(s.task_status),
            agent_locations=dict(s.locations),
            inventories={k: list(v) for k, v in s.inventories.items()},
        )
