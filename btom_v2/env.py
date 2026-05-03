from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schemas import AGENTS, EpisodeSummary, Event, Observation, StepResult


LOCATIONS = ("staging", "red_room", "blue_room", "box_room", "med_room", "victim_room", "decoy_room", "long_decoy_1", "long_decoy_2", "corridor_1")


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
    time_to_red_key_applied: int | None
    time_to_blue_key_applied: int | None
    time_to_box_open: int | None
    time_to_false_belief_conflict: int | None
    time_to_medical_kit_revealed: int | None
    time_to_medical_kit_acquired: int | None
    time_to_rescue: int | None

    locations: Dict[str, str]
    inventories: Dict[str, List[str]]
    room_items: Dict[str, List[str]]
    beliefs: Dict[str, Dict[str, str]]

    task_status: Dict[str, bool]
    trace: List[Event]
    delayed_queue: List[dict]
    inboxes: Dict[str, List[dict]]
    delayed_messages_count: int
    delivered_delayed_messages_count: int


class BTomEnvV2:
    def __init__(self, scenario_id: str, seed: int, max_turns: int = 30) -> None:
        if scenario_id not in {"C1_fully_observable", "C2_partial_observable", "C4_communication_delay", "C5_false_belief_injection", "C5b_costly_false_belief"}:
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
            time_to_red_key_applied=None,
            time_to_blue_key_applied=None,
            time_to_box_open=None,
            time_to_false_belief_conflict=None,
            time_to_medical_kit_revealed=None,
            time_to_medical_kit_acquired=None,
            time_to_rescue=None,
            locations={a: "staging" for a in AGENTS},
            inventories={a: [] for a in AGENTS},
            room_items={
                "staging": [], "red_room": ["red_key"], "blue_room": ["blue_key"], "box_room": ["locked_box"],
                "med_room": [], "victim_room": ["victim"], "decoy_room": [], "long_decoy_1": [], "long_decoy_2": [], "corridor_1": []
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
            delayed_queue=[],
            inboxes={a: [] for a in AGENTS},
            delayed_messages_count=0,
            delivered_delayed_messages_count=0,
        )
        if self.scenario_id in {"C5_false_belief_injection", "C5b_costly_false_belief"}:
            s.beliefs["C"]["medical_kit_location"] = "decoy_room"
            s.false_belief_injections = 1
            s.trace.append(Event(turn=0, event="false_belief_injected", agent="C", details={"medical_kit_location": "decoy_room"}))
        return s

    def get_observation(self, agent: str) -> Observation:
        s = self.state
        loc = s.locations[agent]
        visible = sorted({item for v in s.room_items.values() for item in v}) if self.scenario_id == "C1_fully_observable" else list(s.room_items[loc])
        self._update_belief_from_observation(agent, loc, visible)
        return Observation(agent=agent, location=loc, visible_items=visible, inventory=list(s.inventories[agent]), task_status=dict(s.task_status), beliefs=dict(s.beliefs[agent]), delivered_messages=list(s.inboxes[agent]))

    def _update_belief_from_observation(self, agent: str, loc: str, visible: List[str]) -> None:
        s = self.state
        if agent == "C" and loc == "decoy_room" and s.beliefs["C"]["medical_kit_location"] == "decoy_room" and "medical_kit" not in visible:
            s.belief_conflict_count += 1
            s.false_belief_caused_wasted_action += 1
            s.beliefs["C"]["medical_kit_location"] = "unknown"
            if s.time_to_false_belief_conflict is None:
                s.time_to_false_belief_conflict = s.turn
            self._record("belief_conflict_detected", "C", expected="decoy_room", observed_absent="medical_kit")
            self._record("false_belief_wasted_action", "C", room="decoy_room")
        if "medical_kit" in visible:
            s.beliefs[agent]["medical_kit_location"] = loc


    def neighbors(self, room: str) -> list[str]:
        if self.scenario_id != "C5b_costly_false_belief":
            return [r for r in LOCATIONS if r != room]
        graph = {
            "staging": ["red_room", "blue_room", "corridor_1", "long_decoy_1"],
            "red_room": ["staging"],
            "blue_room": ["staging"],
            "corridor_1": ["staging", "box_room"],
            "box_room": ["corridor_1", "victim_room", "med_room"],
            "victim_room": ["box_room"],
            "med_room": ["box_room"],
            "long_decoy_1": ["staging", "long_decoy_2"],
            "long_decoy_2": ["long_decoy_1", "decoy_room"],
            "decoy_room": ["long_decoy_2"],
        }
        return graph.get(room, [])

    def next_step_toward(self, start: str, goal: str) -> str:
        if start == goal:
            return start
        from collections import deque
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            for n in self.neighbors(cur):
                if n in prev:
                    continue
                prev[n] = cur
                if n == goal:
                    q.clear()
                    break
                q.append(n)
        if goal not in prev:
            return start
        cur = goal
        while prev[cur] != start:
            cur = prev[cur]
            if cur is None:
                return start
        return cur
    def _record(self, event: str, agent: str | None = None, **details: object) -> None:
        self.state.trace.append(Event(turn=self.state.turn, event=event, agent=agent, details=details))


    def _deliver_messages(self) -> None:
        s = self.state
        delivered = []
        still = []
        for m in s.delayed_queue:
            if s.turn >= m["delivery_step"]:
                s.inboxes[m["to"]].append(m)
                delivered.append(m)
                s.delivered_delayed_messages_count += 1
            else:
                still.append(m)
        s.delayed_queue = still
        self._record("message_delivery", details={"delivered": delivered, "pending": len(still)})

    def step(self, agent: str, action: str, target: str, intent: dict | None = None) -> StepResult:
        s = self.state
        if s.done:
            return StepResult(False, True, True, "episode_done")
        s.turn += 1
        self._deliver_messages()
        if intent is not None:
            self._record("action_intent", agent, **intent)
        invalid, reason = False, None
        if action == "move":
            if target not in LOCATIONS:
                invalid, reason = True, "unknown_location"
            elif target not in self.neighbors(s.locations[agent]) and target != s.locations[agent]:
                invalid, reason = True, "non_adjacent_move"
            else:
                s.locations[agent] = target
                self._record("move", agent, target=target)
        elif action == "pickup":
            invalid, reason = self._pickup(agent, target)
        elif action == "open_box":
            invalid, reason = self._open_box(agent)
        elif action == "rescue":
            invalid, reason = self._rescue(agent)
        elif action == "send_message":
            invalid, reason = self._send_message(agent, target)
        else:
            invalid, reason = True, "unknown_action"
        self._record("action_result", agent, action=action, target=target, action_valid=(not invalid), reason=reason,
                     task_status_snapshot=dict(s.task_status),
                     agent_locations_snapshot=dict(s.locations),
                     inventories_snapshot={k:list(v) for k,v in s.inventories.items()})
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
        if agent == "C" and item == "medical_kit" and s.time_to_medical_kit_acquired is None:
            s.time_to_medical_kit_acquired = s.turn
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
            if s.time_to_red_key_applied is None:
                s.time_to_red_key_applied = s.turn
            self._record("red_key_applied", agent)
        elif agent == "B":
            if "blue_key" not in s.inventories["B"] or s.task_status["blue_key_applied"]:
                return True, "blue_key_unavailable"
            s.task_status["blue_key_applied"] = True
            if s.time_to_blue_key_applied is None:
                s.time_to_blue_key_applied = s.turn
            self._record("blue_key_applied", agent)
        else:
            return True, "role_mismatch_open"
        if s.task_status["red_key_applied"] and s.task_status["blue_key_applied"]:
            s.task_status["locked_box_open"] = True
            s.task_status["medical_kit_revealed"] = True
            if s.time_to_box_open is None:
                s.time_to_box_open = s.turn
            if s.time_to_medical_kit_revealed is None:
                s.time_to_medical_kit_revealed = s.turn
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
        if s.time_to_rescue is None:
            s.time_to_rescue = s.turn
        s.success = True
        s.done = True
        self._record("victim_rescued", agent)
        return False, None


    def _send_message(self, agent: str, target: str) -> Tuple[bool, str | None]:
        s = self.state
        try:
            to, content = target.split("|", 1)
        except ValueError:
            return True, "bad_message_format"
        if to not in AGENTS:
            return True, "bad_recipient"
        delay = 1 + ((s.seed + s.turn) % 2)
        msg = {"from": agent, "to": to, "content": content, "sent_step": s.turn, "delivery_step": s.turn + delay}
        s.delayed_queue.append(msg)
        s.delayed_messages_count += 1
        self._record("message_sent", agent, **msg)
        return False, None

    def summary(self) -> EpisodeSummary:
        s = self.state
        return EpisodeSummary(
            scenario_id=s.scenario_id, seed=s.seed, success=s.success, turns=s.turn, invalid_actions=s.invalid_actions,
            false_belief_injections=s.false_belief_injections, belief_conflict_count=s.belief_conflict_count,
            false_belief_caused_wasted_action=s.false_belief_caused_wasted_action,
            time_to_red_key_applied=s.time_to_red_key_applied,
            time_to_blue_key_applied=s.time_to_blue_key_applied,
            time_to_box_open=s.time_to_box_open,
            time_to_false_belief_conflict=s.time_to_false_belief_conflict,
            time_to_medical_kit_revealed=s.time_to_medical_kit_revealed,
            time_to_medical_kit_acquired=s.time_to_medical_kit_acquired,
            time_to_rescue=s.time_to_rescue,
            delayed_messages_count=s.delayed_messages_count,
            delivered_delayed_messages_count=s.delivered_delayed_messages_count,
            pending_messages_final_count=len(s.delayed_queue),
            task_status=dict(s.task_status), agent_locations=dict(s.locations), inventories={k: list(v) for k, v in s.inventories.items()},
        )
