from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .schemas import AGENTS, EpisodeSummary, Event, Observation, StepResult

LOCATIONS = (
    "staging", "wait_room", "box_path_1", "red_room", "blue_room", "box_room", "med_room", "victim_room",
    "decoy_room", "long_decoy_1", "long_decoy_2", "corridor_1", "wrong_branch_1", "wrong_branch_2", "wrong_room",
)

@dataclass
class WorldState:
    scenario_id: str; seed: int; turn: int; max_turns: int; done: bool; success: bool; invalid_actions: int
    false_belief_injections: int; belief_conflict_count: int; false_belief_caused_wasted_action: int
    time_to_red_key_applied: int | None; time_to_blue_key_applied: int | None; time_to_box_open: int | None
    time_to_false_belief_conflict: int | None; time_to_medical_kit_revealed: int | None; time_to_medical_kit_acquired: int | None; time_to_rescue: int | None
    locations: Dict[str, str]; inventories: Dict[str, List[str]]; room_items: Dict[str, List[str]]; beliefs: Dict[str, Dict[str, str]]
    task_status: Dict[str, bool]; trace: List[Event]; delayed_queue: List[dict]; inboxes: Dict[str, List[dict]]
    delayed_messages_count: int; delivered_delayed_messages_count: int; messages_sent_count: int
    premature_shared_memory_assumptions: int; delayed_message_confusion_events: int; second_order_delivery_waits: int
    premature_arrival_or_wrong_positioning_steps: int; wrong_branch_steps: int; recovery_from_wrong_branch_steps: int
    duplicate_resource_attempts: int; wrong_agent_resource_attempts: int; resource_claim_conflicts: int; responsibility_corrections: int; assigned_agent_for_medical_kit: str
    correction_instrument: Dict[str, object]

class BTomEnvV2:
    def __init__(self, scenario_id: str, seed: int, max_turns: int = 30) -> None:
        supported = {"C1_fully_observable", "C2_partial_observable", "C4_communication_delay", "C4b_costly_communication_delay", "C4c_wrong_branch_communication_delay", "C5_false_belief_injection", "C5b_costly_false_belief", "C6_resource_allocation", "C7a_partner_belief_stale", "C7b_partner_belief_current"}
        if scenario_id not in supported: raise ValueError(f"unsupported scenario: {scenario_id}")
        self.scenario_id, self.seed, self.max_turns = scenario_id, seed, max_turns
        self.state = self._init_state()

    def _init_state(self) -> WorldState:
        s = WorldState(self.scenario_id, self.seed, 0, self.max_turns, False, False, 0, 0, 0, 0, None, None, None, None, None, None, None,
            {a: "staging" for a in AGENTS}, {a: [] for a in AGENTS}, {k: [] for k in LOCATIONS}, {a: {"medical_kit_location": "unknown"} for a in AGENTS},
            {"red_key_applied": False, "blue_key_applied": False, "locked_box_open": False, "medical_kit_revealed": False, "victim_rescued": False},
            [], [], {a: [] for a in AGENTS}, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "C",
            {"opportunity_turn": None, "needed": None, "correction_sent_turn": None, "correction_delivered_turn": None,
             "necessary_messages": 0, "unnecessary_messages": 0, "target_stale_belief_steps": 0,
             "target_decoy_branch_steps": 0, "post_correction_decoy_steps": 0})
        s.room_items["red_room"] = ["red_key"]; s.room_items["blue_room"] = ["blue_key"]; s.room_items["box_room"] = ["locked_box"]; s.room_items["victim_room"] = ["victim"]
        if self.scenario_id in {"C5_false_belief_injection", "C5b_costly_false_belief"}:
            s.beliefs["C"]["medical_kit_location"] = "decoy_room"; s.false_belief_injections = 1
            s.trace.append(Event(turn=0, event="false_belief_injected", agent="C", details={"medical_kit_location": "decoy_room"}))
        if self.scenario_id == "C7a_partner_belief_stale":
            s.beliefs["C"]["medical_kit_location"] = "decoy_room"; s.false_belief_injections = 1
            s.trace.append(Event(turn=0, event="partner_belief_initialized", agent="C", details={"medical_kit_location": "decoy_room"}))
        elif self.scenario_id == "C7b_partner_belief_current":
            s.beliefs["C"]["medical_kit_location"] = "box_room"
            s.trace.append(Event(turn=0, event="partner_belief_initialized", agent="C", details={"medical_kit_location": "box_room"}))
        self._assert_state_invariants(s)
        return s

    def get_observation(self, agent: str) -> Observation:
        s = self.state; loc = s.locations[agent]
        visible = sorted({i for v in s.room_items.values() for i in v}) if self.scenario_id == "C1_fully_observable" else list(s.room_items[loc])
        self._update_belief_from_messages(agent)
        self._update_belief_from_observation(agent, loc, visible)
        task_status = self._observable_task_status(agent, loc) if self._is_correction_scenario() else dict(s.task_status)
        observation = Observation(agent, loc, visible, list(s.inventories[agent]), task_status, dict(s.beliefs[agent]), list(s.inboxes[agent]))
        if self._is_correction_scenario(): self._maybe_record_correction_opportunity(agent, observation)
        return observation

    def _is_correction_scenario(self) -> bool:
        return self.scenario_id in {"C7a_partner_belief_stale", "C7b_partner_belief_current"}

    def _update_belief_from_messages(self, agent: str) -> None:
        if not self._is_correction_scenario(): return
        if any(m.get("to") == agent and m.get("content") == "kit_revealed" for m in self.state.inboxes[agent]):
            self.state.beliefs[agent]["medical_kit_location"] = "box_room"

    def _observable_task_status(self, agent: str, loc: str) -> Dict[str, bool]:
        s = self.state
        visible = {k: False for k in s.task_status}
        if agent == "A": visible["red_key_applied"] = s.task_status["red_key_applied"]
        if agent == "B": visible["blue_key_applied"] = s.task_status["blue_key_applied"]
        if loc == "box_room":
            for key in ("red_key_applied", "blue_key_applied", "locked_box_open", "medical_kit_revealed"):
                visible[key] = s.task_status[key]
        if any(m.get("to") == agent and m.get("content") == "kit_revealed" for m in s.inboxes[agent]):
            for key in ("red_key_applied", "blue_key_applied", "locked_box_open", "medical_kit_revealed"):
                visible[key] = True
        if agent == "C" or loc == "victim_room": visible["victim_rescued"] = s.task_status["victim_rescued"]
        return visible

    def _maybe_record_correction_opportunity(self, agent: str, observation: Observation) -> None:
        s, instrument = self.state, self.state.correction_instrument
        if agent != "A" or instrument["opportunity_turn"] is not None or "medical_kit" not in observation.visible_items: return
        evidence = [m for m in observation.delivered_messages if m.get("from") == "C" and str(m.get("content", "")).startswith("belief:medical_kit_location=")]
        if not evidence: return
        believed_value = evidence[-1]["content"].split("=", 1)[1]
        instrument["opportunity_turn"] = s.turn
        instrument["needed"] = s.beliefs["C"]["medical_kit_location"] != "box_room"
        self._record("correction_opportunity", "A", target_agent="C", target_believed_value=believed_value,
                     true_value="box_room", correction_needed=instrument["needed"])

    def _update_belief_from_observation(self, agent: str, loc: str, visible: List[str]) -> None:
        s = self.state
        if agent == "C" and loc == "decoy_room" and s.beliefs["C"]["medical_kit_location"] == "decoy_room" and "medical_kit" not in visible:
            s.belief_conflict_count += 1; s.false_belief_caused_wasted_action += 1; s.beliefs["C"]["medical_kit_location"] = "unknown"
            if s.time_to_false_belief_conflict is None: s.time_to_false_belief_conflict = s.turn
            self._record("belief_conflict_detected", "C", expected="decoy_room", observed_absent="medical_kit")
        if "medical_kit" in visible: s.beliefs[agent]["medical_kit_location"] = loc

    def neighbors(self, room: str) -> list[str]:
        if self.scenario_id in {"C5b_costly_false_belief", "C7a_partner_belief_stale", "C7b_partner_belief_current"}:
            return {"staging": ["red_room", "blue_room", "corridor_1", "long_decoy_1"], "red_room": ["staging"], "blue_room": ["staging"], "corridor_1": ["staging", "box_room"], "box_room": ["corridor_1", "victim_room", "med_room"], "victim_room": ["box_room"], "med_room": ["box_room"], "long_decoy_1": ["staging", "long_decoy_2"], "long_decoy_2": ["long_decoy_1", "decoy_room"], "decoy_room": ["long_decoy_2"]}.get(room, [])
        if self.scenario_id == "C4b_costly_communication_delay":
            return {"staging": ["wait_room", "box_path_1", "red_room", "blue_room"], "wait_room": ["staging"], "box_path_1": ["staging", "box_room"], "red_room": ["staging"], "blue_room": ["staging"], "box_room": ["box_path_1", "victim_room", "med_room"], "victim_room": ["box_room"], "med_room": ["box_room"]}.get(room, [])
        if self.scenario_id == "C4c_wrong_branch_communication_delay":
            return {"staging": ["wait_room", "wrong_branch_1", "red_room", "blue_room"], "wrong_branch_1": ["staging", "wrong_branch_2"], "wrong_branch_2": ["wrong_branch_1", "wrong_room"], "wrong_room": ["wrong_branch_2"], "wait_room": ["staging", "box_path_1"], "box_path_1": ["wait_room", "box_room"], "red_room": ["staging"], "blue_room": ["staging"], "box_room": ["box_path_1", "victim_room", "med_room"], "victim_room": ["box_room"], "med_room": ["box_room"]}.get(room, [])
        return [r for r in LOCATIONS if r != room]

    def next_step_toward(self, start: str, goal: str) -> str:
        if start == goal: return start
        from collections import deque
        q, prev = deque([start]), {start: None}
        while q:
            c = q.popleft()
            for n in self.neighbors(c):
                if n in prev: continue
                prev[n] = c
                if n == goal: q.clear(); break
                q.append(n)
        if goal not in prev: return start
        cur = goal
        while prev[cur] != start: cur = prev[cur]
        return cur

    def _record(self, event: str, agent: str | None = None, **details: object) -> None: self.state.trace.append(Event(turn=self.state.turn, event=event, agent=agent, details=details))
    def _deliver_messages(self) -> None:
        s, delivered, still = self.state, [], []
        for m in s.delayed_queue:
            if s.turn >= m["delivery_step"]:
                s.inboxes[m["to"]].append(m); delivered.append(m); s.delivered_delayed_messages_count += 1
                if self._is_correction_scenario() and m.get("from") == "A" and m.get("to") == "C" and m.get("content") == "kit_revealed":
                    s.beliefs["C"]["medical_kit_location"] = "box_room"
                    s.correction_instrument["correction_delivered_turn"] = s.turn
                    self._record("belief_correction_delivered", "C", sent_step=m["sent_step"], delivery_step=s.turn)
            else: still.append(m)
        s.delayed_queue = still; self._record("message_delivery", details={"delivered": delivered, "pending": len(still)})

    def _apply_delay_cost(self, agent: str, action: str, target: str, intent: dict | None) -> None:
        s = self.state
        if self.scenario_id not in {"C4b_costly_communication_delay", "C4c_wrong_branch_communication_delay"} or agent != "C" or action != "move": return
        assumes = bool((intent or {}).get("premature_shared_memory_assumption", False))
        in_cost_zone = target in {"box_path_1", "box_room", "wrong_branch_1", "wrong_branch_2", "wrong_room"}
        if in_cost_zone and assumes: s.premature_arrival_or_wrong_positioning_steps += 1; s.delayed_message_confusion_events += 1; self._record("premature_arrival_or_wrong_positioning", agent, target=target)
        if self.scenario_id == "C4c_wrong_branch_communication_delay" and target in {"wrong_branch_1", "wrong_branch_2", "wrong_room"}:
            s.wrong_branch_steps += 1
            if (intent or {}).get("recovering_from_wrong_branch", False): s.recovery_from_wrong_branch_steps += 1

    def step(self, agent: str, action: str, target: str, intent: dict | None = None) -> StepResult:
        s = self.state
        if s.done: return StepResult(False, True, True, "episode_done")
        s.turn += 1; self._deliver_messages()
        if self._is_correction_scenario() and agent == "C" and s.correction_instrument["opportunity_turn"] is not None and s.beliefs["C"]["medical_kit_location"] != "box_room":
            s.correction_instrument["target_stale_belief_steps"] += 1
        if intent is not None:
            self._record("action_intent", agent, **intent)
            s.premature_shared_memory_assumptions += int(intent.get("premature_shared_memory_assumption", False)); s.delayed_message_confusion_events += int(intent.get("delayed_message_confusion", False)); s.second_order_delivery_waits += int(intent.get("second_order_delivery_wait", False))
            if intent.get("wrong_agent_resource_attempt_detected", False): s.wrong_agent_resource_attempts += 1; self._record("wrong_agent_resource_attempt_detected", agent)
            if intent.get("duplicate_resource_attempt_detected", False): s.duplicate_resource_attempts += 1; self._record("duplicate_resource_attempt_detected", agent)
            if intent.get("resource_claim_conflict_detected", False): s.resource_claim_conflicts += 1; self._record("resource_claim_conflict_detected", agent)
            if intent.get("responsibility_correction_sent", False): s.responsibility_corrections += 1; self._record("responsibility_correction_sent", agent)
        invalid, reason = False, None
        if action == "move":
            if target not in LOCATIONS: invalid, reason = True, "unknown_location"
            elif target not in self.neighbors(s.locations[agent]) and target != s.locations[agent]: invalid, reason = True, "non_adjacent_move"
            else:
                self._apply_delay_cost(agent, action, target, intent); s.locations[agent] = target; self._record("move", agent, target=target)
                if self._is_correction_scenario() and agent == "C" and target in {"long_decoy_1", "long_decoy_2", "decoy_room"}:
                    s.correction_instrument["target_decoy_branch_steps"] += 1
                    if s.correction_instrument["correction_delivered_turn"] is not None: s.correction_instrument["post_correction_decoy_steps"] += 1
        elif action == "pickup": invalid, reason = self._pickup(agent, target)
        elif action == "open_box": invalid, reason = self._open_box(agent)
        elif action == "rescue": invalid, reason = self._rescue(agent)
        elif action == "send_message": invalid, reason = self._send_message(agent, target)
        else: invalid, reason = True, "unknown_action"
        self._record("action_result", agent, action=action, target=target, action_valid=(not invalid), reason=reason)
        if invalid: s.invalid_actions += 1
        if s.turn >= s.max_turns and not s.done: s.done = True; self._record("episode_timeout")
        if self._is_correction_scenario(): self.assert_invariants()
        return StepResult(success=not invalid, done=s.done, invalid=invalid, reason=reason)

    def _pickup(self, agent: str, item: str) -> Tuple[bool, str | None]:
        s = self.state; loc = s.locations[agent]
        if item not in s.room_items[loc]: return True, "item_not_in_room"
        if item == "red_key" and agent != "A": return True, "role_mismatch_pickup"
        if item == "blue_key" and agent != "B": return True, "role_mismatch_pickup"
        if item == "medical_kit" and agent != "C": return True, "role_mismatch_pickup"
        s.room_items[loc].remove(item); s.inventories[agent].append(item); self._record("pickup", agent, item=item)
        if agent == "C" and item == "medical_kit" and s.time_to_medical_kit_acquired is None: s.time_to_medical_kit_acquired = s.turn
        return False, None

    def _open_box(self, agent: str) -> Tuple[bool, str | None]:
        s = self.state
        if s.locations[agent] != "box_room": return True, "not_at_box_room"
        if s.task_status["locked_box_open"]: return True, "locked_box_already_open"
        if agent == "A":
            if "red_key" not in s.inventories["A"] or s.task_status["red_key_applied"]: return True, "red_key_unavailable"
            s.task_status["red_key_applied"] = True; s.time_to_red_key_applied = s.time_to_red_key_applied or s.turn
        elif agent == "B":
            if "blue_key" not in s.inventories["B"] or s.task_status["blue_key_applied"]: return True, "blue_key_unavailable"
            s.task_status["blue_key_applied"] = True; s.time_to_blue_key_applied = s.time_to_blue_key_applied or s.turn
        else: return True, "role_mismatch_open"
        if s.task_status["red_key_applied"] and s.task_status["blue_key_applied"]:
            s.task_status["locked_box_open"] = True; s.task_status["medical_kit_revealed"] = True; s.time_to_box_open = s.time_to_box_open or s.turn; s.time_to_medical_kit_revealed = s.time_to_medical_kit_revealed or s.turn
            s.room_items["box_room"] = [i for i in s.room_items["box_room"] if i != "locked_box"] + ["medical_kit"]
        return False, None

    def _rescue(self, agent: str) -> Tuple[bool, str | None]:
        s = self.state
        if agent != "C": return True, "role_mismatch_rescue"
        if s.locations[agent] != "victim_room": return True, "not_at_victim"
        if "medical_kit" not in s.inventories["C"]: return True, "medical_kit_missing"
        s.task_status["victim_rescued"] = True; s.time_to_rescue = s.time_to_rescue or s.turn; s.success = True; s.done = True
        return False, None

    def _send_message(self, agent: str, target: str) -> Tuple[bool, str | None]:
        s = self.state
        try: to, content = target.split("|", 1)
        except ValueError: return True, "bad_message_format"
        if to not in AGENTS: return True, "bad_recipient"
        delay = 1 + ((s.seed + s.turn) % 2); msg = {"from": agent, "to": to, "content": content, "sent_step": s.turn, "delivery_step": s.turn + delay}
        s.delayed_queue.append(msg); s.delayed_messages_count += 1; s.messages_sent_count += 1; self._record("message_sent", agent, **msg)
        if self._is_correction_scenario() and agent == "A" and to == "C" and content == "kit_revealed" and s.correction_instrument["opportunity_turn"] is not None:
            needed = bool(s.correction_instrument["needed"])
            s.correction_instrument["correction_sent_turn"] = s.turn
            s.correction_instrument["necessary_messages" if needed else "unnecessary_messages"] += 1
            self._record("belief_correction_sent", "A", target_agent="C", correction_needed=needed)
        return False, None

    def assert_invariants(self) -> None:
        self._assert_state_invariants(self.state)

    def _assert_state_invariants(self, s: WorldState) -> None:
        if s.task_status["medical_kit_revealed"]:
            assert s.task_status["red_key_applied"] and s.task_status["blue_key_applied"]
            assert s.task_status["locked_box_open"]
            kit_available = "medical_kit" in s.room_items["box_room"] or "medical_kit" in s.inventories["C"]
            assert kit_available

    def correction_metrics(self) -> Dict[str, int | None]:
        s, instrument = self.state, self.state.correction_instrument
        opportunity = instrument["opportunity_turn"]
        sent = instrument["correction_sent_turn"]
        delivered = instrument["correction_delivered_turn"]
        needed = bool(instrument["needed"]) if opportunity is not None else False
        return {
            "correction_opportunities": int(opportunity is not None),
            "necessary_correction_messages": int(instrument["necessary_messages"]),
            "missed_necessary_corrections": int(needed and not instrument["necessary_messages"]),
            "unnecessary_correction_messages": int(instrument["unnecessary_messages"]),
            "evidence_to_correction_send_steps": None if opportunity is None or sent is None else int(sent) - int(opportunity),
            "correction_delivery_latency": None if sent is None or delivered is None else int(delivered) - int(sent),
            "target_stale_belief_steps": int(instrument["target_stale_belief_steps"]),
            "target_decoy_branch_steps": int(instrument["target_decoy_branch_steps"]),
            "post_correction_decoy_steps": int(instrument["post_correction_decoy_steps"]),
            "total_messages": s.messages_sent_count,
        }

    def summary(self) -> EpisodeSummary:
        s = self.state
        cm = self.correction_metrics()
        return EpisodeSummary(s.scenario_id, s.seed, s.success, s.turn, s.invalid_actions, s.false_belief_injections, s.belief_conflict_count, s.false_belief_caused_wasted_action,
            s.time_to_red_key_applied, s.time_to_blue_key_applied, s.time_to_box_open, s.time_to_false_belief_conflict, s.time_to_medical_kit_revealed, s.time_to_medical_kit_acquired, s.time_to_rescue,
            s.delayed_messages_count, s.delivered_delayed_messages_count, len(s.delayed_queue), s.messages_sent_count, s.premature_shared_memory_assumptions, s.delayed_message_confusion_events,
            s.second_order_delivery_waits, s.premature_arrival_or_wrong_positioning_steps, s.wrong_branch_steps, s.recovery_from_wrong_branch_steps,
            s.duplicate_resource_attempts, s.wrong_agent_resource_attempts, s.resource_claim_conflicts, s.responsibility_corrections, s.assigned_agent_for_medical_kit,
            dict(s.task_status), dict(s.locations), {k: list(v) for k, v in s.inventories.items()},
            cm["correction_opportunities"], cm["necessary_correction_messages"], cm["missed_necessary_corrections"], cm["unnecessary_correction_messages"],
            cm["evidence_to_correction_send_steps"], cm["correction_delivery_latency"], cm["target_stale_belief_steps"],
            cm["target_decoy_branch_steps"], cm["post_correction_decoy_steps"], cm["total_messages"])
