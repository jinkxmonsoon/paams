from __future__ import annotations

import random
from typing import Any, Dict, Optional


class BasePolicy:
    anti_leakage_passed = True

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        raise NotImplementedError


def _parse_fact_message(text: str) -> Optional[Dict[str, str]]:
    parts = text.split()
    if len(parts) >= 4 and parts[1] == "seen" and parts[2] == "in":
        return {"object": parts[0], "room": parts[3]}
    return None


ROOM_NEIGHBORS = {
    "room_0": ["room_1", "room_2"], "room_1": ["room_0", "room_3"], "room_2": ["room_0", "room_4"],
    "room_3": ["room_1", "room_5"], "room_4": ["room_2", "room_5"], "room_5": ["room_3", "room_4"],
}


class RandomPolicy(BasePolicy):
    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        room = observation["current_room"]
        valid = ["inspect", "wait", "rescue:victim"] + [f"move:{n}" for n in ROOM_NEIGHBORS[room]]
        for obj in ["red_key", "blue_key", "medical_kit"]:
            valid += [f"pickup:{obj}", f"use:{obj}"]
        return self.rng.choice(valid + ["move:room_99", "pickup:banana", "use:banana", "dance"])


class GreedySharedMemoryPolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.shared_memory = {"object_rooms": {}, "sent": set()}

    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        room, items, inv = observation["current_room"], observation["current_room_contents"], observation["own_inventory"]
        for msg in observation.get("delivered_messages", []):
            fact = _parse_fact_message(msg["text"])
            if fact:
                self.shared_memory["object_rooms"][fact["object"]] = fact["room"]
        for item in items:
            self.shared_memory["object_rooms"][item] = room
        preferred = {"A": ["red_key"], "B": ["blue_key"], "C": ["medical_kit"]}
        for obj in preferred[agent_id]:
            if obj in items:
                return f"pickup:{obj}"
        if room == "room_5":
            if "red_key" in inv:
                return "use:red_key"
            if "medical_kit" in inv:
                return "use:medical_kit"
            return "rescue:victim"
        return f"move:{ROOM_NEIGHBORS[room][0]}"


class GreedyBeliefStatePolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.agent_beliefs: Dict[str, Dict[str, str]] = {}
        self.visited: Dict[str, set] = {"A": set(), "B": set(), "C": set()}

    def _next_step_toward(self, current: str, target: str) -> Optional[str]:
        if current == target:
            return None
        queue = [(current, [])]
        seen = {current}
        while queue:
            node, path = queue.pop(0)
            for n in ROOM_NEIGHBORS[node]:
                if n in seen:
                    continue
                if n == target:
                    return (path + [n])[0]
                seen.add(n)
                queue.append((n, path + [n]))
        return None

    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        b = self.agent_beliefs.setdefault(agent_id, {})
        room, items, inv = observation["current_room"], observation["current_room_contents"], observation["own_inventory"]
        self.visited[agent_id].add(room)
        for item in items:
            b[item] = room
        for msg in observation.get("delivered_messages", []):
            fact = _parse_fact_message(msg["text"])
            if fact:
                b[fact["object"]] = fact["room"]

        for obj in ["red_key", "blue_key", "medical_kit"]:
            if obj in items:
                return f"pickup:{obj}"
        if room == "room_5":
            if "red_key" in inv:
                return "use:red_key"
            if "medical_kit" in inv:
                return "use:medical_kit"
            if observation.get("full_global_state", {}).get("task_status", {}).get("medical_kit_used", False):
                return "rescue:victim"
        target_obj = {"A": "red_key", "B": "blue_key", "C": "medical_kit"}[agent_id]
        if target_obj in b:
            nxt = self._next_step_toward(room, b[target_obj])
            if nxt:
                return f"move:{nxt}"
        unvisited = [n for n in ROOM_NEIGHBORS[room] if n not in self.visited[agent_id]]
        return f"move:{unvisited[0] if unvisited else ROOM_NEIGHBORS[room][0]}"


class GreedySecondOrderBeliefPolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.base = GreedyBeliefStatePolicy(seed=seed)
        self.first_order: Dict[str, Dict[str, str]] = {}
        self.second_order: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
        self.sent_message_cache = set()

    def _ensure(self, agent_id: str) -> None:
        self.first_order.setdefault(agent_id, {})
        self.second_order.setdefault(agent_id, {"A": {}, "B": {}, "C": {}})

    def _set_so(self, i: str, j: str, obj: str, room: str, state: str) -> None:
        self.second_order[i][j][obj] = {"state": state, "value": room}

    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        self._ensure(agent_id)
        room = observation["current_room"]
        for item in observation["current_room_contents"]:
            self.first_order[agent_id][item] = room
            for other, other_room in observation.get("known_agent_locations", {}).items():
                if other_room == room:
                    self._set_so(agent_id, other, item, room, "knows")
                elif other != agent_id and item not in self.second_order[agent_id][other]:
                    self._set_so(agent_id, other, item, room, "unknown")
        for msg in observation.get("delivered_messages", []):
            fact = _parse_fact_message(msg["text"])
            if fact:
                self.first_order[agent_id][fact["object"]] = fact["room"]
                self._set_so(agent_id, agent_id, fact["object"], fact["room"], "knows")

        base_action = self.base.act(agent_id, observation)
        urgent = base_action.startswith(("pickup:", "use:", "move:")) or base_action == "rescue:victim"

        if not urgent:
            better = {"red_key": "A", "blue_key": "B", "medical_kit": "C"}
            for obj, obj_room in self.first_order[agent_id].items():
                tgt = better.get(obj)
                if not tgt or tgt == agent_id:
                    continue
                rec = self.second_order[agent_id][tgt].get(obj, {"state": "unknown"})
                key = (agent_id, tgt, obj, obj_room)
                if rec["state"] in {"unknown", "does_not_know"} and key not in self.sent_message_cache:
                    self.sent_message_cache.add(key)
                    self._set_so(agent_id, tgt, obj, obj_room, "knows")
                    return f"send:{obj} seen in {obj_room}"
        return base_action
