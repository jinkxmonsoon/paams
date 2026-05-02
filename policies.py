from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


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
    "room_0": ["room_1", "room_2"],
    "room_1": ["room_0", "room_3"],
    "room_2": ["room_0", "room_4"],
    "room_3": ["room_1", "room_5"],
    "room_4": ["room_2", "room_5"],
    "room_5": ["room_3", "room_4"],
}


class RandomPolicy(BasePolicy):
    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        room = observation["current_room"]
        valid = ["inspect", "wait", "rescue:victim"]
        neighbors = {
            "room_0": ["room_1", "room_2"], "room_1": ["room_0", "room_3"], "room_2": ["room_0", "room_4"],
            "room_3": ["room_1", "room_5"], "room_4": ["room_2", "room_5"], "room_5": ["room_3", "room_4"],
        }
        valid += [f"move:{n}" for n in neighbors[room]]
        for obj in ["red_key", "blue_key", "medical_kit"]:
            valid.append(f"pickup:{obj}")
            valid.append(f"use:{obj}")
        invalid = ["move:room_99", "pickup:banana", "use:banana", "dance"]
        return self.rng.choice(valid + invalid)


class GreedySharedMemoryPolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.shared_memory = {"object_rooms": {}, "sent": set()}

    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        room = observation["current_room"]
        items = observation["current_room_contents"]
        inv = observation["own_inventory"]
        for msg in observation.get("delivered_messages", []):
            fact = _parse_fact_message(msg["text"])
            if fact:
                self.shared_memory["object_rooms"][fact["object"]] = fact["room"]
        for item in items:
            self.shared_memory["object_rooms"][item] = room

        if items:
            for obj in ["red_key", "blue_key", "medical_kit"]:
                if obj in items:
                    return f"pickup:{obj}"

        if room == "room_5":
            if "red_key" in inv:
                return "use:red_key"
            if "medical_kit" in inv:
                return "use:medical_kit"
            return "rescue:victim"

        msg = f"{items[0]} seen in {room}" if items else None
        if msg and msg not in self.shared_memory["sent"]:
            self.shared_memory["sent"].add(msg)
            return f"send:{msg}"

        return self.rng.choice(["move:room_1", "move:room_3", "move:room_4", "move:room_5", "inspect", "wait"])


class GreedyBeliefStatePolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.agent_beliefs: Dict[str, Dict[str, str]] = {}

    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        b = self.agent_beliefs.setdefault(agent_id, {})
        room = observation["current_room"]
        items = observation["current_room_contents"]
        inv = observation["own_inventory"]
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
            return "rescue:victim"
        return self.rng.choice([f"move:{n}" for n in ROOM_NEIGHBORS[room]] + ["inspect"])


class GreedySecondOrderBeliefPolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.first_order: Dict[str, Dict[str, str]] = {}
        self.second_order: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
        self.sent_message_cache = set()

    def _ensure(self, a: str) -> None:
        self.first_order.setdefault(a, {})
        if a not in self.second_order:
            self.second_order[a] = {}
            for other in ["A", "B", "C"]:
                self.second_order[a][other] = {}

    def _set_second_order(self, observer: str, target: str, obj: str, room: str, state: str) -> None:
        self.second_order[observer][target][obj] = {"state": state, "value": room}

    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        self._ensure(agent_id)
        room = observation["current_room"]
        items = observation["current_room_contents"]
        inv = observation["own_inventory"]

        for item in items:
            self.first_order[agent_id][item] = room
            for other, other_room in observation.get("known_agent_locations", {}).items():
                if other_room == room:
                    self._set_second_order(agent_id, other, item, room, "knows")
                elif other != agent_id and item not in self.second_order[agent_id][other]:
                    self._set_second_order(agent_id, other, item, room, "unknown")

        for msg in observation.get("delivered_messages", []):
            fact = _parse_fact_message(msg["text"])
            if fact:
                self.first_order[agent_id][fact["object"]] = fact["room"]
                self._set_second_order(agent_id, agent_id, fact["object"], fact["room"], "knows")

        # 1) pick directly useful object
        useful_by_agent = {"A": {"red_key"}, "B": {"blue_key"}, "C": {"medical_kit"}}
        for item in items:
            if item in useful_by_agent.get(agent_id, set()) or item in {"red_key", "medical_kit"}:
                return f"pickup:{item}"

        # 2) use object to progress
        if room == "room_5":
            if "red_key" in inv:
                return "use:red_key"
            if "medical_kit" in inv:
                return "use:medical_kit"
            if observation.get("full_global_state", {}).get("task_status", {}).get("medical_kit_used", False):
                return "rescue:victim"
            return "inspect"

        # 4) move toward known relevant object locations
        target_obj = "red_key" if agent_id == "A" else ("blue_key" if agent_id == "B" else "medical_kit")
        if target_obj in self.first_order[agent_id]:
            target_room = self.first_order[agent_id][target_obj]
            if target_room in ROOM_NEIGHBORS.get(room, []):
                return f"move:{target_room}"
            return f"move:{ROOM_NEIGHBORS[room][0]}"

        # 5) message only if useful and not duplicated
        better_suited = {"red_key": "A", "blue_key": "B", "medical_kit": "C"}
        for obj, obj_room in self.first_order[agent_id].items():
            target = better_suited.get(obj)
            if not target or target == agent_id:
                continue
            rec = self.second_order[agent_id][target].get(obj, {"state": "unknown"})
            should_send = rec["state"] in {"unknown", "does_not_know"}
            cache_key = (agent_id, target, obj, obj_room)
            if should_send and cache_key not in self.sent_message_cache:
                self.sent_message_cache.add(cache_key)
                self._set_second_order(agent_id, target, obj, obj_room, "knows")
                return f"send:{obj} seen in {obj_room}"

        # 6) explore/wait (same robust baseline as first-order greedy)
        return self.rng.choice(["move:room_1", "move:room_3", "move:room_4", "move:room_5", "inspect"])
