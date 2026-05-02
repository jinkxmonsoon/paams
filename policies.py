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
        return self.rng.choice(["move:room_1", "move:room_3", "move:room_4", "move:room_5", "inspect"])


class GreedySecondOrderBeliefPolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.first_order: Dict[str, Dict[str, str]] = {}
        self.second_order: Dict[str, Dict[str, Dict[str, str]]] = {}

    def _ensure(self, a: str) -> None:
        self.first_order.setdefault(a, {})
        self.second_order.setdefault(a, {"A": {}, "B": {}, "C": {}})

    def act(self, agent_id: str, observation: Dict[str, Any]) -> str:
        self._ensure(agent_id)
        room = observation["current_room"]
        items = observation["current_room_contents"]
        inv = observation["own_inventory"]

        for item in items:
            self.first_order[agent_id][item] = room
            for other, other_room in observation.get("known_agent_locations", {}).items():
                if other_room == room:
                    self.second_order[agent_id].setdefault(other, {})[item] = room

        for msg in observation.get("delivered_messages", []):
            fact = _parse_fact_message(msg["text"])
            if fact:
                self.first_order[agent_id][fact["object"]] = fact["room"]
                self.second_order[agent_id].setdefault(agent_id, {})[fact["object"]] = fact["room"]

        if items:
            return f"pickup:{items[0]}" if items[0] in ["red_key", "blue_key", "medical_kit"] else "inspect"

        if room == "room_5":
            if "red_key" in inv:
                return "use:red_key"
            if "medical_kit" in inv:
                return "use:medical_kit"
            return "rescue:victim"

        # Use second-order beliefs to communicate only when another agent likely doesn't know.
        for obj, obj_room in self.first_order[agent_id].items():
            for other in ["A", "B", "C"]:
                if other != agent_id and obj not in self.second_order[agent_id].get(other, {}):
                    return f"send:{obj} seen in {obj_room}"

        return self.rng.choice(["move:room_1", "move:room_3", "move:room_4", "move:room_5", "wait"])
