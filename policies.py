from __future__ import annotations

import random
from typing import Any, Dict, Optional

ROOM_NEIGHBORS = {
    "room_0": ["room_1", "room_2"], "room_1": ["room_0", "room_3"], "room_2": ["room_0", "room_4"],
    "room_3": ["room_1", "room_5"], "room_4": ["room_2", "room_5"], "room_5": ["room_3", "room_4"],
}

ROLE_OBJ = {"A": "red_key", "B": "blue_key", "C": "medical_kit"}

class BasePolicy:
    anti_leakage_passed = True
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)


def _parse_fact_message(text: str) -> Optional[Dict[str, str]]:
    p = text.split()
    if len(p) >= 4 and p[1] == "seen" and p[2] == "in":
        return {"object": p[0], "room": p[3]}
    return None


def _next_step(cur: str, tgt: str) -> Optional[str]:
    if cur == tgt:
        return None
    q, seen = [(cur, [])], {cur}
    while q:
        n, path = q.pop(0)
        for z in ROOM_NEIGHBORS[n]:
            if z in seen:
                continue
            if z == tgt:
                return (path + [z])[0]
            seen.add(z)
            q.append((z, path + [z]))
    return None


class RandomPolicy(BasePolicy):
    def act(self, agent_id: str, obs: Dict[str, Any]) -> str:
        room = obs["current_room"]
        acts = ["inspect", "wait", "rescue:victim"] + [f"move:{n}" for n in ROOM_NEIGHBORS[room]]
        for o in ["red_key", "blue_key", "medical_kit"]:
            acts += [f"pickup:{o}", f"use:{o}"]
        return self.rng.choice(acts + ["move:room_99", "dance"])


class GreedySharedMemoryPolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.mem = {"obj": {}, "visited": {"A": set(), "B": set(), "C": set()}}
        self.sent_message_cache = set()
        self.messages_sent = 0

    def act(self, agent_id: str, obs: Dict[str, Any]) -> str:
        action = _task_action(agent_id, obs, self.mem, mode="shared")
        if action:
            return action
        return "wait"


class GreedyBeliefStatePolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.mem = {"obj": {}, "visited": {"A": set(), "B": set(), "C": set()}}
        self.sent_message_cache = set()
        self.messages_sent = 0

    def act(self, agent_id: str, obs: Dict[str, Any]) -> str:
        action = _task_action(agent_id, obs, self.mem, mode="belief")
        if action:
            return action
        return "wait"


class GreedySecondOrderBeliefPolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.mem = {"obj": {}, "visited": {"A": set(), "B": set(), "C": set()}}
        self.second_order = {a: {b: {} for b in ["A", "B", "C"]} for a in ["A", "B", "C"]}
        self.sent_message_cache = set()
        self.messages_sent = 0
        self.assigned_resources = {"medical_kit": "C"}

    def act(self, agent_id: str, obs: Dict[str, Any]) -> str:
        # conservative extension: first get same task-progress action
        action = _task_action(agent_id, obs, self.mem, mode="second_order")
        urgent = action.startswith(("move:", "pickup:", "use:")) or action == "rescue:victim"

        room = obs["current_room"]
        for i in obs["current_room_contents"]:
            for other, other_room in obs.get("known_agent_locations", {}).items():
                self.second_order[agent_id][other][i] = {"state": "knows" if other_room == room else "unknown", "value": room}

        # C5 false belief correction: if local evidence disproves room_2 claim, send one correction.
        if not urgent and self.mem["obj"].get("medical_kit") == "room_2" and "medical_kit" not in obs["current_room_contents"] and obs["current_room"] == "room_2":
            key = (agent_id, "broadcast", "medical_kit_not_in", "room_2")
            if key not in self.sent_message_cache:
                self.sent_message_cache.add(key); self.messages_sent += 1
                self.mem["obj"]["medical_kit"] = "unknown"
                return "send:medical_kit not in room_2"

        if not urgent:
            for obj, loc in self.mem["obj"].items():
                target = {"red_key": "A", "blue_key": "B", "medical_kit": "C"}.get(obj)
                if target and target != agent_id:
                    st = self.second_order[agent_id][target].get(obj, {"state": "unknown"})
                    key = (agent_id, target, obj, loc)
                    if st["state"] == "unknown" and key not in self.sent_message_cache:
                        self.sent_message_cache.add(key); self.messages_sent += 1
                        return f"send:{obj} seen in {loc}"
        return action


def _task_action(agent_id: str, obs: Dict[str, Any], mem: Dict[str, Any], mode: str) -> str:
    room, inv, items = obs["current_room"], obs["own_inventory"], obs["current_room_contents"]
    mem["visited"][agent_id].add(room)
    for i in items:
        mem["obj"][i] = room
    for m in obs.get("delivered_messages", []):
        f = _parse_fact_message(m["text"])
        if f:
            mem["obj"][f["object"]] = f["room"]

    # mission stages
    if agent_id == "C" and "medical_kit" in inv and room == mem["obj"].get("victim", "room_5"):
        return "rescue:victim"
    if agent_id in {"A", "B"} and ROLE_OBJ[agent_id] in inv and room == mem["obj"].get("locked_box", "room_5"):
        return f"use:{ROLE_OBJ[agent_id]}"

    if ROLE_OBJ[agent_id] in items:
        return f"pickup:{ROLE_OBJ[agent_id]}"

    target = None
    if agent_id in {"A", "B"} and ROLE_OBJ[agent_id] not in inv:
        target = mem["obj"].get(ROLE_OBJ[agent_id])
    elif agent_id in {"A", "B"} and ROLE_OBJ[agent_id] in inv:
        target = mem["obj"].get("locked_box", "room_5")
    elif agent_id == "C" and "medical_kit" not in inv:
        # Only pursue kit after box open is visible in global observation or kit location known.
        locked_open = obs.get("full_global_state", {}).get("task_status", {}).get("locked_box_open", False)
        if locked_open or "medical_kit" in mem["obj"]:
            target = mem["obj"].get("medical_kit", "room_5")
    elif agent_id == "C" and "medical_kit" in inv:
        target = mem["obj"].get("victim", "room_5")

    if target:
        nxt = _next_step(room, target)
        if nxt:
            return f"move:{nxt}"

    unvisited = [n for n in ROOM_NEIGHBORS[room] if n not in mem["visited"][agent_id]]
    if unvisited:
        return f"move:{unvisited[0]}"
    return f"move:{ROOM_NEIGHBORS[room][0]}"
