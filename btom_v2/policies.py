from __future__ import annotations

from typing import Tuple

from .env import BTomEnvV2


Action = Tuple[str, str]


class DeterministicBaselinePolicy:
    def act(self, env: BTomEnvV2, agent: str, t: int) -> Action:
        s = env.state

        if agent == "A":
            if "red_key" not in s.inventories["A"]:
                return ("move", "red_room") if s.locations["A"] != "red_room" else ("pickup", "red_key")
            if s.locations["A"] != "box_room":
                return ("move", "box_room")
            if (not s.task_status["red_key_applied"]) and (not s.task_status["locked_box_open"]):
                return ("open_box", "locked_box")
            return ("move", s.locations["A"])

        if agent == "B":
            if "blue_key" not in s.inventories["B"]:
                return ("move", "blue_room") if s.locations["B"] != "blue_room" else ("pickup", "blue_key")
            if s.locations["B"] != "box_room":
                return ("move", "box_room")
            if (not s.task_status["blue_key_applied"]) and (not s.task_status["locked_box_open"]):
                return ("open_box", "locked_box")
            return ("move", s.locations["B"])

        if not s.task_status["medical_kit_revealed"]:
            return ("move", s.locations["C"])
        if "medical_kit" not in s.inventories["C"]:
            return ("move", "med_room") if s.locations["C"] != "med_room" else ("pickup", "medical_kit")
        if s.locations["C"] != "victim_room":
            return ("move", "victim_room")
        if not s.task_status["victim_rescued"]:
            return ("rescue", "victim")
        return ("move", s.locations["C"])
