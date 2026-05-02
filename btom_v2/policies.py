from __future__ import annotations

from typing import Dict, Tuple

from .env import BTomEnvV2


Action = Tuple[str, str]


class DeterministicBaselinePolicy:
    """Simple deterministic script policy for A/B/C."""

    def __init__(self) -> None:
        self.plan: Dict[str, list[Action]] = {
            "A": [
                ("move", "red_room"),
                ("pickup", "red_key"),
                ("move", "box_room"),
                ("open_box", "locked_box"),
            ],
            "B": [
                ("move", "blue_room"),
                ("pickup", "blue_key"),
                ("move", "box_room"),
                ("open_box", "locked_box"),
            ],
            "C": [
                ("move", "staging"),
                ("move", "staging"),
                ("move", "staging"),
                ("move", "med_room"),
                ("pickup", "medical_kit"),
                ("move", "victim_room"),
                ("rescue", "victim"),
            ],
        }

    def act(self, env: BTomEnvV2, agent: str, t: int) -> Action:
        steps = self.plan[agent]
        if t < len(steps):
            return steps[t]
        return ("move", env.state.locations[agent])
