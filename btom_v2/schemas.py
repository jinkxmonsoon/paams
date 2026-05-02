from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


AGENTS = ("A", "B", "C")


@dataclass
class Event:
    turn: int
    event: str
    agent: Optional[str] = None
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class Observation:
    agent: str
    location: str
    visible_items: List[str]
    inventory: List[str]
    task_status: Dict[str, bool]


@dataclass
class StepResult:
    success: bool
    done: bool
    invalid: bool
    reason: Optional[str]


@dataclass
class EpisodeSummary:
    scenario_id: str
    seed: int
    success: bool
    turns: int
    invalid_actions: int
    task_status: Dict[str, bool]
    agent_locations: Dict[str, str]
    inventories: Dict[str, List[str]]
