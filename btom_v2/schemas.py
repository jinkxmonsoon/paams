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
    beliefs: Dict[str, str]
    delivered_messages: List[Dict[str, object]]


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
    false_belief_injections: int
    belief_conflict_count: int
    false_belief_caused_wasted_action: int
    time_to_red_key_applied: Optional[int]
    time_to_blue_key_applied: Optional[int]
    time_to_box_open: Optional[int]
    time_to_false_belief_conflict: Optional[int]
    time_to_medical_kit_revealed: Optional[int]
    time_to_medical_kit_acquired: Optional[int]
    time_to_rescue: Optional[int]
    delayed_messages_count: int
    delivered_delayed_messages_count: int
    pending_messages_final_count: int
    task_status: Dict[str, bool]
    agent_locations: Dict[str, str]
    inventories: Dict[str, List[str]]
