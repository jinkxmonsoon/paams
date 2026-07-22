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
    messages_sent_count: int
    premature_shared_memory_assumptions: int
    delayed_message_confusion_events: int
    second_order_delivery_waits: int
    premature_arrival_or_wrong_positioning_steps: int
    wrong_branch_steps: int
    recovery_from_wrong_branch_steps: int
    duplicate_resource_attempts: int
    wrong_agent_resource_attempts: int
    resource_claim_conflicts: int
    responsibility_corrections: int
    assigned_agent_for_medical_kit: str
    task_status: Dict[str, bool]
    agent_locations: Dict[str, str]
    inventories: Dict[str, List[str]]
    correction_opportunities: int
    necessary_correction_messages: int
    missed_necessary_corrections: int
    unnecessary_correction_messages: int
    evidence_to_correction_send_steps: Optional[int]
    correction_delivery_latency: Optional[int]
    target_stale_belief_steps: int
    target_decoy_branch_steps: int
    post_correction_decoy_steps: int
    total_messages: int
