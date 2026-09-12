"""Outcome-independent record-only held-out scenarios for Journal J1 v0.6.0."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARCHETYPES = ("resource_location", "tool_placement", "rendezvous_destination", "delivery_destination", "hazard_assignment", "maintenance_target")
DIFFICULTIES = ("direct", "irrelevant_distractor", "compositional")
ARCH_CODES = {"resource_location": "RL", "tool_placement": "TP", "rendezvous_destination": "RD", "delivery_destination": "DD", "hazard_assignment": "HA", "maintenance_target": "MT"}
DIFF_CODES = {"direct": "DIR", "irrelevant_distractor": "DIS", "compositional": "COM"}
ROLES = (
    ("Maren", "Tobin", "Yvette"), ("Nadia", "Ulric", "Zelda"), ("Oren", "Vesper", "Amabel"),
    ("Petra", "Warren", "Benton"), ("Ronan", "Xanthe", "Cleo"), ("Selene", "Yorick", "Della"),
    ("Theron", "Zinnia", "Eamon"), ("Valora", "Arden", "Faye"), ("Wilfred", "Briony", "Galen"),
    ("Xiomara", "Corwin", "Helena"), ("Yasmin", "Darius", "Isolde"), ("Zephyr", "Elowen", "Jareth"),
    ("Althea", "Fintan", "Kerensa"), ("Bastian", "Gwyneira", "Lorcan"), ("Calista", "Hadley", "Meliora"),
    ("Dorian", "Ianthe", "Nevan"), ("Evadne", "Jago", "Ottilie"), ("Faolan", "Keturah", "Peregrine"),
)
STEMS = ("Aster", "Boreal", "Cinnabar", "Driftwood", "Evergreen", "Foxglove", "Garnet", "Heather", "Ironwood", "Kingfisher", "Lodestar", "Moonstone", "Northwind", "Osprey", "Primrose", "Redwood", "Seabrook", "Windward")
VALUE_STEMS = ("Auburn", "Beryl", "Cirrus", "Dovetail", "Estuary", "Firelight", "Glacier", "Highland", "Inlet", "Jetty", "Keystone", "Lowland", "Monarch", "Nightingale", "Overlook", "Palisade", "Rivermark", "Starling")
ARCH = {
    "resource_location": ("canister", "retrieval", "location", "Bay"),
    "tool_placement": ("instrument", "assembly", "location", "Depot"),
    "rendezvous_destination": ("team marker", "rendezvous", "destination", "Gate"),
    "delivery_destination": ("parcel", "delivery", "destination", "Dock"),
    "hazard_assignment": ("survey packet", "inspection", "sector", "Sector"),
    "maintenance_target": ("service ticket", "maintenance", "component", "Module"),
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def edge(source: str, relation: str, target: str) -> dict[str, str]:
    return {"source": source, "relation": relation, "target": target}


def make_scenario(index: int, archetype: str, difficulty: str) -> dict[str, Any]:
    focal_root, partner_root, operator_root = ROLES[index]
    focal, partner, operator = f"{focal_root} Quill", f"{partner_root} Rook", f"{operator_root} Vale"
    object_type, operation, semantic_type, value_prefix = ARCH[archetype]
    stem = STEMS[index]
    q_entity = f"{object_type.title()} {stem}"
    r_entity = f"{object_type.title()} {VALUE_STEMS[index]}"
    q_true = f"{value_prefix} {stem} North"
    q_false = f"{value_prefix} {stem} South"
    r_true = f"{value_prefix} {VALUE_STEMS[index]} East"
    r_false = f"{value_prefix} {VALUE_STEMS[index]} West"
    depth = 2 if difficulty == "compositional" else 1
    q_intermediate = f"{operation.title()} group {stem}" if depth == 2 else None
    r_intermediate = f"{operation.title()} group {VALUE_STEMS[index]}" if depth == 2 else None
    q_edges = [edge(q_entity, "is assigned to", q_true)] if depth == 1 else [edge(q_entity, "belongs to", q_intermediate), edge(q_intermediate, "is assigned to", q_true)]
    r_edges = [edge(r_entity, "is assigned to", r_true)] if depth == 1 else [edge(r_entity, "belongs to", r_intermediate), edge(r_intermediate, "is assigned to", r_true)]
    ordinal = index
    partner_first = ordinal % 2 == 0
    correction_first = ordinal % 4 in (0, 3)
    visible_invariant = {
        "focal_name": focal, "partner_name": partner, "operator_name": operator,
        "focal_task": f"{focal} is completing the {operation} log.",
        "partner_assignment": f"{partner} immediately executes a {operation} assignment involving {q_entity} after this turn.",
        "operator_assignment": f"At the same time, {operator} independently executes a separate {operation} assignment involving {r_entity}.",
        "isolation_rules": [
            f"A message from {focal} reaches {partner} only and is not relayed to {operator}.",
            f"{partner} cannot alter {operator}'s assignment during this episode.",
            "The two assignments use different objects, targets, routes, and actions.",
            "The episode ends after both immediate assignments execute.",
        ],
        "q_relation": {"entity": q_entity, "intermediate": q_intermediate, "true_value": q_true, "false_value": q_false, "semantic_type": semantic_type, "path_depth": depth, "visible_edges": q_edges},
        "r_relation": {"entity": r_entity, "intermediate": r_intermediate, "true_value": r_true, "false_value": r_false, "semantic_type": semantic_type, "path_depth": depth, "visible_edges": r_edges},
        "static_context": [f"The {operation} roster closes at dusk."] if difficulty == "irrelevant_distractor" else [],
        "partner_proposition_first": partner_first,
        "send_correction_first": correction_first,
    }
    signature_source = {**visible_invariant, "q_relation": {**visible_invariant["q_relation"], "true_value": "VALUE_A", "false_value": "ALTERNATE_A"}, "r_relation": {**visible_invariant["r_relation"], "true_value": "VALUE_B", "false_value": "ALTERNATE_B"}}
    signature = hashlib.sha256(canonical(signature_source)).hexdigest()
    return {"variant_id": f"J1H-{ARCH_CODES[archetype]}-{DIFF_CODES[difficulty]}-01", "namespace": "record_holdout", "archetype": archetype, "difficulty": difficulty, "model_visible_invariant": visible_invariant, "semantic_signature_sha256": signature}


def scenarios() -> list[dict[str, Any]]:
    return [make_scenario(index, archetype, difficulty) for index, (archetype, difficulty) in enumerate((a, d) for a in ARCHETYPES for d in DIFFICULTIES)]
