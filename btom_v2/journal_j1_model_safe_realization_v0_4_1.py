"""Selection-blind, model-safe Journal J1 semantic realizations.

This module performs no tokenization, networking, or model execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FRAME_PATH = ROOT / "btom_v2/journal_j1_confirmatory_candidate_frame_v0_3_0.json"
DEVELOPMENT_PATH = ROOT / "btom_v2/journal_j1_development_scenarios_v0_3_0.json"

ARCHETYPE = {
    "resource_location": ("canister", "retrieval", "location"),
    "tool_placement": ("instrument", "assembly", "location"),
    "rendezvous_destination": ("team", "pickup", "destination"),
    "delivery_destination": ("payload", "delivery", "destination"),
    "hazard_assignment": ("survey", "inspection", "sector"),
    "maintenance_target": ("service ticket", "maintenance", "component"),
}
NAMES = ("Avery", "Blair", "Casey", "Devon", "Ellis", "Finley", "Gray", "Harper", "Indigo", "Jules", "Kai", "Lane", "Morgan", "Noel", "Oakley", "Parker", "Reese", "Sage")
ENTITY_WORDS = ("Cedar", "Maple", "Juniper", "Kestrel", "Lark", "Heron", "Willow", "Falcon", "Laurel", "Birch", "Saffron", "Mosaic", "Dawn", "Ember", "Fjord", "Grove", "Horizon", "Ivory", "Jasper", "Nimbus")
VALUE_WORDS = ("Solace", "Orion", "Meridian", "Harbor", "Pioneer", "Summit", "Cobalt", "Meadow", "Atlas", "Nova", "Lagoon", "Comet", "Briar", "Violet", "Terrace", "Beacon", "Cypress", "Mariner", "Quartz", "Topaz")
PREFIXES = {"location": ("Bay", "Depot", "Station", "Storage"), "destination": ("Hub", "Gate", "Dock", "Terminal"), "sector": ("Sector", "Zone", "Grid", "Area"), "component": ("Valve", "Relay", "Pump", "Sensor")}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _index(variant_id: str, label: str, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{variant_id}:{label}".encode()).digest()[:8], "big") % modulus


def _value(variant_id: str, semantic_type: str, slot: int) -> str:
    prefix = PREFIXES[semantic_type][(_index(variant_id, "prefix", 4) + slot) % 4]
    word = VALUE_WORDS[(_index(variant_id, "value", len(VALUE_WORDS)) + slot * 7) % len(VALUE_WORDS)]
    suffix = chr(65 + ((_index(variant_id, "suffix", 20) + slot * 3) % 20))
    return f"{prefix} {word} {suffix}"


def realize(record: dict[str, Any]) -> dict[str, Any]:
    """Realize one structural record without selection information."""
    variant_id = record["variant_id"]
    kind, context, semantic_type = ARCHETYPE[record["archetype"]]
    entity_a = f"{kind.title()} {ENTITY_WORDS[_index(variant_id, 'entity-a', len(ENTITY_WORDS))]}"
    entity_b = f"{kind.title()} {ENTITY_WORDS[_index(variant_id, 'entity-b', len(ENTITY_WORDS))]}"
    if entity_a == entity_b:
        entity_b += " North"
    true_a, false_a, true_b, false_b = (_value(variant_id, semantic_type, i) for i in range(4))
    depth = int(record["q_relation_depth"])
    intermediate_a = f"{context.title()} group {ENTITY_WORDS[_index(variant_id, 'group-a', len(ENTITY_WORDS))]}" if depth == 2 else None
    intermediate_b = f"{context.title()} group {ENTITY_WORDS[_index(variant_id, 'group-b', len(ENTITY_WORDS))]}" if depth == 2 else None
    if intermediate_a == intermediate_b and depth == 2:
        intermediate_b += " East"

    def relation(entity: str, intermediate: str | None, final: str) -> dict[str, Any]:
        if intermediate is None:
            edges = [{"source": entity, "relation": "is assigned to", "target": final}]
        else:
            edges = [
                {"source": entity, "relation": "belongs to", "target": intermediate},
                {"source": intermediate, "relation": "is assigned to", "target": final},
            ]
        return {"entity": entity, "intermediate": intermediate, "final_value": final, "visible_edges": edges, "path_depth": len(edges)}

    focal = NAMES[_index(variant_id, "focal", len(NAMES))]
    partner = NAMES[(_index(variant_id, "focal", len(NAMES)) + 7) % len(NAMES)]
    causal_first = _index(variant_id, "proposition-order", 2) == 0
    action_correction_first = _index(variant_id, "action-order", 2) == 0
    # Development has exactly 18 records in stable source order. Explicit parity
    # below guarantees the frozen 9/9 balances without consulting outcomes.
    if record["namespace"] == "development":
        ordinal = json.loads(DEVELOPMENT_PATH.read_text())["scenarios"].index(record)
        causal_first = ordinal % 2 == 0
        action_correction_first = ordinal % 4 in (0, 3)
    static = []
    if record["difficulty"] == "irrelevant_distractor":
        static = [f"The {context} roster closes at midday."]
    visible = {
        "focal_agent": focal,
        "partner": partner,
        "immediate_assignment": f"{partner}'s immediate {context} assignment involves {entity_a}.",
        "causal_relation": relation(entity_a, intermediate_a, true_a),
        "noncausal_relation": relation(entity_b, intermediate_b, true_b),
        "represented_values": {"causal_true": true_a, "causal_false": false_a, "noncausal_true": true_b, "noncausal_false": false_b},
        "static_context": static,
        "proposition_order": "causal_first" if causal_first else "noncausal_first",
        "action_order": ["SEND_CORRECTION", "CONTINUE_TASK"] if action_correction_first else ["CONTINUE_TASK", "SEND_CORRECTION"],
    }
    result = {"variant_id": variant_id, "namespace": record["namespace"], "archetype": record["archetype"], "difficulty": record["difficulty"], "model_visible": visible, "structural_semantic_signature_sha256": record["semantic_signature_sha256"]}
    result["realization_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def load_and_realize() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Realize every candidate before any selection artifact is read."""
    frame = json.loads(FRAME_PATH.read_text())
    development = json.loads(DEVELOPMENT_PATH.read_text())
    candidates = [realize(record) for record in frame["candidates"]]
    developments = [realize(record) for record in development["scenarios"]]
    return candidates, developments, development
