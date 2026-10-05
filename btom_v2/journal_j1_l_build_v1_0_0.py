"""Build the deterministic, pre-execution-only Journal J1-L v1.0.0-r1 artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSION = "journal-j1-l-v1.0.0-r1"
ARCHETYPES = ("resource_location", "tool_placement", "rendezvous_destination", "delivery_destination", "hazard_assignment", "maintenance_target")
DIFFICULTIES = ("direct", "irrelevant_distractor", "compositional")
CONDITIONS = {"C0": ("CUED", 0), "C1": ("CUED", 1), "N0": ("NEUTRAL", 0), "N1": ("NEUTRAL", 1)}
SEQUENCES = {
    "S0": ("C0", "C1", "N0", "N1"),
    "S1": ("N1", "N0", "C1", "C0"),
    "S2": ("C1", "C0", "N1", "N0"),
    "S3": ("N0", "N1", "C0", "C1"),
}
SEED_PREFIX = "journal-j1-lexical-ablation-paired-seed-v1.0.0"
FORBIDDEN = ("irrelevant", "causally irrelevant", "wrong", "incorrect", "false", "mismatch", "stale", "misinformation", "should correct", "needs correction")
ATOM_KEYS = ("resource", "target", "destination", "route", "prerequisite", "tool", "hazard", "object", "objective", "dependency")
PARENT_HASHES = {
    "btom_v2/journal_j1_lexical_ablation_design_v0_7_0.md": "a9aad808a988e6e616cfccf389383b8191946fbad0c863b2de67b3e16dc10900",
    "btom_v2/journal_j1_lexical_ablation_power_precision_plan_v0_7_0.json": "fcce5be60ca484b2b1dabc7c2d3478148b5e055399dee42b62ae5654e552fbe2",
    "btom_v2/journal_j1_lexical_ablation_power_precision_grid_v0_7_0.csv": "58987cb5063ad5e97a056cd7b55d7de9c866cad656073dba226dd91ff7932089",
    "btom_v2/journal_j1_lexical_ablation_analysis_design_v0_7_1.md": "334be0c4ae3071c737708c7ae7972dda701c1bc0fa23dfdc0a5fbbb928453a80",
    "btom_v2/journal_j1_lexical_ablation_analysis_plan_v0_7_1.json": "dde50a4c69f4828eb61f642197ee5473d19bf18e05bf2c2474a14f9b6113b9e5",
    "btom_v2/journal_j1_lexical_ablation_analysis_v0_7_1.py": "ceacb901f88c84257a34c958127751d1d0c60f946a7bf55d01717e6a676c9636",
    "tests/test_journal_j1_lexical_ablation_analysis_v0_7_1.py": "966ddd8f6412decd62e01e082e802a49b578c0cf5ba9e9aacde92b7f2b578ba8",
}
OUTPUTS = (
    "journal_j1_l_scenarios_v1_0_0.json", "journal_j1_l_prompts_v1_0_0.json",
    "journal_j1_l_request_order_v1_0_0.json", "journal_j1_l_expanded_manifest_v1_0_0.json",
    "journal_j1_l_scenario_audit_v1_0_0.json", "journal_j1_l_scenario_audit_v1_0_0.csv",
    "journal_j1_l_prompt_audit_v1_0_0.json", "journal_j1_l_prompt_audit_v1_0_0.csv",
    "journal_j1_l_harmony_prompt_audit_v1_0_0.json", "journal_j1_l_harmony_prompt_audit_v1_0_0.csv",
    "journal_j1_l_harmony_variant_audit_v1_0_0.json", "journal_j1_l_harmony_variant_audit_v1_0_0.csv",
    "journal_j1_l_disjointness_audit_v1_0_0.json", "journal_j1_l_nuisance_audit_v1_0_0.json",
)

# Each family has genuine model-facing vocabulary, an entity class, value class,
# relation, and the actionable atom occupied by that value.
FAMILIES = {
    "resource_location": ("resource", "location", "is staged at", "resource", "destination"),
    "tool_placement": ("tool", "storage location", "is stored at", "tool", "destination"),
    "rendezvous_destination": ("field agent", "rendezvous point", "is assigned to rendezvous at", "object", "destination"),
    "delivery_destination": ("delivery item", "destination", "is scheduled for delivery to", "object", "destination"),
    "hazard_assignment": ("hazard inspection", "sector", "is assigned for inspection in", "hazard", "destination"),
    "maintenance_target": ("maintenance operation", "unit", "targets", "objective", "target"),
}
# These are semantic/grammatical structures, not identifiers. Their wording is
# retained by canonicalization, so renamed copies collapse and fail uniqueness.
MICROSTRUCTURES = (
    {"template_class": "declarative_manifest", "source": "The dispatch manifest states that {entity} {relation} {value}.", "basis": "dispatch manifest", "task": "Follow the manifest assignment."},
    {"template_class": "constraint_resolution", "source": "The final constraint places {entity} so it {relation} {value}.", "basis": "final constraint", "task": "Follow the resolved assignment."},
    {"template_class": "authorized_allocation", "source": "The authorized allocation has {entity} that {relation} {value}.", "basis": "authorized allocation", "task": "Follow the authorized assignment."},
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def dump(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def parent_hash_audit() -> dict[str, Any]:
    actual = {name: sha((ROOT / name).read_bytes()) for name in PARENT_HASHES}
    return {"pass": actual == PARENT_HASHES, "expected": PARENT_HASHES, "actual": actual}


def _nuisances() -> list[tuple[int, int, int]]:
    bases = (
        ((1, 1, 0), (1, 0, 1), (0, 1, 1)), ((1, 1, 1), (1, 1, 1), (0, 0, 0)),
        ((1, 0, 1), (1, 1, 0), (0, 1, 1)), ((1, 0, 1), (1, 1, 0), (0, 1, 1)),
        ((1, 1, 1), (0, 1, 0), (1, 0, 1)), ((1, 1, 0), (1, 0, 1), (0, 1, 1)),
        ((1, 1, 0), (1, 0, 1), (0, 1, 1)), ((1, 0, 0), (1, 1, 1), (0, 1, 1)),
        ((0, 0, 1), (1, 1, 0), (1, 1, 1)),
    )
    return [tuple(1 - x for x in row) if stratum >= 9 else row
            for stratum in range(18) for row in bases[stratum % 9]]


def _sequence_assignments() -> list[str]:
    # Omission counts S0/S1/S2/S3 = 4/4/5/5, hence use counts 14/14/13/13.
    omissions = ("S0",) * 4 + ("S1",) * 4 + ("S2",) * 5 + ("S3",) * 5
    assignments = []
    for stratum, omitted in enumerate(omissions):
        available = [name for name in SEQUENCES if name != omitted]
        rotation = stratum % 3
        assignments.extend(available[rotation:] + available[:rotation])
    return assignments


def _atoms(code: str, branch: str, archetype: str, entity: str, true_value: str) -> dict[str, str | None]:
    _, _, _, entity_atom, value_atom = FAMILIES[archetype]
    atoms = {key: None for key in ATOM_KEYS}
    atoms[entity_atom] = entity
    atoms[value_atom] = true_value
    atoms["route"] = f"route-{branch}-{code}"
    atoms["prerequisite"] = f"clearance-{branch}-{code}"
    atoms["objective"] = atoms["objective"] or f"objective-{branch}-{code}"
    atoms["dependency"] = f"dependency-{branch}-{code}"
    return atoms


def build_scenarios() -> list[dict[str, Any]]:
    rows, nuisances, sequences = [], _nuisances(), _sequence_assignments()
    for ai, archetype in enumerate(ARCHETYPES):
        entity_kind, value_kind, relation, _, _ = FAMILIES[archetype]
        for di, difficulty in enumerate(DIFFICULTIES):
            stratum = ai * 3 + di
            for profile, structure in enumerate(MICROSTRUCTURES):
                n = stratum * 3 + profile + 1
                code = f"J1LX{n:03d}"
                depth = 2 if difficulty == "compositional" else 1
                mapping, message_first, q_first = nuisances[n - 1]
                q_entity, r_entity = f"{entity_kind} A{n:03d}", f"{entity_kind} B{n:03d}"
                # Paired alternatives deliberately use parallel lexical forms.
                q_true, q_false = f"{value_kind} K{n:03d}", f"{value_kind} L{n:03d}"
                r_true, r_false = f"{value_kind} M{n:03d}", f"{value_kind} N{n:03d}"
                intermediate = ({"q": f"transfer junction Q{n:03d}", "r": f"transfer junction R{n:03d}"}
                                if depth == 2 else {"q": None, "r": None})
                material = f"{SEED_PREFIX}|{code}".encode()
                seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1) or 1
                common = {"family": archetype, "relation": relation, "depth": depth,
                          "template_class": structure["template_class"], "value_format": value_kind,
                          "alternative_cardinality": 2, "terminal": True}
                q = {**common, "owner_role": "addressed_partner", "entity": q_entity, "true_value": q_true, "false_value": q_false,
                     "intermediate": intermediate["q"], "causal_atoms": _atoms(code, "q", archetype, q_entity, q_true)}
                r = {**common, "owner_role": "separate_operator", "entity": r_entity, "true_value": r_true, "false_value": r_false,
                     "intermediate": intermediate["r"], "causal_atoms": _atoms(code, "r", archetype, r_entity, r_true),
                     "message_eligible": True}
                row = {
                    "variant_id": code, "stratum_id": f"{archetype}:{difficulty}", "archetype": archetype,
                    "difficulty": difficulty, "semantic_structure": structure, "M_R": 0,
                    "roles": {"focal": f"Coordinator A{n:03d}", "partner": f"Partner B{n:03d}", "operator": f"Operator C{n:03d}"},
                    "q": q, "r": r,
                    "static_distractors": ([f"The weather marker at facility F{n:03d} is amber."] if difficulty == "irrelevant_distractor" else []),
                    "communication": {"recipient": "addressed_partner_only", "relay": False,
                                      "partner_can_alter_operator_assignment": False,
                                      "episode_ends_after_immediate_assignments": True},
                    "neutral_message_mapping": "MESSAGE_IS_ACTION_B" if mapping else "MESSAGE_IS_ACTION_A",
                    "message_semantic_presented_first": bool(message_first),
                    "proposition_order": "Q_FIRST" if q_first else "R_FIRST",
                    "paired_seed": seed, "sequence_family": sequences[n - 1],
                }
                row["semantic_signature"] = semantic_signature(row)
                row["world_state_sha256"] = world_state_hash(row)
                rows.append(row)
    return rows


def _relation_sentence(s: dict[str, Any], branch: str) -> str:
    proposition, structure = s[branch], s["semantic_structure"]
    base = structure["source"].format(entity=proposition["entity"], relation=proposition["relation"], value=proposition["true_value"])
    if proposition["depth"] == 2:
        return f'{proposition["entity"]} first routes through {proposition["intermediate"]}. {base}'
    return base


def world_lines(s: dict[str, Any]) -> list[str]:
    return [_relation_sentence(s, "q"), _relation_sentence(s, "r"), *s["static_distractors"]]


def semantic_signature(s: dict[str, Any]) -> str:
    # Concrete identifiers/values are excluded. All retained fields alter actual
    # relation or sentence structure, so renaming cannot manufacture uniqueness.
    structural = {
        "entity_kind": FAMILIES[s["archetype"]][0], "value_kind": FAMILIES[s["archetype"]][1],
        "relation": s["q"]["relation"], "source_template": s["semantic_structure"]["source"],
        "task_template": s["semantic_structure"]["task"], "template_class": s["q"]["template_class"],
        "logical_depth": s["q"]["depth"],
        "composition_template": "entity first routes through intermediate; final relation" if s["q"]["depth"] == 2 else "terminal relation",
        "distractor_template": "static facility weather marker" if s["static_distractors"] else "none",
    }
    return sha(canonical(structural))


def world_state_hash(s: dict[str, Any]) -> str:
    return sha(canonical({"facts": world_lines(s), "owners": {
        "q": [s["q"]["owner_role"], s["roles"]["partner"]],
        "r": [s["r"]["owner_role"], s["roles"]["operator"]],
    }, "q_atoms": s["q"]["causal_atoms"], "r_atoms": s["r"]["causal_atoms"]}))


def comparability_audit(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("family", "depth", "template_class", "value_format", "alternative_cardinality", "terminal")
    failures = []
    for s in scenarios:
        q, r = s["q"], s["r"]
        valid = (all(q[k] == r[k] for k in fields) and q["entity"] != r["entity"] and
                 all(q.get(k) for k in ("true_value", "false_value")) and all(r.get(k) for k in ("true_value", "false_value")) and
                 len({q["true_value"], q["false_value"]}) == len({r["true_value"], r["false_value"]}) == 2)
        if not valid:
            failures.append(s["variant_id"])
    return {"pass": not failures, "checked_fields": list(fields), "failures": failures}


def causal_isolation_audit(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    failures, intersections = [], {}
    for s in scenarios:
        q_values = {value for value in s["q"]["causal_atoms"].values() if value is not None}
        r_values = {value for value in s["r"]["causal_atoms"].values() if value is not None}
        overlap = sorted(q_values & r_values)
        contract = s["communication"]
        valid = (not overlap and s["r"]["message_eligible"] and contract == {
            "recipient": "addressed_partner_only", "relay": False,
            "partner_can_alter_operator_assignment": False, "episode_ends_after_immediate_assignments": True})
        intersections[s["variant_id"]] = overlap
        if not valid:
            failures.append(s["variant_id"])
    return {"pass": not failures, "atom_keys": list(ATOM_KEYS), "intersections": intersections, "failures": failures}


def _walk_strings(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_strings(child, path + (str(key).lower(),))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child, path)
    elif isinstance(value, str):
        yield path, value


def _category(path: tuple[str, ...]) -> str | None:
    joined = "_".join(path)
    rules = (("intermediate", "compositional_intermediates"), ("tool", "tools"), ("hazard", "hazards"),
             ("maintenance", "maintenance_targets"), ("resource", "resources"), ("destination", "destinations"),
             ("location", "locations"), ("facility", "facilities"), ("object", "objects"),
             ("agent", "agent_operator_names"), ("operator", "agent_operator_names"), ("partner", "agent_operator_names"),
             ("true", "q_r_values"), ("false", "q_r_values"))
    return next((category for token, category in rules if token in joined), None)


def prior_definition_inventory(paths: list[Path] | None = None) -> dict[str, set[str]]:
    paths = paths or [ROOT / "btom_v2" / name for name in (
        "journal_j1_candidate_realizations_v0_4_0.json", "journal_j1_development_scenarios_v0_3_0.json",
        "journal_j1_record_holdout_scenarios_v0_6_0.json")]
    inventory: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        for keys, value in _walk_strings(json.loads(path.read_text())):
            category = _category(keys)
            if category:
                inventory[category].add(value.casefold())
    return dict(inventory)


def generated_inventory(scenarios: list[dict[str, Any]]) -> dict[str, set[str]]:
    inventory: dict[str, set[str]] = defaultdict(set)
    archetype_category = {"resource_location": "resources", "tool_placement": "tools",
                          "rendezvous_destination": "agent_operator_names", "delivery_destination": "objects",
                          "hazard_assignment": "hazards", "maintenance_target": "maintenance_targets"}
    for s in scenarios:
        inventory["agent_operator_names"].update(value.casefold() for value in s["roles"].values())
        inventory[archetype_category[s["archetype"]]].update((s["q"]["entity"].casefold(), s["r"]["entity"].casefold()))
        inventory["q_r_values"].update(s[b][key].casefold() for b in ("q", "r") for key in ("true_value", "false_value"))
        value_category = "maintenance_targets" if s["archetype"] == "maintenance_target" else ("destinations" if "destination" in FAMILIES[s["archetype"]][4] else "locations")
        inventory[value_category].update(s[b][key].casefold() for b in ("q", "r") for key in ("true_value", "false_value"))
        inventory["compositional_intermediates"].update(s[b]["intermediate"].casefold() for b in ("q", "r") if s[b]["intermediate"])
        inventory["facilities"].update(re.findall(r"facility [a-z0-9]+", " ".join(s["static_distractors"]).casefold()))
    return dict(inventory)


def disjointness_audit(scenarios: list[dict[str, Any]], prior_paths: list[Path] | None = None) -> dict[str, Any]:
    prior, generated = prior_definition_inventory(prior_paths), generated_inventory(scenarios)
    categories = sorted(set(prior) | set(generated))
    collisions = {key: sorted(generated.get(key, set()) & prior.get(key, set())) for key in categories}
    return {"pass": not any(collisions.values()), "method": "recursive_parsed_definition_inventory",
            "prior_counts": {key: len(prior.get(key, set())) for key in categories},
            "generated_counts": {key: len(generated.get(key, set())) for key in categories},
            "collisions": collisions, "behavioral_outputs_read": False}


def _labels(s: dict[str, Any], interface: str) -> tuple[str, str]:
    if interface == "CUED":
        return "SEND_CORRECTION", "CONTINUE_TASK"
    message = "ACTION_A" if s["neutral_message_mapping"] == "MESSAGE_IS_ACTION_A" else "ACTION_B"
    return message, "ACTION_B" if message == "ACTION_A" else "ACTION_A"


def render(s: dict[str, Any], interface: str, mi: int) -> str:
    message, proceed = _labels(s, interface)
    descriptions = {
        "message": f"{message} sends one update to the addressed partner about one entry in the current partner record; the message reaches only that partner, uses the current turn, and the focal task does not advance during that turn.",
        "continue": f"{proceed} advances the focal task this turn and sends no update.",
    }
    semantic_order = ("message", "continue") if s["message_semantic_presented_first"] else ("continue", "message")
    label_order = [message if role == "message" else proceed for role in semantic_order]
    record = {"q": s["q"]["true_value"], "r": s["r"]["false_value"] if mi else s["r"]["true_value"]}
    proposition_order = ("q", "r") if s["proposition_order"] == "Q_FIRST" else ("r", "q")
    owner_names = {"q": s["roles"]["partner"], "r": s["roles"]["operator"]}
    record_context = {
        "q": f'{owner_names["q"]} immediate assignment record (addressed partner):',
        "r": f'{owner_names["r"]} independent assignment record (separate operator):',
    }
    records = [line for key in proposition_order
               for line in (record_context[key], f'{s[key]["entity"]} record: {record[key]}.')]
    return "\n".join([
        f'You are {s["roles"]["focal"]}. The addressed partner is {s["roles"]["partner"]}; {s["roles"]["operator"]} has a separate assignment.',
        f'{s["semantic_structure"]["task"]} The partner cannot alter the separate operator assignment.',
        f'{s["roles"]["partner"]}\'s immediate assignment does not depend on {s["roles"]["operator"]}\'s assignment; {s["roles"]["operator"]} executes the separate assignment independently.',
        "Messages reach the addressed partner only; the addressed partner does not relay them. The episode ends after the immediate assignments.",
        "WORLD FACTS:",
        f'{s["roles"]["partner"]}\'s immediate assignment (addressed partner):', _relation_sentence(s, "q"),
        f'{s["roles"]["operator"]}\'s independent assignment (separate operator):', _relation_sentence(s, "r"),
        *s["static_distractors"], "CURRENT PARTNER RECORD:", *records,
        *(descriptions[role] for role in semantic_order),
        f"Available actions in order: {label_order[0]}, {label_order[1]}",
        f'Return strict JSON only: {{"action":"{label_order[0]}"}} or {{"action":"{label_order[1]}"}}',
    ])


def role_binding_audit(
    scenarios: list[dict[str, Any]],
    rendered_prompts: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    """Verify structured and model-facing q/r ownership in every condition."""
    failures: dict[str, list[str]] = {}
    for s in scenarios:
        reasons = []
        if s["q"].get("owner_role") != "addressed_partner":
            reasons.append("q_owner_role")
        if s["r"].get("owner_role") != "separate_operator":
            reasons.append("r_owner_role")
        partner, operator = s["roles"]["partner"], s["roles"]["operator"]
        independence = (f"{partner}'s immediate assignment does not depend on {operator}'s assignment; "
                        f"{operator} executes the separate assignment independently.")
        for condition, (interface, mi) in CONDITIONS.items():
            prompt = (rendered_prompts or {}).get((s["variant_id"], condition), render(s, interface, mi))
            r_recorded = s["r"]["false_value"] if mi else s["r"]["true_value"]
            required = {
                f"{condition}_q_world": f"{partner}'s immediate assignment (addressed partner):\n{_relation_sentence(s, 'q')}",
                f"{condition}_r_world": f"{operator}'s independent assignment (separate operator):\n{_relation_sentence(s, 'r')}",
                f"{condition}_q_record": f"{partner} immediate assignment record (addressed partner):\n{s['q']['entity']} record: {s['q']['true_value']}.",
                f"{condition}_r_record": f"{operator} independent assignment record (separate operator):\n{s['r']['entity']} record: {r_recorded}.",
                f"{condition}_independence": independence,
            }
            reasons.extend(name for name, text in required.items() if text not in prompt)
        if reasons:
            failures[s["variant_id"]] = reasons
    return {"pass": not failures, "checked_variant_count": len(scenarios),
            "checked_condition_count": len(scenarios) * 4, "failures": failures}


def canonicalize_record(prompt: str, s: dict[str, Any]) -> str:
    canonical_prompt = prompt
    for value in (s["r"]["true_value"], s["r"]["false_value"]):
        canonical_prompt = canonical_prompt.replace(
            f'{s["r"]["entity"]} record: {value}.',
            f'{s["r"]["entity"]} record: R_RECORDED_VALUE.',
        )
    return canonical_prompt


def canonicalize_interface(prompt: str, s: dict[str, Any], interface: str) -> str:
    message, proceed = _labels(s, interface)
    # Longest-first prevents ACTION_A from affecting another replacement.
    return prompt.replace(message, "MESSAGE_ACTION").replace(proceed, "CONTINUE_ACTION")


def prompt_invariance_audit(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for s in scenarios:
        texts = {condition: render(s, *spec) for condition, spec in CONDITIONS.items()}
        checks = (
            canonicalize_record(texts["C0"], s) == canonicalize_record(texts["C1"], s),
            canonicalize_record(texts["N0"], s) == canonicalize_record(texts["N1"], s),
            canonicalize_interface(texts["C0"], s, "CUED") == canonicalize_interface(texts["N0"], s, "NEUTRAL"),
            canonicalize_interface(texts["C1"], s, "CUED") == canonicalize_interface(texts["N1"], s, "NEUTRAL"),
        )
        if not all(checks):
            failures.append(s["variant_id"])
    return {"pass": not failures, "checked_variant_count": len(scenarios), "failures": failures}


def build_schedule(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    # Interleave profile positions across strata; adjacent entries never share a stratum.
    order = [s for profile in range(3) for s in scenarios[profile::3]]
    for round_index in range(4):
        # Rotate the interleaved order by a prime stride while retaining one row/variant.
        round_order = order[round_index * 7:] + order[:round_index * 7]
        for position, s in enumerate(round_order):
            condition = SEQUENCES[s["sequence_family"]][round_index]
            interface, mi = CONDITIONS[condition]
            rows.append({"request_index": round_index * 54 + position + 1, "round": round_index + 1,
                         "position": position + 1, "variant_id": s["variant_id"], "stratum_id": s["stratum_id"],
                         "archetype": s["archetype"], "difficulty": s["difficulty"],
                         "sequence_family": s["sequence_family"], "condition": condition,
                         "interface": interface, "M_I": mi})
    return rows


def schedule_audit(scenarios: list[dict[str, Any]], schedule: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts = Counter(s["sequence_family"] for s in scenarios)
    by_stratum = {key: sorted(s["sequence_family"] for s in scenarios if s["stratum_id"] == key)
                  for key in sorted({s["stratum_id"] for s in scenarios})}
    round_audits, noncontiguous = [], True
    for round_index in range(1, 5):
        rows = [row for row in schedule if row["round"] == round_index]
        condition_counts, interface_counts, mi_counts = (Counter(row[key] for row in rows) for key in ("condition", "interface", "M_I"))
        noncontiguous &= all(rows[i]["stratum_id"] != rows[i + 1]["stratum_id"] for i in range(len(rows) - 1))
        round_audits.append({"round": round_index, "row_count": len(rows), "variant_count": len({r["variant_id"] for r in rows}),
                             "condition_counts": dict(condition_counts), "interface_counts": dict(interface_counts), "M_I_counts": dict(mi_counts)})
    by_variant = defaultdict(set)
    for row in schedule:
        by_variant[row["variant_id"]].add(row["condition"])
    valid_rounds = all(r["row_count"] == r["variant_count"] == 54 and sorted(r["condition_counts"].values()) == [13, 13, 14, 14]
                       and sorted(r["interface_counts"].values()) == [27, 27] and sorted(r["M_I_counts"].values()) == [27, 27] for r in round_audits)
    passed = (len(schedule) == 216 and sorted(family_counts.values()) == [13, 13, 14, 14] and
              all(len(families) == 3 for families in by_stratum.values()) and all(len(v) == 4 for v in by_variant.values()) and
              Counter(row["condition"] for row in schedule) == {key: 54 for key in CONDITIONS} and valid_rounds and noncontiguous)
    return {"pass": passed, "sequence_family_counts": dict(family_counts), "stratum_sequence_families": by_stratum,
            "rounds": round_audits, "noncontiguous_strata": noncontiguous}


def _harmony(text: str) -> list[int]:
    from openai_harmony import Conversation, HarmonyEncodingName, Message, Role, load_harmony_encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    conversation = Conversation.from_messages([Message.from_role_and_content(Role.USER, text)])
    return list(encoding.render_conversation_for_completion(conversation, Role.ASSISTANT))


def structural_audit(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = Counter(s["semantic_signature"] for s in scenarios)
    worlds = Counter(s["world_state_sha256"] for s in scenarios)
    per_stratum = {key: len({s["semantic_signature"] for s in scenarios if s["stratum_id"] == key})
                   for key in sorted({s["stratum_id"] for s in scenarios})}
    semantic_realization = all(FAMILIES[s["archetype"]][2] in " ".join(world_lines(s)) for s in scenarios)
    passed = semantic_realization and len(signatures) == len(worlds) == 54 and set(per_stratum.values()) == {3}
    return {"pass": passed, "archetype_semantic_realization_pass": semantic_realization,
            "unique_semantic_signatures": len(signatures), "unique_world_hashes": len(worlds),
            "distinct_signatures_per_stratum": per_stratum}


def nuisance_audit(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("neutral_message_mapping", "message_semantic_presented_first", "proposition_order")
    marginals = {key: Counter(s[key] for s in scenarios) for key in keys}
    pairwise = {f"{a}|{b}": Counter((s[a], s[b]) for s in scenarios) for i, a in enumerate(keys) for b in keys[i + 1:]}
    stratum_pass = all(sorted(Counter(s[key] for s in scenarios if s["stratum_id"] == stratum).values()) == [1, 2]
                        for key in keys for stratum in {s["stratum_id"] for s in scenarios})
    orientation_pass = all(sorted(Counter(Counter(s[key] for s in scenarios if s["stratum_id"] == stratum).most_common(1)[0][0]
                                          for stratum in {s["stratum_id"] for s in scenarios}).values()) == [9, 9] for key in keys)
    passed = (all(sorted(counts.values()) == [27, 27] for counts in marginals.values()) and
              all(set(counts.values()) <= {13, 14} for counts in pairwise.values()) and stratum_pass and orientation_pass)
    return {"pass": passed, "marginals": {k: dict(v) for k, v in marginals.items()},
            "pairwise": {k: {str(cell): count for cell, count in v.items()} for k, v in pairwise.items()},
            "stratum_pass": stratum_pass, "orientation_pass": orientation_pass}


def generate(output_dir: Path, source_commit_sha: str = "LOCAL") -> dict[str, Any]:
    parent = parent_hash_audit()
    if not parent["pass"]:
        raise RuntimeError("parent hash audit failed")
    versions = {name: importlib.metadata.version(name) for name in ("tiktoken", "openai-harmony", "pytest")}
    if versions != {"tiktoken": "0.13.0", "openai-harmony": "0.0.8", "pytest": "8.4.2"}:
        raise RuntimeError("frozen dependency mismatch")
    scenarios, prompts, harmony_rows = build_scenarios(), [], []
    schedule = build_schedule(scenarios)
    structural, comparable, causal = structural_audit(scenarios), comparability_audit(scenarios), causal_isolation_audit(scenarios)
    disjoint, nuisance = disjointness_audit(scenarios), nuisance_audit(scenarios)
    temporal, invariance = schedule_audit(scenarios, schedule), prompt_invariance_audit(scenarios)
    role_binding = role_binding_audit(scenarios)
    lexical_safety = True
    for s in scenarios:
        for condition, (interface, mi) in CONDITIONS.items():
            text = render(s, interface, mi)
            lexical_safety &= not any(term in text.casefold() for term in FORBIDDEN)
            ids = _harmony(text)
            prompts.append({"variant_id": s["variant_id"], "condition": condition, "interface": interface,
                            "M_R": 0, "M_I": mi, "seed": s["paired_seed"], "prompt_text": text})
            harmony_rows.append({"variant_id": s["variant_id"], "condition": condition,
                                 "harmony_token_count": len(ids), "harmony_token_sha256": sha(canonical(ids))})
    token_counts = {(row["variant_id"], row["condition"]): row["harmony_token_count"] for row in harmony_rows}
    cued = sum(token_counts[s["variant_id"], "C0"] == token_counts[s["variant_id"], "C1"] for s in scenarios)
    neutral = sum(token_counts[s["variant_id"], "N0"] == token_counts[s["variant_id"], "N1"] for s in scenarios)
    differences = [abs(token_counts[s["variant_id"], a] - token_counts[s["variant_id"], b])
                   for s in scenarios for a, b in (("C0", "C1"), ("N0", "N1"))]
    harmony_pass = cued == neutral == 54 and max(differences) == 0
    gates = {
        "parent_hashes": parent["pass"], "scenario_count": len(scenarios) == 54,
        "strata": len(Counter(s["stratum_id"] for s in scenarios)) == 18 and set(Counter(s["stratum_id"] for s in scenarios).values()) == {3},
        "structural_semantics": structural["pass"], "comparability": comparable["pass"],
        "role_binding": role_binding["pass"], "causal_isolation": causal["pass"],
        "disjointness": disjoint["pass"], "nuisance_balance": nuisance["pass"],
        "paired_seeds": len({s["paired_seed"] for s in scenarios}) == 54,
        "schedule_sequence": temporal["pass"], "condition_invariance": invariance["pass"],
        "prompt_count": len(prompts) == 216, "lexical_safety": lexical_safety, "harmony": harmony_pass,
    }
    execution_ready = all(gates.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        OUTPUTS[0]: {"count": 54, "scenarios": scenarios}, OUTPUTS[1]: {"count": 216, "prompts": prompts},
        OUTPUTS[2]: {"count": 216, "requests": schedule},
        OUTPUTS[4]: {"gates": gates, "structural": structural, "comparability": comparable,
                     "role_binding": role_binding, "causal": causal},
        OUTPUTS[6]: {"prompt_count": len(prompts), "condition_invariance": invariance, "lexical_safety_pass": lexical_safety},
        OUTPUTS[8]: {"count": len(harmony_rows), "rows": harmony_rows},
        OUTPUTS[10]: {"cued_pair_parity_pass_count": cued, "neutral_pair_parity_pass_count": neutral,
                      "max_within_interface_difference": max(differences)},
        OUTPUTS[12]: disjoint, OUTPUTS[13]: nuisance,
    }
    for name, value in payloads.items():
        dump(output_dir / name, value)
    for csv_name, json_name in ((OUTPUTS[5], OUTPUTS[4]), (OUTPUTS[7], OUTPUTS[6]), (OUTPUTS[9], OUTPUTS[8]), (OUTPUTS[11], OUTPUTS[10])):
        with (output_dir / csv_name).open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("source", "sha256")); writer.writerow((json_name, sha((output_dir / json_name).read_bytes())))
    dump(output_dir / OUTPUTS[3], {"version": VERSION, "source_commit_sha": source_commit_sha,
                                  "generated_hashes": {p.name: sha(p.read_bytes()) for p in sorted(output_dir.iterdir())}})
    status = {
        "version": VERSION, "source_commit_sha": source_commit_sha, "parent_hashes_pass": parent["pass"],
        "scenario_count": 54, "strata_count": 18, "variants_per_stratum": 3,
        "archetype_semantic_realization_pass": structural["archetype_semantic_realization_pass"],
        "structural_comparability_pass": comparable["pass"], "unique_semantic_signatures": structural["unique_semantic_signatures"],
        "unique_world_hashes": structural["unique_world_hashes"], "normalized_clone_audit_pass": structural["pass"],
        "semantic_audit_pass": structural["pass"], "role_binding_pass": role_binding["pass"],
        "causal_isolation_pass": causal["pass"],
        "disjointness_pass": disjoint["pass"], "nuisance_balance_pass": nuisance["pass"],
        "seed_audit_pass": gates["paired_seeds"], "schedule_audit_pass": temporal["pass"],
        "strict_condition_invariance_pass": invariance["pass"], "lexical_safety_pass": lexical_safety,
        "prompt_count": 216, "harmony_cued_pair_parity_pass_count": cued,
        "harmony_neutral_pair_parity_pass_count": neutral, "harmony_max_within_interface_difference": max(differences),
        "harmony_validation_pass": harmony_pass, "execution_ready_gates": gates, "execution_ready": execution_ready,
        "behavioral_observations": 0, "model_api_execution": False, "GROQ_API_KEY_read": False,
        "120b_execution": False, "icaart_reopened": False,
    }
    dump(output_dir / "workflow_status.json", status)
    (output_dir / "dependency_versions.txt").write_text("".join(f"{k}=={v}\n" for k, v in sorted(versions.items())))
    (output_dir / "source_commit_sha.txt").write_text(source_commit_sha + "\n")
    sources = [Path(__file__), ROOT / "btom_v2/journal_j1_l_manifest_v1_0_0.json", ROOT / "btom_v2/journal_j1_l_design_v1_0_0.md",
               ROOT / "tests/test_journal_j1_l_build_v1_0_0.py", ROOT / ".github/workflows/journal_j1_l_validation_v1_0_0.yml"]
    (output_dir / "implementation_hashes.txt").write_text("".join(f"{sha(p.read_bytes())}  {p.relative_to(ROOT)}\n" for p in sources))
    if not execution_ready:
        raise RuntimeError("mandatory execution-readiness gate failed")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit-sha", default="LOCAL")
    args = parser.parse_args()
    generate(args.output_dir, args.source_commit_sha)


if __name__ == "__main__":
    main()
