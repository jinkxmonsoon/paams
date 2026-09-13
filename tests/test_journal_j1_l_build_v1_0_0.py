import copy
import json
import re
from collections import Counter, defaultdict

import pytest

from btom_v2 import journal_j1_l_build_v1_0_0 as b


def scenarios():
    return b.build_scenarios()


def test_parent_hashes_and_frozen_design():
    assert b.parent_hash_audit()["pass"]
    assert len(scenarios()) == 54
    assert b.CONDITIONS == {"C0": ("CUED", 0), "C1": ("CUED", 1), "N0": ("NEUTRAL", 0), "N1": ("NEUTRAL", 1)}


def test_archetypes_are_model_facing_semantic_families():
    rows = scenarios()
    rendered = {a: b.render(next(s for s in rows if s["archetype"] == a), "CUED", 0) for a in b.ARCHETYPES}
    for archetype, (_, _, relation, _, _) in b.FAMILIES.items():
        assert relation in rendered[archetype]
    assert len({re.sub(r"[A-Z]\d{3}", "CODE", text) for text in rendered.values()}) == 6


def test_structural_signatures_reject_renaming_clones_and_all_real_variants_are_unique():
    rows = scenarios()
    audit = b.structural_audit(rows)
    assert audit["pass"] and audit["unique_semantic_signatures"] == 54 and audit["unique_world_hashes"] == 54
    assert set(audit["distinct_signatures_per_stratum"].values()) == {3}
    clone = copy.deepcopy(rows[0])
    clone["variant_id"] = "RENAMED"
    clone["roles"] = {k: f"renamed-{k}" for k in clone["roles"]}
    for branch in ("q", "r"):
        clone[branch]["entity"] = f"renamed-{branch}"
        clone[branch]["true_value"] = f"renamed-{branch}-true"
        clone[branch]["false_value"] = f"renamed-{branch}-alternative"
    assert b.semantic_signature(clone) == b.semantic_signature(rows[0])
    assert len({b.semantic_signature(rows[0]), b.semantic_signature(clone)}) == 1


def test_q_r_comparability_has_materialized_alternatives():
    rows = scenarios()
    assert b.comparability_audit(rows)["pass"]
    for s in rows:
        q, r = s["q"], s["r"]
        assert q["true_value"] != q["false_value"] and r["true_value"] != r["false_value"]
        assert all(q[key] == r[key] for key in ("family", "depth", "template_class", "value_format", "alternative_cardinality", "terminal"))
        assert q["alternative_cardinality"] == r["alternative_cardinality"] == 2


def test_causal_isolation_is_derived_from_explicit_atoms():
    rows = scenarios()
    audit = b.causal_isolation_audit(rows)
    assert audit["pass"] and set(audit["atom_keys"]) == set(b.ATOM_KEYS)
    broken = copy.deepcopy(rows[0])
    broken["r"]["causal_atoms"]["route"] = broken["q"]["causal_atoms"]["route"]
    failed = b.causal_isolation_audit([broken])
    assert not failed["pass"] and failed["intersections"][broken["variant_id"]]


def _prompt_sections(text):
    lines = text.splitlines()
    description_indices = [i for i, line in enumerate(lines) if "uses the current turn" in line or "advances the focal task" in line]
    available = next(line for line in lines if line.startswith("Available actions"))
    choices = next(line for line in lines if line.startswith("Return strict JSON"))
    return lines, description_indices, available, choices


@pytest.mark.parametrize("interface", ["CUED", "NEUTRAL"])
def test_message_semantics_and_actual_action_order(interface):
    for s in scenarios():
        text = b.render(s, interface, 0)
        assert "sends one update" in text and "about one entry" in text
        assert "sends the addressed partner the two record entries" not in text
        message, proceed = b._labels(s, interface)
        lines, indices, available, choices = _prompt_sections(text)
        expected = (message, proceed) if s["message_semantic_presented_first"] else (proceed, message)
        assert expected[0] in lines[indices[0]] and expected[1] in lines[indices[1]]
        assert available.endswith(f"{expected[0]}, {expected[1]}")
        assert choices.index(expected[0]) < choices.index(expected[1])


def test_semantic_order_matches_across_interfaces():
    for s in scenarios():
        for mi in (0, 1):
            cued = b.canonicalize_interface(b.render(s, "CUED", mi), s, "CUED")
            neutral = b.canonicalize_interface(b.render(s, "NEUTRAL", mi), s, "NEUTRAL")
            assert cued == neutral


def test_strict_condition_invariance():
    rows = scenarios()
    assert b.prompt_invariance_audit(rows)["pass"]
    for s in rows:
        texts = {condition: b.render(s, *spec) for condition, spec in b.CONDITIONS.items()}
        assert b.canonicalize_record(texts["C0"], s) == b.canonicalize_record(texts["C1"], s)
        assert b.canonicalize_record(texts["N0"], s) == b.canonicalize_record(texts["N1"], s)
        assert b.canonicalize_interface(texts["C0"], s, "CUED") == b.canonicalize_interface(texts["N0"], s, "NEUTRAL")
        assert not any(term in text.casefold() for text in texts.values() for term in b.FORBIDDEN)


def test_disjointness_is_recursive_categorized_inventory(tmp_path):
    prior = {"scenarios": [{"agent": {"name": "Partner Archive"}, "resource": "resource archive", "true_value": "location archive"}]}
    path = tmp_path / "definition.json"
    path.write_text(json.dumps(prior))
    inventory = b.prior_definition_inventory([path])
    assert "partner archive" in inventory["agent_operator_names"]
    assert "resource archive" in inventory["resources"]
    rows = scenarios()
    assert b.disjointness_audit(rows)["method"] == "recursive_parsed_definition_inventory"
    broken = copy.deepcopy(rows)
    broken[0]["roles"]["partner"] = "Partner Archive"
    report = b.disjointness_audit(broken, [path])
    assert not report["pass"] and report["collisions"]["agent_operator_names"] == ["partner archive"]


def test_nuisance_balance():
    audit = b.nuisance_audit(scenarios())
    assert audit["pass"] and audit["stratum_pass"] and audit["orientation_pass"]
    assert all(sorted(values.values()) == [27, 27] for values in audit["marginals"].values())
    assert all(set(values.values()) <= {13, 14} for values in audit["pairwise"].values())


def test_sequence_families_and_temporal_order():
    rows = scenarios()
    schedule = b.build_schedule(rows)
    audit = b.schedule_audit(rows, schedule)
    assert audit["pass"] and sorted(audit["sequence_family_counts"].values()) == [13, 13, 14, 14]
    assert all(len(families) == 3 for families in audit["stratum_sequence_families"].values())
    assert audit["noncontiguous_strata"]
    assert Counter(row["condition"] for row in schedule) == {key: 54 for key in b.CONDITIONS}
    for round_index in range(1, 5):
        current = [row for row in schedule if row["round"] == round_index]
        assert len(current) == 54 and len({row["variant_id"] for row in current}) == 54
        assert sorted(Counter(row["condition"] for row in current).values()) == [13, 13, 14, 14]
        assert Counter(row["interface"] for row in current) == {"CUED": 27, "NEUTRAL": 27}
        assert Counter(row["M_I"] for row in current) == {0: 27, 1: 27}


def test_execution_ready_requires_every_strengthened_gate_and_artifacts_are_deterministic(tmp_path):
    pytest.importorskip("tiktoken")
    pytest.importorskip("openai_harmony")
    import importlib.metadata
    if any(importlib.metadata.version(name) != version for name, version in (("tiktoken", "0.13.0"), ("openai-harmony", "0.0.8"), ("pytest", "8.4.2"))):
        pytest.skip("official frozen dependency set unavailable")
    one, two = tmp_path / "one", tmp_path / "two"
    status = b.generate(one, "TESTSHA")
    b.generate(two, "TESTSHA")
    required = {"parent_hashes", "scenario_count", "strata", "structural_semantics", "comparability", "causal_isolation",
                "disjointness", "nuisance_balance", "paired_seeds", "schedule_sequence", "condition_invariance",
                "prompt_count", "lexical_safety", "harmony"}
    assert set(status["execution_ready_gates"]) == required and all(status["execution_ready_gates"].values())
    assert status["execution_ready"] and status["harmony_cued_pair_parity_pass_count"] == 54
    assert status["harmony_neutral_pair_parity_pass_count"] == 54 and status["harmony_max_within_interface_difference"] == 0
    assert {p.name for p in one.iterdir()} == {p.name for p in two.iterdir()}
    assert all((one / p.name).read_bytes() == (two / p.name).read_bytes() for p in one.iterdir())
