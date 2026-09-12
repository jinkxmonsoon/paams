"""Acceptance tests for the frozen Journal J1 record holdout bank."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from btom_v2 import journal_j1_record_holdout_prompting_v0_6_0 as prompting
from btom_v2 import journal_j1_record_holdout_scenarios_v0_6_0 as scenario_module

ROOT = Path(__file__).resolve().parents[1]
BTOM = ROOT / "btom_v2"
SCENARIOS = json.loads((BTOM / prompting.OUTPUTS[0]).read_text())["scenarios"]
PROMPTS = json.loads((BTOM / prompting.OUTPUTS[1]).read_text())["prompts"]
MANIFEST = json.loads((BTOM / prompting.OUTPUTS[2]).read_text())
AUDIT = json.loads((BTOM / prompting.OUTPUTS[3]).read_text())


def test_population_balance_record_only_and_cells():
    assert len(SCENARIOS) == 18 and len(PROMPTS) == 72
    assert {row["archetype"] for row in SCENARIOS} == set(scenario_module.ARCHETYPES)
    assert {row["difficulty"] for row in SCENARIOS} == set(scenario_module.DIFFICULTIES)
    assert Counter((row["archetype"], row["difficulty"]) for row in SCENARIOS) == {(a, d): 1 for a in scenario_module.ARCHETYPES for d in scenario_module.DIFFICULTIES}
    by_variant = defaultdict(list)
    for prompt in PROMPTS: by_variant[prompt["variant_id"]].append(prompt)
    assert all(len(rows) == 4 and {(r["M_R"], r["M_I"]) for r in rows} == {(0,0),(0,1),(1,0),(1,1)} for rows in by_variant.values())
    assert all(row["F"] == "R" and prompting.FRAMING_LABEL in row["prompt_text"] and "partner belief" not in row["prompt_text"].lower() for row in PROMPTS)


def test_structural_depth_edges_values_and_isolated_roles():
    for scenario in SCENARIOS:
        visible = scenario["model_visible_invariant"]
        q, r = visible["q_relation"], visible["r_relation"]
        expected = 2 if scenario["difficulty"] == "compositional" else 1
        assert q["path_depth"] == r["path_depth"] == expected
        assert len(q["visible_edges"]) == len(r["visible_edges"]) == expected
        assert q["semantic_type"] == r["semantic_type"]
        assert len({q["entity"], r["entity"], q["true_value"], q["false_value"], r["true_value"], r["false_value"]}) == 6
        rows = [p for p in PROMPTS if p["variant_id"] == scenario["variant_id"]]
        for edge in q["visible_edges"] + r["visible_edges"]:
            assert all(prompting.edge_text(edge) in row["prompt_text"] for row in rows)
        assert len(visible["static_context"]) == (1 if scenario["difficulty"] == "irrelevant_distractor" else 0)
        mechanics = ("immediately executes", "independently executes", "reaches", "only and is not relayed", "cannot alter", "different objects, targets, routes, and actions", "episode ends")
        assert all(all(term in row["prompt_text"] for term in mechanics) for row in rows)


def test_only_record_values_change_with_factors():
    for scenario in SCENARIOS:
        rows = [p for p in PROMPTS if p["variant_id"] == scenario["variant_id"]]
        assert len({row["world_state_sha256"] for row in rows}) == 1
        assert len({tuple(row["action_order"]) for row in rows}) == 1 and len({row["proposition_order"] for row in rows}) == 1
        normalized = set()
        relation = scenario["model_visible_invariant"]
        replace = [relation[role][key] for role in ("q_relation", "r_relation") for key in ("true_value", "false_value")]
        for row in rows:
            text = row["prompt_text"]
            for value in replace: text = text.replace(value, "RECORDED_VALUE")
            normalized.add(text)
            assert row["gold_action"] == ("SEND_CORRECTION" if row["M_R"] else "CONTINUE_TASK")
        assert len(normalized) == 1


def test_nuisance_orders_and_signatures():
    proposition = Counter(row["proposition_order"] for row in PROMPTS[::4])
    actions = Counter(row["action_order"][0] for row in PROMPTS[::4])
    joint = Counter((row["proposition_order"], row["action_order"][0]) for row in PROMPTS[::4])
    assert proposition == {"partner_first": 9, "operator_first": 9}
    assert actions == {"SEND_CORRECTION": 9, "CONTINUE_TASK": 9}
    assert joint == {("partner_first", "SEND_CORRECTION"):5, ("partner_first", "CONTINUE_TASK"):4, ("operator_first", "SEND_CORRECTION"):4, ("operator_first", "CONTINUE_TASK"):5}
    assert len({row["semantic_signature_sha256"] for row in SCENARIOS}) == 18
    for row in SCENARIOS:
        assert row["variant_id"].encode() not in scenario_module.canonical(row["model_visible_invariant"])


def test_disjointness_and_selection_blind_source():
    assert all(row["prior_candidate_entity_overlap_count"] == row["prior_development_entity_overlap_count"] == 0 for row in AUDIT["variant_audits"])
    sources = Path(prompting.__file__).read_text() + Path(scenario_module.__file__).read_text()
    assert "journal_j1_confirmatory_selection_v0_3_0.json" not in sources
    forbidden_paths = ("behavioral_results", "raw_api_response", "call_records_v0_5", "cell_rates_v0_5", "variant_estimands_v0_5")
    assert not any(term in sources for term in forbidden_paths)


def test_selection_file_cannot_be_read(monkeypatch):
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if path.name == "journal_j1_confirmatory_selection_v0_3_0.json": raise AssertionError("selection read")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    assert len(prompting.build()[0]) == 18


def test_model_visible_leakage_actions_and_no_reasoning():
    forbidden = ("m_r", "m_i", "q_true", "q_false", "r_true", "r_false", "relevant", "irrelevant", "negative control", "causal control", "causal branch", "mismatch", "experimental condition", "gold action", "ground truth condition", "wrong", "incorrect", "false", "stale", "outdated", "rationale", "explanation", "chain of thought")
    for row in PROMPTS:
        lower = row["prompt_text"].lower()
        assert not any(term in lower for term in forbidden)
        assert row["variant_id"] not in row["prompt_text"]
        assert 'Return strict JSON only: {"action":"SEND_CORRECTION"} or {"action":"CONTINUE_TASK"}' in row["prompt_text"]
    assert AUDIT["checks"]["no_visible_leakage"] and AUDIT["passed"]


def test_seeds_and_four_round_order():
    order = MANIFEST["frozen_request_order"]
    assert len(MANIFEST["frozen_seeds"]) == len({row["requested_seed"] for row in order}) == 72
    assert Counter(row["round"] for row in order) == {1:18,2:18,3:18,4:18}
    assert Counter((row["M_R"], row["M_I"]) for row in order) == {(0,0):18,(0,1):18,(1,0):18,(1,1):18}
    for round_number in range(1,5): assert len({row["variant_id"] for row in order if row["round"] == round_number}) == 18
    by_variant = defaultdict(set)
    for row in order: by_variant[row["variant_id"]].add((row["M_R"], row["M_I"]))
    assert all(cells == {(0,0),(0,1),(1,0),(1,1)} for cells in by_variant.values())
    variants, prompts, regenerated, _, _ = prompting.build()
    assert regenerated == prompting.request_order(variants, prompts)[0]


def test_gates_headroom_stop_rules_and_status():
    assert MANIFEST["five_scenario_control_gates"] == {"record_relevance_pooled": ">= 0.25", "record_relevance_MI0": "> 0", "record_relevance_MI1": "> 0", "abs_record_irrelevant_MR0": "<= 0.25", "abs_record_irrelevant_MR1": "<= 0.25"}
    assert MANIFEST["headroom_diagnostic"]["is_sixth_gate"] is False
    assert MANIFEST["ready_for_confirmatory_realization"] == "scenario_control_pass AND NOT measurement_headroom_warning"
    assert set(MANIFEST["post_execution_classifications"].values()) == {"record_holdout_validated", "generic_mismatch_salience_replication", "record_holdout_relevance_failure", "causal_control_valid_but_measurement_saturated"}
    assert MANIFEST["harmony_validation_status"] == prompting.HARMONY_STATUS == "pending_official_pre_execution_validation"
    assert MANIFEST["behavioral_observations_used_to_select_holdout_variants"] == 0 and MANIFEST["confirmatory_selection_file_read"] is False


def test_no_network_api_model_or_confirmatory_code():
    for module in (prompting, scenario_module):
        source = Path(module.__file__).read_text(); tree = ast.parse(source)
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not imports & {"openai", "groq", "requests", "httpx", "socket", "urllib"}
    assert MANIFEST["model_or_api_execution"] is False and MANIFEST["confirmatory_prompts_generated"] is False and MANIFEST["icaart_reopened"] is False


def test_deterministic_regeneration(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"; one.mkdir(); two.mkdir()
    prompting.generate(one); prompting.generate(two)
    for name in prompting.OUTPUTS:
        assert (one / name).read_bytes() == (two / name).read_bytes() == (BTOM / name).read_bytes()
