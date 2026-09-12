"""Local and Actions gates for the Journal J1 v0.4.1 prompt freeze."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from btom_v2 import journal_j1_development_prompting_v0_4_1 as prompting
from btom_v2 import journal_j1_model_safe_realization_v0_4_1 as realization

ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    "journal_j1_confirmatory_candidate_frame_v0_3_0.json": "64c3b7fad886471830e4dfcb7160254fb83d829725010f1896fcd96396cbb247",
    "journal_j1_confirmatory_selection_v0_3_0.json": "4df0dd30a98bd333d4538fa3aa31e4712253152a5a7c189aabb0f1d0034da81f",
    "journal_j1_development_scenarios_v0_3_0.json": "ecf40e43a6d992311344f385d7b9ff57ca6abb14ccc3d4e5c0257caa5bf21c9f",
    "journal_j1_candidate_realizations_v0_4_0.json": "170ffbe618901b667d43567a2b42a8ad49fa96b5cb7f3bb723b55354e66af24e",
    "journal_j1_development_realizations_v0_4_0.json": "83bb47c38c431946524d6876c233ed4e9589ff47d9e2ea99dbc350a7a66b96fb",
    "journal_j1_development_prompts_v0_4_0.json": "bb4e33c2a47fdb2a47e10bab2b0e4932eb5c7c536b317fc5e3ebc41505349dd1",
}


@pytest.fixture(scope="module")
def built():
    return prompting.build_prospective()


def test_frozen_hashes_and_reference_blob():
    for name, expected in HASHES.items():
        assert hashlib.sha256((ROOT / "btom_v2" / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha1((b"blob " + str((ROOT / "btom_v2/run_decision_point_content_matched_tokenizer_v0_4_1.py").stat().st_size).encode() + b"\0" + (ROOT / "btom_v2/run_decision_point_content_matched_tokenizer_v0_4_1.py").read_bytes())).hexdigest() == "8e0ad715e0f21e6cf107c351e7fbce07399c7d8e"


def test_counts_selection_blind_and_deterministic(built):
    candidates, developments, prompts = built
    assert (len(candidates), len(developments), len(prompts)) == (360, 18, 432)
    assert all("selected" not in record for record in candidates)
    assert prompting.build_prospective() == built


def test_three_candidates_and_complete_factorials(built):
    _, _, prompts = built
    assert prompting.FRAMING_CANDIDATES == (
        {"candidate": 1, "R": "CURRENT PARTNER STATE:", "B": "CURRENT PARTNER BELIEF:"},
        {"candidate": 2, "R": "PARTNER STATE:", "B": "PARTNER BELIEF:"},
        {"candidate": 3, "R": "CURRENT PARTNER STATUS:", "B": "CURRENT PARTNER BELIEF:"},
    )
    groups = defaultdict(set)
    for prompt in prompts:
        groups[prompt["framing_candidate"], prompt["variant_id"]].add((prompt["F"], prompt["M_R"], prompt["M_I"]))
    expected = {(f, mr, mi) for f in "RB" for mr in (0, 1) for mi in (0, 1)}
    assert len(groups) == 54 and all(cells == expected for cells in groups.values())


def test_all_216_rb_pairs_normalize_byte_identically(built):
    _, _, prompts = built
    groups = defaultdict(list)
    for prompt in prompts:
        groups[prompt["framing_candidate"], prompt["variant_id"], prompt["M_R"], prompt["M_I"]].append(prompt)
    assert len(groups) == 216
    for (candidate, *_), pair in groups.items():
        framing = prompting.FRAMING_CANDIDATES[candidate - 1]
        normalized = {prompting.normalize_prompt(row["prompt_text"], framing).encode() for row in pair}
        assert len(pair) == 2 and len(normalized) == 1
        assert len({tuple(row["action_order"]) for row in pair}) == 1
        assert len({row["proposition_order"] for row in pair}) == 1


def test_orders_are_balanced_and_fixed(built):
    _, developments, prompts = built
    assert Counter(row["model_visible"]["proposition_order"] for row in developments) == {"causal_first": 9, "noncausal_first": 9}
    assert Counter(row["model_visible"]["action_order"][0] for row in developments) == {"SEND_CORRECTION": 9, "CONTINUE_TASK": 9}
    assert Counter(
        (row["model_visible"]["proposition_order"], row["model_visible"]["action_order"][0])
        for row in developments
    ) == {
        ("causal_first", "SEND_CORRECTION"): 5,
        ("causal_first", "CONTINUE_TASK"): 4,
        ("noncausal_first", "SEND_CORRECTION"): 4,
        ("noncausal_first", "CONTINUE_TASK"): 5,
    }
    for variant in {row["variant_id"] for row in prompts}:
        rows = [row for row in prompts if row["variant_id"] == variant]
        assert len({row["proposition_order"] for row in rows}) == 1
        assert len({tuple(row["action_order"]) for row in rows}) == 1
        assert len({(row["proposition_order"], tuple(row["action_order"])) for row in rows}) == 1


def test_structured_depth_and_every_edge_is_rendered(built):
    candidates, developments, prompts = built
    for row in candidates + developments:
        expected = 2 if row["difficulty"] == "compositional" else 1
        for role in ("causal_relation", "noncausal_relation"):
            relation = row["model_visible"][role]
            assert relation["path_depth"] == len(relation["visible_edges"]) == expected
        if row["namespace"] == "development":
            rendered = [p["prompt_text"] for p in prompts if p["variant_id"] == row["variant_id"]]
            for relation in (row["model_visible"]["causal_relation"], row["model_visible"]["noncausal_relation"]):
                for edge in relation["visible_edges"]:
                    assert all(prompting.edge_text(edge) in text for text in rendered)


def test_surface_safety_opportunity_cost_and_gold(built):
    _, _, prompts = built
    forbidden = ("task item", "context item", "relevant", "irrelevant", "important fact", "unimportant fact", "active fact", "other fact", "q_true", "q_false", "r_true", "r_false", "rationale", "explanation", "chain-of-thought", "reasoning")
    for row in prompts:
        text = row["prompt_text"]
        assert not any(term in text.lower() for term in forbidden)
        assert "exactly one action for this turn" in text
        assert "uses this turn and does not advance your task this turn" in text
        assert "advances your task this turn and sends no update" in text
        assert row["gold_action"] == ("SEND_CORRECTION" if row["M_R"] else "CONTINUE_TASK")
    groups = defaultdict(set)
    for row in prompts:
        groups[row["framing_candidate"], row["variant_id"], row["M_R"]].add(row["gold_action"])
    assert all(len(values) == 1 for values in groups.values())


def test_no_model_api_or_network_functionality():
    for module in (realization, prompting):
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not imports & {"openai", "groq", "requests", "httpx", "socket", "urllib"}
    assert "offline_gpt_oss_token_count" not in Path(prompting.__file__).read_text()


@pytest.mark.skipif(importlib.util.find_spec("tiktoken") is None or importlib.util.find_spec("openai_harmony") is None, reason="official tokenizer validation runs in Actions")
def test_actions_tokenizer_generation(tmp_path):
    prompting.generate_actions_outputs(tmp_path)
    assert all((tmp_path / name).is_file() for name in prompting.OUTPUT_NAMES)
