"""Prospective and fail-closed tests for Journal J1 framing v0.4.2."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from btom_v2 import journal_j1_development_prompting_v0_4_2 as prompting

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
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


def test_frozen_inputs_and_predecessor_provenance():
    for name, expected in EXPECTED_HASHES.items():
        assert hashlib.sha256((ROOT / "btom_v2" / name).read_bytes()).hexdigest() == expected
    assert prompting.PREDECESSOR == {
        "predecessor_version": "v0.4.1",
        "predecessor_remote_sha": "b87b67d91e6aff93ec51750c6af7b7cbce90cf89",
        "predecessor_actions_run": 34661682287,
        "predecessor_failure": "no prospective v0.4.1 framing candidate achieved complete Harmony parity",
        "behavioral_observations_before_selection": 0,
    }


def test_exact_finite_record_belief_candidates():
    assert prompting.FRAMING_CANDIDATES == (
        {"candidate_id": 1, "R_label": "Current partner record:", "B_label": "Current partner belief:"},
        {"candidate_id": 2, "R_label": "PARTNER RECORD:", "B_label": "PARTNER BELIEF:"},
        {"candidate_id": 3, "R_label": "Partner record:", "B_label": "Partner belief:"},
        {"candidate_id": 4, "R_label": "CURRENT PARTNER RECORD:", "B_label": "CURRENT PARTNER BELIEF:"},
        {"candidate_id": 5, "R_label": "Current partner state record:", "B_label": "Current partner belief state:"},
        {"candidate_id": 6, "R_label": "PARTNER STATE RECORD:", "B_label": "PARTNER BELIEF STATE:"},
        {"candidate_id": 7, "R_label": "Partner state record:", "B_label": "Partner belief state:"},
        {"candidate_id": 8, "R_label": "CURRENT PARTNER STATE RECORD:", "B_label": "CURRENT PARTNER BELIEF STATE:"},
    )
    assert all("record" in row["R_label"].lower() and "belief" in row["B_label"].lower() for row in prompting.FRAMING_CANDIDATES)
    assert all(row["R_label"].lower() not in {"partner state:", "current partner state:", "current partner status:"} for row in prompting.FRAMING_CANDIDATES)


def test_population_and_normalized_equality(built):
    candidates, developments, prompts = built
    assert (len(candidates), len(developments), len(prompts)) == (360, 18, 1152)
    groups = defaultdict(list)
    for row in prompts:
        groups[row["candidate_id"], row["variant_id"], row["M_R"], row["M_I"]].append(row)
    assert len(groups) == 576
    by_candidate = Counter()
    for key, pair in groups.items():
        assert len(pair) == 2 and len({row["normalized_prompt_sha256"] for row in pair}) == 1
        by_candidate[key[0]] += 1
    assert by_candidate == {candidate_id: 72 for candidate_id in range(1, 9)}


def test_frozen_nuisance_orders_depth_gold_and_surface_safety(built):
    candidates, developments, prompts = built
    assert Counter((r["model_visible"]["proposition_order"], r["model_visible"]["action_order"][0]) for r in developments) == {
        ("causal_first", "SEND_CORRECTION"): 5, ("causal_first", "CONTINUE_TASK"): 4,
        ("noncausal_first", "SEND_CORRECTION"): 4, ("noncausal_first", "CONTINUE_TASK"): 5,
    }
    for realization in candidates + developments:
        expected = 2 if realization["difficulty"] == "compositional" else 1
        assert all(realization["model_visible"][role]["path_depth"] == expected for role in ("causal_relation", "noncausal_relation"))
    forbidden = ("task item", "context item", "relevant", "irrelevant", "padding", "filler")
    assert all(row["gold_action"] == ("SEND_CORRECTION" if row["M_R"] else "CONTINUE_TASK") for row in prompts)
    assert not any(term in row["prompt_text"].lower() for row in prompts for term in forbidden)


def _tokens_with_candidate_one_failure(text: str) -> list[int]:
    lines = set(text.splitlines())
    length = 10
    if "Current partner record:" in lines:
        length += 1
    return list(range(length))


def _always_failing_tokens(text: str) -> list[int]:
    belief = any(line in {row["B_label"] for row in prompting.FRAMING_CANDIDATES} for line in text.splitlines())
    return list(range(11 if belief else 10))


def test_first_passing_rule_and_complete_table(built):
    _, _, prompts = built
    table = prompting.evaluate_candidates(prompts, _tokens_with_candidate_one_failure, _tokens_with_candidate_one_failure)
    assert len(table) == 8
    assert table[0]["equal_harmony_token_count_pairs"] == 0
    assert prompting.passing_candidate(table)["candidate_id"] == 2
    required = {"candidate_id", "R_label", "B_label", "matched_pair_count", "normalized_equal_pair_count", "equal_harmony_token_count_pairs", "harmony_parity_rate", "min_signed_token_difference", "max_signed_token_difference", "max_abs_token_difference", "mean_signed_token_difference"}
    assert all(required <= row.keys() for row in table)


def test_diagnostics_persist_before_fail_closed(tmp_path):
    tokenizers = (_always_failing_tokens, _always_failing_tokens, {"tiktoken": "0.13.0", "openai-harmony": "0.0.8", "pytest": "8.4.2"})
    with pytest.raises(RuntimeError, match="no frozen v0.4.2 framing candidate"):
        prompting.run_tokenizer_audit(tmp_path, tokenizers=tokenizers)
    assert (tmp_path / prompting.TABLE_JSON).is_file()
    assert (tmp_path / prompting.TABLE_CSV).is_file()
    assert len(json.loads((tmp_path / prompting.TABLE_JSON).read_text())) == 8
    status = json.loads((tmp_path / "workflow_status.json").read_text())
    assert status["pilot_ready"] is False and status["selected_candidate_id"] is None
    assert not any((tmp_path / name).exists() for name in prompting.FINAL_OUTPUTS)


def test_no_behavior_model_api_network_or_dynamic_search():
    source = Path(prompting.__file__).read_text()
    imports = {alias.name.split(".")[0] for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not imports & {"openai", "groq", "requests", "httpx", "socket", "urllib"}
    assert "while " not in source and "product(" not in source and "offline_gpt_oss_token_count" not in source
