"""Static and synthetic tests for tokenizer-only wording selection v0.3.3."""

import ast
import hashlib
import json
from pathlib import Path

from btom_v2.run_decision_point_natural_wording_selection_v0_3_3 import (
    EXPECTED_CANDIDATES,
    IMMUTABLE_PARENT_SHA256,
    INTERACTIONS,
    PRIMARY,
    STRUCTURAL_FLAGS,
    calculate_token_diagnostics,
    classify_records,
    ranking_key,
    verify_bank_population,
)


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "btom_v2/run_decision_point_natural_wording_selection_v0_3_3.py"
MANIFEST_PATH = ROOT / "btom_v2/decision_point_natural_wording_selection_manifest_v0_3_3.json"
WORKFLOW_PATH = ROOT / ".github/workflows/decision_point_natural_wording_selection_v0_3_3.yml"
MANIFEST = json.loads(MANIFEST_PATH.read_text())
SOURCE = RUNNER.read_text()
WORKFLOW = WORKFLOW_PATH.read_text()


def test_manifest_exact_identity_provenance_dependencies_and_layout():
    assert MANIFEST["calibration_version"] == "btom-v2-decision-point-natural-wording-selection-0.3.3"
    assert MANIFEST["status"] == "tokenizer_only_no_model_execution"
    assert MANIFEST["parent_provenance"] == {
        "wording_bank": "btom-v2-decision-point-natural-wording-bank-0.3.2-mock-only",
        "parent_remote_commit": "b648e29fe73dbb5357777eb5d033fd7dab7fcf61",
        "tokenizer_measurement": "btom-v2-decision-point-natural-control-tokenizer-0.3.1",
        "tokenizer_measurement_run_id": 30311773726,
        "tokenizer_measurement_artifact_digest": "5210d8efb69d9dd74265bff6d80a6556ac66fe9bbe63ec8cd86c22db9d4aea2c",
        "tokenizer_measurement_classification": "nonzero_primary_token_difference",
    }
    assert MANIFEST["dependencies"] == {
        "tiktoken": "0.13.0", "openai-harmony": "0.0.8", "pytest": "8.4.2"
    }
    assert MANIFEST["tokenizer_layout"] == {
        "raw_tokenizer": "o200k_harmony", "harmony_encoding": "HARMONY_GPT_OSS",
        "system_messages": 0, "developer_messages": 0, "user_messages": 1,
        "assistant_generation_prefix": True,
    }


def test_exact_parent_hashes_are_consistent_everywhere():
    assert MANIFEST["immutable_parent_sha256"] == IMMUTABLE_PARENT_SHA256
    for relative, expected in IMMUTABLE_PARENT_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
        assert f"{expected}  {relative}" in WORKFLOW


def test_bank_is_imported_directly_and_population_audit_passes():
    required_imports = (
        "candidate_specifications", "render_candidate_set", "audit_wording_bank",
        "FUTURE_SELECTION_RULE", "EXPECTED_FIELDS_BY_SECTION",
        "PERMITTED_REPLACEMENT_FIELDS",
    )
    for name in required_imports:
        assert name in SOURCE
    specifications, audit = verify_bank_population()
    assert len(specifications) == EXPECTED_CANDIDATES == 1296
    assert len({item.candidate_id for item in specifications}) == 1296
    assert audit["candidate_sets_audited"] == 1296
    assert audit["prompts_audited"] == 20736
    assert audit["candidate_sets_failed"] == 0
    assert audit["failed_candidate_ids"] == []
    assert all(audit[key] is True for key in STRUCTURAL_FLAGS)
    assert tuple(specifications[0].__dataclass_fields__) == (
        "location_surface_id", "h1_operational_variant_id",
        "h2_provenance_variant_id", "candidate_id",
    )
    assert not hasattr(specifications[0], "H1_operational_variant_id")
    assert not hasattr(specifications[0], "H2_provenance_variant_id")


def test_runner_does_not_reconstruct_candidates_or_rewrite_prompts():
    tree = ast.parse(SOURCE)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "itertools" not in imported
    assert "CandidateSpecification" not in SOURCE
    assert "_replace_quoted_field" not in SOURCE
    assert "render_prompt_variant" not in SOURCE
    assert "padding" not in SOURCE.lower()
    assert "filler" not in SOURCE.lower()


def _synthetic_counts(h1a=0, h1b=0, h2a=0, h2b=0):
    counts = {}
    for _, (case_id, informative, reference) in PRIMARY.items():
        counts[(case_id, informative)] = {"raw_token_count": 100, "harmony_token_count": 106}
        counts[(case_id, reference)] = {"raw_token_count": 100, "harmony_token_count": 106}
    for label, delta in (("H1_DP5a", h1a), ("H1_DP5b", h1b), ("H2_DP7a", h2a), ("H2_DP7b", h2b)):
        case_id, informative, _ = PRIMARY[label]
        counts[(case_id, informative)]["raw_token_count"] += delta
        counts[(case_id, informative)]["harmony_token_count"] += delta
    return counts


def test_exact_primary_interaction_and_acceptance_calculations():
    exact = calculate_token_diagnostics(_synthetic_counts())
    assert exact["acceptance_vector"] == [0] * 12
    assert exact["all_token_deltas_exactly_zero"] is True
    nonzero = calculate_token_diagnostics(_synthetic_counts(h1a=1, h2b=2))
    assert nonzero["primary_deltas"]["H1_DP5a"] == {
        "raw_token_count": 1, "harmony_token_count": 1
    }
    assert nonzero["primary_deltas"]["H2_DP7b"] == {
        "raw_token_count": 2, "harmony_token_count": 2
    }
    assert nonzero["interaction_diagnostics"]["H1"] == {
        "raw_token_count": 1, "harmony_token_count": 1
    }
    assert nonzero["interaction_diagnostics"]["H2"] == {
        "raw_token_count": -2, "harmony_token_count": -2
    }
    assert len(nonzero["acceptance_vector"]) == 12
    assert nonzero["all_token_deltas_exactly_zero"] is False
    assert set(INTERACTIONS) == {"H1", "H2"}


def _record(identifier, exact, characters=100, changed=2, canonical="{}"):
    return {
        "candidate_id": identifier,
        "exact_parity": exact,
        "ranking_metrics": {
            "compared_block_character_count": characters,
            "changed_candidate_literal_count": changed,
            "canonical_candidate_specification": canonical,
            "candidate_id_integrity_tiebreaker": identifier,
        },
    }


def test_frozen_ranking_and_success_select_one_deterministically():
    records = [_record(f"candidate-{index:04d}", False) for index in range(1296)]
    records[7] = _record("later", True, characters=90, changed=3, canonical='{"z":1}')
    records[9] = _record("winner", True, characters=90, changed=2, canonical='{"a":1}')
    records[11] = _record("lexically-later", True, characters=90, changed=2, canonical='{"b":1}')
    classification, selected, exit_code = classify_records(records, True)
    assert classification == "success"
    assert selected["candidate_id"] == "winner"
    assert exit_code == 0
    assert ranking_key(selected) < ranking_key(records[11])


def test_no_candidate_and_runtime_failure_exit_semantics():
    records = [_record(f"candidate-{index:04d}", False) for index in range(1296)]
    assert classify_records(records, True) == ("no_candidate", None, 0)
    assert classify_records(records[:-1], False) == ("runtime_failure", None, 1)


def test_manifest_exact_acceptance_ranking_and_no_tolerance():
    assert len(MANIFEST["exact_parity_criteria"]) == 13
    assert MANIFEST["tolerance_threshold"] is None
    assert MANIFEST["ranking"] == [
        "lower total character count across the eight block instances participating in the four case-level primary contrasts",
        "fewer changed candidate-level literal choices relative to v0.3.0: two location surfaces plus differing H1 and H2 metadata values",
        "lexicographic canonical lowercase candidate specification JSON",
        "candidate ID only as a final deterministic integrity tiebreaker",
    ]
    assert MANIFEST["classification_rules"]["exit_codes"] == {
        "success": 0, "no_candidate": 0, "runtime_failure": 1
    }
    assert MANIFEST["candidate_population"]["required_diagnostics"] == 1296
    for flag in (
        "padding_authorized", "filler_authorized", "candidate_generation_authorized",
        "prompt_modification_authorized", "criterion_relaxation_authorized",
        "model_or_api_execution", "real_execution_authorized",
    ):
        assert MANIFEST[flag] is False


def test_tokenizer_imports_are_deferred_and_no_api_or_http_client_exists():
    tree = ast.parse(SOURCE)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "tiktoken" not in top_level_imports
    assert "openai_harmony" not in top_level_imports
    lowered = SOURCE.lower()
    for forbidden in (
        "import groq", "from groq", "openai import", "requests", "httpx",
        "urllib", "api.groq.com", "api.openai.com",
    ):
        assert forbidden not in lowered


def test_exact_workflow_guard_full_tests_single_run_and_always_upload():
    guard = (
        "github.event.head_commit.message ==\n"
        "          'Select exactly token-balanced natural wording [run-wording-selection-v0.3.3]'"
    )
    assert guard in WORKFLOW
    assert "codex/execute-micro-pilot-on-current-branch-vsnbkd" in WORKFLOW
    assert "python -m pytest -q 2>&1 | tee natural-wording-selection-artifact/pytest.log" in WORKFLOW
    assert WORKFLOW.count("python -m btom_v2.run_decision_point_natural_wording_selection_v0_3_3") == 1
    assert "if: always()" in WORKFLOW
    assert "retention-days: 30" in WORKFLOW
    assert "GROQ_API_KEY" not in WORKFLOW
    assert "OPENAI_API_KEY" not in WORKFLOW
