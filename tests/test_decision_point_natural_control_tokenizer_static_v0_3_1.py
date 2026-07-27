import ast
import hashlib
import json
from pathlib import Path

from btom_v2.run_decision_point_natural_control_tokenizer_v0_3_1 import (
    CONTRASTS,
    IMMUTABLE_PARENT_SHA256,
    STRUCTURAL_AUDIT_REQUIREMENTS,
    classify,
    exit_code_for_classification,
    interaction_diagnostics,
    verify_immutable_inputs,
)


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "btom_v2/decision_point_natural_control_tokenizer_manifest_v0_3_1.json"
SCRIPT_PATH = ROOT / "btom_v2/run_decision_point_natural_control_tokenizer_v0_3_1.py"
WORKFLOW_PATH = ROOT / ".github/workflows/decision_point_natural_control_tokenizer_v0_3_1.yml"
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def test_manifest_exact_parent_dependencies_population_and_authorization():
    assert MANIFEST["calibration_version"] == "btom-v2-decision-point-natural-control-tokenizer-0.3.1"
    assert MANIFEST["parent_protocol"] == "btom-v2-decision-point-natural-control-design-0.3.0-mock-only"
    assert MANIFEST["parent_remote_commit"] == "6e4a00395bd7912bdbb27e9373526dd1c548599c"
    assert MANIFEST["dependencies"] == {
        "tiktoken": "0.13.0", "openai-harmony": "0.0.8", "pytest": "8.4.2"
    }
    assert MANIFEST["required_prompt_population"] == {"DP5": 6, "DP7": 10, "total": 16}
    assert MANIFEST["tolerance_threshold"] is None
    assert MANIFEST["prompt_search_authorized"] is False
    assert MANIFEST["prompt_optimization_authorized"] is False
    assert MANIFEST["padding_authorized"] is False
    assert MANIFEST["real_execution_authorized"] is False


def test_exact_parent_hashes_are_current_and_enforced():
    assert MANIFEST["immutable_parent_sha256"] == IMMUTABLE_PARENT_SHA256
    for relative, expected in IMMUTABLE_PARENT_SHA256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert all(record["matches"] for record in verify_immutable_inputs().values())


def test_runtime_only_tokenizer_imports_and_frozen_renderer_import():
    tree = ast.parse(SCRIPT_PATH.read_text())
    top_imports = {
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    assert "tiktoken" not in top_imports
    assert "openai_harmony" not in top_imports
    source = SCRIPT_PATH.read_text()
    assert "from .decision_point_natural_control_prompting_v0_3_0 import" in source
    assert "render_all_prompts" in source and "run_audit" in source
    lowered = source.lower()
    for forbidden in (
        "def generate_candidate", "def generate_filler", "def search", "def optimize",
        "padding =", ".replace(", "renderedprompt(",
    ):
        assert forbidden not in lowered


def test_exact_contrasts_and_structural_population_requirements():
    assert CONTRASTS == {
        "H1": {
            "informative": "explicit_first_order",
            "reference": "operational_metadata_reference",
            "cases": ("DP5a_self_belief_false", "DP5b_self_belief_current"),
            "informative_block": "FIRST-ORDER REPRESENTATION",
            "reference_block": "DECISION METADATA",
        },
        "H2": {
            "informative": "explicit_second_order",
            "reference": "first_order_plus_message_provenance",
            "cases": ("DP7a_partner_belief_stale", "DP7b_partner_belief_current"),
            "informative_block": "SECOND-ORDER REPRESENTATION",
            "reference_block": "MESSAGE PROVENANCE METADATA",
        },
    }
    assert STRUCTURAL_AUDIT_REQUIREMENTS["prompt_count"] == 16
    assert STRUCTURAL_AUDIT_REQUIREMENTS["DP5_prompt_count"] == 6
    assert STRUCTURAL_AUDIT_REQUIREMENTS["DP7_prompt_count"] == 10


def _synthetic_prompt_record(case, condition, raw, harmony, chars, bytes_):
    return {
        "case_id": case, "condition": condition,
        "raw_token_count": raw, "harmony_token_count": harmony,
        "character_count": chars, "utf8_byte_count": bytes_,
    }


def test_interaction_calculation_is_exact_difference_of_differences():
    records = [
        _synthetic_prompt_record("DP5a_self_belief_false", "explicit_first_order", 12, 22, 32, 42),
        _synthetic_prompt_record("DP5b_self_belief_current", "explicit_first_order", 10, 20, 30, 40),
        _synthetic_prompt_record("DP5a_self_belief_false", "operational_metadata_reference", 8, 18, 28, 38),
        _synthetic_prompt_record("DP5b_self_belief_current", "operational_metadata_reference", 7, 17, 27, 37),
        _synthetic_prompt_record("DP7a_partner_belief_stale", "explicit_second_order", 16, 26, 36, 46),
        _synthetic_prompt_record("DP7b_partner_belief_current", "explicit_second_order", 13, 23, 33, 43),
        _synthetic_prompt_record("DP7a_partner_belief_stale", "first_order_plus_message_provenance", 11, 21, 31, 41),
        _synthetic_prompt_record("DP7b_partner_belief_current", "first_order_plus_message_provenance", 9, 19, 29, 39),
    ]
    diagnostics = interaction_diagnostics(records)
    assert diagnostics["H1"]["measurements"]["raw_tokens"] == {
        "informative_first_case_minus_second_case": 2,
        "reference_first_case_minus_second_case": 1,
        "difference_of_pairwise_differences": 1,
    }
    assert diagnostics["H2"]["measurements"]["raw_tokens"]["difference_of_pairwise_differences"] == 1


def test_classification_has_no_tolerance_and_completed_nonzero_returns_zero():
    zero_case = {
        "raw_full_prompt_token_delta": 0,
        "harmony_full_prompt_token_delta": 0,
    }
    nonzero_case = {**zero_case, "raw_full_prompt_token_delta": 1}
    interactions = {
        name: {"measurements": {
            "raw_tokens": {"difference_of_pairwise_differences": 0},
            "harmony_tokens": {"difference_of_pairwise_differences": 0},
        }} for name in ("H1", "H2")
    }
    exact = classify({"H1": {"a": zero_case}, "H2": {"b": zero_case}}, interactions)
    nonzero = classify({"H1": {"a": nonzero_case}, "H2": {"b": zero_case}}, interactions)
    assert exact == "exact_primary_token_parity"
    assert nonzero == "nonzero_primary_token_difference"
    assert exit_code_for_classification(exact) == 0
    assert exit_code_for_classification(nonzero) == 0
    assert exit_code_for_classification("runtime_failure") == 1


def test_no_api_client_http_dependency_or_execution_authorization():
    source = SCRIPT_PATH.read_text().lower()
    for forbidden in (
        "from groq", "import groq", "openai import", "requests", "httpx", "urllib",
        "api.groq.com", "api.openai.com",
    ):
        assert forbidden not in source
    assert '"real_execution_authorized": true' not in source
    assert '"model_or_api_execution": true' not in source


def test_workflow_exact_guard_full_tests_single_measurement_and_always_upload():
    workflow = WORKFLOW_PATH.read_text()
    assert (
        "github.event.head_commit.message ==\n"
        "          'Measure natural-control tokenizer differences [run-natural-control-tokenizer-v0.3.1]'"
    ) in workflow
    assert "tiktoken==0.13.0 openai-harmony==0.0.8" in workflow
    assert "pytest==8.4.2" in workflow
    assert "python -m pytest -q 2>&1 |" in workflow and "set -o pipefail" in workflow
    assert workflow.count("run_decision_point_natural_control_tokenizer_v0_3_1") == 1
    assert "if: always()" in workflow and "retention-days: 30" in workflow
    assert "GROQ_API_KEY" not in workflow and "OPENAI_API_KEY" not in workflow
    assert "continue-on-error" not in workflow and "|| true" not in workflow
