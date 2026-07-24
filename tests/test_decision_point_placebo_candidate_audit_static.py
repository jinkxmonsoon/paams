import ast
import json
from pathlib import Path

from btom_v2.run_decision_point_placebo_candidate_audit_v0_2_4 import (
    ADDITIONAL_IMMUTABLE_INPUTS, LEGACY_SECOND_ORDER_VARIANT_INDEX, PANEL_SIZE,
    PAYLOAD_ALPHABET, PREFIX, SEARCH_BOUNDS, _legacy_rejections,
    _rank_key, anti_salience_requirements, typed_placebo_candidates,
    verify_immutable_inputs,
)


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "btom_v2/decision_point_placebo_candidate_manifest_v0_2_4.json"
SCRIPT_PATH = ROOT / "btom_v2/run_decision_point_placebo_candidate_audit_v0_2_4.py"
WORKFLOW_PATH = ROOT / ".github/workflows/decision_point_placebo_candidate_audit_v0_2_4.yml"
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def test_manifest_freezes_provenance_dependencies_and_no_execution():
    assert MANIFEST["calibration_version"] == "btom-v2-decision-point-placebo-candidate-audit-0.2.4"
    assert MANIFEST["status"] == "mock_only_tokenizer_only_no_model_execution"
    assert MANIFEST["parent_provenance"]["run_id"] == 30066731354
    assert MANIFEST["parent_provenance"]["artifact_sha256"] == "0294867f039827410c9e245324f52fe8ed19b80a8adf813e3f3ed5baf440d72b"
    assert MANIFEST["dependencies"] == {"tiktoken": "0.13.0", "openai-harmony": "0.0.8", "pytest": "8.4.2"}
    assert MANIFEST["real_execution_authorized"] is False


def test_exact_typed_grammar_and_payload_exclusions():
    grammar = MANIFEST["typed_placebo_grammar"]
    assert PREFIX == grammar["literal_prefix"] == "loc_"
    assert grammar["short_form"] == "loc_XXXX" and grammar["short_length"] == 8
    assert grammar["long_form"] == "loc_XXXXYY" and grammar["long_length"] == 10
    assert PAYLOAD_ALPHABET == grammar["payload_alphabet"] == "DFGHJKLMNPQRTVWXYZ23456789"
    assert not set("ABCaeiouAEIOU01") & set(PAYLOAD_ALPHABET)
    candidate = next(typed_placebo_candidates(4))
    assert len(candidate) == 8 and candidate.startswith("loc_")
    assert all(anti_salience_requirements(candidate).values())


def test_anti_salience_rejects_runs_periodicity_underscores_and_legacy_values():
    assert not anti_salience_requirements("loc_DDDD")["maximum_run_two"]
    assert not anti_salience_requirements("loc_DFD F" )["no_whitespace"]
    assert not anti_salience_requirements("loc_DFDF")["not_period_two"]
    assert not anti_salience_requirements("loc_DF_G")["underscore_only_in_prefix"]
    assert not anti_salience_requirements("BBBBBBBB")["typed_prefix"]
    assert not anti_salience_requirements("989_____")["typed_prefix"]


def test_three_independent_searches_bounds_panels_and_ranking_are_frozen():
    assert PANEL_SIZE == 20
    assert SEARCH_BOUNDS == {"DP5_first_order": 500000, "DP7_first_order": 500000, "DP7_second_order": 500000}
    assert all(search["required_panel_size"] == 20 for search in MANIFEST["searches"].values())
    assert MANIFEST["deterministic_ranking"] == [
        "lower_maximum_repeated_character_run", "higher_distinct_payload_character_count",
        "lower_repeated_bigram_count", "lower_raw_isolated_value_token_count",
        "lower_harmony_isolated_value_token_count", "lexicographic_candidate_text",
    ]
    better = {"maximum_repeated_character_run": 1, "distinct_payload_character_count": 4, "repeated_bigram_count": 0, "raw_isolated_value_token_count": 3, "harmony_isolated_value_token_count": 9, "candidate_text": "loc_DFGH"}
    worse = {**better, "maximum_repeated_character_run": 2, "candidate_text": "loc_DFFH"}
    assert _rank_key(better) < _rank_key(worse)


def test_provenance_corrects_structured_candidates_and_rejects_legacy_values():
    schema = MANIFEST["provenance_schema"]
    assert "filler_variant_index" not in schema
    assert schema == ["generator_family", "candidate_rank", "candidate_id", "search_name", "short_value", "long_value", "parent_run_id", "parent_artifact_digest"]
    source = SCRIPT_PATH.read_text()
    assert '"generator_family": "typed_placebo_location_code"' in source
    assert "filler_variant_index" not in source
    assert LEGACY_SECOND_ORDER_VARIANT_INDEX == 3087
    rejected = _legacy_rejections()
    assert all(item["reference_only"] and item["rejected_for_final_selection"] for item in rejected)
    assert "legacy_second_order_variant_index" in rejected[1]
    assert "legacy_second_order_variant_index" not in rejected[0]


def test_immutable_hashes_are_enforced_and_current():
    assert ADDITIONAL_IMMUTABLE_INPUTS == {
        "btom_v2/decision_point_first_order_feasibility_manifest_v0_2_3.json": "0791b2b318577d3f312f3da147facaefb5819f13b5d54d4c3ba599557f10e480",
        "btom_v2/run_decision_point_first_order_feasibility_v0_2_3.py": "aa563de6f61ae93e36b02641104106253d345a498be7830b48a67e3298798bd8",
    }
    records = verify_immutable_inputs()
    assert records["parent_protocol_authorization_valid"] is True
    assert all(record["matches"] for record in records.values() if isinstance(record, dict))


def test_runtime_only_dependencies_and_no_api_client():
    tree = ast.parse(SCRIPT_PATH.read_text())
    top_imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names} | {node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "tiktoken" not in top_imports and "openai_harmony" not in top_imports
    source = SCRIPT_PATH.read_text().lower()
    for forbidden in ("from groq", "import groq", "openai import", "requests", "httpx", "urllib", "api.groq.com", "api.openai.com"):
        assert forbidden not in source
    assert '"real_execution_authorized": True' not in SCRIPT_PATH.read_text()


def test_workflow_guard_full_tests_and_always_upload():
    workflow = WORKFLOW_PATH.read_text()
    assert "github.event.head_commit.message ==\n          'Audit typed placebo location codes [run-placebo-audit-v0.2.4]'" in workflow
    assert "tiktoken==0.13.0 openai-harmony==0.0.8" in workflow
    assert "pytest==8.4.2" in workflow and "python -m pytest -q 2>&1 |" in workflow
    assert "set -o pipefail" in workflow and "if: always()" in workflow
    assert workflow.count("run_decision_point_placebo_candidate_audit_v0_2_4") == 1
    assert "retention-days: 30" in workflow
    assert "GROQ_API_KEY" not in workflow and "OPENAI_API_KEY" not in workflow
    assert "continue-on-error" not in workflow and "|| true" not in workflow
