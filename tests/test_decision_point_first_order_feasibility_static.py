import ast
import json
from pathlib import Path

from btom_v2.run_decision_point_first_order_feasibility_v0_2_3 import (
    ADDITIONAL_IMMUTABLE_INPUTS,
    EXPECTED_SECOND_ORDER_FILLERS,
    MAX_DP5_CANDIDATES,
    MAX_DP7_CANDIDATES,
    SECOND_ORDER_VARIANT,
    STRUCTURED_ALPHABET,
    periodic_candidates,
    verify_immutable_inputs,
)


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "btom_v2/decision_point_first_order_feasibility_manifest_v0_2_3.json"
SCRIPT_PATH = ROOT / "btom_v2/run_decision_point_first_order_feasibility_v0_2_3.py"
WORKFLOW_PATH = ROOT / ".github/workflows/decision_point_first_order_feasibility_v0_2_3.yml"
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def test_manifest_freezes_parent_dependencies_searches_and_authorization():
    assert MANIFEST["calibration_version"] == "btom-v2-decision-point-first-order-feasibility-0.2.3"
    assert MANIFEST["status"] == "mock_only_tokenizer_only_no_model_execution"
    assert MANIFEST["parent_run_id"] == 30062353572
    assert MANIFEST["parent_remote_commit"] == "88ecc49fd45651dcd442d683284fea89b0bdf8c8"
    assert MANIFEST["dependencies"] == {
        "tiktoken": "0.13.0", "openai-harmony": "0.0.8", "pytest": "8.4.2"
    }
    assert MANIFEST["raw_tokenizer"] == "o200k_harmony"
    assert MANIFEST["harmony_encoding"] == "HARMONY_GPT_OSS"
    assert MANIFEST["searches"]["DP5_first_order"]["maximum_unique_candidates"] == 250000
    assert MANIFEST["searches"]["DP7_first_order"]["maximum_unique_candidates"] == 250000
    assert MANIFEST["candidate_families"]["periodic_structured"]["seed_lengths"] == [1, 2, 3, 4]
    assert "tokenizer_vocabulary_fragments" in MANIFEST["candidate_families"]
    assert MANIFEST["real_execution_authorized"] is False


def test_immutable_parent_hashes_are_enforced_and_current():
    assert ADDITIONAL_IMMUTABLE_INPUTS == {
        "btom_v2/decision_point_tokenizer_calibration_manifest_v0_2_2.json": "18940981899f4338e23fdab74da830484294afe9d66339b70dc8a98e907718c2",
        "btom_v2/run_decision_point_tokenizer_calibration_v0_2_2.py": "4b6733199b2b802f9c7c5265ae6fb776fad2da33301d8383c19a10c2a9cb36ab",
    }
    records = verify_immutable_inputs()
    assert records["parent_protocol_authorization_valid"] is True
    assert all(record["matches"] for record in records.values() if isinstance(record, dict))


def test_second_order_reproduction_is_fixed_not_searched():
    assert SECOND_ORDER_VARIANT == 3087
    assert EXPECTED_SECOND_ORDER_FILLERS == {8: "989_____", 10: "989_____ZZ"}
    assert MANIFEST["second_order_reproduction"] == {
        "namespace": "DP7_SECOND_ORDER_PAIR",
        "variant": 3087,
        "expected_fillers": {"8": "989_____", "10": "989_____ZZ"},
    }
    source = SCRIPT_PATH.read_text()
    assert 'pair_stream("DP7_SECOND_ORDER_PAIR", SECOND_ORDER_VARIANT, length)' in source
    assert "recomputed != EXPECTED_SECOND_ORDER_FILLERS" in source


def test_candidate_families_bounds_and_independent_searches_are_explicit():
    assert STRUCTURED_ALPHABET == "BCDFGHJKLMNPQRSTVWXYZ0123456789_"
    assert MAX_DP5_CANDIDATES == MAX_DP7_CANDIDATES == 250000
    assert next(periodic_candidates(8))[0] == "periodic_structured"
    source = SCRIPT_PATH.read_text()
    assert "def periodic_candidates" in source
    assert "def vocabulary_fragments" in source
    assert "def vocabulary_compositions" in source
    assert '"DP5_first_order"' in source
    assert '"DP7_first_order"' in source
    assert "global_variant" not in source
    generator = source[source.index("def periodic_candidates"):source.index("def _parents")]
    for forbidden in ("case_id", "DP5a", "DP5b", "DP7a", "DP7b", "condition"):
        assert forbidden not in generator


def test_third_party_imports_are_runtime_only_and_no_api_client_exists():
    tree = ast.parse(SCRIPT_PATH.read_text())
    top_imports = {
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    assert "tiktoken" not in top_imports
    assert "openai_harmony" not in top_imports
    source = SCRIPT_PATH.read_text().lower()
    for forbidden in (
        "from groq", "import groq", "openai import", "requests", "httpx",
        "urllib", "api.groq.com", "api.openai.com",
    ):
        assert forbidden not in source
    assert '"real_execution_authorized": True' not in SCRIPT_PATH.read_text()


def test_workflow_is_exactly_guarded_and_tokenizer_only():
    workflow = WORKFLOW_PATH.read_text()
    assert (
        "github.event.head_commit.message ==\n"
        "          'Run structured first-order tokenizer feasibility [run-first-order-feasibility-v0.2.3]'"
    ) in workflow
    assert "tiktoken==0.13.0 openai-harmony==0.0.8" in workflow
    assert "pytest==8.4.2" in workflow
    assert "python -m pytest -q 2>&1 |" in workflow
    assert "set -o pipefail" in workflow
    assert workflow.count("run_decision_point_first_order_feasibility_v0_2_3") == 1
    assert "if: always()" in workflow
    assert "retention-days: 30" in workflow
    assert "GROQ_API_KEY" not in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "continue-on-error" not in workflow
    assert "|| true" not in workflow
