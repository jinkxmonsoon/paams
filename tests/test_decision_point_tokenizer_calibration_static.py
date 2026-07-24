import ast
import json
from pathlib import Path

from btom_v2.run_decision_point_tokenizer_calibration_v0_2_2 import (
    EXPECTED_CHANGED,
    IMMUTABLE_INPUTS,
    PAIR_NAMESPACES,
    construct_candidate,
    pair_stream,
    verify_immutable_inputs,
)


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "btom_v2/decision_point_tokenizer_calibration_manifest_v0_2_2.json"
SCRIPT_PATH = ROOT / "btom_v2/run_decision_point_tokenizer_calibration_v0_2_2.py"
WORKFLOW_PATH = ROOT / ".github/workflows/decision_point_tokenizer_calibration_v0_2_2.yml"
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def test_manifest_is_exactly_tokenizer_only_and_pinned():
    assert MANIFEST == {
        "calibration_version": "btom-v2-decision-point-tokenizer-calibration-0.2.2",
        "status": "tokenizer_only_no_model_execution",
        "parent_prompt_protocol": "btom-v2-decision-point-prompt-protocol-0.2.1-provenance-separated-mock-only",
        "parent_remote_commit": "c2a2d66475564406e0f5625c73c848e8e4cb8c9b",
        "provider": "Groq",
        "model": "openai/gpt-oss-20b",
        "raw_tokenizer": "o200k_harmony",
        "harmony_encoding": "HARMONY_GPT_OSS",
        "python_version": "3.11",
        "dependencies": {"tiktoken": "0.13.0", "openai-harmony": "0.0.8"},
        "conversation_layout": {
            "system_messages": 0,
            "developer_messages": 0,
            "user_messages": 1,
            "assistant_generation_prefix": True,
        },
        "search_range": {"minimum_variant": 0, "maximum_variant": 9999},
        "filler_alphabet": "QZX789_",
        "pair_namespaces": [
            "DP5_FIRST_ORDER_PAIR",
            "DP7_FIRST_ORDER_PAIR",
            "DP7_SECOND_ORDER_PAIR",
        ],
        "real_execution_authorized": False,
        "interpretation_constraint": "This tokenizer-only calibration does not test H1, H2, or H3 and cannot demonstrate Theory of Mind, coordination improvement, autonomous planning, or cognition.",
    }


def test_immutable_hashes_are_enforced_and_current():
    expected = {
        "btom_v2/decision_point_prompting_v0_2_1.py": "a2a6140e4e7676187c589a7e10c61fc6ff1b96f5aa0ba90258d54bb265e8b80f",
        "btom_v2/decision_point_prompt_protocol_v0_2_1.json": "9acec8dba50f0e33082cb8b0b238b8f4d27875b6ee4382fdee7c7e4ea806e8f7",
        "btom_v2/audit_decision_point_prompts_v0_2_1.py": "d4f7f9dbf0eda3219716987b379776207f1ad0e8c8131e704ec88a99ddcf55ec",
        "btom_v2/decision_point_prompt_amendment_v0_2_1.md": "032acd30bf00e562530c849b47a0c76608bbd07a1fdfe3982a51c092832054d9",
    }
    assert IMMUTABLE_INPUTS == expected
    assert all(record["matches"] for record in verify_immutable_inputs().values() if isinstance(record, dict))


def test_third_party_imports_are_deferred_and_no_api_clients_are_imported():
    tree = ast.parse(SCRIPT_PATH.read_text())
    top_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert "tiktoken" not in top_imports
    assert "openai_harmony" not in top_imports
    source = SCRIPT_PATH.read_text().lower()
    assert "from groq" not in source
    assert "import groq" not in source
    assert "openai import" not in source
    assert "requests" not in source
    assert "httpx" not in source


def test_script_has_runtime_tokenizer_calls_and_never_authorizes_execution():
    source = SCRIPT_PATH.read_text()
    assert 'tiktoken.get_encoding("o200k_harmony")' in source
    assert "load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)" in source
    assert "Conversation.from_messages([" in source
    assert "Message.from_role_and_content(Role.USER, prompt)" in source
    assert "render_conversation_for_completion(" in source
    assert '"real_execution_authorized": True' not in source
    assert '"real_execution_authorized": False' in source


def test_pair_streams_are_deterministic_prefix_coupled_and_case_agnostic():
    for namespace in PAIR_NAMESPACES:
        short = pair_stream(namespace, 17, 8)
        long = pair_stream(namespace, 17, 10)
        assert long.startswith(short)
        assert pair_stream(namespace, 17, 10) == long
        assert set(long) <= set(MANIFEST["filler_alphabet"])
    source = SCRIPT_PATH.read_text()
    stream_function = source[source.index("def pair_stream"):source.index("def _replace_section")]
    for forbidden in ("case_id", "condition", "DP5a", "DP5b", "DP7a", "DP7b"):
        assert forbidden not in stream_function


def test_in_memory_candidates_change_only_six_neutral_prompts():
    parents = {
        item.prompt_id: item
        for item in __import__(
            "btom_v2.decision_point_prompting_v0_2_1",
            fromlist=["render_all_prompts"],
        ).render_all_prompts()
    }
    candidates = construct_candidate(0)
    changed = {
        item.prompt_id for item in candidates
        if item.prompt != parents[item.prompt_id].prompt
    }
    assert len(candidates) == 16
    assert changed == EXPECTED_CHANGED
    assert all(
        item.prompt == parents[item.prompt_id].prompt
        for item in candidates
        if item.condition in {"reactive_no_explicit_belief", "explicit_first_order", "explicit_second_order"}
    )


def test_workflow_has_exact_guard_no_secrets_and_tokenizer_only_steps():
    workflow = WORKFLOW_PATH.read_text()
    guard = (
        "github.event.head_commit.message ==\n"
        "          'Retry tokenizer calibration with pinned pytest [run-tokenizer-calibration-v0.2.2-r2]'"
    )
    assert guard in workflow
    assert "GROQ_API_KEY" not in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "tiktoken==0.13.0" in workflow
    assert "openai-harmony==0.0.8" in workflow
    assert "pytest==8.4.2" in workflow
    assert "python -m pytest -q 2>&1 |" in workflow
    assert "tee tokenizer-calibration-artifact/pytest.log" in workflow
    assert "set -o pipefail" in workflow
    assert "|| true" not in workflow
    assert "continue-on-error" not in workflow
    assert workflow.count("run_decision_point_tokenizer_calibration_v0_2_2") == 1
    assert "if: always()" in workflow
    assert "actions/upload-artifact@v4" in workflow
    for forbidden in ("curl ", "wget ", "api.groq.com", "api.openai.com"):
        assert forbidden not in workflow.lower()

    tokenizer_step = workflow.split("- name: Install frozen tokenizer dependencies", 1)[1].split(
        "- name: Install pinned test runner", 1
    )[0]
    pytest_step = workflow.split("- name: Install pinned test runner", 1)[1].split(
        "- name: Run complete test suite", 1
    )[0]
    assert "tiktoken==0.13.0" in tokenizer_step
    assert "openai-harmony==0.0.8" in tokenizer_step
    assert "pytest==8.4.2" not in tokenizer_step
    assert "pytest==8.4.2" in pytest_step
    assert "tiktoken==0.13.0" not in pytest_step
    assert "openai-harmony==0.0.8" not in pytest_step
    assert "tokenizer-calibration-artifact/pytest_install.log" in pytest_step
    assert "tokenizer-calibration-artifact/pytest_resolved_version.txt" in pytest_step
    assert "python -m pip show pytest" in pytest_step


def test_new_activation_message_does_not_match_old_workflow_guards():
    message = "Retry tokenizer calibration with pinned pytest [run-tokenizer-calibration-v0.2.2-r2]"
    old_workflows = (
        ROOT / ".github/workflows/frozen_real_llm_micro_pilot.yml",
        ROOT / ".github/workflows/frozen_real_llm_micro_pilot_v1_1.yml",
    )
    assert all(message not in path.read_text() for path in old_workflows)
    assert message != "Run tokenizer-only decision-point calibration [run-tokenizer-calibration-v0.2.2]"
