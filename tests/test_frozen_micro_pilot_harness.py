import json
from pathlib import Path

import pytest

from btom_v2 import run_frozen_micro_pilot as harness


class SuccessfulClient:
    calls = 0
    transport = "mock_transport"
    last_latency_sec = 0.01
    last_usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}

    def __init__(self, model):
        self.model = model

    def generate(self, prompt, **kwargs):
        type(self).calls += 1
        return '{"action":"move","target":"staging","message":"","reason":"mock"}'


class FailingClient(SuccessfulClient):
    calls = 0

    def generate(self, prompt, **kwargs):
        type(self).calls += 1
        raise RuntimeError("mock connectivity failure")


def fake_episode_runner(scenario, seed, policy_class, client, args, trace_sink, episode_id):
    policy = "DeterministicBaseline" if policy_class.__name__ == "DeterministicBaselinePolicy" else policy_class.name
    trace_sink.append({
        "episode_id": episode_id,
        "scenario": scenario,
        "policy": policy,
        "seed": seed,
        "event": {"turn": 1, "event": "mock_event", "agent": "A", "details": {}},
    })
    return {
        "scenario_id": scenario, "seed": seed, "policy": policy, "success": False,
        "turns": 1, "time_to_rescue": None, "time_to_medical_kit_acquired": None,
        "llm_calls": 0, "api_success_rate": None, "parse_success_rate": None,
        "environment_action_valid_rate": 1.0, "parse_failures": 0, "invalid_targets": 0,
        "fallback_actions": 0, "progress_actions": 0, "valid_but_strategically_poor_actions": 0,
        "budget_cap_hits": 0, "false_belief_driven_decoy_pursuits": 0,
        "post_conflict_false_belief_pursuits": 0, "wrong_branch_steps": 0,
        "delayed_message_confusion_events": 0, "premature_shared_memory_assumptions": 0,
        "second_order_delivery_waits": 0, "correction_opportunities": 0,
        "necessary_correction_messages": 0, "missed_necessary_corrections": 0,
        "unnecessary_correction_messages": 0, "evidence_to_correction_send_steps": None,
        "correction_delivery_latency": None, "target_stale_belief_steps": 0,
        "target_decoy_branch_steps": 0, "post_correction_decoy_steps": 0,
        "total_messages": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "mean_latency_sec": None, "latency_per_call_sec": [], "parser_error_type_counts": {},
        "parsed_action_counts": {}, "api_failure_count": 0,
        "environment_invalid_action_count": 0, "call_audit": [],
    }


@pytest.fixture(autouse=True)
def credential(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-secret-that-must-never-be-written")
    SuccessfulClient.calls = 0
    FailingClient.calls = 0


def test_manifest_is_source_and_matrix_is_exact():
    manifest, _ = harness.load_manifest()
    matrix = harness.execution_matrix(manifest)
    assert len(matrix) == 12
    assert matrix == [
        (scenario, policy, seed)
        for scenario in manifest["scenarios"]
        for policy in manifest["policies"]
        for seed in manifest["seeds"]
    ]


def test_duplicate_combinations_are_rejected():
    manifest, _ = harness.load_manifest()
    manifest["policies"].append(manifest["policies"][0])
    with pytest.raises(ValueError, match="duplicate"):
        harness.execution_matrix(manifest)


def test_connectivity_failure_is_one_shot_and_creates_all_artifacts(tmp_path):
    status, output_dir = harness.execute(
        output_root=tmp_path, client_factory=FailingClient,
        episode_runner=lambda *args, **kwargs: pytest.fail("episode executed"),
        now="20260101T000000Z",
    )
    assert status == 3
    assert FailingClient.calls == 1
    assert output_dir.name == f"{harness.EXPECTED_PROTOCOL}_20260101T000000Z"
    assert {path.name for path in output_dir.iterdir()} == set(harness.REQUIRED_OUTPUT_FILES)
    assert (output_dir / "episode_summaries.jsonl").read_text() == ""


def test_success_writes_identity_null_times_exclusions_and_no_secret(tmp_path):
    status, output_dir = harness.execute(
        output_root=tmp_path, client_factory=SuccessfulClient,
        episode_runner=fake_episode_runner, now="20260101T000001Z",
    )
    assert status == 0
    assert SuccessfulClient.calls == 1
    rows = [json.loads(line) for line in (output_dir / "episode_summaries.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    assert all(row["time_to_rescue"] is None for row in rows)
    traces = [json.loads(line) for line in (output_dir / "complete_environment_traces.jsonl").read_text().splitlines()]
    assert len(traces) == 12
    assert all({"episode_id", "scenario", "policy", "seed"} <= trace.keys() for trace in traces)
    exclusions = json.loads((output_dir / "exclusion_decisions.json").read_text())
    assert all(item["excluded"] is False for item in exclusions)
    assert all(item["evidence"] == {"task_success": False} for item in exclusions)
    serialized = "".join(path.read_text() for path in output_dir.iterdir())
    assert "test-secret-that-must-never-be-written" not in serialized


def test_api_failure_is_technical_but_task_failure_alone_is_not():
    task_failure = fake_episode_runner("C5b_costly_false_belief", 0, type("P", (), {"name": "P"}), None, None, [], "id")
    assert harness._exclusion(task_failure)["excluded"] is False
    task_failure["api_failure_count"] = 1
    decision = harness._exclusion(task_failure)
    assert decision["excluded"] is True
    assert decision["criterion"].startswith("episode has an API failure")


def test_workflow_is_one_shot_and_secret_safe():
    workflow = Path(".github/workflows/frozen_real_llm_micro_pilot.yml").read_text()
    assert "secrets.GROQ_API_KEY" in workflow
    assert "gsk_" not in workflow
    assert "codex/execute-micro-pilot-on-current-branch-vsnbkd" in workflow
    assert "github.event.head_commit.message == 'Add GitHub Actions frozen pilot executor [run-frozen-pilot-v1]'" in workflow
    assert "python -m btom_v2.run_frozen_micro_pilot" in workflow
    assert "if: always()" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "path: btom_v2/outputs/" in workflow
