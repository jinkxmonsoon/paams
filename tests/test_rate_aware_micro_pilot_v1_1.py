import hashlib
import json
from pathlib import Path

import pytest

from btom_v2 import run_frozen_micro_pilot_v1_1 as harness


class FakeClock:
    def __init__(self):
        self.value = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class Mock429(Exception):
    http_status = 429
    error_type = "rate_limit"
    sanitized_error_message = (
        "organization org_private123 tokens per minute Limit 6000, Used 5900, "
        "Requested 500, retry-after 10; Authorization: Bearer private-token"
    )


class SequencedClient:
    calls = 0
    fail_on = None
    transport = "mock"

    def __init__(self, model):
        self.model = model
        self.last_usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        self.last_latency_sec = 0.1
        self.last_error = None

    def generate(self, prompt, **kwargs):
        type(self).calls += 1
        if type(self).calls == type(self).fail_on:
            raise Mock429()
        return '{"action":"move","target":"staging","message":"","reason":"mock"}'


def fake_summary(scenario, seed, policy, episode_id, parse_success=True):
    call = {
        "parse_success": parse_success,
        "parsed_action": "move",
        "parsed_target": "red_room",
        "valid_actions": [{"action": "move", "target": "staging"}],
        "environment_result": {"valid": True},
    }
    return {
        "episode_id": episode_id,
        "scenario_id": scenario,
        "seed": seed,
        "policy": policy,
        "success": False,
        "call_audit": [] if policy == "DeterministicBaseline" else [call],
        "fallback_actions": 0,
        "budget_cap_hits": 0,
        "input_tokens": 10 if policy != "DeterministicBaseline" else 0,
        "output_tokens": 2 if policy != "DeterministicBaseline" else 0,
        "total_tokens": 12 if policy != "DeterministicBaseline" else 0,
        "latency_per_call_sec": [0.1] if policy != "DeterministicBaseline" else [],
        "parser_error_type_counts": {},
        "parsed_action_counts": {"move": 1} if policy != "DeterministicBaseline" else {},
    }


def episode_runner(scenario, seed, policy_class, client, args, trace_sink, episode_id):
    policy = "DeterministicBaseline" if policy_class.__name__ == "DeterministicBaselinePolicy" else policy_class.name
    if policy != "DeterministicBaseline":
        client.generate("episode prompt", model=args.model)
    trace_sink.append({
        "episode_id": episode_id, "scenario": scenario, "policy": policy, "seed": seed,
        "event": {"turn": 1, "event": "action_result", "agent": "A", "details": {
            "action": "move", "target": "red_room", "action_valid": True,
        }},
    })
    return fake_summary(scenario, seed, policy, episode_id)


@pytest.fixture(autouse=True)
def secret(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-secret-value")
    SequencedClient.calls = 0
    SequencedClient.fail_on = None


def test_v1_manifest_is_immutable_and_v1_1_matrix_is_unique():
    assert hashlib.sha256(harness.V1_MANIFEST_PATH.read_bytes()).hexdigest() == harness.V1_MANIFEST_SHA256
    manifest, _ = harness.load_manifest()
    parent = json.loads(harness.V1_MANIFEST_PATH.read_text())
    for field in (
        "scenarios", "policies", "seeds", "model", "temperature", "top_p",
        "max_tokens", "max_steps", "max_llm_calls_per_episode",
        "primary_functional_criteria", "primary_behavioral_metrics", "exclusion_criteria",
    ):
        assert manifest[field] == parent[field]
    matrix = harness.execution_matrix(manifest)
    assert len(matrix) == 12
    assert len(set(matrix)) == 12
    assert matrix == [
        (scenario, policy, seed)
        for scenario in manifest["scenarios"]
        for policy in manifest["policies"]
        for seed in manifest["seeds"]
    ]


def test_global_scheduler_spaces_only_api_calls():
    clock = FakeClock()
    scheduler = harness.GlobalCallScheduler(10.0, clock=clock.monotonic, sleep=clock.sleep)
    first = scheduler.before_call({"phase": "connectivity"})
    clock.value += 2.0
    second = scheduler.before_call({"phase": "episode"})
    assert second["call_start_monotonic"] - first["call_start_monotonic"] >= 10.0
    assert second["requested_sleep_seconds"] == 8.0
    assert second["actual_sleep_seconds"] == 8.0
    assert clock.sleeps == [8.0]
    # No scheduler method is invoked for deterministic environment actions.
    assert len(scheduler.records) == 2


def test_first_429_aborts_without_retry_and_preserves_partial_artifacts(tmp_path):
    clock = FakeClock()
    SequencedClient.fail_on = 3  # connectivity, first LLM response, then first 429
    status, output_dir = harness.execute(
        output_root=tmp_path,
        client_factory=SequencedClient,
        episode_runner=episode_runner,
        clock=clock.monotonic,
        sleep=clock.sleep,
        timestamp="20260101T000000Z",
    )
    assert status == 5
    assert SequencedClient.calls == 3
    metadata = json.loads((output_dir / "execution_metadata.json").read_text())
    assert metadata["run_status"] == "rate_limit_abort"
    assert metadata["episodes_completed"] == 2
    expected = set(harness.REQUIRED_OUTPUT_FILES + harness.EXTRA_OUTPUT_FILES)
    assert expected <= {path.name for path in output_dir.iterdir()}
    schedule = json.loads((output_dir / "api_call_schedule.json").read_text())
    assert len(schedule) == 3
    assert all(
        later["call_start_monotonic"] - earlier["call_start_monotonic"] >= 10.0
        for earlier, later in zip(schedule, schedule[1:])
    )
    failure = json.loads((output_dir / "api_failure_audit.json").read_text())[0]
    assert failure["http_status"] == 429
    assert failure["run_aborted"] is True
    assert failure["limit"] == 6000
    assert "org_private123" not in json.dumps(failure)
    assert "private-token" not in json.dumps(failure)
    assert "test-secret-value" not in "".join(path.read_text() for path in output_dir.iterdir())


def test_corrected_rates_have_distinct_denominators_and_milestones_are_objective():
    row = fake_summary("C5b_costly_false_belief", 0, "LLMReactiveReal", "episode-x")
    records = [
        {"success": True},
        {"success": False},
    ]
    traces = [
        {"episode_id": "episode-x", "event": {"event": "action_result", "agent": "A", "details": {"action": "open_box", "action_valid": True}}},
        {"episode_id": "episode-x", "event": {"event": "action_result", "agent": "B", "details": {"action": "open_box", "action_valid": True}}},
        {"episode_id": "episode-x", "event": {"event": "action_result", "agent": "C", "details": {"action": "pickup", "target": "medical_kit", "action_valid": True}}},
    ]
    metrics = harness.corrected_metrics(row, records, traces)
    assert metrics["api_response_success_rate"] == 0.5
    assert metrics["parser_success_given_api_response"] == 1.0
    assert metrics["end_to_end_parseable_action_rate"] == 0.5
    assert metrics["valid_non_stay_model_actions"] == 1
    assert metrics["objective_task_milestones_reached"] == {
        "red_key_applied": True,
        "blue_key_applied": True,
        "locked_box_open": True,
        "medical_kit_acquired": True,
        "victim_rescued": False,
    }
    assert "progress" not in json.dumps(metrics).lower()


def test_api_failure_requires_manual_causal_audit():
    row = fake_summary("C5b_costly_false_belief", 0, "LLMReactiveReal", "episode-x")
    decision = harness._exclusion(row, [{"success": False, "call_order": 2}])
    assert decision["excluded"] is None
    assert decision["exclusion_status"] == "requires_manual_audit"


def test_workflow_is_future_guarded_and_always_uploads():
    workflow = Path(".github/workflows/frozen_real_llm_micro_pilot_v1_1.yml").read_text()
    future_message = "Run rate-aware functional pilot v1.1 [run-rate-aware-pilot-v1.1]"
    assert future_message in workflow
    assert future_message != "Prepare rate-aware functional pilot v1.1"
    assert "secrets.GROQ_API_KEY" in workflow
    assert "python -m btom_v2.run_frozen_micro_pilot_v1_1" in workflow
    assert workflow.count("python -m btom_v2.run_frozen_micro_pilot_v1_1") == 1
    assert "if: always()" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "timeout-minutes: 120" in workflow
