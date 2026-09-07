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


def episode_runner(
    scenario, seed, policy_class, client, args, trace_sink, episode_id, partial_sink=None
):
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


def test_scientific_manifest_mismatch_stops_before_client_construction(tmp_path):
    manifest = json.loads(harness.MANIFEST_PATH.read_text())
    manifest["temperature"] = 0.5
    altered = tmp_path / "altered_manifest.json"
    altered.write_text(json.dumps(manifest))

    class MustNotConstruct:
        def __init__(self, **kwargs):
            pytest.fail("client constructed before manifest freeze validation")

    with pytest.raises(AssertionError, match="temperature"):
        harness.execute(
            output_root=tmp_path,
            client_factory=MustNotConstruct,
            manifest_path=altered,
        )
    assert list(tmp_path.iterdir()) == [altered]


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
    SequencedClient.fail_on = 4  # connectivity, two successful episode calls, then 429
    status, output_dir = harness.execute(
        output_root=tmp_path,
        client_factory=SequencedClient,
        clock=clock.monotonic,
        sleep=clock.sleep,
        timestamp="20260101T000000Z",
    )
    assert status == 5
    assert SequencedClient.calls == 4
    metadata = json.loads((output_dir / "execution_metadata.json").read_text())
    assert metadata["run_status"] == "rate_limit_abort"
    assert metadata["episodes_completed"] == 1
    expected = set(harness.REQUIRED_OUTPUT_FILES + harness.EXTRA_OUTPUT_FILES)
    assert expected <= {path.name for path in output_dir.iterdir()}
    schedule = json.loads((output_dir / "api_call_schedule.json").read_text())
    assert len(schedule) == 4
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
    partial = [
        json.loads(line)
        for line in (output_dir / "partial_episode_records.jsonl").read_text().splitlines()
    ]
    assert len(partial) == 1
    record = partial[0]
    assert record["completed"] is False
    assert record["policy"] == "LLMReactiveReal"
    assert record["abort_type"] == "RateLimitAbort"
    assert record["abort_api_call_order"] == 4
    assert len(record["call_audit"]) == 2
    assert record["environment_trace"]
    assert record["current_turn"] >= 2
    assert set(record["current_agent_locations"]) == {"A", "B", "C"}
    assert set(record["current_inventories"]) == {"A", "B", "C"}
    complete_audit = [
        json.loads(line)
        for line in (output_dir / "complete_call_audit.jsonl").read_text().splitlines()
    ]
    assert len(complete_audit) == 2
    assert all(item["completed"] is False for item in complete_audit)
    assert len({item["call"]["call_idx"] for item in complete_audit}) == 2
    traces = [
        json.loads(line)
        for line in (output_dir / "complete_environment_traces.jsonl").read_text().splitlines()
    ]
    assert traces
    assert all(item["episode_id"] == record["episode_id"] or item["policy"] == "DeterministicBaseline" for item in traces)
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


@pytest.mark.parametrize(
    ("parse_failures", "invalid_actions"),
    [(1, 0), (0, 1)],
)
def test_c7_technical_block_requires_manual_audit(parse_failures, invalid_actions):
    row = fake_summary("C7a_partner_belief_stale", 0, "LLMReactiveReal", "episode-x")
    row.update({
        "correction_opportunities": 0,
        "parse_failures": parse_failures,
        "parser_error_type_counts": {"invalid_json": parse_failures},
        "environment_invalid_action_count": invalid_actions,
    })
    row["call_audit"][0].update({
        "call_idx": 1,
        "parse_success": not bool(parse_failures),
        "environment_result": {"valid": not bool(invalid_actions)},
    })
    decision = harness._exclusion(row, [])
    assert decision["excluded"] is None
    assert decision["exclusion_status"] == "requires_manual_audit"
    assert decision["criterion"].startswith("episode never reaches the correction opportunity")
    assert decision["evidence"]["correction_opportunities"] == 0
    assert decision["evidence"]["relevant_call_orders"] == [1]


def test_c7_behavioral_task_failure_without_blockers_is_not_excluded():
    row = fake_summary("C7b_partner_belief_current", 0, "LLMBToMReal", "episode-x")
    row.update({
        "correction_opportunities": 0,
        "parse_failures": 0,
        "environment_invalid_action_count": 0,
    })
    decision = harness._exclusion(row, [])
    assert decision["excluded"] is False
    assert decision["exclusion_status"] == "not_excluded"


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
