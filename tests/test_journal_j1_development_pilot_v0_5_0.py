"""Pre-execution gates for the Journal J1 v0.5.0 development pilot."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from btom_v2 import journal_j1_development_prompting_v0_4_2 as prompting
from btom_v2 import run_journal_j1_development_pilot_v0_5_0 as pilot

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(pilot.MANIFEST_PATH.read_text())


def selected_prompts():
    return [row for row in prompting.build_prospective()[2] if row["candidate_id"] == 1]


def test_prompt_freeze_and_manifest_configuration():
    assert MANIFEST["prompt_freeze"]["immutable_output_sha256"] == pilot.PROMPT_FREEZE_HASHES
    assert MANIFEST["prompt_freeze"]["selected_candidate_id"] == 1
    assert MANIFEST["prompt_freeze"]["selected_R_label"] == "Current partner record:"
    assert MANIFEST["prompt_freeze"]["selected_B_label"] == "Current partner belief:"
    assert MANIFEST["status"] == "exploratory_development_scenario_discrimination"
    api = MANIFEST["api"]
    assert (MANIFEST["model"], api["temperature"], api["top_p"], api["max_completion_tokens"]) == ("openai/gpt-oss-20b", 0, 1, 1024)
    assert api["reasoning_effort"] == "low" and api["include_reasoning"] is False and api["stream"] is False
    assert api["behavioral_retries"] == 0 and api["delay_seconds"] == 20 and api["max_behavioral_requests"] == 144


def test_request_order_pair_seeds_and_balance():
    prompts = selected_prompts(); first = pilot.request_order(prompts); second = pilot.request_order(prompts)
    assert first == second == MANIFEST["frozen_request_order"] and len(first) == 144
    assert len({row["variant_id"] for row in first}) == 18
    pairs = [first[i:i + 2] for i in range(0, 144, 2)]
    assert len(pairs) == 72 and all(a["variant_id"] == b["variant_id"] and (a["M_R"], a["M_I"]) == (b["M_R"], b["M_I"]) and a["requested_seed"] == b["requested_seed"] and {a["F"], b["F"]} == {"R", "B"} for a, b in pairs)
    assert len({pair[0]["requested_seed"] for pair in pairs}) == 72
    assert Counter(pair[0]["F"] for pair in pairs) == {"R": 36, "B": 36}
    by_variant = defaultdict(Counter)
    for pair in pairs:
        by_variant[pair[0]["variant_id"]][pair[0]["F"]] += 1
    assert len(by_variant) == 18 and all(counts == {"R": 2, "B": 2} for counts in by_variant.values())


def test_seed_algorithm_exact():
    for key, expected in MANIFEST["pair_seeds"].items():
        variant, mr, mi = key.split("|")
        digest = hashlib.sha256(f"{pilot.PAIR_SEED_LITERAL}|{variant}|{mr}|{mi}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") % 2147483647 or 1
        assert pilot.pair_seed(variant, int(mr), int(mi)) == expected == value


def test_strict_schema_parser_and_no_fallback():
    schema = pilot.response_schema()["json_schema"]["schema"]
    assert schema == MANIFEST["api"]["strict_response_schema"]
    assert set(schema["properties"]) == {"action"} and schema["additionalProperties"] is False
    assert pilot.strict_parse('{"action":"SEND_CORRECTION"}') == "SEND_CORRECTION"
    assert pilot.strict_parse('{"action":"CONTINUE_TASK"}') == "CONTINUE_TASK"
    for invalid in ('SEND_CORRECTION', '{"action":"BAD"}', '{"action":"SEND_CORRECTION","reason":"x"}', 'prefix {"action":"SEND_CORRECTION"}'):
        assert pilot.strict_parse(invalid) is None
    body = pilot.request_body("prompt", 7)
    assert body["messages"] == [{"role": "user", "content": "prompt"}] and "tools" not in body
    assert not {"rationale", "reason", "explanation"} & schema["properties"].keys()


def passing_rates():
    return {(f, mr, mi): float(mr) for f in "RB" for mr in (0, 1) for mi in (0, 1)}


def test_all_five_discrimination_criteria_and_synthetic_pass():
    result = pilot.discrimination(passing_rates(), True)
    assert result["record_relevance_pooled"] == result["record_relevance_MI0"] == result["record_relevance_MI1"] == 1
    assert result["record_irrelevant_MR0"] == result["record_irrelevant_MR1"] == 0
    assert result["criteria"] == {"record_relevance_pooled_gte_0_25": True, "record_relevance_MI0_gt_0": True, "record_relevance_MI1_gt_0": True, "record_irrelevant_MR0_abs_lte_0_25": True, "record_irrelevant_MR1_abs_lte_0_25": True}
    assert result["scenario_discrimination_pass"]


@pytest.mark.parametrize("rates,key", [
    ({("R", 0, 0): 0.4, ("R", 0, 1): 0.4, ("R", 1, 0): 0.5, ("R", 1, 1): 0.5}, "record_relevance_pooled_gte_0_25"),
    ({("R", 0, 0): 0.5, ("R", 0, 1): 0, ("R", 1, 0): 0.5, ("R", 1, 1): 1}, "record_relevance_MI0_gt_0"),
    ({("R", 0, 0): 0, ("R", 0, 1): 0.5, ("R", 1, 0): 1, ("R", 1, 1): 0.5}, "record_relevance_MI1_gt_0"),
    ({("R", 0, 0): 0, ("R", 0, 1): 0.5, ("R", 1, 0): 1, ("R", 1, 1): 1}, "record_irrelevant_MR0_abs_lte_0_25"),
    ({("R", 0, 0): 0, ("R", 0, 1): 0, ("R", 1, 0): 0.5, ("R", 1, 1): 1}, "record_irrelevant_MR1_abs_lte_0_25"),
])
def test_each_discrimination_gate_can_fail(rates, key):
    rates.update({("B", mr, mi): 0 for mr in (0, 1) for mi in (0, 1)})
    result = pilot.discrimination(rates, True)
    assert result["criteria"][key] is False and result["scenario_discrimination_pass"] is False


def test_technical_failure_prevents_scenario_pass():
    result = pilot.discrimination({}, False)
    assert result["scenario_discrimination_pass"] is False and result["criteria"] == {}


def test_preclient_failure_does_not_construct_client(tmp_path):
    called = False
    def factory(_key):
        nonlocal called; called = True
        raise AssertionError
    assert pilot.execute(tmp_path / "output", tmp_path / "missing", sleep=lambda _: None, client_factory=factory, environ={"GROQ_API_KEY": "secret"}) == 1
    assert called is False
    status = json.loads((tmp_path / "output/workflow_status.json").read_text())
    assert not status["technical_pass"] and status["behavioral_observations"] == 0


def test_no_inference_confirmatory_execution_or_secret_persistence():
    source = Path(pilot.__file__).read_text().lower()
    tree = ast.parse(source)
    assert all(term not in source for term in ("scipy", "bootstrap", "holm", "p_value", "p-value"))
    assert MANIFEST["inference"] == {"bootstrap": False, "confidence_intervals": False, "confirmatory_significance_claims": False, "holm": False, "p_values": False}
    assert MANIFEST["confirmatory_prompts_generated"] is False and MANIFEST["icaart_reopened"] is False
    assert "groq_api_key_secondary" not in source
    assert pilot.sanitize("Bearer secret GROQ_API_KEY", "secret") == "[REDACTED][REDACTED] [REDACTED]"
