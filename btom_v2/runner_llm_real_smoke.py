from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import asdict

from .env import BTomEnvV2
from .llm_policy import LLMPolicyAdapter, ValidMockLLMClient
from .policies import DeterministicBaselinePolicy
from .real_llm_clients import GroqClient, OllamaClient, RealLLMSetupError, TransformersLocalClient


OUTDIR = "btom_v2/outputs"


class LLMReactiveReal(LLMPolicyAdapter):
    name = "LLMReactiveReal"
    variant = "LLMReactive"


class LLMBeliefStateReal(LLMPolicyAdapter):
    name = "LLMBeliefStateReal"
    variant = "LLMBeliefState"


class LLMBToMReal(LLMPolicyAdapter):
    name = "LLMBToMReal"
    variant = "LLMBToM"


POLICY_MAP = {
    "DeterministicBaseline": DeterministicBaselinePolicy,
    LLMReactiveReal.name: LLMReactiveReal,
    LLMBeliefStateReal.name: LLMBeliefStateReal,
    LLMBToMReal.name: LLMBToMReal,
}


def select_client(args):
    if args.backend == "mock":
        return ValidMockLLMClient()
    if args.backend == "groq":
        return GroqClient(model=args.model or "llama-3.1-8b-instant")
    if args.backend == "ollama":
        return OllamaClient(model=args.model or "llama3.1:8b")
    if args.backend == "transformers_local":
        return TransformersLocalClient(model=args.model, allow_download=args.allow_download)
    raise ValueError(f"unsupported backend: {args.backend}")


def _trace_flag_count(env, flag):
    return sum(
        int(event.event == "action_intent" and bool(event.details.get(flag)))
        for event in env.state.trace
    )


def run_episode(
    scenario,
    seed,
    policy_class,
    client,
    args,
    trace_sink=None,
    episode_id=None,
    partial_sink=None,
):
    env = BTomEnvV2(scenario, seed, max_turns=args.max_steps)
    is_llm = issubclass(policy_class, LLMPolicyAdapter)
    if is_llm:
        policy = policy_class(
            client=client,
            budget_kwargs={"max_llm_calls": args.max_llm_calls},
            llm_kwargs={
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
        )
    else:
        policy = policy_class()

    baseline_valid = 0
    baseline_invalid = 0
    progressed_agents = set()
    try:
        for turn in range(env.max_turns):
            for agent in ("A", "B", "C"):
                action, target, meta = policy.act(env, agent, turn)
                result = env.step(agent, action, target, meta)
                if not result.invalid and env.state.locations[agent] != "staging":
                    progressed_agents.add(agent)
                if is_llm:
                    policy.record_step_result(result, meta)
                    if policy.call_audit:
                        policy.call_audit[-1]["environment_result"] = {
                            "valid": not result.invalid,
                            "success": result.success,
                            "done": result.done,
                            "reason": result.reason,
                        }
                elif result.invalid:
                    baseline_invalid += 1
                else:
                    baseline_valid += 1
                if env.state.done:
                    break
            if env.state.done:
                break
    except BaseException as abort:
        if partial_sink is not None:
            partial_sink.append({
                "episode_id": episode_id,
                "scenario": scenario,
                "policy": "DeterministicBaseline" if not is_llm else policy_class.name,
                "seed": seed,
                "completed": False,
                "abort_type": type(abort).__name__,
                "abort_api_call_order": getattr(abort, "audit_record", {}).get("call_order"),
                "call_audit": list(policy.call_audit) if is_llm else [],
                "environment_trace": [asdict(event) for event in env.state.trace],
                "current_turn": env.state.turn,
                "current_task_status": dict(env.state.task_status),
                "current_agent_locations": dict(env.state.locations),
                "current_inventories": {agent: list(items) for agent, items in env.state.inventories.items()},
                "policy_budget_counters": policy.budget.as_dict() if is_llm else {},
                "parser_error_type_counts": dict(policy.parser_error_type_counts) if is_llm else {},
                "parsed_action_counts": dict(policy.parsed_action_counts) if is_llm else {},
                "api_failure_count_before_abort": policy.api_failures if is_llm else 0,
            })
        raise
    finally:
        if trace_sink is not None:
            identity = {
                "episode_id": episode_id,
                "scenario": scenario,
                "policy": "DeterministicBaseline" if not is_llm else policy_class.name,
                "seed": seed,
            }
            trace_sink.extend({**identity, "event": asdict(event)} for event in env.state.trace)

    summary = asdict(env.summary())
    summary["policy"] = "DeterministicBaseline" if not is_llm else policy_class.name
    total_steps = baseline_valid + baseline_invalid
    if is_llm:
        summary.update(policy.budget.as_dict())
        llm_calls = policy.budget.llm_calls
        parsed = sum(policy.parsed_action_counts.values())
        attempted_env_actions = policy.environment_valid_actions + policy.environment_invalid_actions
        summary.update({
            "api_success_rate": None if llm_calls == 0 else (llm_calls - policy.api_failures) / llm_calls,
            "parse_success_rate": None if llm_calls == 0 else parsed / llm_calls,
            "environment_action_valid_rate": None if attempted_env_actions == 0 else policy.environment_valid_actions / attempted_env_actions,
            "parser_error_type_counts": dict(policy.parser_error_type_counts),
            "parsed_action_counts": dict(policy.parsed_action_counts),
            "api_failure_count": policy.api_failures,
            "environment_invalid_action_count": policy.environment_invalid_actions,
            "valid_but_strategically_poor_actions": policy.valid_but_strategically_poor_actions,
            "progress_actions": policy.progress_actions,
            "agents_progressing_beyond_staging": sorted(progressed_agents),
            "input_tokens": policy.input_tokens,
            "output_tokens": policy.output_tokens,
            "total_tokens": policy.total_tokens,
            "latency_per_call_sec": policy.latencies,
            "mean_latency_sec": None if not policy.latencies else sum(policy.latencies) / len(policy.latencies),
            "call_audit": policy.call_audit,
            "raw_model_error_types": dict(Counter(
                record["raw_model_error_type"]
                for record in policy.call_audit
                if record["raw_model_error"]
            )),
        })
    else:
        summary.update({
            "llm_calls": 0,
            "input_tokens_approx": 0,
            "output_tokens_approx": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "parse_failures": 0,
            "invalid_actions_from_llm": 0,
            "budget_cap_hits": 0,
            "invalid_targets": 0,
            "fallback_actions": 0,
            "api_success_rate": None,
            "parse_success_rate": None,
            "environment_action_valid_rate": None if total_steps == 0 else baseline_valid / total_steps,
            "parser_error_type_counts": {},
            "parsed_action_counts": {},
            "api_failure_count": 0,
            "environment_invalid_action_count": baseline_invalid,
            "valid_but_strategically_poor_actions": None,
            "progress_actions": None,
            "agents_progressing_beyond_staging": sorted(progressed_agents),
            "latency_per_call_sec": [],
            "mean_latency_sec": None,
            "call_audit": [],
            "raw_model_error_types": {},
        })
    summary.update({
        "false_belief_driven_decoy_pursuits": _trace_flag_count(env, "false_belief_driven_decoy_pursuit"),
        "post_conflict_false_belief_pursuits": _trace_flag_count(env, "post_conflict_false_belief_pursuit"),
        "post_conflict_decoy_dwell_steps": _trace_flag_count(env, "post_conflict_decoy_dwell_step"),
        "final_task_status": dict(env.state.task_status),
        "final_agent_locations": dict(env.state.locations),
    })
    return summary


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def _nullable_mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return None if not values else sum(values) / len(values)


def group(rows, scenario, policy):
    selected = [row for row in rows if row["scenario_id"] == scenario and row["policy"] == policy]
    return {
        "scenario_id": scenario,
        "policy": policy,
        "N": len(selected),
        "success_rate": _mean(selected, "success"),
        "mean_turns": _mean(selected, "turns"),
        "mean_time_to_rescue": _nullable_mean(selected, "time_to_rescue"),
        "mean_llm_calls": _mean(selected, "llm_calls"),
        "api_success_rate": _nullable_mean(selected, "api_success_rate"),
        "parse_success_rate": _nullable_mean(selected, "parse_success_rate"),
        "environment_action_valid_rate": _nullable_mean(selected, "environment_action_valid_rate"),
        "mean_parse_failures": _mean(selected, "parse_failures"),
        "mean_invalid_targets": _mean(selected, "invalid_targets"),
        "mean_fallback_actions": _mean(selected, "fallback_actions"),
        "mean_progress_actions": _nullable_mean(selected, "progress_actions"),
        "mean_valid_but_strategically_poor_actions": _nullable_mean(selected, "valid_but_strategically_poor_actions"),
        "mean_budget_cap_hits": _mean(selected, "budget_cap_hits"),
        "mean_false_belief_driven_decoy_pursuits": _mean(selected, "false_belief_driven_decoy_pursuits"),
        "mean_post_conflict_false_belief_pursuits": _mean(selected, "post_conflict_false_belief_pursuits"),
        "mean_wrong_branch_steps": _mean(selected, "wrong_branch_steps"),
        "mean_delayed_message_confusion_events": _mean(selected, "delayed_message_confusion_events"),
        "mean_premature_shared_memory_assumptions": _mean(selected, "premature_shared_memory_assumptions"),
        "mean_second_order_delivery_waits": _mean(selected, "second_order_delivery_waits"),
        "mean_correction_opportunities": _mean(selected, "correction_opportunities"),
        "mean_necessary_correction_messages": _mean(selected, "necessary_correction_messages"),
        "mean_missed_necessary_corrections": _mean(selected, "missed_necessary_corrections"),
        "mean_unnecessary_correction_messages": _mean(selected, "unnecessary_correction_messages"),
        "mean_evidence_to_correction_send_steps": _nullable_mean(selected, "evidence_to_correction_send_steps"),
        "mean_correction_delivery_latency": _nullable_mean(selected, "correction_delivery_latency"),
        "mean_target_stale_belief_steps": _mean(selected, "target_stale_belief_steps"),
        "mean_target_decoy_branch_steps": _mean(selected, "target_decoy_branch_steps"),
        "mean_post_correction_decoy_steps": _mean(selected, "post_correction_decoy_steps"),
        "mean_total_messages": _mean(selected, "total_messages"),
        "mean_input_tokens": _mean(selected, "input_tokens"),
        "mean_output_tokens": _mean(selected, "output_tokens"),
        "mean_total_tokens": _mean(selected, "total_tokens"),
        "mean_latency_sec": _nullable_mean(selected, "mean_latency_sec"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mock", "groq", "ollama", "transformers_local"], default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--scenario", default="C5b_costly_false_belief")
    parser.add_argument("--policies", default="LLMReactiveReal,LLMBToMReal")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-llm-calls", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    scenarios = [item for item in args.scenario.split(",") if item]
    policy_classes = [POLICY_MAP[item.strip()] for item in args.policies.split(",") if item.strip() in POLICY_MAP]
    seeds = [int(item) for item in args.seeds.split(",") if item]
    needs_client = any(issubclass(policy, LLMPolicyAdapter) for policy in policy_classes)
    try:
        client = select_client(args) if needs_client else None
        skipped = False
        skip_reason = ""
    except RealLLMSetupError as error:
        client = None
        skipped = True
        skip_reason = error.reason

    print("LLM_REAL_PILOT_CONFIG")
    print(vars(args))
    rows = []
    if not skipped:
        for scenario in scenarios:
            for policy in policy_classes:
                for seed in seeds:
                    rows.append(run_episode(scenario, seed, policy, client, args))
    grouped = [group(rows, scenario, policy.name if policy is not DeterministicBaselinePolicy else "DeterministicBaseline") for scenario in scenarios for policy in policy_classes] if rows else []

    logs = f"{OUTDIR}/llm_real_pilot_logs.jsonl"
    metrics = f"{OUTDIR}/llm_real_pilot_metrics.csv"
    audit_path = f"{OUTDIR}/llm_real_pilot_call_audit.jsonl"
    summary_path = f"{OUTDIR}/llm_real_pilot_summary.json"
    audits = [record for row in rows for record in row["call_audit"]]
    with open(logs, "w") as handle:
        if rows:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        else:
            handle.write(json.dumps({"skipped": True, "skip_reason": skip_reason}) + "\n")
    with open(audit_path, "w") as handle:
        for record in audits:
            handle.write(json.dumps(record) + "\n")
    with open(metrics, "w", newline="") as handle:
        if grouped:
            writer = csv.DictWriter(handle, fieldnames=list(grouped[0]))
            writer.writeheader()
            writer.writerows(grouped)

    summary = {
        "backend": args.backend,
        "transport": getattr(client, "transport", "mock" if args.backend == "mock" else None),
        "model": args.model or "default",
        "skipped": skipped,
        "skip_reason": skip_reason,
        "configuration": {
            "scenarios": scenarios,
            "policies": ["DeterministicBaseline" if policy is DeterministicBaselinePolicy else policy.name for policy in policy_classes],
            "seeds": seeds,
            "max_llm_calls": args.max_llm_calls,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "grouped_metrics": grouped,
        "episodes": rows,
        "limitations": [
            "Functional micro-pilot only; not a hypothesis-validation experiment.",
            "One seed cannot support H1 or H2.",
            "Executable parsed actions do not establish autonomous planning or Theory-of-Mind capability.",
        ],
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("LLM_REAL_PILOT_GROUPED_METRICS")
    print(json.dumps(grouped, indent=2))
    print("LLM_CALL_AUDIT_FIRST_FIVE_PER_CONDITION")
    print(json.dumps({policy: [record for record in audits if record["policy"] == policy][:5] for policy in (LLMReactiveReal.name, LLMBeliefStateReal.name, LLMBToMReal.name)}, indent=2))
    print("LLM_REAL_PILOT_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))
    print("OUTPUT_FILES", logs, metrics, audit_path, summary_path)


if __name__ == "__main__":
    main()
