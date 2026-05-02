from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional


def _safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in vals if v is not None]
    return mean(nums) if nums else None


def first_order_belief_accuracy(ep: Dict[str, Any]) -> Optional[float]:
    if not ep.get("steps"):
        return None
    final = ep["steps"][-1]
    truth = final.get("global_state_snapshot", {}).get("object_locations", {})
    if not truth:
        return None
    beliefs = [s.get("agent_first_order_belief", {}) for s in ep["steps"][-3:]]
    scores = []
    for b in beliefs:
        known = b.get("known_object_locations", {})
        comp = 0
        correct = 0
        for k, v in known.items():
            if k in truth:
                comp += 1
                correct += 1 if truth[k] == v else 0
        scores.append(correct / comp if comp else None)
    return _safe_mean(scores)


def first_order_precision_coverage(ep: Dict[str, Any]) -> Dict[str, Optional[float]]:
    if not ep.get("steps"):
        return {"precision": None, "coverage": None}
    truth = ep["steps"][-1].get("global_state_snapshot", {}).get("object_locations", {})
    if not truth:
        return {"precision": None, "coverage": None}
    task_keys = {"red_key", "blue_key", "medical_kit", "victim"}
    b = ep["steps"][-1].get("agent_first_order_belief", {}).get("known_object_locations", {})
    asserted = {k: v for k, v in b.items() if k in task_keys}
    correct = sum(1 for k, v in asserted.items() if truth.get(k) == v)
    precision = (correct / len(asserted)) if asserted else None
    coverage = (len(asserted) / len(task_keys)) if task_keys else None
    return {"precision": precision, "coverage": coverage}


def second_order_belief_accuracy(ep: Dict[str, Any]) -> Optional[float]:
    rec = ep.get("policy_records", {}).get("second_order_beliefs")
    if not rec:
        return None
    # symbolic accuracy proxy: percentage of entries that map to string room labels.
    total = 0
    ok = 0
    for agent_data in rec.values():
        for other_data in agent_data.values():
            for _, state_val in other_data.items():
                state = state_val.get("state") if isinstance(state_val, dict) else "unknown"
                val = state_val.get("value") if isinstance(state_val, dict) else None
                if state == "unknown":
                    continue
                total += 1
                if state == "knows" and isinstance(val, str) and val.startswith("room_"):
                    ok += 1
                if state == "does_not_know":
                    ok += 1
    return (ok / total) if total else None


def second_order_coverage(ep: Dict[str, Any]) -> Optional[float]:
    rec = ep.get("policy_records", {}).get("second_order_beliefs")
    if not rec:
        return None
    total = 0
    non_unknown = 0
    for a in rec.values():
        for o in a.values():
            for v in o.values():
                total += 1
                if isinstance(v, dict) and v.get("state") != "unknown":
                    non_unknown += 1
    return (non_unknown / total) if total else None


def aggregate_metrics(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        grouped[(ep["scenario_id"], ep["variant"])].append(ep)

    rows = []
    for (scenario, variant), eps in grouped.items():
        total_steps = sum(len(ep.get("steps", [])) for ep in eps)
        total_invalid = sum(ep.get("total_invalid_actions", 0) for ep in eps)
        msg_tokens = 0
        f1 = []
        f2 = []
        f1p, f1c, f2c = [], [], []
        comm_use = []
        for ep in eps:
            for st in ep.get("steps", []):
                msg = st.get("message_sent")
                if msg and msg.get("text"):
                    msg_tokens += len(msg["text"].split())
            f1.append(first_order_belief_accuracy(ep))
            f2.append(second_order_belief_accuracy(ep))
            pc = first_order_precision_coverage(ep)
            f1p.append(pc["precision"])
            f1c.append(pc["coverage"])
            f2c.append(second_order_coverage(ep))
            sent = ep.get("policy_records", {}).get("sent_message_cache_size")
            useful = ep.get("policy_records", {}).get("useful_messages_estimate")
            comm_use.append((useful / sent) if sent else 0.0)

        conflict_total = sum(ep.get("belief_conflict_count", 0) for ep in eps)
        resolved = sum(ep.get("resolved_belief_conflict_count", 0) for ep in eps)

        rows.append(
            {
                "scenario_id": scenario,
                "variant": variant,
                "N": len(eps),
                "task_success_rate": mean([1.0 if ep.get("success") else 0.0 for ep in eps]),
                "mean_normalized_team_score": mean([ep.get("normalized_team_score", 0.0) for ep in eps]),
                "mean_turns_to_completion": mean([ep.get("turns_to_completion", 0) for ep in eps]),
                "invalid_action_rate": (total_invalid / total_steps) if total_steps else 0.0,
                "mean_messages_per_episode": mean([ep.get("total_messages", 0) for ep in eps]),
                "mean_message_tokens_approx": msg_tokens / len(eps),
                "mean_first_order_belief_accuracy": _safe_mean(f1),
                "mean_second_order_belief_accuracy": _safe_mean(f2),
                "first_order_precision": _safe_mean(f1p),
                "first_order_coverage": _safe_mean(f1c),
                "second_order_precision": _safe_mean(f2),
                "second_order_coverage": _safe_mean(f2c),
                "communication_usefulness_proxy": _safe_mean(comm_use),
                "belief_conflict_resolution_rate": (resolved / conflict_total) if conflict_total else 0.0,
            }
        )
    return rows


def parse_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out
