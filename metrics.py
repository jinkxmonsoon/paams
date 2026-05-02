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


def second_order_belief_accuracy(ep: Dict[str, Any]) -> Optional[float]:
    rec = ep.get("policy_records", {}).get("second_order_beliefs")
    if not rec:
        return None
    # symbolic accuracy proxy: percentage of entries that map to string room labels.
    total = 0
    ok = 0
    for agent_data in rec.values():
        for other_data in agent_data.values():
            for _, room in other_data.items():
                total += 1
                if isinstance(room, str) and room.startswith("room_"):
                    ok += 1
    return (ok / total) if total else 0.0


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
        for ep in eps:
            for st in ep.get("steps", []):
                msg = st.get("message_sent")
                if msg and msg.get("text"):
                    msg_tokens += len(msg["text"].split())
            f1.append(first_order_belief_accuracy(ep))
            f2.append(second_order_belief_accuracy(ep))

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
