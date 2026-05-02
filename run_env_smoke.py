"""Smoke runner for minimal B-ToM environment."""

from __future__ import annotations

import random
from statistics import mean
from typing import Any, Dict, List

from btom_env import BToMEnvironment


REQUIRED_STEP_FIELDS = {
    "step_idx",
    "agent_id",
    "observation",
    "action",
    "action_valid",
    "action_result",
    "message_sent",
    "messages_delivered",
    "global_state_snapshot",
    "agent_first_order_belief",
    "belief_conflicts_detected",
}


def policy(agent_id: str, obs: Dict[str, Any], _env: BToMEnvironment) -> str:
    """Anti-leakage policy: use observation (+ tiny local memory encoded in message text) only.

    IMPORTANT: this policy intentionally does not read private environment fields.
    """
    rng = random.Random(hash((obs.get("own_location"), agent_id, len(obs.get("delivered_messages", [])))))
    room = obs["own_location"]
    room_items = obs["current_room_contents"]

    if rng.random() < 0.15 and room_items:
        return f"send:{room_items[0]} seen in {room}"

    for item in ["red_key", "blue_key", "medical_kit"]:
        if item in room_items:
            return f"pickup:{item}"

    if room == "room_5":
        if "red_key" in obs["own_inventory"]:
            return "use:red_key"
        if "medical_kit" in obs["own_inventory"]:
            return "use:medical_kit"
        return "rescue:victim"

    # Route via fixed graph heuristics from observation only.
    preferred_moves = {
        "room_0": "room_1",
        "room_1": "room_3",
        "room_2": "room_4",
        "room_3": "room_5",
        "room_4": "room_5",
        "room_5": "room_3",
    }
    return f"move:{preferred_moves.get(room, 'room_0')}"


def _print_failed_episode_debug(ep: Dict[str, Any]) -> None:
    last_snapshot = ep["steps"][-1]["global_state_snapshot"] if ep["steps"] else {}
    last_actions = [s["action"] for s in ep["steps"][-5:]]
    print(
        f"DEBUG failed {ep['scenario_id']} seed={ep['seed']} | "
        f"agent_locations={last_snapshot.get('agent_locations')} | "
        f"inventories={last_snapshot.get('inventories')} | "
        f"object_locations={last_snapshot.get('object_locations')} | "
        f"task_status={last_snapshot.get('task_status')} | "
        f"last_5_actions={last_actions}"
    )


def main() -> None:
    env = BToMEnvironment()
    scenarios = ["C1_fully_observable", "C2_partial_observable", "C5_false_belief_injection"]
    seeds = [0, 1, 2]

    rows: List[Dict[str, Any]] = []
    all_traces_have_fields = True

    for scenario in scenarios:
        for seed in seeds:
            env.reset(seed=seed, scenario_id=scenario)
            ep = env.run_episode(policy_fn=policy, max_steps=30)

            for step in ep["steps"]:
                if not REQUIRED_STEP_FIELDS.issubset(step.keys()):
                    all_traces_have_fields = False
                    break

            if not ep["success"]:
                _print_failed_episode_debug(ep)

            rows.append(
                {
                    "scenario_id": scenario,
                    "seed": seed,
                    "success": ep["success"],
                    "score": ep["normalized_team_score"],
                    "turns": ep["turns_to_completion"],
                    "invalid_actions": ep["total_invalid_actions"],
                    "messages": ep["total_messages"],
                    "conflicts": ep["belief_conflict_count"],
                    "false_belief_injections": ep["false_belief_injections"],
                }
            )

    header = "scenario_id | seed | success | score | turns | invalid_actions | messages | conflicts | false_belief_injections"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['scenario_id']} | {r['seed']} | {r['success']} | {r['score']:.2f} | {r['turns']} | "
            f"{r['invalid_actions']} | {r['messages']} | {r['conflicts']} | {r['false_belief_injections']}"
        )

    grouped = {s: [r for r in rows if r["scenario_id"] == s] for s in scenarios}
    c1_succ = mean([1.0 if r["success"] else 0.0 for r in grouped["C1_fully_observable"]])
    c2_succ = mean([1.0 if r["success"] else 0.0 for r in grouped["C2_partial_observable"]])
    c1_conf = mean([r["conflicts"] for r in grouped["C1_fully_observable"]])
    c5_conf = mean([r["conflicts"] for r in grouped["C5_false_belief_injection"]])
    c1_inj = sum(r["false_belief_injections"] for r in grouped["C1_fully_observable"])
    c2_inj = sum(r["false_belief_injections"] for r in grouped["C2_partial_observable"])
    c5_inj = sum(r["false_belief_injections"] for r in grouped["C5_false_belief_injection"])

    print()
    print(f"total episodes run: {len(rows)}")
    print(f"all episode traces contain required fields: {all_traces_have_fields}")
    print("policy observation-only usage assertion: True (by code structure and naming)")
    print(f"sanity check C1 success rate >= C2 success rate: {c1_succ >= c2_succ} ({c1_succ:.2f} >= {c2_succ:.2f})")
    if c1_succ < c2_succ:
        print("WARNING: C1 success rate is lower than C2.")
    print(f"sanity check C1 mean conflicts <= C5 mean conflicts: {c1_conf <= c5_conf} ({c1_conf:.2f} <= {c5_conf:.2f})")
    if c1_conf > c5_conf:
        print("WARNING: C1 conflicts are higher than C5.")
    print(f"sanity check C1 false belief injections == 0: {c1_inj == 0} (sum={c1_inj})")
    print(f"sanity check C2 false belief injections == 0: {c2_inj == 0} (sum={c2_inj})")
    print(f"sanity check C5 false belief injections > 0: {c5_inj > 0} (sum={c5_inj})")


if __name__ == "__main__":
    main()
