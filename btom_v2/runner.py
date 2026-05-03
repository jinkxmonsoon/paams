from __future__ import annotations
from dataclasses import asdict
from .env import BTomEnvV2


def run_c4_smoke(seed: int):
    env = BTomEnvV2("C4_communication_delay", seed=seed, max_turns=12)
    # tiny smoke: send one message then wait/move
    plan = [
        ("A", "send_message", "C|box_info"),
        ("B", "move", "blue_room"),
        ("C", "move", "staging"),
        ("A", "move", "red_room"),
        ("B", "move", "staging"),
        ("C", "move", "staging"),
    ]
    i = 0
    while not env.state.done and env.state.turn < env.max_turns:
        a, act, tgt = plan[i % len(plan)]
        env.step(a, act, tgt, {"agent_goal": "smoke", "target_object": None, "target_room": tgt, "action_reason": "smoke", "belief_used_for_action": None})
        i += 1
        if env.state.turn >= env.max_turns:
            env.state.done = True

    summ = asdict(env.summary())
    sent = [e for e in env.state.trace if e.event == "message_sent"]
    delivered = [e for e in env.state.trace if e.event == "message_delivery" and e.details.get("details",{}).get("delivered")]
    first_sent = sent[0].turn if sent else None
    # find first non-empty delivered list
    first_delivery = None
    for e in env.state.trace:
        if e.event == "message_delivery":
            d = e.details.get("details", {}).get("delivered", [])
            if d:
                first_delivery = e.turn
                break
    # hidden before delivery check (proxy): delivery strictly after first send
    hidden_ok = (first_delivery is not None and first_sent is not None and first_delivery > first_sent)
    row = {
        "scenario_id": summ["scenario_id"], "seed": seed, "success": summ["success"], "turns": summ["turns"],
        "delayed_messages_count": summ["delayed_messages_count"], "delivered_delayed_messages_count": summ["delivered_delayed_messages_count"],
        "pending_messages_final_count": summ["pending_messages_final_count"], "first_sent_step": first_sent, "first_delivery_step": first_delivery,
        "pending_hidden_before_delivery": hidden_ok,
    }
    return row


def main():
    rows = [run_c4_smoke(s) for s in (0,1,2)]
    for r in rows:
        print(r)

    assert rows, "C4 runs"
    assert all(r["delayed_messages_count"] > 0 for r in rows)
    assert all(r["delivered_delayed_messages_count"] > 0 for r in rows)
    assert all((r["first_delivery_step"] or 0) > (r["first_sent_step"] or 0) for r in rows)
    assert all(r["pending_hidden_before_delivery"] for r in rows)
    print("sanity_checks=PASS")

if __name__ == "__main__":
    main()
