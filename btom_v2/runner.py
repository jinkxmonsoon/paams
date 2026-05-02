from __future__ import annotations

from dataclasses import asdict

from .env import BTomEnvV2
from .policies import DeterministicBaselinePolicy


def run_episode(scenario_id: str, seed: int):
    env = BTomEnvV2(scenario_id=scenario_id, seed=seed)
    policy = DeterministicBaselinePolicy()

    # fixed deterministic round-robin schedule
    for t in range(env.max_turns):
        for agent in ("A", "B", "C"):
            action, target = policy.act(env, agent, t)
            env.step(agent, action, target)
            if env.state.done:
                return env
    return env


def validate_sanity(episodes):
    c1 = [e for e in episodes if e.state.scenario_id == "C1_fully_observable"]
    c2 = [e for e in episodes if e.state.scenario_id == "C2_partial_observable"]

    assert c1, "C1 runs"
    assert c2, "C2 runs"
    assert any(e.state.success for e in c1), "At least one C1 episode succeeds"
    assert any(e.state.success for e in c2), "At least one C2 episode succeeds"

    # no role mismatch pickup success
    for e in episodes:
        for ev in e.state.trace:
            if ev.event == "pickup" and ev.details.get("item") == "red_key":
                assert ev.agent == "A"
            if ev.event == "pickup" and ev.details.get("item") == "blue_key":
                assert ev.agent == "B"
            if ev.event == "pickup" and ev.details.get("item") == "medical_kit":
                assert ev.agent == "C"

    assert any(any(ev.event == "locked_box_open" for ev in e.state.trace) for e in episodes)
    assert any(any(ev.event == "victim_rescued" for ev in e.state.trace) for e in episodes)


def main():
    episodes = []
    for scenario_id in ("C1_fully_observable", "C2_partial_observable"):
        for seed in (0, 1, 2):
            env = run_episode(scenario_id, seed)
            episodes.append(env)
            summary = asdict(env.summary())
            print(summary)

    validate_sanity(episodes)
    print("sanity_checks=PASS")


if __name__ == "__main__":
    main()
