from __future__ import annotations

from dataclasses import asdict

from .env import BTomEnvV2
from .policies import DeterministicBaselinePolicy


def run_episode(scenario_id: str, seed: int):
    env = BTomEnvV2(scenario_id=scenario_id, seed=seed)
    policy = DeterministicBaselinePolicy()
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
    c5 = [e for e in episodes if e.state.scenario_id == "C5_false_belief_injection"]
    assert all(e.state.success and e.state.invalid_actions == 0 for e in c1)
    assert all(e.state.success and e.state.invalid_actions == 0 for e in c2)
    assert any(e.state.success for e in c5)
    assert all(e.state.false_belief_injections > 0 for e in c5)
    assert any(e.state.belief_conflict_count > 0 for e in c5)
    assert any(e.state.false_belief_caused_wasted_action > 0 for e in c5)
    assert any(e.state.task_status["victim_rescued"] for e in c5)


def main():
    episodes = []
    for scenario_id in ("C1_fully_observable", "C2_partial_observable", "C5_false_belief_injection"):
        for seed in (0, 1, 2):
            env = run_episode(scenario_id, seed)
            episodes.append(env)
            print(asdict(env.summary()))
    validate_sanity(episodes)
    print("sanity_checks=PASS")


if __name__ == "__main__":
    main()
