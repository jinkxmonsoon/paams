# Prospectively frozen scenario-discrimination experiment v0.6.0

This stage validates state-contingent behavior under reference conditions before treatment effects are tested. Parent run 30372084567 showed saturated actions; H1a, H1b, and H2 remain untested.

## Semantic-input correction

H1 model-visible prompts contain only the route map, legal actions, and—where applicable—the represented room. The internal current location and state label are not exposed. Representation consistency is not objective correctness in the stale case.

H2 prompts expose an observed concrete resource location and the partner's delivered expected concrete location. The model must infer their match or mismatch. They state only the generic one-decision trade-off: an update consumes the decision, progress advances the mission, and the partner acts from its delivered location. Two variants list update first and two list progress first.

Acceptance remains prospective and reference-only: `matched_decision_record` for H1 and `self_belief_plus_message_record` for H2. At least three of four reference variants must pass. Treatment outputs are descriptive only. This is a heuristic validation gate, not significance testing, and no scientific hypothesis is tested.
