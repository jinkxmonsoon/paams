# Mock-only decision-point design

## Scientific question and hypotheses

Can action-relevant first- and second-order belief manipulations be evaluated
at isolated decision points without requiring completion of the rescue chain?
The scaffold prepares later tests of H1a (first-order representation changes
action selection), H1b/H3 (a false belief can select a worse legal branch), and
H2 (second-order representation changes necessary versus unnecessary
correction). Scripted audits in this scaffold test only instrument sensitivity,
not these hypotheses.

## Why the full-chain run was insufficient

The immutable v1.1 run established rate-aware technical execution but yielded
no LLM task success and no C7 correction opportunity. Full-chain navigation,
resource acquisition, parser interaction, and repeated decisions obscured the
specific epistemic decision points. The pickup defect further compromised one
C7 episode. Isolating one decision removes mission-completion as a prerequisite
without retrospectively changing the frozen run.

## Independent and dependent variables

For DP5, the independent variable is the explicit first-order value of
`medical_kit_location` (`decoy_room` versus `box_room`) while physical state,
visible observation, actions, ordering, and scoring remain fixed. Dependent
variables are productive-, decoy-, neutral-branch, and invalid/unparseable
selection indicators.

For DP7, the independent variable is C's communicated and explicitly modeled
expected kit location (`decoy_room` versus `box_room`). The raw statement and
nested representation vary together; correction necessity is their derived
scoring consequence. Dependent variables are necessary correction, missed
necessary correction, unnecessary correction, appropriate no-correction, and
invalid/unparseable indicators.

## Controls

All paired cases require identical physical state, direct observation, action
order, mission rules, acting agent, and one-decision termination. Raw delivered
evidence must be available to every representation arm. The future arms are:

1. `reactive_no_explicit_belief`
2. `reactive_neutral_first_order_matched`
3. `explicit_first_order`
4. `first_order_neutral_second_order_matched`
5. `explicit_second_order`

The neutral first- and second-order arms are mandatory prompt-length,
block-structure, and added-context controls. Real execution is forbidden until
these controls and their prompt-layer audits are implemented.

## ICAART contribution and journal-relevant extension

The ICAART-level contribution would be a transparent paired instrument showing
whether explicit epistemic blocks alter legal action outputs at matched
decision points. A journal-relevant extension requires multiple models and
seeds, preregistered matched prompts, repeated isolated propositions, causal
mediation analyses, and robustness to raw-message reconstruction strategies.

## Confounds and information boundaries

Prompt length and salience may explain representation-arm differences. A model
may reconstruct C's belief directly from the raw message, reducing the contrast
with explicit second-order structure. Hidden global truth is kept outside the
visible observation, and nested beliefs are reconstructed only from delivered
evidence. No arm may delegate to a deterministic baseline. Model explanations
or reasons are not evidence of cognition.

The adjacency-constrained action set and one-decision design improve internal
diagnosis but limit ecological and temporal validity. No future reward, message
delivery, or full-chain success is fabricated.

## Future technical metrics and exclusions

Future execution must distinguish API response success, parser success given an
API response, direct-action validity, token use, prompt characters, latency,
and fallback count. API failure, parser failure, illegal action, missing
prompt-layer audit, and missing decision record are separate technical
exclusion categories. A behaviorally poor but technically valid action is not
excluded. Parser failure must never be interpreted as strategic failure.

## Readiness criteria

Model, seed count, call budget, and token budget remain undecided. The scaffold
becomes eligible for a separately authorized real protocol only after matched
neutral prompts are implemented, prompt lengths and layers are audited, action
serialization is tested without fallback, exclusion machinery is frozen,
sample size is justified, and all information-boundary tests pass. Metrics may
not be added after seeing real results without a declared amendment.

This scaffold authorizes no API call or experiment and supports no claim about
H1, H2, H3, Theory of Mind, coordination improvement, or autonomous planning.
