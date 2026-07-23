# Matched decision-point prompt design

## Scientific question and frozen contrasts

Can planned representation conditions be rendered so primary contrasts differ
in epistemic content while controlling prompt structure, length, visible
evidence, legal actions, and output instructions? This mock-only work tests
prompt construction, not H1, H2, or H3.

The H1 primary contrast is `explicit_first_order` versus
`reactive_neutral_first_order_matched`, evaluated as the interaction between
the DP5 false/current pair. The future estimand is the between-arm difference
in the pairwise change in decoy/productive selection.

The H2 primary contrast is `explicit_second_order` versus
`first_order_neutral_second_order_matched`, evaluated as the interaction
between the DP7 stale/current pair. The future estimand is the between-arm
difference in stale/current correction sensitivity. Secondary descriptive
references are explicit first order versus no explicit belief and explicit
second order versus explicit first order; these cannot be redesignated as
primary after results are seen.

## Condition information

- `reactive_no_explicit_belief` receives task instructions, local observation,
  raw messages, valid actions, and output format only.
- `reactive_neutral_first_order_matched` adds a structurally and
  character-matched opaque first-order block.
- `explicit_first_order` adds only the case-authorized first-order state.
- `first_order_neutral_second_order_matched` adds the informative first-order
  block and a structurally and character-matched opaque second-order block.
- `explicit_second_order` adds the same informative first-order block and only
  nested content reconstructed from the raw delivered statement.

Case names, purposes, family labels, scoring, hidden fields, condition names,
and outcome labels are excluded from model-visible prompts. Reasons may be
logged but are never evidence of cognition.

## Why neutral blocks are mandatory

Adding a block changes prompt length, visual structure, salience, and the
amount of apparent context independently of its epistemic content. The two
neutral arms preserve keys, ordering, punctuation structure, lines,
characters, and approximate whitespace-token counts while replacing only
action-relevant values. They are mandatory for the primary contrasts.

Opaque fillers use deterministic uppercase ASCII letters, digits, and
underscores. Exact character matching does not establish semantic neutrality:
models may react to unusual strings, repeated symbols, or apparent identifiers.
Character and whitespace-token equality also do not imply equality under a
model-specific tokenizer.

## Evidence and reconstruction risks

Raw messages are byte-identical across conditions within a case. DP7 reactive
arms can reconstruct C's represented belief directly from that statement,
which may reduce the incremental effect of an explicit nested block. This is
an intentional evidence-boundary control and a substantive reconstruction
risk. Explicit nested fields must retain an audit link to that message and may
not add hidden global truth.

## Remaining blockers and future requirements

The model and tokenizer are not frozen. Exact tokenizer-specific counts must
be audited after both are selected and before any real execution. A future
protocol must also preregister prompt-order randomization, justify sample size,
freeze parser and exclusion handling, and audit every prompt layer and
decision record. API failure, parser failure, illegal action, missing layer
audit, and missing decision record remain distinct technical exclusions; a
poor but valid action is not excluded.

No API call or real execution is authorized. The scaffold supports no claim of
Theory of Mind, coordination improvement, autonomous planning, or cognition
from model explanations.
