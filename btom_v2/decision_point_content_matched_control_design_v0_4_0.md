# Content-matched epistemic framing controls v0.4.0

## Prospective status

Tokenizer-selection run `30314244024` evaluated all 1,296 frozen wording-bank
candidates and returned `no_candidate`. It skipped no candidates and reported no
structural failures. Every candidate retained a +1 raw and Harmony interaction
differential for H1 and H2. The minimum H2 primary deltas remained +4 for DP7a
and +3 for DP7b.

No model result exists. H1a, H1b, and H2 remain untested.

## Why the primary references changed

The operational-metadata reference omitted the represented location supplied by
the explicit first-order arm. The message-provenance reference likewise omitted
the structured represented content that the explicit second-order arm supplied,
while the latter repeated and interpreted message content. Those content and
structure differences prevent the old contrasts from isolating epistemic
framing. Operational and provenance controls are therefore removed as primary
references and retained only as possible journal ablations.

The v0.4.0 primary pairs contain the same proposition, represented location,
field order, field count, line count, and section position. H1 differs only
between `decision_record` and `self_belief`. H2 independently obtains the
reference value by parsing the raw message and the treatment value from the
frozen second-order representation, requires equality, and differs only between
`message_record` and `partner_belief`. Raw evidence remains visible exactly
once.

## Causal estimands and secondary comparisons

H1 estimates the incremental effect of self-belief framing over an otherwise
identical explicit decision-state record, interacted with DP5a versus DP5b. H2
estimates the incremental effect of attributing structured message content as a
partner belief, interacted with DP7a versus DP7b.

Reactive remains a secondary baseline because it measures the broader effect of
adding any explicit representation, not the primary epistemic-framing effect.
Explicit self belief remains an additional secondary comparator in DP7.

## Remaining limitations

The role labels can still differ in lexical salience even though content and
structure are matched. A later tokenizer-only calibration must measure the new
prompts; this design makes no tokenizer-parity claim. Content matching also does
not establish semantic equivalence or cognitive neutrality.

This task runs no tokenizer, API, model, workflow, or experiment, authorizes no
real execution, and tests no scientific hypothesis.
