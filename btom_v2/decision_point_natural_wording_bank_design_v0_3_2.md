# Finite natural wording bank v0.3.2

## Motivation and status

Tokenizer measurement run `30311773726` classified the frozen v0.3.0 prompts
as `nonzero_primary_token_difference`. H1 full-prompt deltas were +5 and +4
raw/Harmony tokens, while H2 deltas were +9 and +8. Both interactions retained
a one-token raw and Harmony differential associated with `decoy_room` versus
`box_room` tokenization.

No model result exists. Model execution remains blocked because these measured
interaction differences confound a future case-by-condition comparison. H1a,
H1b, and H2 remain untested.

## Finite prospective redesign

The bank contains four natural surface mappings, 36 operational metadata
variants, and nine message-provenance variants. Their Cartesian product yields
1,296 deterministic candidate specifications. The bank is finite to prevent
open-ended prompt optimization and researcher degrees of freedom.

Only represented values are surface-normalized. Frozen scenario values remain
symbolic internally, and raw messages are not rewritten because changing raw
evidence would alter the evidence available to every DP7 condition. Common
sections, actions, field order, field count, and section order remain frozen. The
only permitted model-visible replacements are the seven field values
`represented_value`, `believed_value`, `decision_scope`, `action_source`,
`output_mode`, `message_type`, and `delivery_status`.

Future selection is tokenizer-only and non-behavioral. A wording may be selected
only under exact raw and Harmony parity for every primary contrast and both
interaction differentials. There is no tolerance. If none passes, the outcome
is `no_candidate`; criteria may not be relaxed automatically.

Among exactly passing candidates, ranking uses total compared-block characters,
then the number of changed literals relative to v0.3.0, then canonical
lexicographic specification. No wording is selected in this task.

## Remaining risks

Natural wording can still create semantic or attentional salience even when it
preserves the intended symbolic mapping. Token parity cannot establish semantic
equivalence or cognitive neutrality. Those limitations must remain explicit in
any future protocol.

This mock-only design loads no tokenizer, calls no API or model, runs no
experiment, and tests no scientific hypothesis. It does not establish Theory of
Mind, coordination improvement, autonomous planning, or model cognition.


## Prospective v0.3.2 schema and audit amendment

The original implementation inconsistently named the candidate fields
`H1_operational_variant_id` and `H2_provenance_variant_id`. The frozen schema
uses lowercase `h1_operational_variant_id` and `h2_provenance_variant_id`; the
implementation and manifest now use those names without compatibility aliases.
All 1,296 candidate IDs changed prospectively because each ID is the SHA-256 of
the corrected canonical JSON specification. No tokenizer or model result existed
for any wording-bank candidate when the IDs were regenerated.

The original aggregate audit also returned a constant successful
`field_and_line_counts_frozen` value. It has been replaced by executable checks
of section order, field names, field order, field counts, line counts, and the
permitted replacement boundary. The audit renders every one of the 1,296
candidate sets and checks all 20,736 prompts without sampling or truncation.
This amendment tests no scientific hypothesis.
