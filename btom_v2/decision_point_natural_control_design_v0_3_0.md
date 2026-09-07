# Decision-point natural metadata controls v0.3.0

## Status and negative placebo result

This is a mock-only prospective design. Placebo audit run `30068749379`
produced no valid DP5 or DP7 second-order panel and only three DP7 first-order
codes: `loc_HTML`, `loc_RDWR`, and `loc_HTTP`. Those strings are semantically
loaded technical abbreviations. Opaque and typed-placebo fillers are therefore
abandoned before any model result exists.

## Causal interpretation

The operational metadata reference adds true structured context about the
current interface: one action is selected from the listed actions and returned
as JSON. It carries no represented belief, location inference, recommendation,
score, or outcome. The H1 contrast asks whether the DP5 pairwise response changes
when this action-invariant structure is replaced by an explicit first-order
representation.

The message-provenance reference adds true structure about the already visible
message without repeating, paraphrasing, or interpreting its content. It is a
stronger H2 reference than an opaque filler because it controls for observer,
sender, message availability, and evidence linkage while withholding the nested
proposition and represented value. The H2 contrast asks whether stale-versus-
current correction sensitivity differs between explicit second-order content
and this provenance-only reference.

Reactive prompting remains a secondary H1 baseline. Explicit first order remains
a secondary H2 comparator. These secondary comparisons cannot replace the
preregistered primary contrasts after results are observed.

## Remaining confounds and risks

Equal field and line counts do not imply equal character counts or tokenizer
counts. No tokenizer parity is claimed in v0.3.0; a later tokenizer-only
calibration is required before any real execution. Natural metadata can itself
be salient, and labels such as `json_object` or `structured_statement` may alter
attention even though they are true and action-invariant. This risk must be
reported and, if necessary, addressed prospectively.

Raw DP7 evidence remains visible exactly once. Audit-only provenance is never
shown to the model. Model-written reasons cannot be used as evidence of
cognition.

## Scientific status

No scientific hypothesis is tested here. H1a, H1b, and H2 remain untested. This
design does not authorize tokenizer execution, model execution, API calls, or an
experiment, and it does not establish Theory of Mind, coordination improvement,
autonomous planning, or model cognition.
