# Decision-point prompt amendment v0.2.1

## Prior design and discovered confound

Protocol v0.2 placed the complete C statement in `RAW DELIVERED MESSAGES` for
every DP7 condition. Its explicit second-order block then repeated that exact
statement as `source_message`, while the matched-neutral block substituted an
opaque string. This made the H2 contrast differ in both nested belief content
and informative evidence repetition and lexical salience.

Evidence repetition threatens interpretation because a response difference
could arise from seeing the same informative statement twice rather than from
the explicit nested relation. The issue was identified before any real result
under v0.2; no observed model outcome motivated this prospective amendment.

## Corrected design

The raw statement remains exactly once in the common raw-message section. Both
second-order arms now expose the invariant `evidence_ref="raw_message_1"` and
never expose `source_message`. Audit-only metadata resolves that reference to
the raw-message index and records its SHA-256, parsed proposition, parsed value,
and agreement with the explicit nested value. Audit metadata is not included
in model-visible prompts.

The H1 contrast, H2 estimand, case matrix, condition applicability, first-order
behavior, valid actions, instructions, output schema, metrics, and
interpretation constraint remain unchanged. Exactly four second-order prompts
change relative to v0.2.

## Remaining risks

Every DP7 arm still sees the raw C statement and may reconstruct the target
belief without an explicit nested block. Opaque fillers may themselves be
salient or interpreted as identifiers. Character and whitespace matching do
not establish model-token matching; model and tokenizer selection and exact
token audits remain prerequisites for execution.

A repetition/salience ablation is reserved for journal-level extension rather
than folded into the ICAART primary design. For ICAART, this amendment improves
the interpretability of the matched H2 contrast by removing unequal informative
repetition. For a journal study, dedicated repetition, formatting, and lexical
salience arms would be needed to isolate these mechanisms.

No real result motivated this amendment. No API call, model run, tokenizer
download, or experiment is authorized by this protocol amendment.
