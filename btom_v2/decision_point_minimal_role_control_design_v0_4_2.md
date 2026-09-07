# Context-anchored minimal representation roles v0.4.2

## Change-of-direction record

Run `30324745794` classified the v0.4.1 content-matched prompts as
`nonzero_primary_token_difference`. H1 DP5a and DP5b and H2 DP7a and DP7b each
had +2 raw and +2 Harmony treatment-minus-reference deltas. Both raw and Harmony
interaction differentials were zero for H1 and H2.

The result is not exact parity because the preregistered criterion requires every
case-level raw and Harmony delta to equal zero. That criterion is not relaxed
post hoc merely because the interaction differentials were zero.

## Deterministic normalization

The prefixes in `decision_record`, `self_belief`, `message_record`, and
`partner_belief` repeat context already supplied by the representation titles,
agents, proposition, represented value, and—on H2—message evidence. v0.4.2
therefore deterministically maps both references to `record` and both treatments
to `belief`. This is one fixed normalization, not a prompt bank or search.

Both terms contain six ASCII characters and one whitespace-delimited word. This
character match does not establish tokenizer parity or semantic equivalence. The
record arm remains an explicit content-matched representation; it is not a
no-representation control.

## Preserved design and provenance

Condition identifiers, propositions, represented values, sections, messages,
observations, actions, output format, contrasts, and interactions are unchanged.
H2 reference values remain sourced only from `raw_delivered_messages`; treatment
values remain sourced only from `second_order_representation`; their equality is
still required and raw evidence remains visible exactly once.

Lexical salience may remain despite character equality. A later tokenizer-only
v0.4.3 measurement must test T-CAL-3. This task runs no tokenizer, API, model,
workflow, or experiment and tests no scientific hypothesis.


## Test-infrastructure amendment

The original test used the repository-global assertion
`glob("*minimal_role*")`, which would prospectively reject a legitimate later
tokenizer workflow. It is replaced by an existence check scoped only to
`.github/workflows/decision_point_minimal_role_control_v0_4_2.yml`. Later-version
workflows are permitted. This amendment changes no prompt, scenario, role,
contrast, or parent hash; it tests no scientific or technical hypothesis, and
T-CAL-3 remains untested.
