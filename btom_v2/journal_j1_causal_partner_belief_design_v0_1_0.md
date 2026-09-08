# Journal J1 Causal Partner-Belief Intervention

**PRE-IMPLEMENTATION STATISTICAL DESIGN**
**NOT A CONFIRMATORY MANIFEST**

## Status and scientific question

This journal-only design aid asks whether explicit partner-belief framing selectively changes communication when partner misinformation is causally relevant to the next coordination action, rather than inducing a general communication or salience prior. The proposed mechanism is belief representation → policy change → communication behavior → coordination outcome → cost. Shared ICAART evidence motivates J1, but the ICAART confirmatory phase is closed and is not reopened, modified, rerun, reinterpreted, or used for numerical outcome adaptation. No J1 data, prompt bank, or experiment exists in this task.

**J-H3 (functional calibration):** explicit partner-belief framing increases the relevant-mismatch communication calibration gap relative to a matched current, non-mentalistic partner-state record. **J-H4 (negative control):** this role interaction is substantially weaker for explicitly decision-irrelevant mismatch. These labels pre-specify questions, not supported findings.

Possible future patterns remain distinct: functional calibration (`Delta_role > 0` with substantially weaker/null irrelevant sensitivity); a communication-prior alternative (communication rises broadly regardless of relevant mismatch); generic false-belief salience (material response to irrelevant mismatch); a partner-state/content account (little difference between matched current record and belief framing); and harmful epistemic framing (unnecessary correction increases enough to reduce calibration). None is empirically supported here.

## Factorial structure and observables

Each independent new-bank variant will hold objective world state fixed and cross three binary factors: framing `F ∈ {R,B}`, relevant mismatch `M_R ∈ {0,1}`, and irrelevant mismatch `M_I ∈ {0,1}`. `R` is a matched current non-mentalistic partner-state record; `B` is explicit partner-belief representation. This is not historical-message versus current-belief framing, and exact wording is not designed here. The `2 × 2 × 2` design has eight canonical decisions per variant per model.

The binary observable `C_i(F,M_R,M_I)` equals 1 for the communication/correction action and 0 for task-progress/non-communication. The inferential unit is the **variant**, never an individual call.

## Estimands

Marginalize irrelevant mismatch: `A_i(F,M_R)=[C_i(F,M_R,0)+C_i(F,M_R,1)]/2`. Let `G_i(F)=A_i(F,1)-A_i(F,0)` and `D_i=G_i(B)-G_i(R)`. The primary bank estimand is `Delta_role=mean_i(D_i)`, equivalently the belief-framing relevant-mismatch gap minus the matched-record relevant-mismatch gap. `D_i ∈ [-2,2]`; positive values indicate a larger differential response between correction-needed and correction-not-needed states under belief framing.

For the negative control, marginalize relevant mismatch: `Q_i(F,M_I)=[C_i(F,0,M_I)+C_i(F,1,M_I)]/2`. Let `H_i(F)=Q_i(F,1)-Q_i(F,0)`, `E_i=H_i(B)-H_i(R)`, and `Delta_irrelevant=mean_i(E_i)`. This measures framing sensitivity to incorrect but action-irrelevant partner information and is not the primary endpoint.

## Planned bank structure and separation

The intended independent bank retains six broad coordination archetypes: `resource_location`, `tool_placement`, `rendezvous_destination`, `delivery_destination`, `hazard_assignment`, and `maintenance_target`; and three difficulty strata: `direct`, `irrelevant_distractor`, and `compositional`. Their 18 archetype-by-difficulty cells require candidate N values divisible by 18. No prior sensitive variant, behavioral sensitivity, action, target, classification, or result may select J1 content. Development and confirmatory test banks must be separated, with held-out entities/templates and no scoring or state leakage; those implementation details will be frozen later.

## Power and precision design aid

Candidate N is `[36,54,72,90,108,126,144,180,216]`; assumed SD(D_i) is `[0.50,0.75,1.00,1.25,1.50,2.00]`; and planning effects are `[0.10,0.15,0.20,0.25,0.30]`. The theoretical maximum assumed SD is 2 because `D_i ∈ [-2,2]`. Calculations use two-sided alpha 0.05 and target powers 0.80 and 0.90. Approximate power uses the two-sided normal mean approximation; MDE is `(z_0.975+z_power) SD/sqrt(N)`; approximate 95% half-width is `z_0.975 SD/sqrt(N)`. These are planning sensitivities, not scientifically important effects or the final analysis. A later preregistration may freeze block-aware randomization/permutation inference and variant-level confidence intervals.

Each N implies `8N` calls per model and `16N` for two models. Tables also show reference-only fixed delay at 20 seconds per call; this neither freezes J1 delay nor estimates deployment latency.

**Final N is NOT yet frozen. Final SOEI is NOT yet frozen. The final inferential test is NOT yet frozen.** Selection follows scientific review of the full sensitivity table. The numerical calculation uses only the factorial structure, declared grids, alpha, power targets, and formulas—no prior treatment outcome. This task performs no API or experiment execution.
