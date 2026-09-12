# Journal J1 Causal Partner-Belief Intervention — Inference and Sampling Freeze v0.2.1

**PRE-IMPLEMENTATION STATISTICAL AND SAMPLING DESIGN — NOT AN EXPERIMENT MANIFEST**

## Status and unchanged mechanistic design

ICAART remains closed and unmodified. Journal J1 has no behavioral observations, scenarios, candidate contents, prompts, bank, manifest, runner, workflow, or API configuration. The factorial design remains `F∈{R,B} × M_R∈{0,1} × M_I∈{0,1}`: eight binary communication decisions per variant/model, fixed objective world state within each future counterfactual block, and variant-level inference.

For `C_i(F,M_R,M_I)∈{0,1}`, define `A_i(F,M_R)=[C_i(F,M_R,0)+C_i(F,M_R,1)]/2` and `D_i=[A_i(B,1)-A_i(B,0)]-[A_i(R,1)-A_i(R,0)]`. Define `Q_i(F,M_I)=[C_i(F,0,M_I)+C_i(F,1,M_I)]/2` and `E_i=[Q_i(B,1)-Q_i(B,0)]-[Q_i(R,1)-Q_i(R,0)]`. The unchanged primary contrast is `S_i=D_i-E_i`, and `Delta_specificity=mean_i(S_i)=Delta_role-Delta_irrelevant`. Attainable values remain `D_i,E_i∈{-2,-1.5,-1,-.5,0,.5,1,1.5,2}` and `S_i∈{-2,-1,0,1,2}`. `Delta_role`, `Delta_irrelevant`, and `Delta_communication_prior` remain secondary and outside primary multiplicity.

## Frozen size, SOEI, and prospective sampling

Retain N=180 variants/model, SOEI=.25, models `openai/gpt-oss-20b` and `openai/gpt-oss-120b`, six named coordination archetypes, three difficulty strata, 18 equal cells, 10 final variants/cell, 1,440 calls/model, and 2,880 calls/two models. Statistical support and practical magnitude remain distinct; positive practical-support language requires hypothesis-consistent direction and interpretation against SOEI, and no automatic “validated” field is authorized.

Before any execution or behavioral observation, future construction must provide 20 unique confirmatory-eligible candidates in each cell (360 total). Independently within each cell, select 10 of 20 without replacement using deterministic seed literal `journal-j1-confirmatory-stratified-selection-v0.2.1`, whose first eight SHA-256 bytes interpreted unsigned big-endian give `8030438251456861422`. The 180 unselected candidates are frame members only: no replacement, backfill, failure recovery, selective extension, or sensitivity enrichment absent a separate pre-outcome preregistration.

Development is structurally separate: at least 18 development variants, one per cell, using disjoint namespaces/templates/entities. Development variants never enter the candidate frame, selection, or analysis; development responses never influence selection.

The inferential target is the balanced scenario-generating distribution prospectively represented by six archetypes × three difficulties under the prespecified models and protocol. Stratified outcome-independent sampling prevents manual cherry-picking. Claims remain limited to that distribution and protocol. No finite-population correction is used.

## Primary stratified estimator and uncertainty

For each model independently, cell `h` has `n_h=10`, weight `W_h=1/18`, mean `mean_h`, and unbiased sample variance `s_h²`. Estimate

`Delta_hat = sum_h W_h mean_h`,

which equals the ordinary mean of 180 values because cell sizes are equal. Its standard error is

`SE_hat = sqrt(sum_h W_h² s_h²/n_h)`.

The two-sided primary test of `Delta_specificity=0` is a deterministic stratified studentized bootstrap—not exact, distribution-free, or randomization inference. Observe `T_obs=Delta_hat/SE_hat`. Globally null-center each value as `S_i_null=S_i-Delta_hat`; this forces weighted mean zero while preserving between-cell mean differences. For each of 100,000 replicates, resample 10 null-centered values with replacement inside every original cell, compute `Delta_b`, the same stratified `SE_b`, and `T_b=Delta_b/SE_b`. Report `(1 + count(|T_b|>=|T_obs|))/(100000+1)`.

If SE is zero, the deterministic fail-closed statistic is zero only when its estimate is also zero, otherwise signed infinity. The primary seed literal `journal-j1-specificity-primary-test-bootstrap-v0.2.1` derives integer `17273625087858402091` by the same SHA-256 rule.

The model-specific 95% CI is a separate stratified percentile bootstrap over original, non-centered `S_i`: 100,000 within-cell replicates preserving 10 draws/cell and percentiles 2.5/97.5. Literal `journal-j1-specificity-ci-bootstrap-v0.2.1` derives independent seed `14934827325440938879`.

Apply Holm correction to exactly the two model-specific primary bootstrap p-values. No secondary metric enters this family. Report communication rate by framing, necessary/unnecessary correction rates, cell estimates, and the `S_i` distribution descriptively, with no subgroup significance tests.

## Sign-flip sensitivity and planning continuity

Sign-flip is not primary inference. A sensitivity-only calculation may exhaustively enumerate sign assignments conditional on observed absolute contrast magnitudes. Its computational enumeration is exact, but no exact inferential randomization validity is claimed; agreement or disagreement with the bootstrap is descriptive, and its p-value receives no Holm correction.

The outcome-independent design references remain unchanged: N=180, SD=1, SOEI=.25, alpha=.05, normal approximate power `0.9183620828255178`, MDE90 `0.24160830400373037`, and approximate half-width `0.1460870900960969`. These were sample-size design calculations. Replacing inference uses no J1 outcome—none exists—and does not alter N or SOEI.

This task performs no experiment or API/network operation and reads no ICAART or model behavioral artifact.
