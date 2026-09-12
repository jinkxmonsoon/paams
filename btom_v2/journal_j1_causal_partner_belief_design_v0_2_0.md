# Journal J1 Causal Partner-Belief Intervention — Statistical Freeze v0.2.0

**PRE-IMPLEMENTATION STATISTICAL DESIGN — NOT A MANIFEST OR EXPERIMENT IMPLEMENTATION**

## Scientific status and factorial design

ICAART is closed and remains frozen. Journal J1 is an independent journal-only experiment; no J1 behavioral response exists. This freeze uses the completed outcome-independent planning sensitivity table and no ICAART or prior-model behavioral result. It creates no scenarios, prompts, bank, manifest, runner, credential routing, or workflow.

Each future counterfactual variant holds objective world state fixed and crosses framing `F∈{R,B}`, action-relevant mismatch `M_R∈{0,1}`, and explicitly action-irrelevant mismatch `M_I∈{0,1}`. `R` is a matched current non-mentalistic partner-state representation; `B` is an explicit partner-belief representation. Zero denotes correct partner state and one denotes incorrect partner state. The observable `C_i(F,M_R,M_I)∈{0,1}` is one for communication/correction and zero for task progress/non-communication. Thus each variant has `2×2×2=8` canonical decisions per model, while the inferential unit is the variant.

## Frozen estimands

Let `A_i(F,M_R)=[C_i(F,M_R,0)+C_i(F,M_R,1)]/2`, `G_i(F)=A_i(F,1)-A_i(F,0)`, and `D_i=G_i(B)-G_i(R)`. `Delta_role=mean_i(D_i)` is the key secondary relevant-calibration estimand.

Let `Q_i(F,M_I)=[C_i(F,0,M_I)+C_i(F,1,M_I)]/2`, `H_i(F)=Q_i(F,1)-Q_i(F,0)`, and `E_i=H_i(B)-H_i(R)`. `Delta_irrelevant=mean_i(E_i)` is the prespecified negative-control component.

The primary mechanistic contrast is `S_i=D_i-E_i`, with `Delta_specificity=mean_i(S_i)=Delta_role-Delta_irrelevant`. Positive specificity means belief framing changes response to action-relevant misinformation more than to action-irrelevant misinformation, preventing generic false-information response from masquerading as selective epistemic calibration. Exhaustive enumeration gives `D_i,E_i∈{-2,-1.5,-1,-.5,0,.5,1,1.5,2}` and `S_i∈{-2,-1,0,1,2}`.

The descriptive communication-prior contrast is `M_i=mean_{M_R,M_I}C_i(B,M_R,M_I)-mean_{M_R,M_I}C_i(R,M_R,M_I)`, with `Delta_communication_prior=mean_i(M_i)`. Positive communication prior without specificity is not evidence of functional ToM.

## Frozen bank size, balance, and practical magnitude

Freeze `N=180` variants per model for `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. These are two fixed sizes in one model family, not cross-family generalization. Six archetypes (`resource_location`, `tool_placement`, `rendezvous_destination`, `delivery_destination`, `hazard_assignment`, `maintenance_target`) crossed with three difficulties (`direct`, `irrelevant_distractor`, `compositional`) form 18 cells, with exactly 10 confirmatory variants per cell. This yields 1,440 canonical calls per model and 2,880 across both; development calls are excluded. An independent development bank must contain at least 18 variants, one per cell, and can never enter confirmatory analysis. No sensitive ICAART variant or observed outcome may influence templates, entities, or selection.

The outcome-independent planning reference is SD=1.00—not an empirical estimate. At N=180, effect=.25, and two-sided alpha=.05, normal planning power is 0.9183620828, 90% MDE is 0.2416083040, and approximate 95% half-width is 0.1460870901. The full SD sensitivity grid `.50,.75,1,1.25,1.5,2` remains recorded. These normal calculations are design aids, not final inference.

Freeze `SOEI_primary=0.25` for `Delta_specificity`. Eight binary decisions make variant contrasts discrete; .25 is a materially non-trivial average selective policy shift. Effects may be detectable below .25 but are not predeclared practically substantial for the main claim. Because usefulness is evaluated against communication and computational overhead, arbitrarily tiny effects are insufficient. This threshold is a scientific decision, not an estimate from previous behavior.

## Frozen inference and multiplicity

For each model independently, test `Delta_specificity=0` with an exact two-sided paired variant-level sign-flip randomization distribution at nominal alpha .05. Dynamic programming accumulates counts of every attainable signed sum; the p-value is the fraction with absolute sum at least the observed absolute sum. It is deterministic and uses no Monte Carlo or call-level independence.

Apply Holm correction to exactly the two model-specific primary p-values. Both positive estimates with both adjusted p-values below .05 support cross-size replication only; one supported model is model-dependent; neither means the primary hypothesis is unsupported; sign reversal indicates instability and must not be pooled away. Statistical significance below SOEI cannot be called practically substantial.

Model-specific 95% CIs use a deterministic stratified nonparametric percentile bootstrap: resample variants with replacement within each of 18 archetype×difficulty cells, preserving 10 draws per cell, 100,000 replicates, and percentiles 2.5/97.5. The seed is the unsigned big-endian integer from the first eight SHA-256 bytes of literal `journal-j1-specificity-bootstrap-v0.2.0`: `18140943157235233773`. The CI is secondary to the exact primary p-value.

## Secondary outputs and interpretation constraints

Report outside the primary Holm family: `Delta_role`, `Delta_irrelevant`, `Delta_communication_prior`, communication rate by framing, necessary- and unnecessary-correction rates by framing, descriptive archetype×difficulty estimates, and the variant-level `S_i` distribution. No post-hoc subgroup significance test is authorized.

Interpretation patterns are behavioral, not proof of human-like mental states: functional calibration requires positive specificity and relevant calibration not explained by irrelevant response; broad communication with weak specificity is a communication prior; similar relevant and irrelevant effects are generic false-belief salience; little framing difference supports a partner-state/content account; increased unnecessary communication, reduced calibration, or negative specificity is harmful framing.

No J1 execution or API/network call occurs in this freeze. ICAART remains closed and unmodified.
