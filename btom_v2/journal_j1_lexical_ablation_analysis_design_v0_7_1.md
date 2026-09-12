# Journal J1 lexical-cue ablation analysis freeze v0.7.1

This is a **pre-behavioral journal-only analysis protocol**. ICAART remains frozen. No scenarios, prompts, workflow, or behavioral files are created or read.

## Eligibility and primary analysis

Analysis requires technical success, 54 distinct scenario variants, a complete valid C0/C1/N0/N1 tuple for each, and the frozen semantic decoding of neutral labels. Individual calls are not inferential units; the complete scenario variant is the inferential and bootstrap unit.

The primary estimand is `Delta_I_neutral = mean_i(Y_i,N1-Y_i,N0)`. Positive discordance is N0=0,N1=1 and reverse discordance is N0=1,N1=0. The sole primary test is the one-sided exact paired McNemar/binomial discordance test at alpha .05.

Persistence requires both `exact p <= .05` and `Delta_I_neutral >= .25`. The .25 boundary is the inherited protocol-specific SOEI used in prospective power planning, not a universal threshold. Significance below .25 or magnitude without statistical support is insufficient.

## Cued effect, moderation, and uncertainty

`Delta_I_cued = mean_i(Y_i,C1-Y_i,C0)` is secondary. `Lambda = mean_i[(Y_i,C1-Y_i,C0)-(Y_i,N1-Y_i,N0)]` is the key secondary moderation estimand. Its cued-arm McNemar result is nominal and secondary. No Lambda equivalence margin is frozen.

A single SHA-derived RNG stream resamples 54 complete four-condition tuples 100,000 times. Ordinary percentile 95% intervals use explicit R/NumPy type-7 interpolation for Delta_I_neutral, Delta_I_cued, and Lambda. Amplification requires the Lambda CI lower bound above zero; reverse moderation requires its upper bound below zero. A CI containing zero is no detected moderation—not equivalence.

## Frozen classification priority

Ineligible data yield `technical_or_analysis_ineligible`. Among eligible data, specific moderation evidence is resolved before the broad no-positive-moderation branch:

1. persistence plus positive Lambda CI: `mismatch_effect_with_lexical_amplification`;
2. persistence plus negative Lambda CI: `label_independent_effect_neutral_stronger`;
3. persistence without positive moderation: `label_independent_mismatch_effect`;
4. no persistence, cued effect at least .25, and positive Lambda CI: `primarily_lexical_affordance_compatible`;
5. otherwise: `mechanism_inconclusive`.

`label_independent_mismatch_effect` means the neutral effect survived removal of the cue; it does not establish interface equivalence.

## Diagnostics and reporting

Rates at C0/C1/N0/N1 receive <=.05 or >=.95 headroom warnings. Neutral mapping and action-order differences receive heuristic warnings only when absolute differences exceed .25. These diagnostics never change classification or exclude variants. Archetype and difficulty estimates are descriptive only and cannot redefine results, cause exclusion/rerun, alter thresholds, or select later replication.

No imputation, fallback, parser repair, outcome-dependent retry, or replacement is allowed. Future reporting includes all estimates and CIs, discordances and primary p-value, persistence/moderation indicators, diagnostics, classification, completeness, and technical provenance; verbalized reasoning cannot support a claim.
