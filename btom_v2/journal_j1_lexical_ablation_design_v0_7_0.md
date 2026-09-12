# Journal J1 lexical-cue ablation v0.7.0

**PRE-BEHAVIORAL DESIGN AND POWER/PRECISION PLAN — NOT AN EXPERIMENT MANIFEST**

## Question and hypotheses

This journal-only experiment asks whether a causally non-operative mismatch in represented partner state still increases communication when action labels are lexically opaque rather than containing `CORRECTION`. J-H3b-L1 predicts a positive neutral-label mismatch effect, `Delta_I_neutral`. J-H3b-L2 asks whether the cued effect is larger; `Lambda` is the key secondary moderation estimand. ICAART remains frozen and closed.

## Design and estimands

The scenario **variant is the inferential unit**. `M_R` is fixed at zero. Every variant supplies the four conditions C0, C1, N0, and N1 from `M_I ∈ {0,1}` × lexical interface `{CUED,NEUTRAL}`. The cued labels are `SEND_CORRECTION` and `CONTINUE_TASK`. Neutral labels are `ACTION_A` and `ACTION_B`; the MESSAGE mapping is counterbalanced globally 50/50 and frozen before behavior.

Let `Y_i,C0`, `Y_i,C1`, `Y_i,N0`, and `Y_i,N1` be semantic MESSAGE indicators. The primary contrast is `d_i,N = Y_i,N1 - Y_i,N0`, and `Delta_I_neutral = mean_i(d_i,N)`. The cued contrast is `d_i,C = Y_i,C1 - Y_i,C0`. The key secondary is `Lambda = mean_i(d_i,C - d_i,N)`. No Lambda equivalence threshold is frozen; uncertainty details will be preregistered in a later analysis freeze.

The sole primary test is neutral N1 versus N0 using a one-sided exact paired McNemar/binomial discordance test at alpha 0.05. The inherited protocol-specific SOEI is 0.25, originating in the earlier frozen J1 scenario-control criterion—not the observed v0.6.2 effect. No primary multiplicity correction is needed.

## Exact planning and selection

For each `p_minus ∈ {0,.025,.05,.10}`, planning fixes `p_plus=p_minus+.25`. If `D~Binomial(N,p_plus+p_minus)` and, conditional on `D=d`, `X~Binomial(d,p_plus/(p_plus+p_minus))`, rejection uses the smallest `x` with `P(Binomial(d,.5)>=x)<=.05`; exact power integrates the conditional rejection probability over D. The smallest N with exact balance over 18 archetype × difficulty strata and minimum grid power at least .80 is selected. Feasibility is assessed only afterward against `(4N-1)×20` delay seconds and the current 90-minute budget.

The historical `12/18` effect is context only and never enters selection. Lambda precision rows report only conservative range-based bounds and `1/sqrt(N)` scale—not moderation power.

## Future construction controls

Final scenarios are not generated here and must be new and disjoint from all 18 v0.4.x development variants, 18 v0.6.x holdouts, and 360 prior candidates. They retain six archetypes and three difficulties. Causal isolation, world state, record content, opportunity cost, and action semantics are held constant across interfaces. Action order, neutral mapping, proposition order, strata, and four-round temporal position are prospectively balanced. One paired seed is reused across all four conditions.

Within-interface C0/C1 and N0/N1 complete-conversation Harmony parity is mandatory later using `tiktoken==0.13.0` and `openai-harmony==0.0.8`. Cross-interface parity is desirable but will not be fabricated. No label search occurs here.

Only GPT-OSS-20B is planned first; 120B is deferred. No scenario, prompt, workflow, seed, model call, API access, or behavioral observation is produced by this task.
