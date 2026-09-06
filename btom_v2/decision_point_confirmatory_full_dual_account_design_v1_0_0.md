# Fresh full dual-account confirmatory design v1.0.0

## Scope and freeze

This preparation defines one fresh sequential collection batch, `full_confirmatory_20b_dual_account_fresh`, containing all 432 prompts (72 variants, two states, three conditions per state) from the frozen confirmatory bank. Prompt text, request bodies, explicit seeds, valid actions, global order, schema, parser, classification, estimands, alpha, SOEI, McNemar, Clopper–Pearson, Holm, H1a and subgroup interpretations, and fingerprint sensitivity remain frozen. The unchanged analyzer `analyze_decision_point_confirmatory_v1_0_0.py` is used only after the complete gate passes.

The model is `openai/gpt-oss-20b`; temperature 0, top-p 1, maximum completion tokens 1024, low reasoning effort, reasoning excluded, non-streaming, no tools, strict frozen JSON Schema, 120-second timeout, 20-second sequential delay, and zero behavioral retries.

## Prospective assignment

The manifest freezes an explicit variant-level primary/secondary assignment before collection. It was derived only from family, difficulty, archetype, frozen order-pattern metadata, variant ID, and frozen ordinal: within every family × difficulty × archetype pair, suffixes 1/3/5 are primary and 2/4/6 are secondary. No prior action, target, classification, task success, role sensitivity, fingerprint, response, or HTTP outcome was loaded or inspected. Each slot has 36 variants (18 H1 and 18 H2), and each slot × family × difficulty has six variants. Every required order-pattern, role-first, reactive-placement, and role-position balance is exact.

Credential values, prefixes, hashes, and Authorization values are never persisted. A variant's six canonical calls must use one final slot. Nominal requests retain exact original global order.

## Failover and completeness

Failover is allowed only after strict confirmation of HTTP 429 tokens-per-day exhaustion. RPM, RPD, TPM, ITPM, OTPM, generic 429, transport/timeout, other HTTP, parse, finish, content, and legality failures never fail over. An exhausted slot is permanently suppressed. If exhaustion interrupts a variant, all its prior canonical records are moved to discarded transport telemetry and the complete six-call variant is replayed in its frozen internal order on the other slot. A second confirmed TPD is terminal, removes that partial variant, and makes final analysis unavailable. Complete earlier variants remain canonical.

The final analysis gate requires all 432 unique frozen prompt IDs and technically complete calls, 72 homogeneous complete variants (36 per family), matching returned seeds when supplied, all immutable hashes and prompt/scenario digests, request identity, raw and Harmony parity, content matching, no development overlap or leakage, and no dual-account exhaustion. Analysis is absent otherwise. Operational `scientific_inference` is always null.

## Exclusions

Confirmatory attempts 1, 2, and 3; compositional recovery run 30581466480; dual-account compositional recovery run 34001864149; and every other prior confirmatory observation are excluded. No response from any earlier run may be imported, reused, selected, copied, backfilled, or assembled into this fresh dataset.

## Credential-slot diagnostic

A pre-specified descriptive-only diagnostic reports final variant count, H1/H2 and difficulty counts, role-sensitive counts by family, and H1a task-pass counts by condition for each final slot. It performs no credential-slot significance testing and is not used for exclusion or primary inference.
