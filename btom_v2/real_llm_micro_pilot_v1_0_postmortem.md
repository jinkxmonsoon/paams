# Functional micro-pilot v1.0 postmortem

## Immutable run record

- GitHub Actions run: `29891488868`
- Run commit: `c7c8d4fe6c91abf61ef44793b813924b981e9302`
- Protocol: `btom-v2-functional-micro-pilot-1.0`
- Manifest SHA-256: `d0495c6f5eca130ef9cbaf153b860ffc6fe257adf215e525c156e7425a073bf9`
- The v1.0 manifest and run artifacts remain immutable.

## Technical outcome

- The run made 360 episode API attempts.
- 21 attempts received successful API responses.
- 339 attempts failed with HTTP 429 token-per-minute rate limits.
- The applicable organization-scoped TPM limit was 6000.
- All nine LLM episodes were excluded.
- All three deterministic episodes succeeded.
- LLM episodes reached zero correction opportunities.
- There were zero usable observations for H1, H2, or H3.

The fixed condition order was confounded with depletion of an organization-wide
quota. Consequently, differences among LLM conditions cannot be separated from
quota timing. Preliminary model actions are not scientific evidence.

## Measurement corrections

The v1.0 `parse_success_rate` name mixed parser outcomes with calls that never
received an API response. Version 1.1 separately reports API response success,
parser success conditional on an API response, and end-to-end parseable-action
rates. It also separates valid non-stay model actions from objective task
milestones; movement or messaging is not itself labeled task progression.

## Decision

Version 1.1 retains the complete scientific matrix and adds only execution and
measurement controls: one organization-global scheduler, no retries, and an
immediate run-level abort on the first HTTP 429. No inference about H1, H2, H3,
Theory of Mind, or autonomous planning is warranted from v1.0.
