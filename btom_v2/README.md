# btom_v2

Clean, auditable v2 scaffold for the PAAMS B-ToM-MAS benchmark.

## Scope in this first commit

Implemented:
- deterministic symbolic environment;
- scenarios `C1_fully_observable` and `C2_partial_observable`;
- core mission chain:
  1. A gets `red_key`
  2. B gets `blue_key`
  3. both keys open `locked_box`
  4. `medical_kit` is revealed
  5. C gets `medical_kit`
  6. C rescues `victim`
- role-gated actions;
- structured episode trace;
- one deterministic baseline policy;
- runner for scenarios over seeds `0,1,2`.

Not yet implemented:
- C4/C5/C6;
- second-order belief modeling;
- LLM calls.
