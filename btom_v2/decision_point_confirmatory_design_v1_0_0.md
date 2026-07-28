# Held-out confirmatory epistemic-role experiment v1.0.0

The v0.6.0 bank is development data and is excluded. The confirmatory unit is one held-out variant; its two states are nested observations, never independent replicates. Each family contains six archetypes, six lexical instances per archetype, and 12 variants in each frozen difficulty stratum.

H1a compares reactive content absence with explicit belief content and is a content-availability manipulation, not an isolated Theory-of-Mind effect. H1b and H2 compare content-identical record and belief roles using two-sided variant-level paired McNemar tests. Role sensitivity means any action-target difference across a variant's two states. Practical invariance requires all 36 variants complete and an exact 97.5% one-sided Clopper-Pearson upper bound below the prespecified 10% effect threshold. A nonsignificant McNemar result is not equivalence.

Seeds are deterministic per family/variant/state triplet and shared across reactive, record, and belief calls. Role calls are adjacent and role-first/reactive placement is counterbalanced. The exact bank and analysis must be replicated with `openai/gpt-oss-120b` regardless of the 20B outcome; that replication is not executed here and prompts may not be altered after the 20B result.
