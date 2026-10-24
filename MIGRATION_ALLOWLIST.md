# Migration Allowlist

This is a deny-by-default migration contract. Source paths are inspected only at
commit `a28093110325968c26906223e9eb0f1e078f6aad`. No file may be copied wholesale
before dependency, naming, license, and private-data review.

## PORT_AND_REFACTOR

The following files are candidates for logic-level porting. They must be rewritten
under the public package name, stripped of internal runtime dependencies, and
covered by tests. Frozen scientific settings and their configuration identities
remain explicit:

```text
bomi/eval/stage3_admissibility_metrics.py
bomi/eval/stage3_admissibility_runner.py
bomi/bench/stage3_memops_mapping.py
bomi/bench/stage3_rhelm_evidence_resolution.py
bomi/bench/stage3_natural_corpus_admissibility.py
bomi/bench/stage3_natural_dev_analysis.py  # selected analysis logic only
bomi/bench/stage3_admissibility_audit.py  # public record readers only
bomi/bench/stage3_natural_embedding_pilot.py  # content-free input identities only
bomi/bench/stage3_natural_embedding_store.py
bomi/bench/stage3_natural_route_store.py
bomi/bench/stage3_natural_dev_runtime.py  # exact retrieval operations only
bomi/experiments/stage3_natural_dev_routing.py  # frozen configuration and stable ranks
scripts/run_stage3_natural_corpus_embedding_pilot.py  # text preparation and encoding only
scripts/run_stage3_natural_corpus_public_eval.py  # population and execution assembly only
```

Required Phase 2 tests include complete method-by-query coverage, unresolved-label
bounds, rejection of `NaN` and infinity, and query-intent-sensitive lifecycle cases.
Because per-query route rows remain excluded, coverage is verified from the frozen
execution manifest: nine selected arms times 3,767 queries must equal both 33,903
route rows and 33,903 score rows.

The public-source runtime completion adds source adapters, embedding/route stores,
and the nine frozen retrieval configurations to the same package. It replaces
private archive imports with explicit public source and run directories. Source
mapping, split rules, tie breaking, and selected settings are preserved; archive
authorization gates and private payloads are excluded. Synthetic fixtures exercise
construction, routing, serialization, and fresh provider-run binding without model
or API calls.

## IMPORT_AS_FROZEN_AGGREGATE

Only the aggregate and manifest files enumerated in `SOURCE_ARTIFACTS.yaml` are
eligible. Phase 2 must:

1. verify the source SHA-256 against the read-only snapshot;
2. inspect for raw text, private content, identifiers, and license restrictions;
3. preserve the original source label and hash;
4. produce a deterministic normalized artifact with a new hash; and
5. keep the mechanism smoke separate from the full natural-corpus evaluation.

The eligible evidence families are:

- two separately executed GateMem reader aggregates and manifests;
- the 16-packet candidate-pool mechanism smoke; and
- the 87-group, 3,767-query public-source natural evaluation.

Raw response bundles and private scored derivatives are not eligible.

## REFERENCE_ONLY

Historical narrative, planning, and status documents may be read for source
interpretation but must not be copied into this repository. Their evidence status
is stale relative to the frozen full public-source evaluation.

## PROHIBITED

The following are outside the new repository's scientific and release scope:

```text
historical project branding and development timeline
Bayesian or nonparametric mixture implementations
novelty sidecars and retired indexing experiments
internal experiment and milestone numbering
owner-unlock and provider-recovery records
failed execution checkpoints
private responses and raw prompts
GPU embedding shards and route checkpoints
data/private/**
reports/**/cache/**
old status, agent-instruction, review-index, and README files
old .git directory and commit history
```

## Public Method Labels

Phase 2 may apply this mapping only through a deterministic transformation script.
The source file must retain its original hash, and the normalized output must receive
a new hash.

| Source label | Public label |
| --- | --- |
| `flat_dense` | `global_dense` |
| `namespace_dense` | `namespace_dense` |
| `ncr_threshold` | `threshold_router` |
| `ncr_a5` | `cluster_router` |
| `namespace_current_only` | `current_only_filter` |
| `namespace_source_intent_lifecycle` | `released_intent_lifecycle_upper_bound` |
| `flat_bm25_dense_rrf` | `hybrid_rrf` |
