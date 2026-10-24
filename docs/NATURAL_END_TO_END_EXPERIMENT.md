# Natural route-to-reader evaluation

This experiment tests the full observable path on one frozen natural population:

`retrieval -> admissibility filter -> prompt exposure -> reader answer -> blinded score`

It uses the 3,767 evaluation queries already routed in the pinned RHELM and MemOps
source snapshots. It is a non-official diagnostic, not an official benchmark result.

## Frozen arms

All arms expose at most 20 candidates. The three filtering arms delete candidates from
the same frozen namespace-dense top-20 list; they do not backfill, rerank, or use answer
gold.

| Arm | Candidate construction |
|---|---|
| `global_dense` | Frozen global dense top-20 |
| `namespace_dense` | Frozen trusted-namespace dense top-20 |
| `namespace_policy_gate` | Drop source-released policy violations |
| `namespace_text_verifier` | Drop candidates whose fixed dev-selected violation probability is at least 0.95 |
| `released_field_oracle` | Drop known released-field scope, policy, or lifecycle violations; retain unresolved items |

The reader sees the query, optional as-of time, candidate text, and visible record order.
It never sees arm names, hidden memory IDs, relevance labels, or admissibility gold. Exact
reader-visible prompts may be evaluated once and shared across arms only when their bytes
are identical. Reader outputs are capped at 2,048 provider tokens; the earlier 512-token
pilot was frozen unscored after a valid Claude response reached the cap.

The v2 protocol may reuse the completed v1 verifier bundle because the only substantive
execution change is the downstream reader output cap. The import is allowed only after
all 3,767 verifier payload hashes, OpenAI request-body hashes, current-schema parses, and
the response-set hash match exactly. Imported envelopes retain source-record hashes and
a private compatibility manifest; verifier response content is unchanged.

## Gold coverage

- Answer utility: 3,767/3,767 queries.
- Source-defined protected targets: 2,548/3,244 MemOps queries.
- RHELM protected disclosure: unevaluable, because RHELM does not release an equivalent
  answer-level protected target.

RHELM eligibility is restricted to `conversation_only` questions whose released
supporting references are all uniquely resolvable (`all_references_unique`). Of 1,305
RHELM QA pairs at the pinned source revision, 859 meet both filters; the frozen split
assigns 523 of those eligible questions to evaluation.

Protected disclosure is therefore reported with an explicit denominator and never
imputed for RHELM. Retrieval and prompt-exposure metrics cover both sources.

## Metrics

Retrieval metrics are evidence recall, feasible rate at target recall 0.8, penalized
admissibility upper risk, route width, and typed wrong-namespace, policy, lifecycle, and
unresolved exposure. Answer metrics are blinded semantic accuracy, answer quality,
non-answer rate, reference containment, source-defined protected disclosure, and stale
disclosure. The frozen aggregate CSVs retain the legacy field name `over_refusal`, but
its exact implementation is `reader_action != "answer"`; it is not a gold-action-aware
over-refusal label. Reader models are reported separately. Paired 95% intervals use
10,000 namespace-group bootstrap replicates.

## Execution

First build and validate a public retrieval run as described in
[the reproducibility guide](../REPRODUCIBILITY.md#6-natural-corpus-reproduction).
Then materialize its top-20 cases:

```powershell
uv run --no-sync python -m scripts.materialize_natural_end_to_end_cases `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction
```

Cases, provider records, and new results are written below
`tmp/natural_reproduction/natural_end_to_end/`. All provider commands use this prefix:

```powershell
uv run --no-sync python -m scripts.run_natural_end_to_end_experiment `
  --mode reproduction --run-root tmp/natural_reproduction validate
```

Replace the final `validate` with the commands below, in order. `plan-*` commands
prepare requests and cost estimates without contacting providers. `fixture` makes
one synthetic API call to verify the configured provider contract; `execute-*`
runs the actual stage. Credentials are read locally from the repository's ignored
`.env`, or the file specified with `--dotenv` before the command.

| Stage | Commands after the common prefix |
| --- | --- |
| Verifier | `plan-verifier`, `plan-fixture --stage verifier`, `fixture --stage verifier`, `execute-verifier` |
| Readers | `plan-readers`; then `plan-fixture --stage reader --provider Gemini`, `fixture --stage reader --provider Gemini`, `execute-reader --provider Gemini`; repeat for `DeepSeek` |
| Judge | `plan-judge`, `plan-fixture --stage judge`, `fixture --stage judge`, `execute-judge` |
| Aggregation | `score` |

The default reproduction profile uses the frozen Gemini/DeepSeek reader panel and
the final dated Haiku judge configuration. The verifier and readers use the 3,767
query population; semantic scoring uses all 523 RHELM cases and the fixed,
hash-selected 1,000 MemOps cases. The run records its protocol, prompts,
materialization, and sampling rule; the score manifest records the selected case-ID
set's hash. The separately released GPT reader
replication and its results retain their own protocol and execution records.

A fresh run creates its own receipts. Historical mode and its receipt checks remain
available for replaying original archives; they are not required by reproduction
mode. Model outputs can differ across new executions, so new results are saved
separately from the checked-in paper results.

Every successful call is checkpointed separately. A provider failure stops new
submissions after the small in-flight window drains; completed calls remain
resumable. Each provider stage holds an atomic single-writer lock. Full-bundle
coverage, request hashes, and cost caps are checked before scoring. No semantic
repair or outcome-selective rerun is performed.
