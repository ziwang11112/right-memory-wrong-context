# Evidence Package

The `normalized/` directory contains content-free aggregate measurements for claims
C2-C14. It does not contain benchmark conversations, query text, memory text, model
responses, embeddings, or per-query identifiers.

The `examples/` directory contains four deliberately selected, abridged
public-source query-memory cases for Figure 1 and claim C1. It contains no private
user data, model responses, or embeddings. Its packet and memory
identifiers are retained solely to make the qualitative examples auditable.

## Evidence Families

| File | Population boundary |
| --- | --- |
| `normalized/gatemem.csv` | Separate fixed-reader exposure/disclosure associations and a clean-reader leakage-utility trade-off |
| `normalized/mechanism_smoke.csv` | Sixteen curated candidate-pool evaluation packets only |
| `normalized/natural_evaluation.csv` | Source-macro evaluation over 87 groups, 182,908 memories, and 3,767 queries |
| `normalized/counterfactual_exposure.csv` | Reader-separated paired prompt interventions over 16 controlled scenarios |
| `normalized/claude_opus5_exposure_replication.csv` | Separately executed fourth-provider replication of the same paired prompt construction; never pooled with the original three-reader execution |
| `normalized/fixed_budget_support.csv` | Frozen top-k support, recall, risk, policy-axis sensitivity, evaluator-label missingness, gold-preserving oracle, and algorithmic-cost contrasts |
| `normalized/metadata_reliability.csv` | Corrected axis attribution and observed metadata-corruption brackets |
| `normalized/text_inferred_admissibility.csv` | Label-aligned released-field reference versus text-inferred filtering on fixed natural-development cases |
| `normalized/controlled_selective_verification.csv` | Controlled focal flips, stable controls, and selective-verification error rates |
| `normalized/natural_end_to_end.csv` | Reader-separated route-to-answer deltas on the frozen 1,523-case natural sample; GPT is sequential and all readers share one blinded judge |
| `normalized/cross_judge_audit.csv` | Post-hoc, outcome-independent agreement audit on 200 exact-deduplicated natural outputs; one alternate judge and no full-population re-score |

Every CSV uses the same schema:

```text
claim_id,family,population,source,contrast,metric,estimate,
ci95_lower,ci95_upper,n,notes
```

Blank confidence bounds mean that the frozen source did not report an interval for
that row. They are not zeroes. Delta rows are method minus reference, as named in
`contrast`. Source-macro values weight the RHELM and MemOps source estimates equally;
they are not pooled-query estimates.

## Qualitative Examples

`examples/retrieval_admissibility_cases.json` records the four Figure 1 cases:
wrong namespace, superseded current state, an explicit forget request, and a history
query that legitimately requires older states. The source artifact hashes and frozen
repository commit are recorded in the file. Text is abridged for legibility, while
the source labels and use/drop verdicts are unchanged. These cases illustrate failure
modes; they do not estimate prevalence.

## Integrity

Each file has a JSON receipt in `manifests/` containing:

- its normalized SHA-256 and row count;
- every frozen source path and SHA-256 used to derive it;
- the read-only source snapshot identity; and
- the exact transformation-script SHA-256.

Run:

```powershell
uv run --extra dev python scripts/verify_evidence.py
```

The verifier rejects changed CSVs, changed transformation code, unindexed or changed
source hashes, path traversal, non-finite values, malformed intervals, retired method
labels, values or intervals that drift from `claims/claims.yaml`, and missing
claim-specific interpretation boundaries. The original paired-exposure evidence and
the Claude Opus 5 replication are each bound to their own exact execution commit and
content-free publication manifest. The natural end-to-end family is bound to both
complete result bundles and preserves its no-pooling, shared-judge, and sequential-
replication boundaries. The cross-judge family is separately bound to its exact
provider-execution commit and preserves its post-hoc, 200-output, one-alternate-judge
scope. The qualitative example JSON is separately hash-bound to its upstream source
artifacts and validated by the evidence tests.
