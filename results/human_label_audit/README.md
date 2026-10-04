# Human-label repeatability audit

This is the content-free aggregate release for the paper's audit of **207
record-query pairs in 20 blinded packets**, independently annotated by two humans
before adjudication (claim C15). It is distinct from the 200-output model
cross-judge audit in `results/natural_cross_judge_audit/`.

| Axis | Exact agreement | Nominal Krippendorff alpha |
| --- | ---: | ---: |
| Relevance | 0.8937 | 0.8135 |
| Scope | 0.9952 | 0.9808 |
| Lifecycle state | 0.8599 | 0.6703 |
| Prohibited status | 0.8841 | 0.0829 |

`summary.csv` retains the source precision and count denominators. Exact agreement
uses all 207 pairs (including uncertainty as a category); uncertainty rate uses
414 individual labels. Alpha omits pairs with an uncertain label: 184 eligible
pairs for lifecycle state, 207 for the other axes. No confidence intervals were
reported for this audit, so none are manufactured in the release.

The frozen historical summary's `usable_evidence` composite omits lifecycle. It is
not exported as an independent human-label axis and must not be interpreted as the
paper's current definition of usable evidence.

## Verification and boundary

From the repository root:

```sh
uv run --locked --extra dev python -m scripts.publish_human_label_audit verify
uv run --locked --extra dev python scripts/verify_evidence.py
```

The first command checks the summary hash, count/rate arithmetic and its parity
with normalized evidence. The second checks the claim values, source receipts and
transformation hash. The original aggregate JSON is bound to a frozen source
revision and SHA-256 in `manifest.json` and `SOURCE_ARTIFACTS.yaml`.

**Independent raw annotator labels and original packet text are not included.**
This release preserves reported alpha values; it does not recompute them from
independent observations or establish that the annotators were independent. The
source report and paper record that study design. Low prohibited-status alpha
remains a limitation, and this sampled audit is not full-population human gold.

Maintainers who hold the exact frozen aggregate can recreate this public release
without provider calls or reviewer identities:

```sh
uv run --locked --extra dev python -m scripts.publish_human_label_audit publish --source /path/to/human_agreement_summary.json
```
