# Result Packages

All tracked result packages are content-free derivatives. They may contain aggregate
metrics, pair-level binary/numeric scores, bootstrap intervals, model/token/cost
receipts, and hashes. They do not contain benchmark conversations, reference answers,
reader prompts, raw provider responses, embeddings, or credentials.

| Directory | Evaluation | Main artifacts |
| --- | --- | --- |
| `supplemental_natural/` | Natural retrieval, top-k Pareto, released-field attribution, metadata break-even | CSV/JSON summaries and manifests |
| `support_controls/` | Size-matched support, pre/post-filter depth, recall/cost sensitivity, few-group robustness | Aggregate CSVs, summary, manifest |
| `policy_axis_sensitivity/` | Frozen top-20 admissibility rescore with and without the released policy predicate | Source and macro summaries, paired bootstrap deltas, manifest |
| `submission_zero_call_diagnostics/` | Evaluator-label missingness and gold-preserving same-size support diagnostics | Seed summaries, aggregate tables, manifest |
| `inferred_admissibility/` | Natural-development text verifier | Classification, route, threshold, sensitivity, usage, and manifest files |
| `posthoc_robustness/` | Fixed verifier operating curves and paired-exposure scenario robustness | Curve CSV, leave-one-out summary, manifest |
| `counterfactual_admissibility/` | Controlled focal/stable verifier diagnostic | Pair scores, bootstrap intervals, error taxonomy, figures, manifest |
| `counterfactual_exposure/` | Original three-reader paired exposure intervention | Pair scores, cell metrics, bootstrap intervals, usage, figures, manifest |
| `claude_opus5_exposure_replication/` | Separately executed fourth-reader replication | Pair scores, cell metrics, bootstrap intervals, usage, manifest |
| `natural_end_to_end_two_reader_deterministic/` | Deterministic retrieval-side closure analysis | Main table, paired deltas, summary, manifest |
| `natural_end_to_end_two_reader_judged/` | DeepSeek/Gemini natural route-to-reader results | Main table, paired deltas, completion receipt, summary, manifest |
| `natural_end_to_end_gpt_luna_judged/` | Sequential GPT-reader replication | Main table, paired deltas, completion receipt, summary, manifest |
| `natural_end_to_end_case_audit/` | Content-free case-level natural closure audit | Tokenized case scores, source-specific intervals, case-weighted sensitivity, population summary, manifest |
| `natural_cross_judge_audit/` | Outcome-independent 200-output alternate-judge audit | Agreement, confusion, subgroup, usage, summary, manifest |

## Verification

The normalized subset used for claim-level reporting is under `evidence/normalized/`.
Each evidence family has a manifest binding source paths, source SHA-256 values,
transformation code, row count, and normalized output hash.

Run all content and provenance checks without network or provider calls:

```powershell
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
uv run --extra dev python -m scripts.check_reproducibility_package
uv run --extra dev python -m scripts.publish_supplemental_results verify
uv run --extra dev python -m scripts.publish_counterfactual_exposure_results verify
uv run --extra dev python -m scripts.publish_claude_opus5_exposure_results verify
uv run --extra dev python -m scripts.publish_natural_case_audit verify
uv run --extra dev python -m pytest tests/test_posthoc_robustness_results.py
uv run --extra dev python -m pytest tests/test_policy_axis_sensitivity.py
uv run --extra dev python -m pytest tests/test_submission_diagnostic_results.py
```

Interpretation boundaries are maintained in `CLAIM_CONTRACT.md` and
`claims/claims.yaml`. Results from distinct readers, populations, or sequential
replications must not be pooled unless the corresponding contract defines that
estimand.

The original support-size control is deliberately harsh: it preserves the source-level
namespace label counts while randomly reassigning those labels. Its failure shows
that an arbitrary smaller pool alone does not recover the released evidence support.
A separate non-deployable control retains every released required anchor and samples
non-anchors to exactly the namespace support size. It reaches `0.840` recall and
`0.747` feasibility, versus `0.533` and `0.311` for namespace pre-filtering. This
oracle exposes substantial support/ranking headroom: trusted namespace is an
available provenance constraint, not a claim that namespace identity is necessary
or sufficient for optimal retrieval. At the same
time, the scope-excluded conditional risk among feasible queries is `0.1261` for
namespace pre-filtering and `0.1229` for global dense. The lower primary penalized
residual risk therefore comes from improved feasibility under the preregistered
infeasibility cost, not from a demonstrated policy/lifecycle improvement. Namespace
support should not be interpreted as a substitute for those checks.

The evaluator-label missingness diagnostic changes neither routes nor recall. Hiding
20% of established composite admissibility judgments lowers matched-prefix coverage
from about `0.992` to `0.786`/`0.793` and widens the risk interval from about `0.008`
to `0.214`/`0.207` for global/namespace routes. It demonstrates why unresolved labels
need explicit coverage and partial-identification bounds; it is not a deployment-time
metadata-corruption model.

The policy-axis sensitivity keeps the same rankings, anchors, scope labels,
lifecycle labels, route limit, recall target, and infeasibility penalty. Omitting
only the released policy-disallowed predicate leaves the namespace-minus-global
penalized upper-risk delta at `-0.0933` (95% CI `[-0.1147, -0.0714]`). This is a
robustness analysis of the namespace result, not evidence that policy verification
is unnecessary.
