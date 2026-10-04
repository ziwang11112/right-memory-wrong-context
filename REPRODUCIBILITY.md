# Reproducibility Guide

This guide separates checks that are fully reproducible from this checkout from
historical executions that require external public data, model artifacts, or provider
access. All commands are run from the repository root.

The accompanying paper is **The Right Memory in the Wrong Context: Verifying
Retrieval Admissibility in Long-Term Agent Memory**. The
[experiment guide](docs/PAPER_ALIGNMENT.md) maps the core evaluations to their
implementations and released assets.
Passing file-integrity checks establishes agreement with the released snapshot; it
does not by itself independently validate the original labels or model responses.

## 1. Environment

Requirements:

- Python 3.11 or newer;
- Git 2.30 or newer;
- `uv` for the locked environment; and
- no GPU or API key for the default verification path.

Install the test and plotting dependencies:

```powershell
uv sync --locked --extra dev --extra plots
```

## 2. Verify the Released Package

This tier is deterministic, offline after dependency installation, and makes no model
or provider call:

```powershell
uv run --extra dev --extra plots python -m pytest
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
uv run --extra dev python -m scripts.check_reproducibility_package
```

The checks validate schemas, formulas, frozen settings, source hashes, normalized
evidence, result manifests, strict JSON/CSV parsing, secret exclusions, and the absence
of release-excluded private-data files.

## 3. Run the Local Retrieval Smoke

The synthetic fixture exercises the same normalized schema as the natural evaluation:

```powershell
uv run --extra dev python -m scripts.run_retrieval_experiment validate `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json

uv run --extra dev python -m scripts.run_retrieval_experiment run `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/retrieval_routes.jsonl

uv run --extra dev python -m scripts.run_retrieval_experiment select `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/selected_settings.jsonl
```

Run deterministic metadata and fixed-budget diagnostics on the same fixture:

```powershell
uv run --extra dev python -m scripts.run_metadata_robustness `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --retrieval-protocol experiments/metadata_robustness_retrieval_protocol.json `
  --robustness-protocol experiments/metadata_robustness_protocol.json `
  --output tmp/metadata_curve.jsonl `
  --break-even-output tmp/metadata_break_even.jsonl

uv run --extra dev python -m scripts.run_top_k_pareto `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --retrieval-protocol experiments/frozen_natural_protocol.json `
  --pareto-protocol experiments/top_k_pareto_protocol.json `
  --output tmp/top_k_pareto.jsonl
```

These are implementation checks, not benchmark estimates.

## 4. Validate Controlled Inputs and Results

Materialize provider-neutral paired exposure requests without calling a provider:

```powershell
uv run --extra dev python -m scripts.run_counterfactual_exposure_intervention validate
uv run --extra dev python -m scripts.run_counterfactual_exposure_intervention requests `
  --model local-contract-smoke `
  --output tmp/counterfactual_exposure/requests.jsonl
```

Verify the checked-in derived result packages and regenerate their public plots:

```powershell
uv run --extra dev python -m scripts.publish_supplemental_results verify
uv run --extra dev python -m scripts.publish_counterfactual_exposure_results verify
uv run --extra dev python -m scripts.publish_claude_opus5_exposure_results verify
uv run --extra dev python -m scripts.publish_natural_case_audit verify
uv run --extra dev python -m pytest tests/test_posthoc_robustness_results.py
uv run --extra dev python -m pytest tests/test_policy_axis_sensitivity.py
uv run --extra dev --extra plots python -m scripts.plot_counterfactual_selectivity_figure
uv run --extra dev --extra plots python -m scripts.plot_counterfactual_exposure_figure
```

Plot scripts consume checked-in score tables only. They do not read credentials or raw
provider responses.

The checked-in support-control and operating-curve bundles are fully hash-verifiable
from Git. Recomputing them from frozen rankings or provider predictions additionally
requires the excluded provenance archive. The following commands are for researchers
who already possess those historical inputs; they are not a clean-checkout public
reproduction path. Replace the archive location with the local copy you hold:

```powershell
python -m scripts.run_frozen_natural_support_controls `
  --archive-root ../bomi-codex-starter `
  --output-dir tmp/support_controls

python scripts/run_policy_axis_sensitivity.py `
  --archive-root ../bomi-codex-starter `
  --output-dir tmp/policy_axis_sensitivity

python -m scripts.run_submission_zero_call_diagnostics `
  --archive-root ../bomi-codex-starter `
  --output-dir tmp/submission_zero_call_diagnostics

python -m scripts.run_frozen_posthoc_robustness `
  --cases tmp/inferred_admissibility/cases.jsonl `
  --response-dir tmp/inferred_admissibility_canonical/primary_first_committed `
  --output-dir tmp/posthoc_robustness
```

The support-control, policy-sensitivity, and submission-diagnostic commands read
frozen route bundles; the submission diagnostic additionally reuses the frozen
embedding table for an exact gold-preserving same-size oracle control. The
operating-curve command reads frozen structured verifier responses. All four are
zero-call post-hoc analyses and none reads credentials or contacts a provider.

## 5. Acquire Exact Public Sources

Validate the pinned source registry without network access:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources validate
```

Fetch and verify GateMem, RHELM, and MemOps from their owners:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources fetch
uv run --extra dev python -m scripts.fetch_public_sources verify
```

Checkouts are placed in ignored `data/raw/upstream/` directories. The tool verifies
both exact commit and Git tree hashes and refuses to replace a non-Git path or use a
mismatched remote.

## 6. Full Re-execution

The supported levels differ by experiment. Verification of a result package,
regeneration from released numeric scores, and a new data-to-model execution are
separate operations:

| Tier | Supported path | Additional material |
| --- | --- | --- |
| Code behavior and synthetic smoke | Yes | None |
| Checked-in result/evidence file integrity and claim consistency | Yes | None |
| Natural case-level aggregation, equal-source results, and case-weighted sensitivity | Yes | Tokenized `results/natural_end_to_end_case_audit/case_scores.csv` |
| Controlled paired-exposure request construction and score aggregation | Yes | Included scenarios/targets; a complete response bundle for a new scoring run |
| Exact historical natural rankings and post-hoc control recomputation | Archive-dependent | Excluded population, embeddings, route checkpoints, and archive implementation |
| Historical text-verifier/provider execution replay | Archive-dependent | Excluded case/materialization and complete provider-response bundles |
| Fresh natural-corpus/provider execution | Requires a new execution setup | Pinned upstream sources, input construction, model artifacts, credentials, and the runner's exact execution contract |

The exact reported natural execution used 182,908 memories, 3,767 queries, 33,903
route rows, and 33,903 score rows. Its source revisions, config, population,
embedding, route, and score hashes are recorded in `PROVENANCE.md`. Those hashes
identify the historical inputs; they do not make the inputs downloadable from this
repository. Fetching the same upstream revisions does not alone reproduce the
omitted construction, embeddings, rankings, or provider responses. The public
retrieval runner accepts normalized case bundles, and natural provider runners
require materialized cases and matching receipts in addition to upstream sources.
No single public command currently reconstructs every historical input from a fresh
clone. New execution outputs belong in ignored local directories and must be
reported separately from the frozen paper results.

The case-level audit exposes parsed numeric labels and SHA-256 bindings, so the public
package can recompute source-specific intervals, the post-hoc case-weighted
sensitivity, and natural closure aggregates from the released numeric labels.
It cannot independently replay the historical payload-to-score boundary because
the benchmark payloads and original provider records are absent. The provider
runners construct requests from supplied materialized cases and enforce complete-
bundle, cost-cap, and frozen-contract requirements. Credentials remain local to the
researcher; historical execution receipts do not unlock a new paid run.

Independent human-annotation evidence is outside this release's validated scope.
The 200-output alternate-model-judge audit evaluates model outputs and is a
separate analysis. See the [experiment guide](docs/PAPER_ALIGNMENT.md) for the
implemented evaluations and their reproduction boundaries.

## 7. Directory Contract

```text
src/           pure evaluation library
scripts/       local runners, importers, publishers, and audits
experiments/   frozen protocols, prompts, and constructed inputs
data/          upstream revision registry and redistribution policy
results/       content-free derived result packages
evidence/      normalized claim-bound evidence and hash manifests
claims/        machine-readable interpretation boundaries
tests/         unit, invariant, provenance, and end-to-end smoke tests
docs/          method and execution contracts
```

See `data/README.md`, `experiments/README.md`, and `results/README.md` for complete
inventories.
