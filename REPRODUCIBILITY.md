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

The checked-in support-control and operating-curve bundles are hash-verifiable from
Git. Section 6 builds new natural rankings for the support diagnostics. Operating
curves can also be recalculated from a complete local verifier-response bundle:

```powershell
python -m scripts.run_frozen_posthoc_robustness `
  --cases tmp/inferred_admissibility/cases.jsonl `
  --response-dir tmp/inferred_admissibility_canonical/primary_first_committed `
  --output-dir tmp/posthoc_robustness
```

This command reads structured verifier responses and makes no provider calls.

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

## 6. Natural-Corpus Reproduction

The public pipeline constructs inputs from the pinned sources, generates Qwen
embeddings, runs the nine frozen retrieval configurations, and materializes
route-to-reader cases. It imports only this repository's modules and installed
dependencies. Choose a new output directory for each execution; the examples use
the ignored `tmp/natural_reproduction/` directory.

### Build the inputs

```powershell
uv sync --locked --extra dev --extra plots --extra reproduction
uv run --no-sync python -m scripts.fetch_public_sources fetch --source rhelm --source memops
uv run --no-sync python -m scripts.run_public_natural_retrieval build `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction
```

The builder checks exact source revisions, split assignments, and the expected
182,908 memories and 3,767 evaluation queries. It reconstructs 176,512 unique
embedding identities. The query prefix, source-text hashes, and frozen retrieval
settings are bound to the run manifest. This step needs no model or API key.

### Encode and retrieve

Install the optional local-model runtime and download the pinned model snapshot:

```powershell
uv sync --locked --extra dev --extra plots --extra encoder
uv run --no-sync python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-Embedding-8B', revision='1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'))"
```

Pass the returned snapshot directory as `--model-path`:

```powershell
uv run --no-sync python -m scripts.run_public_natural_retrieval embed `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction `
  --model-path /path/to/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af
uv run --no-sync python -m scripts.run_public_natural_retrieval route `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction
uv run --no-sync python -m scripts.run_public_natural_retrieval validate `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction
```

Embedding runs the 8B model locally and requires sufficient GPU/system memory.
The encoder uses the frozen instruction, 512-token truncation, last-token pooling,
and 4,096-dimensional normalized float32 vectors. Retrieval preserves the selected
parameters, source visibility rules, and deterministic memory-ID tie breaking.
The complete run contains 18 source-by-method bundles and 33,903 route rows.
Completed shards and route bundles can be resumed using the same run directory;
their input bindings and hashes are checked before reuse.

### Recompute support diagnostics

```powershell
uv run --no-sync python -m scripts.run_frozen_natural_support_controls `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction `
  --output-dir tmp/natural_reproduction/support_controls
uv run --no-sync python -m scripts.run_policy_axis_sensitivity `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction `
  --output-dir tmp/natural_reproduction/policy_axis_sensitivity
uv run --no-sync python -m scripts.run_submission_zero_call_diagnostics `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction `
  --output-dir tmp/natural_reproduction/submission_diagnostics
```

These commands read the new run's rankings and embeddings without model calls.
They compare arms within that run and retain its input hashes. The explicit
`--archive-root` compatibility option remains available for historical archives.

### Materialize and run the readers

```powershell
uv run --no-sync python -m scripts.materialize_natural_end_to_end_cases `
  --source-root data/raw/upstream --run-root tmp/natural_reproduction
uv run --no-sync python -m scripts.run_natural_end_to_end_experiment `
  --mode reproduction --run-root tmp/natural_reproduction validate
uv run --no-sync python -m scripts.run_natural_end_to_end_experiment `
  --mode reproduction --run-root tmp/natural_reproduction plan-verifier
```

Validation and planning make no provider calls. Follow the
[provider execution guide](docs/NATURAL_END_TO_END_EXPERIMENT.md#execution) to run
the verifier, readers, and judge with your own credentials. Fresh execution binds
receipts to this run's actual input hashes; historical receipts cannot be reused
as authorization for it. Stage completion, model settings, and cost limits remain
checked by the runner.

## 7. Reproduction Levels

| Tier | Supported path | Additional material |
| --- | --- | --- |
| Code behavior and synthetic smoke | Yes | None |
| Checked-in result/evidence file integrity and claim consistency | Yes | None |
| Natural case-level aggregation, equal-source results, and case-weighted sensitivity | Yes | Tokenized `results/natural_end_to_end_case_audit/case_scores.csv` |
| Controlled paired-exposure request construction and score aggregation | Yes | Included scenarios/targets; a complete response bundle for a new scoring run |
| Natural input construction and nine-arm retrieval | Public pipeline in Section 6 | Pinned RHELM/MemOps checkouts and Qwen weights |
| Support controls and policy sensitivity on new rankings | Public pipeline in Section 6 | This run's rankings and embeddings |
| Fresh natural route-to-reader execution | Public reproduction mode | Materialized cases, provider access, and complete stage bundles |
| Byte-for-byte historical ranking or provider replay | Archive-dependent | Original embeddings, rankings, or raw provider responses |

The exact reported natural execution used 182,908 memories, 3,767 queries, 33,903
route rows, and 33,903 score rows. Its source revisions, config, population,
embedding, route, and score hashes are recorded in `PROVENANCE.md`. Those hashes
identify the historical execution. The public pipeline reconstructs the inputs and
generates new model outputs; it does not download historical caches. New execution
outputs belong in ignored local directories and are recorded separately from the
frozen paper results.

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

## 8. Directory Contract

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
