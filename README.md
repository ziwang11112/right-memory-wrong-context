# The Right Memory in the Wrong Context: Verifying Retrieval Admissibility in Long-Term Agent Memory

Code and released evidence accompanying the paper by **Zi Wang, Xingqiao Wang,
Emmanuel Addai, Devika Ambekar, and Xiaowei Xu**.

University of Arkansas at Little Rock.

[Experiment guide](docs/PAPER_ALIGNMENT.md) ·
[Reproducibility](REPRODUCIBILITY.md) ·
[Results](results/README.md) ·
[Citation](CITATION.cff)

## Overview

A memory can be relevant to a query yet inadmissible for the current principal,
policy, intent, or lifecycle state. This repository implements the paper's
evaluation framework and core experiments, tracing memory use through four stages:

```text
Stored -> Retrieved -> Exposed -> Disclosed
```

| Stage | What the code measures |
| --- | --- |
| Candidate support | Evidence recall, target-recall feasibility, and candidate work |
| Eligibility | Three-valued admissibility decisions, violations, coverage, and risk bounds |
| Exposure | Which memory records enter the reader's prompt |
| Disclosure | Reader-specific answer and literal-disclosure effects |

The implementation uses the paper's definitions:

```text
admissible_z(m) = scope_z(m) AND policy_z(m) AND lifecycle_z(m)
usable_z(m)     = relevant_z(m) AND admissible_z(m)
```

Decisions may be positive, negative, or unresolved. Comparisons at matched evidence
recall separate admissibility from the loss of useful context. See the
[method guide](docs/EXPERIMENT_METHODS.md) for the evaluation details.

## Main Results

On frozen top-20 rankings covering 3,767 RHELM/MemOps queries, trusted namespace
support improves required-evidence recall from **0.432 to 0.533**, increases the
fraction reaching the 0.8 recall target from **0.237 to 0.311**, and reduces exact
similarity evaluations by **98.3%**.

On the separate 1,523-case route-to-reader subset, answer-accuracy differences are
positive across separately reported readers (**+0.053 to +0.068**). These route
contrasts are observational. Controlled exposure experiments report reader-specific
effects and do not establish a general disclosure-reduction claim.

The top-20 analysis is a post-hoc rescore of frozen rankings, distinct from the
top-100 evaluation. Detailed estimates, confidence intervals, sensitivity analyses,
and interpretation boundaries are in [results](results/README.md) and the
[claim contract](CLAIM_CONTRACT.md).

## Quick Start

Requirements: **Python 3.11+**, Git, and [uv](https://docs.astral.sh/uv/).
The example runs locally without a GPU, API key, or private data.

```sh
git clone https://github.com/ziwang11112/right-memory-wrong-context.git
cd right-memory-wrong-context
uv sync --locked --extra dev --extra plots

uv run --no-sync python -m scripts.run_retrieval_experiment validate --cases tests/fixtures/retrieval_cases.jsonl --protocol experiments/frozen_natural_protocol.json
uv run --no-sync python -m scripts.run_retrieval_experiment run --cases tests/fixtures/retrieval_cases.jsonl --protocol experiments/frozen_natural_protocol.json --output tmp/retrieval_routes.jsonl
```

The example writes route-level results to `tmp/retrieval_routes.jsonl`. It uses
synthetic fixtures to demonstrate the interfaces; benchmark reproduction paths are
documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Verify the Release

After installing the locked environment above:

```sh
uv run --no-sync python -m pytest
uv run --no-sync python -m ruff check .
uv run --no-sync python -m ruff format --check .
uv run --no-sync python -m scripts.check_claim_contract
uv run --no-sync python -m scripts.verify_evidence
uv run --no-sync python -m scripts.check_reproducibility_package
```

CI runs these checks and validates an independently installed anonymous export.

## Release Scope

The release includes the evaluation library, experiment protocols, constructed
inputs, derived results, and verification tools. Public pair-level and tokenized
case scores support recalculation of the corresponding aggregates and intervals.
Historical embeddings, complete natural-corpus rankings, and raw provider responses
are not distributed. Recreating those executions requires additional inputs and,
where applicable, model access.

See the [experiment guide](docs/PAPER_ALIGNMENT.md) for code-to-paper mappings and
the [reproducibility guide](REPRODUCIBILITY.md) for runnable commands. The repository
covers the released experiments; it does not claim to reproduce every manuscript
detail.

## Repository Structure

```text
src/           evaluation library
scripts/       experiment runners and verification tools
experiments/   frozen protocols, prompts, and constructed inputs
data/          public-source registry and redistribution guidance
results/       derived experiment results
evidence/      normalized measurements and provenance manifests
claims/        machine-readable claim boundaries
tests/         unit and reproducibility tests
docs/          method and execution guides
```

## Citation and License

Use [CITATION.cff](CITATION.cff) for the author list and repository citation.
A paper identifier will be added when available. The Python package name remains
`verify-agent-memory`.

Original code and documentation are released under the [MIT License](LICENSE).
Third-party materials retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
