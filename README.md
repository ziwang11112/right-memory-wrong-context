# The Right Memory in the Wrong Context: Verifying Retrieval Admissibility in Long-Term Agent Memory

**Code and released evidence accompanying the paper by Zi Wang, Xingqiao Wang,
Emmanuel Addai, Devika Ambekar, and Xiaowei Xu.**

University of Arkansas at Little Rock.

The repository slug and Python package remain `verify-agent-memory`. Earlier
repository descriptions used the working title *The Wrong Memory at the Right
Time*; this is the same research project under the manuscript title above.
See [paper-to-artifact alignment](docs/PAPER_ALIGNMENT.md) for the experiment map
and the limits of the public release. No arXiv identifier or DOI is assigned in
this repository yet.

Agents with long-term memory can retrieve a record that is topically relevant but still
ineligible for the current principal, policy, intent, lifecycle state, or time. This
repository provides the evaluation code, frozen protocols, derived evidence, and
integrity checks used to trace that failure across four observable stages:

```text
Stored -> Retrieved -> Exposed -> Disclosed
```

This is an evaluation and verification artifact, not a new memory index, a production
policy engine, or an official benchmark leaderboard.

## What Is Verified

| Stage | Verification question | Main outputs |
| --- | --- | --- |
| Candidate support | Was the right evidence reachable under the allowed namespace? | Evidence recall, target-recall feasibility, candidate work |
| Eligibility | Was each retrieved record allowed for this query and time? | Three-valued decisions, typed violations, coverage, risk bounds |
| Exposure | Which record IDs crossed the prompt boundary? | Identity-preserving prompt traces |
| Disclosure | Did exposed content appear in the answer? | Reader-specific answer and literal-disclosure effects |

For memory `m` and query context `z = (q, principal, policy, intent, time)`:

```text
admissible_z(m) = scope_z(m) AND policy_z(m) AND lifecycle_z(m)
usable_z(m)     = relevant_z(m) AND admissible_z(m)
```

Every component is three-valued: positive, negative, or unresolved. Known violations
are excluded, while missing evidence remains unresolved and is reflected in coverage
and lower/upper risk bounds. Routes are compared at matched evidence recall so that
returning less useful context cannot appear artificially safe.

See [`docs/EXPERIMENT_METHODS.md`](docs/EXPERIMENT_METHODS.md) for the executable
method contract and [`CLAIM_CONTRACT.md`](CLAIM_CONTRACT.md) for reporting limits.

## Results at a Glance

The checked-in natural evaluation contains 87 namespace groups, 182,908 memories,
and 3,767 RHELM/MemOps queries. On frozen top-20 rankings, trusted namespace support:

- raises evidence recall from `0.432` to `0.533`;
- raises the fraction reaching the `0.8` recall target from `0.237` to `0.311`;
- reduces exact candidate scoring from 90,122 to 1,551 candidates (`98.3%` fewer
  exact similarity evaluations, or `58.1x` fewer);
- reduces feasible prefixes containing any known admissibility violation from
  `0.528` to `0.396`; and
- reduces the mean known-violation count from `2.080` to `1.393`, without reaching
  zero.

An arbitrary size-matched random partition performs poorly, and global retrieval
requires depth 500 before late namespace filtering approaches namespace pre-filter
recall. A non-deployable gold-preserving same-size control instead reaches `0.840`
recall, exposing substantial headroom when released required anchors are known.
Trusted namespace is therefore a practical provenance constraint that improves
anchor reachability, not proof that namespace identity is necessary or sufficient
for optimal retrieval, a safety certificate, or a demonstrated policy/lifecycle
solution.

On a frozen 1,523-case route-to-reader subset, namespace-minus-global answer-accuracy
effects are positive for separately reported DeepSeek, Gemini, and sequential GPT
readers (`+0.053` to `+0.068`). These route contrasts are observational and use one
shared blinded primary judge.

The remaining diagnostics establish important boundaries:

- on 72 public-development cases, a label-aligned released-field reference lowers
  recall-constrained upper loss by `0.032`, while the two tested text-only gates do
  not realize that headroom;
- a zero-call sensitivity that omits the low-agreement policy predicate preserves
  the namespace-support advantage: penalized upper risk changes from `0.787` to
  `0.693` (`delta = -0.093`, 95% CI `[-0.115, -0.071]`);
- a zero-call evaluator-missingness diagnostic leaves retrieval fixed but widens the
  matched-prefix risk interval from about `0.008` at baseline to about `0.21` after
  hiding 20% of established judgments;
- all four controlled readers show positive selectivity between relevant-admissible
  and relevant-inadmissible evidence, but only DeepSeek's relevant-inadmissible
  disclosure effect has a 95% interval excluding zero (`+0.156`, `[0.031, 0.312]`); and
- protected- and stale-disclosure changes are inconclusive, so the artifact makes no
  general disclosure-reduction claim.

The top-100 v1 route-family evaluation is the held-out evaluation of its frozen
non-usable metric. The top-20 admissibility result above is a post-hoc v2 rescore of
the same frozen rankings: it changes no route, ranking, setting, or hyperparameter.
Exact estimates, intervals, and provenance are indexed in
[`results/README.md`](results/README.md), [`evidence/README.md`](evidence/README.md),
and [`claims/claims.yaml`](claims/claims.yaml).

## Quick Start

Requirements: Python 3.11+, Git, and [`uv`](https://docs.astral.sh/uv/). The default
path requires no API key, provider call, GPU, or private data.

```powershell
uv sync --locked --extra dev --extra plots

uv run --extra dev python -m scripts.run_retrieval_experiment validate `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json

uv run --extra dev python -m scripts.run_retrieval_experiment run `
  --cases tests/fixtures/retrieval_cases.jsonl `
  --protocol experiments/frozen_natural_protocol.json `
  --output tmp/retrieval_routes.jsonl
```

These synthetic fixtures exercise the same normalized interfaces and invariants as
the natural evaluation. They are implementation checks, not benchmark estimates.

## Verify the Released Artifact

Run the complete offline verification path:

```powershell
uv run --extra dev --extra plots python -m pytest
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python scripts/check_claim_contract.py
uv run --extra dev python scripts/verify_evidence.py
uv run --extra dev python -m scripts.check_reproducibility_package
uv run --extra dev python -m scripts.publish_human_label_audit verify
```

Verify each derived result family and regenerate released plots with the commands in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The checks reject schema drift,
non-finite values, changed hashes, incomplete result families, release-excluded
paths, and common secret patterns.

## Reproduction Paths

| Material | Availability |
| --- | --- |
| Evaluation library, protocols, prompts, and synthetic fixtures | Included |
| Content-free aggregate, pair-level, and tokenized derived scores | Included and hash-bound |
| GateMem, RHELM, and MemOps source repositories | Fetchable at pinned commits and tree hashes |
| Historical full natural rankings, embeddings, and raw provider bundles | Excluded; hashes document their provenance |
| Human label-repeatability agreement aggregates and provenance | Included in [the audit summary](results/human_label_audit/) |
| Independent human labels for the paper's 207-pair repeatability audit | Not included in this release |

Fetch and verify redistributable public sources from their owners:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources validate
uv run --extra dev python -m scripts.fetch_public_sources fetch
uv run --extra dev python -m scripts.fetch_public_sources verify
```

The default path verifies released result files and their provenance without a
provider call. Public pair-level and tokenized case scores also regenerate the
corresponding aggregates and bootstrap intervals. Recomputing every historical
natural-corpus ranking or verifier operating curve requires inputs from the excluded
provenance archive; fetching upstream repositories alone does not restore them.
Controlled request construction is public, while provider execution has additional
input, credential, and frozen-contract requirements. Fresh model outputs constitute
a new execution, not a replay of the historical provider responses. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`data/README.md`](data/README.md),
[`PROVENANCE.md`](PROVENANCE.md), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Repository Map

```text
src/           pure evaluation library
scripts/       experiment runners, publishers, and integrity checks
experiments/   frozen protocols, prompts, and constructed inputs
data/          public-source registry and redistribution policy
results/       content-free derived result packages
evidence/      normalized measurements and provenance manifests
claims/        machine-readable interpretation boundaries
tests/         unit, invariant, provenance, and reproducibility tests
docs/          method and execution contracts
```

## Citation and License

Use [CITATION.cff](CITATION.cff) for the title, author order, and repository citation.
When referring to the paper, use its full title above. A paper identifier will be
added once available; the repository URL identifies the software artifact and is
not an arXiv publication link.

Original code and documentation are MIT licensed. Third-party datasets and source
repositories retain their own terms. The public repository names the authors;
anonymous review exports are separate artifacts and do not define its citation.
