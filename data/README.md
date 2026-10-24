# Data

This repository separates checked-in evaluation inputs from externally acquired raw
sources and non-redistributable execution material.

## What Is Included

| Input | Location | Purpose |
| --- | --- | --- |
| Synthetic retrieval cases | `tests/fixtures/retrieval_cases.jsonl` | Deterministic local smoke for every retrieval and scoring path |
| Controlled admissibility cases | `experiments/counterfactual_admissibility_cases.json` | Public fixed-pool verifier diagnostic |
| Controlled exposure targets | `experiments/counterfactual_exposure_targets.json` | Public paired include/withhold construction |
| Frozen protocols and prompts | `experiments/` | Exact settings, grids, model contracts, and prompt templates |
| Content-free results | `results/` | Aggregate metrics, pair-level derived scores, bootstrap intervals, manifests, and receipts |
| Normalized evidence | `evidence/` | Claim-bound measurements and source-hash manifests |

No checked-in file contains a private user conversation, credential, embedding shard,
raw provider response, or hidden benchmark payload.

## Public Sources

`upstream_sources.json` pins the exact repository commit and Git tree used for each
public source:

| Source | Revision | Role |
| --- | --- | --- |
| GateMem | `603f9f4b4ba4b77f043c20f85687fa016fd720b0` | Observational exposure/disclosure audit |
| RHELM | `726fc42ea0d5c11085c1a961ade10d702ba187d3` | Natural retrieval and route-to-reader evaluation |
| MemOps | `0b85a27bba856c287405a36e53a39e59e424c4e9` | Natural retrieval and lifecycle-operation evaluation |

Validate the manifest without network access:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources validate
```

Fetch the exact upstream checkouts into the ignored `data/raw/upstream/` directory:

```powershell
uv run --extra dev python -m scripts.fetch_public_sources fetch
uv run --extra dev python -m scripts.fetch_public_sources verify
```

Use `--source rhelm`, for example, to fetch or verify one source. The fetcher never
deletes an existing path, refuses a mismatched Git remote, and verifies both commit
and tree identities after checkout.

## Redistribution Boundary

The upstream repositories are public and have repository-level MIT licenses at the
pinned revisions. That does not automatically settle redistribution rights for every
incorporated conversation or dataset. In particular, MemOps reports use of UltraChat.
For that reason this repository fetches raw data from upstream owners but does not
copy raw benchmark text into its own Git history.

The following derived material also remains excluded:

- 4,096-dimensional Qwen3 embedding shards;
- full route and score checkpoints containing per-query identifiers;
- reader and judge prompts with benchmark text;
- raw provider responses and private scored derivatives; and
- credentials or local `.env` files.

Their frozen identities, counts, and SHA-256 receipts are recorded in
`PROVENANCE.md`, `SOURCE_ARTIFACTS.yaml`, and result manifests. Checked-in aggregate
results can be verified without these materials. The
[public reproduction pipeline](../REPRODUCIBILITY.md#6-natural-corpus-reproduction)
builds inputs from these source checkouts and writes new embeddings, rankings, and
provider outputs to an ignored run directory. It uses the released source adapters
and does not require the original private archive.

See `docs/LICENSE_AUDIT.md` and `THIRD_PARTY_NOTICES.md` for the complete license
decision.
