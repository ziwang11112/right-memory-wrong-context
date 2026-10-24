# Third-Party Sources

The frozen repository-level license audit found MIT licenses for all three upstream
repositories. Benchmark payloads are fetched from their owners rather than
redistributed here. See `docs/LICENSE_AUDIT.md` for the data boundary.

## GateMem

- Repository: `https://github.com/rzhub/GateMem.git`
- Frozen revision: `603f9f4b4ba4b77f043c20f85687fa016fd720b0`
- Repository license: MIT, Copyright (c) 2026 Renz
- Use here: source identity and content-free aggregate evaluation evidence only.

## RHELM

- Repository: `https://github.com/microsoft/RHELM.git`
- Frozen revision: `726fc42ea0d5c11085c1a961ade10d702ba187d3`
- Repository license: MIT, Copyright (c) Microsoft Corporation
- Use here: public-source input construction and content-free evaluation evidence.
  This repository provides project-authored adapters; raw benchmark data are
  fetched from the pinned upstream revision.

## MemOps

- Repository: `https://github.com/MemTensor/MemOps.git`
- Frozen revision: `0b85a27bba856c287405a36e53a39e59e424c4e9`
- Repository license: MIT, Copyright (c) 2026 MemTensor
- Use here: public-source input construction and content-free evaluation evidence.
  This repository provides project-authored adapters; raw benchmark data are
  fetched from the pinned upstream revision.

## Optional retrieval and encoder dependencies

- `rank-bm25` is installed as a dependency under its Apache-2.0 license; its source
  is not vendored.
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) weights retain
  their Apache-2.0 license. The model revision is pinned in the experiment
  protocol; weights are downloaded separately and are not included in this release.

Repository visibility, public availability, and a repository-level MIT license do
not by themselves settle the terms of incorporated source datasets. Raw source data
remain excluded. The root `LICENSE` covers this repository's original code and
documentation only; it does not broaden rights in upstream datasets or artifacts.
