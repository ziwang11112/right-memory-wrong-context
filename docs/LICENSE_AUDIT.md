# Phase 2 License Audit

Audit date: 2026-07-27

This audit covers the exact upstream revisions recorded in `PROVENANCE.md`. It
supports clean-room-style reimplementation of evaluation logic and import of
content-free aggregate measurements. It does not authorize redistribution of raw
benchmark text.

## Repository-Level Licenses

| Source | Frozen revision | Repository license | Copyright notice |
| --- | --- | --- | --- |
| GateMem | `603f9f4b4ba4b77f043c20f85687fa016fd720b0` | MIT | Copyright (c) 2026 Renz |
| RHELM | `726fc42ea0d5c11085c1a961ade10d702ba187d3` | MIT | Copyright (c) Microsoft Corporation |
| MemOps | `0b85a27bba856c287405a36e53a39e59e424c4e9` | MIT | Copyright (c) 2026 MemTensor |

The license texts were read from each repository's `LICENSE` file at the frozen
revision. GateMem's README also displays an MIT badge. RHELM and MemOps include
benchmark data inside or alongside their repositories, but this audit does not
assume that every upstream or incorporated data source inherits the repository
license.

## Additional Data Boundary

MemOps states that its data construction uses UltraChat. The licensing and
redistribution conditions of incorporated source text therefore require a separate
review before any raw MemOps conversation is copied. This project does not copy that
text in Phase 2.

RHELM and GateMem raw benchmark payloads are likewise excluded. Phase 2 imports only
content-free aggregate measurements produced by the internal provenance archive.

## Internal Source Boundary

The original internal source snapshot has no root `LICENSE` file. The authors have
authorized release of their experiment implementation. The public package adapts
the source construction, retrieval, and scoring logic under its own package
structure and preserves source-file identities in `SOURCE_ARTIFACTS.yaml`.
Internal execution locks, private caches, and provider recovery records are not
part of the port. This does not change the upstream data boundary above.

## Decision

- Pure evaluation ideas and standard statistical formulas may be reimplemented.
- Upstream MIT notices remain recorded in `THIRD_PARTY_NOTICES.md`.
- Content-free aggregate evidence may be normalized after hash verification.
- Raw benchmark text, prompts, responses, embeddings, and checkpoints remain
  prohibited.
- The repository's original code and documentation are released under the MIT
  License in the root `LICENSE` file. That license does not broaden any upstream data
  right: raw benchmark text, prompts, responses, embeddings, and checkpoints remain
  excluded, and third-party notices remain controlling for incorporated material.
