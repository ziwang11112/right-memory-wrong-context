# Provenance

This repository starts from an empty Git history. It is not a fork and does not
rewrite, replace, or supersede the source experiment history.

The public repository is now `ziwang11112/right-memory-wrong-context`.
Frozen source receipts retain the earlier name `ziwang11112/verify-agent-memory`
where it was recorded at execution time; both names refer to this same project.

## Source Boundaries

| Role | Identity |
| --- | --- |
| Internal provenance archive | `ziwang11112/bomi` |
| Read-only migration snapshot | `a28093110325968c26906223e9eb0f1e078f6aad` |
| Natural-corpus execution commit | `8e34e3d41c56e1699696bc27be95cdac7c9528e5` |
| Natural-corpus config SHA-256 | `ce547d04faa9dac8a1ff213abf971e7e8dd37902c8ddb56e7d395ebe95d9eba5` |
| Population bundle SHA-256 | `14e8dd5a618aa4b830901853a46ba8ce8109a1352fc5730a0fba2cf95ceb9bca` |
| Embedding checkpoint SHA-256 | `d333477116735f49b7628ee5d324e92a9f608190079a7b1d699308f5ec4f1ae6` |
| Route checkpoint SHA-256 | `70ff69fb3d36e8367a16bfe70c796e88c4b3cec5ff4096f0fa7ab1bd0ad709ec` |
| Score rows SHA-256 | `434bb94d089ef06f8ded5156ac52c843554fe4ffe1c058c3d9efefce0440ee71` |

The migration snapshot is the source tree used to inspect and select artifacts. The
execution commit is the implementation revision recorded by the frozen natural-corpus
manifest. These identities are intentionally distinct.

The natural-corpus execution contains 33,903 route rows and 33,903 score rows for
3,767 queries. It recorded zero provider, reader, judge, and paid calls.

## Upstream Public Sources

The following identities were recorded in the frozen source audit:

| Source | Repository | Revision | Tree |
| --- | --- | --- | --- |
| GateMem | `https://github.com/rzhub/GateMem.git` | `603f9f4b4ba4b77f043c20f85687fa016fd720b0` | `1cc06ce6dfcdaf2d34d631db094ef49d3e4d8a09` |
| RHELM | `https://github.com/microsoft/RHELM.git` | `726fc42ea0d5c11085c1a961ade10d702ba187d3` | `15bada413dc52f41177f5817563181243ed936d8` |
| MemOps | `https://github.com/MemTensor/MemOps.git` | `0b85a27bba856c287405a36e53a39e59e424c4e9` | `f9d5cf4ac3c600eecccfec008db4d487f961b5af` |

These identities do not establish redistribution permission for incorporated data.
The repository-level license audit is complete and recorded in
`docs/LICENSE_AUDIT.md`. Raw benchmark data remain excluded.

## Normalized Evidence

Phase 2 reads eligible aggregate artifacts only through
`git show a28093110325968c26906223e9eb0f1e078f6aad:<path>`. Every input is checked
against `SOURCE_ARTIFACTS.yaml` before parsing. Each normalized CSV is bound to its
source paths, source hashes, transformation script, and output hash in
`evidence/manifests/`.

The natural-evaluation importer additionally verifies the frozen execution
manifest: nine selected arms cover all 3,767 queries in both 33,903 route rows and
33,903 score rows, and provider, reader, judge, and paid-call counts are zero.

## Reader Executions

The two GateMem reader estimates are separate same-provider results:

| Reader | Provider | Execution commit | Response bundle SHA-256 |
| --- | --- | --- | --- |
| `gpt-4o-mini-2024-07-18` | OpenAI | `b5d5e4cf49395b4079d16984b877af7b1abbc974` | `ebd2cdf942dff31ab5f8de42b460c6ceb88411babf7dcac724ebf9cc76cf22f2` |
| `gpt-4o-2024-08-06` | OpenAI | `a3d6814cff1f3c3927e473fb6573397e3205ee19` | `10aac8e7900f9498672f51f556f547b1620007c222418d020a78bbc0feeac99a` |

No raw response bundle is authorized for migration. The estimates must not be pooled
or described as a cross-provider replication.

## Controlled Prompt Executions

The paired-exposure evidence comes from two separately hash-bound executions over the
same constructed 16-scenario contract:

| Execution | Readers | Commit | Published manifest SHA-256 |
| --- | --- | --- | --- |
| Original panel | `gpt-5.6-sol`, `gemini-3.6-flash`, `deepseek-v4-pro` | `82d3bce8023d1ccc97bb21b0bbb36e15a4b3c6af` | `a8574e39b2fa23a65491ead0329954251bd9cde51344330ee94e39e3c4afeda1` |
| Reader replication | `claude-opus-5` | `938320909b4c2d13e286987dc4297a7cb6ef73a7` | `b738a330efbf4072dbba30a1a5ff879072414523053e261ec58a52500eaa0963` |

Each reader completed 384 scored requests and one compatibility fixture. The original
panel and Claude replication are reported separately and never pooled. Their public
packages contain only content-free pair scores, aggregates, execution receipts, and
hashes; prompts, candidate text, answer text, and raw provider responses remain
excluded.

## Frozen Post-Hoc Controls

The support-control analysis is bound to natural execution commit
`8e34e3d41c56e1699696bc27be95cdac7c9528e5`, embedding checkpoint
`d333477116735f49b7628ee5d324e92a9f608190079a7b1d699308f5ec4f1ae6`, and the
route-bundle hashes listed in `results/support_controls/manifest.json`. It recomputes
exact rankings locally and makes zero provider calls.

The final submission diagnostics use the same natural execution commit, embedding
checkpoint, and route bundles. Their implementation is frozen at commit
`616d84ec3a49eade0c6cef946f741ce0ea92565e`; exact hashes and 3,767-query global and
namespace ranking-parity counts appear in
`results/submission_zero_call_diagnostics/manifest.json`. The missingness analysis
changes evaluator judgments only. The gold-preserving support arm reads released
required anchors and is explicitly non-deployable. Both diagnostics make zero
provider calls.

The verifier operating curves are bound to complete OpenAI and Gemini structured
response hashes `fa25549ef10ce36408fc1c88312ab801ecd5ca0a447dac8f4a18f5ff0f8ec8ab`
and `b56bd3a1793334cbc0ff5ec75fb7b280ade5c143b13274ca0681da9dd1ce37db`.
Only aggregate curve points are released. The paired-exposure robustness calculation
uses the already public content-free pair-score hash
`855f98897b9e1963d98cd0d6da7ad043c93602febd8f3813583930b644572938`.
