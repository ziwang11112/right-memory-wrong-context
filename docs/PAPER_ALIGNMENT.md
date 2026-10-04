# Paper and Public Artifact Alignment

This repository accompanies **The Right Memory in the Wrong Context: Verifying
Retrieval Admissibility in Long-Term Agent Memory**, by Zi Wang, Xingqiao Wang,
Emmanuel Addai, Devika Ambekar, and Xiaowei Xu, University of Arkansas at Little Rock.
The package and repository name is `verify-agent-memory`; the earlier working title
*The Wrong Memory at the Right Time* refers to the same research project.

## Manuscript Reference

The alignment reference is the author-supplied manuscript inspected on 2026-10-04:

| Reference | SHA-256 |
| --- | --- |
| PDF `44_The_Right_Memory_in_the_Wro.pdf` | `f81f1b03dd2d7a43ed25ebf8264232ca9cfd23f163a7bafcba55c8041ce05a2c` |
| `main.tex` in the supplied LaTeX source archive | `324c3963473eb6e28c942eecbad32fef71172189e42e00015fe81802451b51a8` |

These hashes identify the manuscript used for this comparison. The PDF, LaTeX
source, and conference submission are not distributed in this code repository.
[CITATION.cff](../CITATION.cff) records the public identity without inventing an
arXiv ID, DOI, acceptance, or publication date.

## Experiment Map

The populations below remain separate. A row identifies the corresponding code or
evidence, not a claim that every original input is included or every manuscript
figure can be regenerated from a clean checkout.

| Manuscript component | Population and design | Public implementation and evidence | Reproduction boundary |
| --- | --- | --- | --- |
| Retrieval-admissibility framework and matched-recall accounting | Three-valued relevance/admissibility; recall target 0.8 | [Admissibility](../src/verify_agent_memory/admissibility.py), [metrics](../src/verify_agent_memory/metrics.py), [method guide](EXPERIMENT_METHODS.md) | Library behavior and synthetic fixtures run locally; unresolved labels remain unresolved. |
| RQ1: candidate support | 87 namespace groups, 182,908 memories, 3,767 RHELM/MemOps queries; frozen top-20 post-hoc audit | [Frozen protocol](../experiments/frozen_natural_protocol.json), [retrieval runner](../scripts/run_retrieval_experiment.py), [fixed-budget evidence](../evidence/normalized/fixed_budget_support.csv), [supplemental results](../results/supplemental_natural/) | The top-20 audit rescored frozen rankings. Full historical population/embedding/ranking checkpoints are excluded. The frozen top-100 v1 evaluation used a different non-usable metric. |
| RQ1: support and attribution controls | The same natural query population; fixed settings and rankings | [Support controls](../results/support_controls/), [policy sensitivity](../results/policy_axis_sensitivity/), [missingness/oracle diagnostics](../results/submission_zero_call_diagnostics/) | Published aggregate files and hashes are available. Recomputing these controls requires the historical archive and, for exact reranking, its embeddings. |
| RQ2: text-only eligibility diagnostic | 96 development queries: 24 calibration and 72 analysis, balanced across three source/intent strata | [Protocol](../experiments/inferred_admissibility_protocol.json), [runner](../scripts/run_inferred_admissibility_experiment.py), [results](../results/inferred_admissibility/), [operating curves](../results/posthoc_robustness/) | Published metrics are verifiable. Original materialized cases and complete structured provider responses are excluded; operating curves cannot be independently rebuilt from the aggregate curve points alone. |
| RQ2: controlled selective verification | 16 constructed scenarios, 32 query-condition pairs, 64 cases per complete provider | [Constructed cases](../experiments/counterfactual_admissibility_cases.json), [runner](../scripts/run_counterfactual_admissibility_experiment.py), [results](../results/counterfactual_admissibility/) | Construction and scoring logic are public. A new provider execution requires its execution contract and complete responses; it is distinct from the frozen result. |
| RQ3: natural route-to-reader association | 1,523 paired benchmark-native cases; DeepSeek/Gemini panel and a sequential GPT reader, all with one shared blinded judge | [Protocols](../experiments/README.md#natural-route-to-reader-evaluation), [runner](../scripts/run_natural_end_to_end_experiment.py), [normalized results](../evidence/normalized/natural_end_to_end.csv), [tokenized case audit](../results/natural_end_to_end_case_audit/) | Released numeric case scores regenerate aggregates and paired intervals. Original prompts, answers, judge rationales, and raw responses are absent. Recall changes along with routes, so the contrasts are observational. |
| RQ3: controlled exposure-to-disclosure intervention | 16 scenarios; 192 include/omit pairs and 384 requests per reader; three-reader panel plus separate Claude replication | [Construction protocol](../experiments/counterfactual_exposure_protocol.json), [request/scoring CLI](../scripts/run_counterfactual_exposure_intervention.py), [original results](../results/counterfactual_exposure/), [separate replication](../results/claude_opus5_exposure_replication/) | Public construction, pair scores, and bootstrap calculations support local checks. Reader estimates are never pooled; a fresh call cannot reproduce an omitted historical response byte for byte. |
| Alternate-model-judge audit | Post-hoc, outcome-independent sample of 200 deduplicated model outputs | [Protocol](../experiments/natural_cross_judge_audit_protocol.json), [sample manifest](../experiments/manifests/natural_cross_judge_sample.json), [results](../results/natural_cross_judge_audit/) | Aggregate agreement is released; this does not re-score all 1,523 cases or independently replicate route effects. It is separate from the human label audit below. |
| Qualitative admissibility examples | Four selected, abridged public-source query-memory examples | [Example manifest](../evidence/examples/retrieval_admissibility_cases.json) | The examples preserve source IDs and verdicts for illustration; four examples do not establish population prevalence or replace the full human audit. |
| GateMem and mechanism-smoke appendices | Separate observational reader evidence and 16 curated candidate-pool packets | [GateMem evidence](../evidence/normalized/gatemem.csv), [smoke evidence](../evidence/normalized/mechanism_smoke.csv), [claim boundaries](../CLAIM_CONTRACT.md) | Historical, bounded evidence families; they are not additional samples for the three primary evaluations. |

## Human Label Repeatability

The manuscript's **Blinded Dual-Rater Label Repeatability Audit** reports two
PhD-level annotators independently labeling 207 record-query pairs in 20 blinded
packets (10 RHELM, 10 MemOps). It reports unadjudicated nominal Krippendorff's alpha
of 0.8135 for relevance, 0.9808 for scope, 0.6703 for lifecycle state, and 0.0829 for
prohibited evidence. The sparse prohibited axis is not independently reliable;
the high raw agreement must not be presented as establishing its reliability.

Claim C15 in [the claim contract](../CLAIM_CONTRACT.md) bounds this result.
The release includes [content-free aggregate agreement values](../results/human_label_audit/summary.csv),
[normalized evidence](../evidence/normalized/human_label_audit.csv), and provenance
bindings for this audit. The [audit verifier](../scripts/publish_human_label_audit.py)
checks the released snapshot and count arithmetic. The two reviewers' original
independent labels, complete packet text, and identities are not included. Thus the
release can check the reported aggregate snapshot, but cannot independently
recompute the human agreement statistics from raw annotations. Neither the four
qualitative examples nor the
200-output alternate-model-judge audit closes this gap. The derived legacy usable
composite excludes lifecycle and does not validate the full admissibility target.

## Main-Result Checks and Interpretation

The manuscript and released evidence agree on these distinctive reported results:

| Quantity | Manuscript rounding | Public evidence |
| --- | --- | --- |
| Top-20 required-evidence recall, global to namespace | 0.432 to 0.533 | [Fixed-budget support](../evidence/normalized/fixed_budget_support.csv) |
| Fraction reaching 80% recall | 0.237 to 0.311 | [Top-k results](../results/supplemental_natural/natural_top_k_pareto.csv) |
| Exact similarity evaluations | 98.3% fewer, approximately 90,122 to 1,551 | [Top-k results](../results/supplemental_natural/natural_top_k_pareto.csv) |
| Natural namespace-minus-global judged-accuracy change | +0.053 to +0.068 across separately reported readers | [Natural end-to-end evidence](../evidence/normalized/natural_end_to_end.csv) |
| Relevant-inadmissible literal-disclosure effect for DeepSeek | +0.156, 95% CI [0.031, 0.312] | [Controlled exposure evidence](../evidence/normalized/counterfactual_exposure.csv) |

These checks establish correspondence for the named values and populations. They do
not certify every sentence, figure layout, or unpublished input in the manuscript.
Keep source-macro and case-weighted summaries distinct; do not average independent
reader estimates or replace the frozen historical metric with a newer one.
The natural support advantage follows trusted namespace/anchor assumptions and does
not demonstrate reliable policy or lifecycle inference. Only one of the four
controlled readers has a relevant-inadmissible disclosure interval excluding zero.

## What a Public Reproduction Establishes

Follow [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for executable commands.

1. **Code verification:** local fixtures and tests exercise schema, filtering,
   ranking, and scoring behavior without external data or provider calls.
2. **Released-evidence verification:** hash and claim checks establish integrity
   and consistency of the distributed snapshot. Public numeric pair/case scores
   enable regeneration of the corresponding statistics.
3. **Historical execution reconstruction:** several runners read a separately held
   provenance archive. The private archive's hashes document provenance but are
   not a public download mechanism. Fetching pinned upstream datasets alone does
   not supply the frozen embeddings, case construction, rankings, or responses.
4. **Independent new execution:** uses separately acquired source data, a complete
   input construction, model access, and the runner's execution contract. New model
   outputs belong to a separate result bundle; agreement with a paper number must
   be measured rather than assumed.

Public identity and citation metadata can change without rewriting frozen
experimental evidence. Anonymous exports are separate review artifacts; historical
protocol names and execution identifiers remain intact so that provenance can be
checked.
