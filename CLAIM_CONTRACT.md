# Claim Contract

This document renders the machine-readable contract in `claims/claims.yaml`. The
YAML file is authoritative for automated checks; this document is the readable
reporting and evidence boundary.

## Formal Distinction

### C1: Retrieval admissibility

**Status:** Definition

**Claim:** A memory can be relevant and still be inadmissible for the current query.

```text
admissible(m, q, p, t)
  = scopeAllowed(m, q, p)
    AND policyAllowed(m, q, p)
    AND lifecycleCompatible(m, intent(q), t)

usable(m, q, p, t)
  = relevant(m, q)
    AND admissible(m, q, p, t)
```

**Allowed:** Admissibility is conditioned on the query, policy, lifecycle, and
time.

**Forbidden:** Do not equate every inadmissible memory with security harm. Do not
claim that stale or superseded memories are never useful.

**Boundary:** The definition is not an inference procedure for latent intent or
policy.

## Association

### C2: Exposure and answer leakage

**Status:** Frozen association

**Population:** 721 exposure-discordant privacy or safety checkpoints from a frozen
GateMem route bundle.

| Reader | Risk difference | 95% CI | Brier delta | Log-loss delta |
| --- | ---: | ---: | ---: | ---: |
| `gpt-4o-mini-2024-07-18` | 0.4286 | [0.3919, 0.4642] | -0.022532 | -0.052176 |
| `gpt-4o-2024-08-06` | 0.5298 | [0.4915, 0.5665] | -0.025471 | -0.063268 |

Table values are rounded for display; the YAML contract and normalized evidence
retain source precision.

The Brier and log-loss intervals are recorded in the YAML contract. Both models are
OpenAI readers. The estimates are reported separately and are never pooled.

**Allowed:** Across fixed checkpoints and separately executed readers, exposing the
protected target was strongly associated with whether it appeared in the answer.
Adding exposure improved held-out leakage prediction for each reader.

**Forbidden:** Do not call this causal mediation, a randomized treatment effect, a
cross-provider replication, a pooled estimate, or an official GateMem result.

**Boundary:** This is same-provider cross-reader robustness and a non-causal
association.

## Leakage-Utility Trade-Off

### C3: Reducing exposure has a utility cost

**Status:** Frozen trade-off

The paired G1-minus-G0 comparison produced:

| Metric | Difference | 95% CI | Eligible checkpoints |
| --- | ---: | ---: | ---: |
| Answer leakage | -0.220134 | [-0.249196, -0.191904] | 1,490 |
| Utility accuracy | -0.168956 | [-0.207922, -0.130790] | 728 |
| Over-refusal | 0.287088 | [0.233783, 0.340488] | 728 |

**Allowed:** Reducing exposure lowered leakage but incurred a bounded utility and
over-refusal cost. The result motivates a safety-utility verification problem.

**Forbidden:** Do not say the stricter policy is uniformly better, that filtering is
free, or that this comparison identifies a causal treatment effect.

**Boundary:** Leakage and bounded utility use different eligible populations. The
estimate comes from one OpenAI reader and is not cross-provider evidence.

## Mechanism Smoke

### C4: Candidate-pool scorer check

**Status:** Mechanism smoke only

The audit contained four development packets and 16 evaluation packets, with 207
records across the full audit and 7-12 candidates per packet.

| Method | Recall | Wrong-namespace leakage | Measured non-usable rate |
| --- | ---: | ---: | ---: |
| Global dense | 1.0000 | 0.0250 | 0.1189 |
| Namespace dense | 1.0000 | 0.0000 | 0.0824 |

Table values are rounded for display; the YAML contract and normalized evidence
retain source precision.

**Allowed:** The small candidate-pool smoke verifies that the scorer responds to a
known scope constraint and detects recall loss from aggressive routing.

**Forbidden:** Do not present these 16 packets as a natural-corpus estimate, an
official benchmark result, or part of the same population as the 3,767-query
evaluation.

**Boundary:** Candidate pools were curated and development selection used only four
packets. No reader or answer outcome was evaluated.

## Full Natural-Corpus Evaluation

### C5: Trusted namespace support

**Status:** Frozen public-source evaluation

The evaluation applied settings selected once on development data to 87 groups,
182,908 memories, and 3,767 queries.

| Quantity | Estimate | 95% CI |
| --- | ---: | ---: |
| Global-dense recall | 0.716835 | n/a |
| Namespace-dense recall | 0.865663 | n/a |
| Global-dense feasible rate | 0.538871 | n/a |
| Namespace-dense feasible rate | 0.761867 | n/a |
| Global-dense penalized non-usable upper risk | 0.872626 | n/a |
| Namespace-dense penalized non-usable upper risk | 0.836668 | n/a |
| Recall difference | 0.148828 | [0.129460, 0.168372] |
| Feasible-rate difference | 0.222996 | [0.192156, 0.255019] |
| Penalized non-usable upper-risk difference | -0.035958 | [-0.041611, -0.030301] |
| Namespace wrong-scope leakage | 0.000000 | n/a |

The preregistered penalized non-usable upper risk assigns `1.0` to an infeasible
query or a feasible query without a resolved non-usability label. It combines
relevance and admissibility and is therefore neither a pure admissibility estimate
nor a pure conditional contamination estimate. Among feasible matched prefixes:

| Support | Known non-usable rate | Non-usable label coverage | Lower bound | Upper bound |
| --- | ---: | ---: | ---: | ---: |
| Global dense | 0.575314 | 0.538031 | 0.307860 | 0.769830 |
| Namespace dense | 0.347858 | 0.355480 | 0.133686 | 0.778206 |

**Allowed:** Trusted namespace support improved recall and feasible rate while
reducing the preregistered penalized non-usable upper risk on this public-source
evaluation. Namespace arms had zero measured wrong-scope leakage under trusted
released namespaces. Descriptively, the resolved non-usable rate and lower bound
were smaller inside feasible matched prefixes, but non-usable label coverage was
also lower and the conditional upper bound remained high.

**Forbidden:** Do not call this an official RHELM or MemOps submission, claim a
downstream answer-quality effect, generalize beyond the tested corruption mechanisms
or deployment environments, or describe this bound as pure admissibility
contamination. Do not describe the penalized risk as a pure conditional estimate or
claim that namespace support reduced the conditional matched-prefix upper bound.

**Boundary:** Non-required same-namespace memories remain unresolved. There is no
reader outcome for this population. Intervals are source-macro bootstrap intervals.

## Diagnostic Routing

### C6: No established value beyond trusted namespace

**Status:** Diagnostic negative result

| Contrast against namespace dense | Recall difference | 95% CI | Penalized non-usable upper-risk difference | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Threshold router | -0.023001 | [-0.028848, -0.017318] | 0.002100 | [0.001476, 0.002854] |
| Cluster router | -0.000609 | [-0.001875, 0.000000] | 0.000166 | [-0.000001, 0.000513] |

**Allowed:** Additional threshold and cluster routing did not establish practical
incremental utility beyond trusted namespace support. The threshold arm lost recall;
the cluster arm was practically indistinguishable on primary quality metrics.
The selected threshold arm used namespace-local fallback on 0.965443 of source-macro
queries, while the selected cluster arm routed to one cluster on average.

**Forbidden:** Do not present a new router as the contribution, claim clustering is
universally useless, or claim exact identity on every full-evaluation query.

**Boundary:** This result covers two frozen diagnostic variants, not the class of all
possible routing methods.

## Historical Released-Field Result

### C7: Frozen historical released-field v1

**Status:** Frozen historical released-field v1

| Upper-bound minus namespace dense | Difference | 95% CI |
| --- | ---: | ---: |
| Recall | 0.002902 | [-0.002336, 0.007897] |
| Penalized non-usable upper risk | -0.008667 | [-0.010697, -0.006750] |
| Prohibited-stale exposure | -0.002475 | [-0.003401, -0.001653] |
| Prohibited-superseded exposure | -0.010397 | [-0.011768, -0.009086] |

**Allowed:** The historical v1 released-field arm reduced the preregistered penalized
non-usable upper risk and prohibited stale or superseded exposure without a detected
recall loss. This result is retained for reproducibility and is not evidence that
lifecycle filtering alone helps.

**Forbidden:** Do not describe this as a deployable lifecycle filter, claim the
system inferred intent or state, recommend deleting every stale memory, or claim
that lifecycle metadata drove the incremental retrieval gain.

**Boundary:** Query intent and lifecycle state are released inputs. The historical
v1 arm applies released policy filtering only to current-state queries. The corrected
v2 governance arm applies policy independently of lifecycle intent and is reported
separately; the two arms are not interchangeable. Historical queries may require
records that are stale for current-state queries.

## Controlled Prompt Intervention

### C8: Paired exposure and target disclosure

**Status:** Controlled prompt intervention

Sixteen constructed scenarios independently vary relevance and admissibility across
principal scope, policy purpose, lifecycle intent, and as-of time. For each
query-condition-candidate unit, the reader receives two stateless requests that differ
only in whether the manipulated candidate is exposed. Each reader completed 192 paired
units (384 requests). Reader estimates are separate and are never pooled.

| Reader | Relevant + admissible effect | Relevant + inadmissible effect | Selectivity gap (95% CI) |
| --- | ---: | ---: | ---: |
| `gpt-5.6-sol` | 0.90625 | 0.00000 | 0.90625 [0.81250, 0.96875] |
| `gemini-3.6-flash` | 0.81250 | 0.00000 | 0.81250 [0.68750, 0.93750] |
| `deepseek-v4-pro` | 0.81250 | 0.15625 | 0.65625 [0.50000, 0.81250] |

The DeepSeek relevant-inadmissible effect has a 95% scenario-bootstrap interval of
`[0.03125, 0.31250]`. OpenAI and Gemini have aggregate estimates of zero in that
cell, but their exposed disclosure rates are not zero and a zero paired effect is not
a zero-risk statement. Irrelevant-admissible effects are 0.046875, 0.03125, and 0.0;
irrelevant-inadmissible effects are 0.0 for all three readers.

In a post-hoc zero-call check, every leave-one-scenario-out selectivity gap remained
positive. The ranges were `[0.9000, 0.9333]`, `[0.8000, 0.8667]`, and
`[0.6333, 0.7000]` for OpenAI, Gemini, and DeepSeek; exact two-sided sign-flip
`p`-values were at most `0.000244`.

**Allowed:** Assigned exposure had different disclosure effects under
construction-defined admissible and inadmissible query conditions while focal
relevance was held fixed. In this controlled prompt-level intervention, exposure
increased target disclosure substantially more in the relevant-admissible cell than
in the relevant-inadmissible cell for each separately reported reader. DeepSeek
retained a positive relevant-inadmissible exposure effect, supporting pre-prompt
verification rather than reliance on reader restraint.

**Forbidden:** Do not call disclosure internal causal use, pool readers, infer
natural-history prevalence, claim production safety, describe OpenAI or Gemini as
having zero inadmissible-disclosure risk, claim admissibility itself was isolated as
the only causal moderator, or present this as an official benchmark.

**Boundary:** The population contains sixteen constructed scenarios with literal
markers and a fixed rule-based disclosure scorer. Axis-specific estimates have four
scenarios each. There was no judge, retry, output repair, or selective rerun. The
leave-one-scenario-out and sign-flip analyses are post-hoc checks of frozen pair
scores.

## Post-Hoc Frozen-Ranking Results

### C9: Fixed-budget support advantage

**Status:** Post-hoc frozen-ranking decomposition

This analysis re-scores frozen natural-corpus rankings with the v2 penalized
admissibility upper risk. It did not participate in development selection and no
setting was retuned.

| Top-k | Global recall | Namespace recall | Recall delta | Feasible delta | Admissibility-risk delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.322337 | 0.389469 | 0.067132 | 0.031790 | -0.038626 |
| 20 | 0.431570 | 0.532878 | 0.101309 | 0.073910 | -0.085362 |
| 50 | 0.603221 | 0.739892 | 0.136671 | 0.152614 | -0.179821 |
| 100 | 0.716835 | 0.865663 | 0.148828 | 0.222996 | -0.273935 |

At each depth, namespace-group bootstrap intervals under equal-source
macro-averaging exclude zero for recall, feasibility, total v2 risk, its infeasibility component, and its feasible-prefix
admissibility component.

Frozen zero-call controls separate support identity from pool size and filter order:

| Top-20 support | Mean candidates | Recall | Feasible rate |
| --- | ---: | ---: | ---: |
| Global dense | 90121.7 | 0.431570 | 0.237404 |
| Size-matched random partition (10 seeds) | 1540.1 | 0.060954 | 0.024233 |
| Gold-preserving same-size oracle (10 seeds) | 1551.2 | 0.839636 | 0.746616 |
| Trusted namespace pre-filter | 1551.2 | 0.532878 | 0.311314 |

Global top-20 followed by namespace deletion retains `0.431570` recall; global
post-filter depth 100 reaches `0.515895`, and depth 500 reaches `0.531262`. After
scope violations are removed from the numerator, conditional upper risk is
`0.122873` for global dense and `0.126079` for namespace support. At zero
infeasibility cost, the namespace-minus-global residual-risk delta is `+0.007732`;
at unit cost it is `-0.066178`. Leave-one-group-out recall deltas remain in
`[0.097427, 0.107711]`; the exact seven-group RHELM sign-flip `p`-value is
`0.015625`.

The policy-axis sensitivity reuses the same frozen top-20 routes and source-required
anchors while omitting only the released policy-disallowed predicate. Penalized upper
risk changes from `0.786818` for global dense to `0.693498` for namespace dense, a
paired delta of `-0.093320` (95% CI `[-0.114739, -0.071374]`). The any-known-violation
and known-violation-count deltas are `-0.222094` and `-0.899191`, respectively, with
intervals excluding zero.

The gold-preserving arm is a non-deployable attribution control: it retains every
released required anchor and samples non-anchors to exactly the namespace support
size. Its `0.839636` recall and `0.746616` feasibility expose substantial headroom
when gold anchors are known; they do not make the control a routing method. In a
separate evaluator-label missingness sensitivity, routes and recall remain fixed.
At 20% hiding, global/namespace coverage falls from about `0.992` to
`0.785896`/`0.792754`, while bound width grows from about `0.008` to
`0.214104`/`0.207246`.

**Allowed:** On frozen rankings, trusted namespace support improved recall,
feasibility, and post-hoc v2 penalized admissibility upper risk at all four reported
depths. Both risk components contributed. An arbitrary size-matched partition does
not explain the support gain, while the gold-preserving oracle shows that retaining
released anchors plus matched support size has much higher headroom than namespace
alone. Shallow late filtering does not recover the pre-filtered result. Omitting the
released policy predicate preserves the direction of the namespace-support result.

**Forbidden:** Do not say v2 risk selected the original settings, call this a new
held-out run, claim namespace support guarantees admissibility, generalize the random
control to every candidate-pool effect, or claim conditional policy/lifecycle risk
improved. Do not call namespace identity necessary or sufficient for optimal
retrieval, call the gold-preserving oracle deployable, or say evaluator-label
missingness changes retrieval performance.

**Boundary:** This is a no-retuning post-hoc rescore of released trusted namespaces.
Original selection used the v1 penalized non-usable upper risk. The random partition
is not a semantic-cluster control, and the residual-risk advantage depends on a
positive penalty for infeasibility. The policy sensitivity does not validate policy
labels or imply that policy verification is unnecessary. The gold-preserving arm
uses released required anchors. The missingness sensitivity hides evaluator labels
after retrieval and is not a deployment-time metadata corruption model.

### C10: Metadata-error asymmetry

**Status:** Post-hoc metadata robustness diagnostic

Corrected v2 attribution against namespace dense is:

| Arm | Recall delta | Feasible delta | Admissibility-risk delta |
| --- | ---: | ---: | ---: |
| Policy only | 0.026547 | 0.057645 | -0.146723 |
| Lifecycle only | -0.015783 | -0.037916 | 0.017592 |
| Policy + lifecycle (`released_governance_oracle_v2`, label-aligned reference) | 0.010489 | 0.016954 | -0.123554 |

The observed ten-seed weak-dominance brackets are mechanism-specific. Namespace
channels use global dense as reference; policy, lifecycle, and intent channels use
clean namespace dense:

| Corruption channel | Last dominant | First non-dominant |
| --- | ---: | ---: |
| Namespace false deny | 0.10 | 0.20 |
| Namespace missing | 0.10 | 0.20 |
| Policy false deny | 0.00 | 0.02 |
| Policy false allow / fail-open missing | 0.30 | 0.40 |
| Lifecycle false stale | 0.00 | 0.02 |
| Source-level namespace swap | 0.10 | 0.20 |

No loss of dominance was observed through 0.50 for query-memory namespace false
allow, lifecycle false current, lifecycle missing, or query-intent flip. At 0.50
namespace false allow, matched-prefix wrong-scope exposure was 0.167211 even though
aggregate weak dominance remained. This is not a zero-leakage result.

**Allowed:** Error direction matters. In this frozen grid, false-deny mechanisms
crossed their channel-specific weak-dominance boundary earlier than tested fail-open
mechanisms. Released
policy metadata drove the corrected incremental gain, while the coarse lifecycle-only
rule hurt retrieval.

**Forbidden:** Do not call grid points universal thresholds, reduce all metadata
error to one accuracy scalar, generalize beyond tested mechanisms, or claim lifecycle
filtering is generally beneficial.

**Boundary:** Results are post-hoc diagnostics with fixed settings. Support-expanding
corruptions use full natural reranking with the original embeddings and ranker.

## Public-Development Verification

### C11: Oracle-inference gap

**Status:** Public-development inference-gap diagnostic

On 72 fixed top-20 public-development cases, route deltas relative to namespace dense
were:

| Verifier | Recall delta | Feasible delta | Admissibility-risk delta |
| --- | ---: | ---: | ---: |
| Released-field oracle | 0.000000 | 0.000000 | -0.031957 |
| `gpt-5.6-sol` text-only | -0.041667 | -0.083333 | 0.056744 |
| `gemini-3.6-flash` text-only | -0.013889 | -0.027778 | 0.017956 |

The corresponding admissibility ROC-AUC values were 0.626179 and 0.522180. On a
separate controlled population of 16 scenarios, strict focal-pair consistency was
1.000000, 1.000000, and 0.718750 for GPT, Gemini, and DeepSeek, while stable-control
overflip was 0.203125, 0.296875, and 0.265625. All overflip estimates exceeded the
preregistered 0.05 limit. Stable-admissible false-deny rates were 0.265625, 0.531250,
and 0.593750, respectively.

A fixed zero-call threshold sweep does not reveal a safe useful hard-filter region.
GPT reaches at most 1% required-anchor false denial only at the retain-all endpoint,
where violation recall is zero; Gemini has no tested threshold at or below 1% false
denial. The development-selected threshold remains unchanged.

**Allowed:** Released fields expose headroom, but these text-only verifiers did not
recover it on fixed public-development candidates. In the controlled diagnostic,
errors concentrated in false denial of stable admissible memories.

**Forbidden:** Do not claim universal impossibility, call the oracle deployable,
pool natural and controlled populations, pool models, or present an official
benchmark result.

**Boundary:** Both populations are public development diagnostics. Natural candidate
pools and thresholds are fixed. The controlled set is constructed. Populations and
models are always reported separately. The threshold curve is post-hoc and does not
select a new operating point.

## Reader Replication

### C12: Claude Opus 5 paired-exposure replication

**Status:** Controlled prompt reader replication

Claude Opus 5 was executed separately on the same 16 constructed scenarios and 192
paired units as C8. Its relevant-admissible exposure effect was 0.968750 [0.906250,
1.000000], its relevant-inadmissible effect was 0.125000 [0.000000, 0.281250], and
the resulting selectivity gap was 0.843750 [0.687500, 0.968750]. The model completed
384 scored requests plus one fixture with no retry, output repair, selective rerun,
or judge.

**Allowed:** Claude Opus 5 separately replicated a positive prompt-level selectivity
gap. Its estimate may be displayed alongside the original readers when the separate
execution and no-pooling boundary is explicit.

**Forbidden:** Do not describe all four readers as one jointly executed panel, pool
their estimates, claim zero inadmissible-disclosure risk, infer natural prevalence or
deployment safety, or call this an official benchmark result.

**Boundary:** This is a separate fourth-reader replication using the exact frozen
construction, `claude-opus-5`, disabled thinking, and medium effort. The scenarios are
constructed and literal disclosure does not identify internal model reasoning.

## Natural Same-Population Closure

### C13: Natural route-to-reader utility and risk

**Status:** Natural same-population route-to-reader evaluation

The frozen sample contains all 523 eligible RHELM evaluation cases and an
outcome-independent sample of 1,000/3,244 MemOps cases. RHELM eligibility requires a
conversation-only question and uniquely resolved supporting references: 859 of 1,305
source QA pairs meet both filters before the frozen split. The same 1,523 cases are
paired across routes for each reader. Namespace dense versus global dense produced:

| Reader | Answer-accuracy delta (95% CI) | Answer-quality delta (95% CI) | Non-answer delta (95% CI) |
| --- | ---: | ---: | ---: |
| DeepSeek V4 Pro | 0.05252 [0.02030, 0.08374] | 0.04259 [0.02268, 0.06166] | -0.04849 [-0.06798, -0.03051] |
| Gemini 3.6 Flash | 0.06764 [0.04670, 0.09021] | 0.04992 [0.03630, 0.06475] | -0.04053 [-0.05739, -0.02490] |
| GPT-5.6 Luna | 0.06594 [0.03895, 0.09610] | 0.04762 [0.03104, 0.06646] | -0.03814 [-0.05892, -0.01484] |

The corresponding route-level evidence-recall delta was 0.10402 and penalized
admissibility-upper-risk delta was -0.08475. The released-policy gate reduced risk a
further -0.01920 relative to namespace dense, but answer-accuracy intervals included
zero for all three readers. For the two original readers, the text-only verifier
increased route risk by 0.00675; Gemini answer accuracy declined while the DeepSeek
interval included zero.

The last column is not a gold-action-aware over-refusal measure. The frozen result
files use the legacy field name `over_refusal`, defined exactly as
`reader_action != "answer"`. Source-specific answer-accuracy intervals also show
heterogeneity: DeepSeek is +0.0210 [-0.0369, 0.0746] on RHELM and +0.0840 [0.0520,
0.1164] on MemOps; Gemini is +0.0593 [0.0322, 0.0943] and +0.0760 [0.0453, 0.1063];
GPT Luna is +0.0459 [0.0037, 0.0939] and +0.0860 [0.0525, 0.1208], respectively.

A post-hoc case-weighted sensitivity gives every evaluable case equal weight across
RHELM and MemOps while resampling namespace groups within source. Namespace-minus-
global answer-accuracy deltas remain positive for DeepSeek (+0.0624 [0.0326, 0.0915]),
Gemini (+0.0703 [0.0480, 0.0940]), and GPT Luna (+0.0722 [0.0455, 0.1007]). This is a
sensitivity analysis of the frozen records, not a replacement for the prespecified
equal-source estimand.

**Allowed:** Trusted namespace support improved route utility-risk and separately
judged answer utility on the same natural sample for each reader. Additional
released-policy deletion reduced route risk without establishing an answer-utility
gain. The text-only verifier did not provide a consistent substitute for trusted
namespace support.

**Forbidden:** Do not pool readers, call the shared judge independent cross-judge
replication, describe GPT-5.6 Luna as an independently preregistered replication,
claim general protected- or stale-disclosure reduction, claim monotonic benefit from
stricter deletion, or present this as an official RHELM/MemOps benchmark result.

**Boundary:** All reader estimates are separate and share one blinded Claude Haiku
judge. GPT-5.6 Luna was a sequential reader replication after the positive
two-reader continuation gate. Namespace effects on protected and stale disclosure
have intervals containing zero for every reader. GPT evaluated only the three
primary routes; the text-verifier and released-field-oracle diagnostics were not
rerun for GPT. The case-weighted analysis is explicitly post-hoc and must be reported
as a sensitivity check rather than a confirmatory estimand.

## Natural Cross-Judge Audit

### C14: Post-hoc alternate-judge agreement

**Status:** Post-hoc outcome-independent cross-judge audit

We selected 200 exact-deduplicated outputs without inspecting outcomes, using near-
equal reader-by-source-by-common-route strata, and asked GPT-5.1 to apply the frozen
judge contract independently of the primary Claude Haiku labels. Answer-correctness
exact agreement was 0.865 (95% output-bootstrap CI [0.815, 0.910]), Cohen's kappa was
0.728, and Gwet's AC1 was 0.732. The judges disagreed in nearly symmetric directions:
14 outputs were Claude-positive/GPT-negative and 13 were Claude-negative/GPT-positive.

Agreement was 0.881 [0.821, 0.933] on outputs from the original DeepSeek/Gemini
reader panel and 0.833 [0.742, 0.924] on the sequential GPT-reader subgroup. Ordinal
answer-quality exact agreement was only 0.240, although within-one agreement was
0.755 and quadratic weighted kappa was 0.884. Protected- and stale-disclosure exact
agreement were 0.975 and 0.970, respectively, as descriptive secondary checks.

**Allowed:** The outcome-independent 200-output audit found substantial but imperfect
answer-correctness agreement and materially reduces concern that the natural result
is specific to one judge. The overall point estimate crossed the frozen 0.85 audit
threshold, while its confidence interval and the GPT-reader subgroup require caution.

**Forbidden:** Do not call this independently preregistered cross-judge replication,
claim that GPT-5.1 re-scored the full 1,523-case population, say that every reader
subgroup met the threshold, treat the judges as interchangeable, infer that route-
effect estimates were independently reproduced, or present an official RHELM or
MemOps benchmark result.

**Boundary:** This is a post-hoc, outcome-independent agreement audit over one
alternate judge. It does not re-score the full natural closure, re-estimate route or
reader effects, or remove the shared-primary-judge limitation from C13. Reader
effects remain separate and are never pooled.

## Reporting Rule

Every empirical statement derived from this repository must map to one claim ID.
When a report combines claims, it must preserve the population, estimand, and
limitations of each source claim. Mechanism-smoke values and full-evaluation values
must never be pooled, averaged, or represented as one experiment.
