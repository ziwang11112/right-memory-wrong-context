# Paper Companion: Experiment Guide

This repository accompanies **The Right Memory in the Wrong Context: Verifying
Retrieval Admissibility in Long-Term Agent Memory**, by Zi Wang, Xingqiao Wang,
Emmanuel Addai, Devika Ambekar, and Xiaowei Xu, University of Arkansas at Little Rock.

The table below maps the paper's core experiments to the released implementation.
Installation and execution commands are in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

## Code-to-Paper Map

| Paper component | Implementation | Released results |
| --- | --- | --- |
| Three-valued admissibility and matched-recall evaluation | [Admissibility](../src/verify_agent_memory/admissibility.py), [metrics](../src/verify_agent_memory/metrics.py), [method guide](EXPERIMENT_METHODS.md) | [Claim contract](../CLAIM_CONTRACT.md) |
| RQ1: candidate support and natural retrieval | [Retrieval runner](../scripts/run_retrieval_experiment.py), [frozen protocol](../experiments/frozen_natural_protocol.json) | [Natural results](../results/supplemental_natural/), [support controls](../results/support_controls/), [policy sensitivity](../results/policy_axis_sensitivity/) |
| RQ2: eligibility verification | [Inferred-admissibility runner](../scripts/run_inferred_admissibility_experiment.py), [controlled cases](../experiments/counterfactual_admissibility_cases.json) | [Text-only diagnostic](../results/inferred_admissibility/), [controlled verification](../results/counterfactual_admissibility/) |
| RQ3: natural route-to-reader association | [Natural end-to-end runner](../scripts/run_natural_end_to_end_experiment.py), [protocols](../experiments/README.md#natural-route-to-reader-evaluation) | [Normalized results](../evidence/normalized/natural_end_to_end.csv), [tokenized case scores](../results/natural_end_to_end_case_audit/) |
| RQ3: controlled exposure-to-disclosure intervention | [Exposure runner](../scripts/run_counterfactual_exposure_intervention.py), [protocol](../experiments/counterfactual_exposure_protocol.json) | [Exposure results](../results/counterfactual_exposure/), [separate reader replication](../results/claude_opus5_exposure_replication/) |

Additional diagnostics and their evidence boundaries are indexed in
[results/README.md](../results/README.md) and [evidence/README.md](../evidence/README.md).

## Evaluation Populations

- **Natural retrieval:** 87 namespace groups, 182,908 memories, and 3,767 queries.
  The paper's top-20 admissibility audit rescored frozen rankings; it is distinct
  from the frozen top-100 evaluation.
- **Text-only eligibility:** 96 development queries, split into 24 calibration and
  72 analysis queries.
- **Natural route-to-reader:** 1,523 paired cases, with readers reported separately
  and a shared blinded primary judge. Route contrasts are observational.
- **Controlled exposure:** 16 scenarios, with 192 include/omit pairs and 384 requests
  per reader. Reader estimates and separate replications are not pooled.

## Reproduction Scope

The library, synthetic examples, constructed inputs, and released-result checks run
from this repository. Public pair-level and tokenized case scores regenerate their
corresponding aggregates and bootstrap intervals. Historical embeddings, complete
natural rankings, raw prompts, and provider responses are outside the release;
fresh provider calls are new executions.

Independent human-annotation evidence is outside the validated release. The
separate [alternate-model-judge audit](../results/natural_cross_judge_audit/)
evaluates model outputs and is not a human annotation study.

This guide maps the implemented experiments to the paper without claiming that
every manuscript detail can be independently regenerated from the public package.
