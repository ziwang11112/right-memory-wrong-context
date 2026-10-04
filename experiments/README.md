# Experiment Registry

Every experiment is represented by a checked-in protocol, immutable input contract,
and executable script. Development selection and evaluation are kept separate, and
provider-backed paths fail closed unless their exact execution contract is satisfied.

## Retrieval and Robustness

| Family | Protocol | Entry point |
| --- | --- | --- |
| Frozen natural retrieval | `frozen_natural_protocol.json` | `scripts.run_retrieval_experiment` |
| Namespace support expansion | `natural_namespace_support_expansion_protocol.json` | `scripts.run_frozen_natural_support_expansion` |
| Metadata corruption | `metadata_robustness_protocol.json`, `metadata_robustness_retrieval_protocol.json` | `scripts.run_metadata_robustness` |
| Fixed-budget Pareto | `top_k_pareto_protocol.json` | `scripts.run_top_k_pareto` |
| Released-field attribution | `admissibility_attribution_protocol.json` | `scripts.publish_supplemental_results` |
| Policy-axis sensitivity | `policy_axis_sensitivity_protocol.json` | `scripts/run_policy_axis_sensitivity.py` |

## Verification and Exposure

| Family | Protocol/input | Entry point |
| --- | --- | --- |
| Text-inferred admissibility | `inferred_admissibility_protocol.json`, `prompts/inferred_admissibility_v1.txt` | `scripts.run_inferred_admissibility_experiment` |
| Controlled verifier | `counterfactual_admissibility_protocol.json`, `counterfactual_admissibility_cases.json` | `scripts.run_counterfactual_admissibility_experiment` |
| Paired exposure construction | `counterfactual_exposure_protocol.json`, `counterfactual_exposure_targets.json` | `scripts.run_counterfactual_exposure_intervention` |
| Original exposure execution | `counterfactual_exposure_execution_protocol.json` | `scripts.run_counterfactual_exposure_execution` |
| Separate Claude replication | `claude_opus5_exposure_replication_protocol.json` | `scripts.run_claude_opus5_exposure_replication` |

## Natural Route-to-Reader Evaluation

| Stage | Protocol | Entry point |
| --- | --- | --- |
| Case and route contract | `natural_end_to_end_protocol.json` | `scripts.run_natural_end_to_end_experiment` |
| Original two-reader execution | `natural_end_to_end_two_reader_protocol.json` | `scripts.run_natural_end_to_end_experiment` |
| Original judge execution | `natural_end_to_end_two_reader_judge_protocol.json` | `scripts.run_natural_two_reader_judge` |
| Frozen analysis | `natural_end_to_end_two_reader_analysis_protocol.json` | `scripts.analyze_natural_two_reader_deterministic` |
| Sequential GPT reader | `natural_end_to_end_gpt_reader_protocol.json` | `scripts.run_natural_gpt_reader_replication` |
| Shared judge for GPT reader | `natural_end_to_end_gpt_judge_protocol.json` | `scripts.run_natural_gpt_reader_replication` |
| Alternate-judge audit | `natural_cross_judge_audit_protocol.json`, `natural_cross_judge_recovery_protocol.json` | `scripts.run_natural_cross_judge_audit` |

The cross-judge sample IDs are frozen in `manifests/natural_cross_judge_sample.json`.
Prompt files define semantic contracts; they contain no API key. Result packages
contain no raw prompt or response text.

## Execution Rules

- Run local validation before any materialization or provider command.
- Tune only on the development split named in a protocol.
- Never infer an evaluation setting from test outcomes.
- Keep reader/model estimates separate unless a protocol explicitly defines an
  aggregate estimand.
- Require complete response bundles; do not score prefixes or selectively repair
  outputs.
- Treat the checked-in provider receipts as historical evidence, not authorization to
  rerun paid calls.

See `REPRODUCIBILITY.md` for commands and `docs/EXPERIMENT_METHODS.md` for formulas,
schemas, and edge-case behavior.
