"""Normalize content-free post-hoc diagnostics for claims C9--C11."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from verify_agent_memory.provenance import canonical_json_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[1]
NATURAL_COMMIT = "009ca3657fb9ebe2bad2f107d5f374c69a4afab3"
POLICY_SENSITIVITY_COMMIT = "5ffcf28f912c89d1c81b835a752943305a6f2e0c"
SUBMISSION_DIAGNOSTICS_COMMIT = "616d84ec3a49eade0c6cef946f741ce0ea92565e"
INFERRED_COMMIT = "c4f93cad3e55ae816cda868d6591e57b48e342da"
CONTROLLED_COMMIT = "1ff8c13911b3165773162b31954ac7601c0092ea"
FIELDS = (
    "claim_id",
    "family",
    "population",
    "source",
    "contrast",
    "metric",
    "estimate",
    "ci95_lower",
    "ci95_upper",
    "n",
    "notes",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Sequence[Mapping[str, str]], **selectors: str) -> Mapping[str, str]:
    selected = [
        row for row in rows if all(row.get(field) == value for field, value in selectors.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {selectors}, found {len(selected)}")
    return selected[0]


def _row(
    claim_id: str,
    family: str,
    population: str,
    source: str,
    contrast: str,
    metric: str,
    estimate: str | float,
    *,
    lower: str | float = "",
    upper: str | float = "",
    n: int,
    notes: str,
) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "family": family,
        "population": population,
        "source": source,
        "contrast": contrast,
        "metric": metric,
        "estimate": str(estimate),
        "ci95_lower": str(lower),
        "ci95_upper": str(upper),
        "n": str(n),
        "notes": notes,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    path: Path,
    normalized_path: Path,
    source_paths: Sequence[Path],
    *,
    source_commit: str,
    additional_source_commits: Sequence[str] = (),
) -> None:
    payload = {
        "schema_version": 1,
        "contains_raw_text_or_private_content": False,
        "normalized_file": normalized_path.relative_to(ROOT).as_posix(),
        "normalized_sha256": sha256_file(normalized_path),
        "row_count": len(_csv_rows(normalized_path)),
        "source_artifacts": [
            {
                "path": source.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(source),
            }
            for source in source_paths
        ],
        "source_repository": "ziwang11112/verify-agent-memory",
        "source_snapshot_commit": source_commit,
        "transformation_script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "transformation_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    if additional_source_commits:
        payload["additional_source_snapshot_commits"] = list(additional_source_commits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _natural_fixed_budget_rows(
    result_root: Path,
    policy_result_root: Path | None = None,
    submission_result_root: Path | None = None,
) -> list[dict[str, str]]:
    pareto = _csv_rows(result_root / "natural_top_k_pareto.csv")
    deltas = _csv_rows(result_root / "natural_top_k_paired_deltas.csv")
    output: list[dict[str, str]] = []
    population = "87 groups; 182908 memories; 3767 queries"
    notes = "post_hoc_v2;frozen_rankings;no_retuning;source_clustered_bootstrap"
    for top_k in (10, 20, 50, 100):
        contrast = f"top_k_{top_k}"
        global_row = _one(pareto, arm="global_dense", top_k=str(top_k))
        namespace_row = _one(pareto, arm="namespace_dense", top_k=str(top_k))
        for prefix, arm_row in (("global", global_row), ("namespace", namespace_row)):
            for source_metric, metric in (
                ("evidence_recall", f"{prefix}_evidence_recall"),
                ("feasible_rate", f"{prefix}_feasible_rate"),
                (
                    "penalized_admissibility_upper_risk",
                    f"{prefix}_penalized_admissibility_upper_risk",
                ),
            ):
                output.append(
                    _row(
                        "C9",
                        "fixed_budget_support",
                        population,
                        "RHELM and MemOps public sources",
                        contrast,
                        metric,
                        arm_row[source_metric],
                        n=3767,
                        notes=notes,
                    )
                )
        for source_metric, metric in (
            ("evidence_recall", "evidence_recall_delta"),
            ("feasible_rate", "feasible_rate_delta"),
            (
                "penalized_admissibility_upper_risk",
                "penalized_admissibility_upper_risk_delta",
            ),
            ("infeasibility_risk_component", "infeasibility_risk_component_delta"),
            (
                "admissibility_conditional_risk_component",
                "admissibility_conditional_risk_component_delta",
            ),
        ):
            delta = _one(
                deltas,
                method="namespace_dense",
                reference="global_dense",
                top_k=str(top_k),
                metric=source_metric,
            )
            output.append(
                _row(
                    "C9",
                    "fixed_budget_support",
                    population,
                    "RHELM and MemOps public sources",
                    contrast,
                    metric,
                    delta["source_macro_mean_delta"],
                    lower=delta["bootstrap_ci95_lower"],
                    upper=delta["bootstrap_ci95_upper"],
                    n=3767,
                    notes=notes,
                )
            )

    if policy_result_root is not None:
        policy_deltas = _csv_rows(policy_result_root / "paired_deltas.csv")
        for source_metric, metric in (
            ("penalized_upper_risk", "penalized_upper_risk_delta_without_policy"),
            ("any_known_violation", "any_known_violation_delta_without_policy"),
            ("known_violation_count", "known_violation_count_delta_without_policy"),
        ):
            delta = _one(
                policy_deltas,
                axis_family="scope_lifecycle_no_policy",
                comparison="namespace_dense_minus_global_dense",
                metric=source_metric,
            )
            output.append(
                _row(
                    "C9",
                    "fixed_budget_support",
                    population,
                    "RHELM and MemOps public sources",
                    "policy_axis_sensitivity_top_k_20",
                    metric,
                    delta["estimate"],
                    lower=delta["ci_lower"],
                    upper=delta["ci_upper"],
                    n=3767,
                    notes=(
                        "post_hoc_v2;frozen_rankings;no_retuning;policy_axis_omitted;"
                        "source_clustered_bootstrap"
                    ),
                )
            )

    if submission_result_root is not None:
        missingness = _csv_rows(submission_result_root / "label_missingness_summary.csv")
        for arm, arm_label in (("global_dense", "global"), ("namespace_dense", "namespace")):
            for rate, rate_label in ((0.0, "0pct"), (0.2, "20pct"), (0.5, "50pct"), (0.9, "90pct")):
                source = _one(missingness, arm=arm, missing_rate=str(rate))
                contrast = f"missingness_{arm_label}_{rate_label}"
                for source_metric, metric in (
                    ("mean_coverage", "coverage"),
                    ("mean_lower_bound", "lower_bound"),
                    ("mean_upper_bound", "upper_bound"),
                    ("mean_bound_width", "bound_width"),
                ):
                    output.append(
                        _row(
                            "C9",
                            "fixed_budget_support",
                            population,
                            "RHELM and MemOps public sources",
                            contrast,
                            metric,
                            source[source_metric],
                            n=3767,
                            notes=(
                                "post_hoc_v2;frozen_routes;no_retuning;"
                                "evaluator_label_missingness;ten_seed_mean"
                            ),
                        )
                    )

        gold = _one(
            _csv_rows(submission_result_root / "gold_preserving_summary.csv"),
            arm="gold_preserving_same_size",
        )
        for source_metric, metric in (
            ("mean_evidence_recall", "evidence_recall"),
            ("mean_feasible_rate", "feasible_rate"),
            ("mean_penalized_upper_risk", "penalized_upper_risk"),
            ("mean_candidates_scored", "candidates_scored"),
        ):
            output.append(
                _row(
                    "C9",
                    "fixed_budget_support",
                    population,
                    "RHELM and MemOps public sources",
                    "gold_preserving_same_size_top_k_20",
                    metric,
                    gold[source_metric],
                    n=3767,
                    notes=(
                        "post_hoc_v2;frozen_embeddings;no_retuning;"
                        "non_deployable_released_gold_oracle;ten_seed_mean"
                    ),
                )
            )
    return output


def _metadata_reliability_rows(result_root: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    population = "87 groups; 182908 memories; 3767 queries"
    attribution = _csv_rows(result_root / "natural_admissibility_attribution_deltas.csv")
    method_names = {
        "namespace_policy_only": "policy_only",
        "namespace_lifecycle_only": "lifecycle_only",
        "released_governance_oracle_v2": "governance_v2",
    }
    for method, contrast in method_names.items():
        for metric in (
            "evidence_recall",
            "feasible_rate",
            "penalized_admissibility_upper_risk",
        ):
            source = _one(attribution, method=method, reference="namespace_dense", metric=metric)
            output.append(
                _row(
                    "C10",
                    "metadata_reliability",
                    population,
                    "RHELM and MemOps public sources",
                    contrast,
                    f"{metric}_delta",
                    source["source_macro_mean_delta"],
                    lower=source["bootstrap_ci95_lower"],
                    upper=source["bootstrap_ci95_upper"],
                    n=3767,
                    notes="post_hoc_v2;corrected_governance_semantics;source_clustered_bootstrap",
                )
            )

    break_even_files = (
        "natural_namespace_break_even.json",
        "natural_governance_break_even.json",
        "natural_namespace_support_expansion_break_even.json",
    )
    for name in break_even_files:
        payload = json.loads((result_root / name).read_text(encoding="utf-8"))
        for channel, values in payload["break_even"].items():
            output.append(
                _row(
                    "C10",
                    "metadata_reliability",
                    population,
                    "RHELM and MemOps public sources",
                    channel,
                    "last_observed_dominating_rate",
                    values["last_observed_dominating_rate"],
                    n=10,
                    notes="nested_corruption;ten_seed_mean;observed_grid_bracket;not_population_threshold",
                )
            )
            first = values["first_observed_non_dominating_rate"]
            if first is not None:
                output.append(
                    _row(
                        "C10",
                        "metadata_reliability",
                        population,
                        "RHELM and MemOps public sources",
                        channel,
                        "first_observed_non_dominating_rate",
                        first,
                        n=10,
                        notes=(
                            "nested_corruption;ten_seed_mean;observed_grid_bracket;"
                            "not_population_threshold"
                        ),
                    )
                )

    expansion = _csv_rows(result_root / "natural_namespace_support_expansion_summary.csv")
    for channel, rate, contrast in (
        ("namespace_false_allow", "0.5", "false_allow_at_0_5"),
        ("namespace_swap", "0.2", "namespace_swap_at_0_2"),
    ):
        source = _one(expansion, channel=channel, rate=rate)
        for source_metric, metric in (
            ("mean_evidence_recall", "evidence_recall"),
            ("mean_feasible_rate", "feasible_rate"),
            (
                "mean_penalized_admissibility_upper_risk",
                "penalized_admissibility_upper_risk",
            ),
            (
                "mean_matched_prefix_wrong_scope_exposure_rate",
                "matched_prefix_wrong_scope_exposure_rate",
            ),
            ("mean_candidates_scored", "candidates_scored"),
        ):
            output.append(
                _row(
                    "C10",
                    "metadata_reliability",
                    population,
                    "RHELM and MemOps public sources",
                    contrast,
                    metric,
                    source[source_metric],
                    n=10,
                    notes="full_natural_reranking;ten_seed_mean;fixed_ranker;no_retuning",
                )
            )
    return output


def _inferred_rows(result_root: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    population = "72 public-development analysis cases; fixed top-20 candidate pools"
    route = _csv_rows(result_root / "paired_route_deltas.csv")
    classification = _csv_rows(result_root / "classification_metrics.csv")
    filters = _csv_rows(result_root / "filter_metrics.csv")
    route_contrasts = (
        ("OpenAI", "gpt-5.6-sol", "released_oracle", "released_oracle"),
        ("OpenAI", "gpt-5.6-sol", "text_inferred", "openai_text_inferred"),
        ("Gemini", "gemini-3.6-flash", "text_inferred", "gemini_text_inferred"),
    )
    for provider, model, arm, contrast in route_contrasts:
        for metric in (
            "evidence_recall",
            "feasible_rate",
            "penalized_admissibility_upper_risk",
        ):
            source = _one(
                route,
                provider=provider,
                model=model,
                arm=arm,
                comparator="namespace_dense",
                metric=metric,
            )
            output.append(
                _row(
                    "C11",
                    "text_inferred_admissibility",
                    population,
                    model if arm != "released_oracle" else "released fields",
                    contrast,
                    f"{metric}_delta",
                    source["estimate"],
                    lower=source["ci_lower"],
                    upper=source["ci_upper"],
                    n=72,
                    notes="public_development;fixed_candidates;dev_selected_threshold;no_model_pooling",
                )
            )

    for provider, model, contrast in (
        ("OpenAI", "gpt-5.6-sol", "openai_text_inferred"),
        ("Gemini", "gemini-3.6-flash", "gemini_text_inferred"),
    ):
        classifier = _one(
            classification,
            provider=provider,
            model=model,
            stratum="equal_stratum_macro",
            axis="admissibility",
        )
        filter_row = _one(
            filters,
            provider=provider,
            model=model,
            arm="text_inferred",
            stratum="equal_stratum_macro",
        )
        for metric, value in (
            ("roc_auc", classifier["roc_auc"]),
            ("violation_precision", filter_row["violation_precision"]),
            ("violation_recall", filter_row["violation_recall"]),
            ("required_anchor_false_deny_rate", filter_row["required_anchor_false_deny_rate"]),
        ):
            output.append(
                _row(
                    "C11",
                    "text_inferred_admissibility",
                    population,
                    model,
                    contrast,
                    metric,
                    value,
                    n=1411 if metric != "required_anchor_false_deny_rate" else 113,
                    notes="public_development;equal_stratum_macro;no_model_pooling",
                )
            )
    return output


def _controlled_rows(result_root: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    population = "16 controlled scenarios; 32 focal pairs; 64 stable candidate pairs"
    aggregate = _csv_rows(result_root / "aggregate_metrics.csv")
    intervals = _csv_rows(result_root / "bootstrap_ci.csv")
    taxonomy = _csv_rows(result_root / "error_taxonomy.csv")
    models = (
        ("OpenAI/gpt-5.6-sol", "gpt56_controlled"),
        ("Gemini/gemini-3.6-flash", "gemini_controlled"),
        ("DeepSeek/deepseek-v4-pro", "deepseek_controlled"),
    )
    for model, contrast in models:
        aggregate_row = _one(aggregate, model=model, scope="overall", axis="all")
        for metric in ("strict_focal_pair_consistency", "stable_control_overflip_rate"):
            interval = _one(intervals, model=model, scope="overall", axis="all", metric=metric)
            output.append(
                _row(
                    "C11",
                    "controlled_selective_verification",
                    population,
                    model,
                    contrast,
                    metric,
                    aggregate_row[metric],
                    lower=interval["ci_lower"],
                    upper=interval["ci_upper"],
                    n=32 if metric == "strict_focal_pair_consistency" else 64,
                    notes="controlled_public_development;scenario_bootstrap;no_model_pooling",
                )
            )
        admissible = _one(
            taxonomy,
            model=model,
            axis="all",
            role="stable_admissible",
            condition="all",
        )
        inadmissible = _one(
            taxonomy,
            model=model,
            axis="all",
            role="stable_inadmissible",
            condition="all",
        )
        output.extend(
            (
                _row(
                    "C11",
                    "controlled_selective_verification",
                    population,
                    model,
                    contrast,
                    "stable_admissible_false_deny_rate",
                    admissible["false_deny_rate"],
                    n=64,
                    notes="controlled_public_development;posthoc_descriptive;no_model_pooling",
                ),
                _row(
                    "C11",
                    "controlled_selective_verification",
                    population,
                    model,
                    contrast,
                    "stable_inadmissible_false_admit_rate",
                    inadmissible["false_admit_rate"],
                    n=64,
                    notes="controlled_public_development;posthoc_descriptive;no_model_pooling",
                ),
            )
        )
    return output


def build(repository_root: Path = ROOT) -> tuple[Path, ...]:
    supplemental = repository_root / "results" / "supplemental_natural"
    inferred = repository_root / "results" / "inferred_admissibility"
    controlled = repository_root / "results" / "counterfactual_admissibility"
    policy_sensitivity = repository_root / "results" / "policy_axis_sensitivity"
    submission_diagnostics = repository_root / "results" / "submission_zero_call_diagnostics"
    normalized = repository_root / "evidence" / "normalized"
    manifests = repository_root / "evidence" / "manifests"

    outputs = (
        normalized / "fixed_budget_support.csv",
        normalized / "metadata_reliability.csv",
        normalized / "text_inferred_admissibility.csv",
        normalized / "controlled_selective_verification.csv",
    )
    rows = (
        _natural_fixed_budget_rows(supplemental, policy_sensitivity, submission_diagnostics),
        _metadata_reliability_rows(supplemental),
        _inferred_rows(inferred),
        _controlled_rows(controlled),
    )
    for path, values in zip(outputs, rows, strict=True):
        _write_csv(path, values)

    manifest_specs = (
        (
            manifests / "fixed_budget_support.json",
            outputs[0],
            (
                supplemental / "natural_top_k_pareto.csv",
                supplemental / "natural_top_k_paired_deltas.csv",
                supplemental / "natural_top_k_pareto_manifest.json",
                policy_sensitivity / "summary.csv",
                policy_sensitivity / "paired_deltas.csv",
                policy_sensitivity / "manifest.json",
                submission_diagnostics / "label_missingness_summary.csv",
                submission_diagnostics / "gold_preserving_summary.csv",
                submission_diagnostics / "manifest.json",
            ),
            NATURAL_COMMIT,
            (POLICY_SENSITIVITY_COMMIT, SUBMISSION_DIAGNOSTICS_COMMIT),
        ),
        (
            manifests / "metadata_reliability.json",
            outputs[1],
            (
                supplemental / "natural_admissibility_attribution_deltas.csv",
                supplemental / "natural_namespace_break_even.json",
                supplemental / "natural_governance_break_even.json",
                supplemental / "natural_namespace_support_expansion_summary.csv",
                supplemental / "natural_namespace_support_expansion_break_even.json",
                supplemental / "manifest.json",
            ),
            NATURAL_COMMIT,
            (),
        ),
        (
            manifests / "text_inferred_admissibility.json",
            outputs[2],
            (
                inferred / "paired_route_deltas.csv",
                inferred / "classification_metrics.csv",
                inferred / "filter_metrics.csv",
                inferred / "manifest.json",
            ),
            INFERRED_COMMIT,
            (),
        ),
        (
            manifests / "controlled_selective_verification.json",
            outputs[3],
            (
                controlled / "aggregate_metrics.csv",
                controlled / "bootstrap_ci.csv",
                controlled / "error_taxonomy.csv",
                controlled / "manifest.json",
            ),
            CONTROLLED_COMMIT,
            (),
        ),
    )
    for manifest_path, normalized_path, source_paths, commit, additional_commits in manifest_specs:
        _write_manifest(
            manifest_path,
            normalized_path,
            source_paths,
            source_commit=commit,
            additional_source_commits=additional_commits,
        )
    return (*outputs, *(spec[0] for spec in manifest_specs))


def main() -> int:
    outputs = build()
    print(json.dumps({"outputs": len(outputs), "status": "normalized"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
