"""Recompute frozen natural-route metrics with the policy axis omitted."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import scripts as scripts_package
from scripts.run_frozen_natural_support_expansion import (
    _prepare_archive_imports,
    _sha256_file,
    _write_csv,
    _write_json,
)
from verify_agent_memory.policy_axis_sensitivity import (
    admissibility_status,
    score_axis_family,
    select_common_feasible_records,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT.parent / "bomi-codex-starter"
PROTOCOL_PATH = ROOT / "experiments" / "policy_axis_sensitivity_protocol.json"
COMMON_PROTOCOL_PATH = ROOT / "experiments" / "common_feasible_sensitivity_protocol.json"
PARETO_PATH = ROOT / "results" / "supplemental_natural" / "natural_top_k_pareto.csv"
SUPPORT_MAIN_PATH = ROOT / "results" / "support_controls" / "support_control_main.csv"
OUTPUT_NAMES = {
    "summary": "summary.csv",
    "sources": "source_metrics.csv",
    "deltas": "paired_deltas.csv",
    "common_summary": "common_feasible_summary.csv",
    "common_sources": "common_feasible_source_metrics.csv",
    "common_deltas": "common_feasible_paired_deltas.csv",
    "readme": "README.md",
    "manifest": "manifest.json",
}
METRICS = (
    "evidence_recall",
    "feasible",
    "penalized_upper_risk",
    "known_risk",
    "coverage",
    "lower_bound",
    "upper_bound",
    "any_known_violation",
    "known_violation_count",
)
COMMON_METRICS = (
    "evidence_recall",
    "known_risk",
    "coverage",
    "lower_bound",
    "upper_bound",
    "any_known_violation",
    "known_violation_count",
    "matched_prefix_count",
)
ARMS = ("global_dense", "namespace_dense")
FAMILIES = ("full_scope_policy_lifecycle", "scope_lifecycle_no_policy")


def _load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported policy-axis sensitivity protocol")
    if value.get("protocol_id") != "natural-policy-axis-sensitivity-v1":
        raise ValueError("policy-axis sensitivity protocol identity drifted")
    if tuple(value["arms"]) != ARMS or tuple(value["axis_families"]) != FAMILIES:
        raise ValueError("policy-axis sensitivity arms or families drifted")
    if int(value["route_limit"]) != 20 or float(value["target_recall"]) != 0.8:
        raise ValueError("policy-axis sensitivity route or recall contract drifted")
    return value


def _load_common_protocol() -> dict[str, Any]:
    value = json.loads(COMMON_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported common-feasible sensitivity protocol")
    if value.get("protocol_id") != "natural-common-feasible-sensitivity-v1":
        raise ValueError("common-feasible sensitivity protocol identity drifted")
    if tuple(value["arms"]) != ARMS or tuple(value["axis_families"]) != FAMILIES:
        raise ValueError("common-feasible sensitivity arms or families drifted")
    if int(value["route_limit"]) != 20 or float(value["target_recall"]) != 0.8:
        raise ValueError("common-feasible sensitivity route or recall contract drifted")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("cannot aggregate an empty metric")
    return float(sum(values) / len(values))


def _source_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources = sorted({str(row["source"]) for row in records})
    for family in FAMILIES:
        for arm in ARMS:
            for source in sources:
                selected = [
                    row
                    for row in records
                    if row["axis_family"] == family
                    and row["arm"] == arm
                    and row["source"] == source
                ]
                output: dict[str, object] = {
                    "axis_family": family,
                    "arm": arm,
                    "source": source,
                    "query_count": len(selected),
                    "matched_query_count": sum(bool(row["feasible"]) for row in selected),
                }
                for metric in METRICS:
                    values = [float(row[metric]) for row in selected if row[metric] is not None]
                    output[metric] = _mean(values)
                rows.append(output)
    return rows


def _macro_rows(source_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for arm in ARMS:
            selected = [
                row for row in source_rows if row["axis_family"] == family and row["arm"] == arm
            ]
            output: dict[str, object] = {
                "axis_family": family,
                "arm": arm,
                "source_count": len(selected),
                "query_count": sum(int(row["query_count"]) for row in selected),
                "matched_query_count": sum(int(row["matched_query_count"]) for row in selected),
            }
            for metric in METRICS:
                output[metric] = _mean([float(row[metric]) for row in selected])
            rows.append(output)
    return rows


def _common_source_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources = sorted({str(row["source"]) for row in records})
    for family in FAMILIES:
        for arm in ARMS:
            for source in sources:
                selected = [
                    row
                    for row in records
                    if row["axis_family"] == family
                    and row["arm"] == arm
                    and row["source"] == source
                ]
                if not selected:
                    raise RuntimeError(f"no common-feasible rows for {family}/{arm}/{source}")
                output: dict[str, object] = {
                    "axis_family": family,
                    "arm": arm,
                    "source": source,
                    "common_feasible_query_count": len(selected),
                }
                for metric in COMMON_METRICS:
                    values = [float(row[metric]) for row in selected if row[metric] is not None]
                    output[metric] = _mean(values)
                rows.append(output)
    return rows


def _common_macro_rows(source_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for arm in ARMS:
            selected = [
                row for row in source_rows if row["axis_family"] == family and row["arm"] == arm
            ]
            output: dict[str, object] = {
                "axis_family": family,
                "arm": arm,
                "source_count": len(selected),
                "common_feasible_query_count": sum(
                    int(row["common_feasible_query_count"]) for row in selected
                ),
            }
            for metric in COMMON_METRICS:
                output[metric] = _mean([float(row[metric]) for row in selected])
            rows.append(output)
    return rows


def _group_arrays(
    records: Sequence[Mapping[str, object]],
    *,
    source: str,
    family: str,
    arm: str,
    metric: str,
    groups: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if (
            row["source"] == source
            and row["axis_family"] == family
            and row["arm"] == arm
            and row[metric] is not None
        ):
            grouped[str(row["group"])].append(float(row[metric]))
    sums = np.asarray([sum(grouped[group]) for group in groups], dtype=np.float64)
    counts = np.asarray([len(grouped[group]) for group in groups], dtype=np.float64)
    return sums, counts


def _paired_deltas(
    records: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    *,
    samples: int,
    seed: int,
    metrics: Sequence[str] = METRICS,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    sources = sorted({str(row["source"]) for row in records})
    group_names = {
        source: sorted({str(row["group"]) for row in records if row["source"] == source})
        for source in sources
    }
    sample_indices = {
        source: rng.integers(
            0,
            len(group_names[source]),
            size=(samples, len(group_names[source])),
        )
        for source in sources
    }
    indexed_summary = {(str(row["axis_family"]), str(row["arm"])): row for row in summary_rows}
    output: list[dict[str, object]] = []
    for family in FAMILIES:
        for metric in metrics:
            replicates = np.zeros(samples, dtype=np.float64)
            for source in sources:
                groups = group_names[source]
                indices = sample_indices[source]
                arm_estimates: dict[str, np.ndarray] = {}
                for arm in ARMS:
                    sums, counts = _group_arrays(
                        records,
                        source=source,
                        family=family,
                        arm=arm,
                        metric=metric,
                        groups=groups,
                    )
                    sampled_sum = sums[indices].sum(axis=1)
                    sampled_count = counts[indices].sum(axis=1)
                    if np.any(sampled_count == 0):
                        raise RuntimeError(f"bootstrap produced no observations for {metric}")
                    arm_estimates[arm] = sampled_sum / sampled_count
                replicates += (
                    arm_estimates["namespace_dense"] - arm_estimates["global_dense"]
                ) / len(sources)
            estimate = float(indexed_summary[(family, "namespace_dense")][metric]) - float(
                indexed_summary[(family, "global_dense")][metric]
            )
            output.append(
                {
                    "axis_family": family,
                    "comparison": "namespace_dense_minus_global_dense",
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": float(np.quantile(replicates, 0.025)),
                    "ci_upper": float(np.quantile(replicates, 0.975)),
                    "bootstrap_unit": "namespace_group_within_source",
                    "bootstrap_samples": samples,
                    "bootstrap_seed": seed,
                }
            )
    return output


def _published_reference() -> dict[str, dict[str, float]]:
    with PARETO_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        "evidence_recall": "evidence_recall",
        "feasible": "feasible_rate",
        "penalized_upper_risk": "penalized_admissibility_upper_risk",
        "known_risk": "matched_admissibility_violation_known_rate",
        "coverage": "matched_admissibility_label_coverage",
        "lower_bound": "matched_admissibility_violation_lower_bound",
        "upper_bound": "matched_admissibility_violation_upper_bound",
    }
    return {
        str(row["arm"]): {metric: float(row[column]) for metric, column in mapping.items()}
        for row in rows
        if row["top_k"] == "20" and row["arm"] in ARMS
    }


def _validate_full_family(summary_rows: Sequence[Mapping[str, object]]) -> None:
    published = _published_reference()
    current = {
        str(row["arm"]): row
        for row in summary_rows
        if row["axis_family"] == "full_scope_policy_lifecycle"
    }
    for arm in ARMS:
        for metric, expected in published[arm].items():
            if abs(float(current[arm][metric]) - expected) > 1e-12:
                raise RuntimeError(f"published top-20 parity failed for {arm}/{metric}")

    with SUPPORT_MAIN_PATH.open(encoding="utf-8", newline="") as handle:
        controls = {row["arm"]: row for row in csv.DictReader(handle)}
    control_arms = (
        ("global_dense", "global_dense"),
        ("namespace_dense", "namespace_pre_filter"),
    )
    for arm, control_arm in control_arms:
        expected = controls[control_arm]
        checks = {
            "any_known_violation": "mean_matched_prefix_any_known_admissibility_violation_rate",
            "known_violation_count": "mean_matched_prefix_mean_known_admissibility_violation_count",
        }
        for metric, column in checks.items():
            if abs(float(current[arm][metric]) - float(expected[column])) > 1e-12:
                raise RuntimeError(f"support-control parity failed for {arm}/{metric}")


def _write_readme(
    output_dir: Path,
    summary: Sequence[Mapping[str, object]],
    common_summary: Sequence[Mapping[str, object]],
) -> None:
    indexed = {(str(row["axis_family"]), str(row["arm"])): row for row in summary}
    lines = [
        "# Policy-Axis Sensitivity",
        "",
        "This frozen, zero-call sensitivity reuses the published global-dense and",
        "namespace-dense top-20 routes. It omits only the released policy-disallowed",
        "predicate from admissibility scoring; source-required anchors, scope, lifecycle,",
        "rankings, route limits, and the 0.8 recall target remain unchanged.",
        "",
        "| Axis family | Arm | Recall | Feasible | Penalized upper risk | Known risk | "
        "Coverage | Bounds | Any violation | Mean count |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "full_scope_policy_lifecycle": "Scope + policy + lifecycle",
        "scope_lifecycle_no_policy": "Scope + lifecycle (policy omitted)",
    }
    for family in FAMILIES:
        for arm in ARMS:
            row = indexed[(family, arm)]
            lines.append(
                "| {family} | {arm} | {recall:.4f} | {feasible:.4f} | {risk:.4f} | "
                "{known:.4f} | {coverage:.4f} | [{lower:.4f}, {upper:.4f}] | "
                "{any_v:.4f} | {count:.3f} |".format(
                    family=labels[family],
                    arm=arm.replace("_", " "),
                    recall=float(row["evidence_recall"]),
                    feasible=float(row["feasible"]),
                    risk=float(row["penalized_upper_risk"]),
                    known=float(row["known_risk"]),
                    coverage=float(row["coverage"]),
                    lower=float(row["lower_bound"]),
                    upper=float(row["upper_bound"]),
                    any_v=float(row["any_known_violation"]),
                    count=float(row["known_violation_count"]),
                )
            )
    lines.extend(
        [
            "",
            "The direction of the namespace comparison is evaluated in `paired_deltas.csv`",
            "with a paired 10,000-sample namespace-group bootstrap within each source.",
            "",
            "## Common-feasible sensitivity",
            "",
            "The tables below condition both arms on the identical intersection of queries",
            "for which global dense and namespace dense each reach 0.8 evidence recall.",
            "This removes the arm-specific feasible-set difference from the conditional",
            "matched-prefix comparison.",
            "",
            "| Axis family | Arm | Common queries | Known risk | Coverage | Bounds | "
            "Any violation | Mean count | Mean prefix |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    common_indexed = {(str(row["axis_family"]), str(row["arm"])): row for row in common_summary}
    for family in FAMILIES:
        for arm in ARMS:
            row = common_indexed[(family, arm)]
            lines.append(
                "| {family} | {arm} | {queries} | {known:.4f} | {coverage:.4f} | "
                "[{lower:.4f}, {upper:.4f}] | {any_v:.4f} | {count:.3f} | "
                "{prefix:.2f} |".format(
                    family=labels[family],
                    arm=arm.replace("_", " "),
                    queries=int(row["common_feasible_query_count"]),
                    known=float(row["known_risk"]),
                    coverage=float(row["coverage"]),
                    lower=float(row["lower_bound"]),
                    upper=float(row["upper_bound"]),
                    any_v=float(row["any_known_violation"]),
                    count=float(row["known_violation_count"]),
                    prefix=float(row["matched_prefix_count"]),
                )
            )
    lines.extend(
        [
            "",
            "`common_feasible_paired_deltas.csv` applies the same paired 10,000-sample",
            "namespace-group bootstrap to this fixed intersection.",
            "No query IDs, memory IDs, source text, embeddings, prompts, or responses are",
            "written by this analysis.",
            "",
        ]
    )
    (output_dir / OUTPUT_NAMES["readme"]).write_bytes("\n".join(lines).encode("utf-8"))


def _run(archive_root: Path, output_dir: Path) -> None:
    protocol = _load_protocol()
    common_protocol = _load_common_protocol()
    archive_scripts = str(archive_root / "scripts")
    if archive_scripts not in scripts_package.__path__:
        scripts_package.__path__.append(archive_scripts)
    api = _prepare_archive_imports(archive_root)
    config_path = (
        archive_root / "experiments" / "configs" / "stage3_natural_corpus_public_eval.json"
    )
    route_root = (
        archive_root
        / "data"
        / "private"
        / "stage3_natural_corpus_admissibility_eval"
        / "route_bundles"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = {row["arm"]: row for row in config["selected_settings"]}
    config_ids = {
        "global_dense": selected["flat_dense"]["ranking_config_id"],
        "namespace_dense": selected["namespace_dense"]["ranking_config_id"],
    }
    route_files = {
        (
            path.name.split("-", maxsplit=2)[1],
            path.name.split("-", maxsplit=2)[2].split(".")[0],
        ): path
        for path in route_root.glob("*.jsonl.gz")
    }
    inputs = api["load_eval_inputs"](include_texts=False, include_model=False)
    records: list[dict[str, object]] = []
    route_hashes: dict[str, str] = {}
    for source, corpus in sorted(inputs.corpora.items()):
        memories = tuple(memory for memory in corpus.memories if memory.split == "eval")
        queries = tuple(query for query in corpus.queries if query.split == "eval")
        memory_by_id = {memory.memory_id: memory for memory in memories}
        routes_by_arm: dict[str, Mapping[str, Any]] = {}
        for arm in ARMS:
            route_path = route_files[(source, config_ids[arm])]
            route_hashes[route_path.name] = _sha256_file(route_path)
            routes_by_arm[arm] = {row.query_id: row for row in api["load_route_bundle"](route_path)}
        for query in queries:
            anchors = frozenset(query.required_memory_ids)
            if not anchors:
                raise RuntimeError(f"natural query lacks a required anchor: {source}")
            for arm in ARMS:
                ranked = tuple(
                    routes_by_arm[arm][query.query_id].ranked_memory_ids[
                        : int(protocol["route_limit"])
                    ]
                )
                for family, include_policy in (
                    ("full_scope_policy_lifecycle", True),
                    ("scope_lifecycle_no_policy", False),
                ):
                    statuses = {
                        memory_id: admissibility_status(
                            same_namespace=(
                                memory_by_id[memory_id].namespace_id == query.namespace_id
                            ),
                            required_anchor=(memory_id in anchors),
                            memory_state=memory_by_id[memory_id].state,
                            query_intent=query.query_intent,
                            policy_disallowed=bool(memory_by_id[memory_id].released_distractor),
                            include_policy=include_policy,
                        )
                        for memory_id in ranked
                    }
                    score = score_axis_family(
                        ranked,
                        anchors,
                        statuses,
                        target_recall=float(protocol["target_recall"]),
                        infeasibility_cost=float(protocol["infeasibility_cost"]),
                    )
                    records.append(
                        {
                            "source": source,
                            "group": query.namespace_id,
                            "query_key": query.query_id,
                            "axis_family": family,
                            "arm": arm,
                            **asdict(score),
                        }
                    )

    source_rows = _source_rows(records)
    summary_rows = _macro_rows(source_rows)
    _validate_full_family(summary_rows)
    delta_rows = _paired_deltas(
        records,
        summary_rows,
        samples=int(protocol["bootstrap_samples"]),
        seed=int(protocol["bootstrap_seed"]),
    )
    common_records = select_common_feasible_records(
        records,
        arms=ARMS,
        identity_fields=("source", "query_key", "axis_family"),
    )
    common_source_rows = _common_source_rows(common_records)
    common_summary_rows = _common_macro_rows(common_source_rows)
    common_delta_rows = _paired_deltas(
        common_records,
        common_summary_rows,
        samples=int(common_protocol["bootstrap_samples"]),
        seed=int(common_protocol["bootstrap_seed"]),
        metrics=COMMON_METRICS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / OUTPUT_NAMES["summary"], summary_rows)
    _write_csv(output_dir / OUTPUT_NAMES["sources"], source_rows)
    _write_csv(output_dir / OUTPUT_NAMES["deltas"], delta_rows)
    _write_csv(output_dir / OUTPUT_NAMES["common_summary"], common_summary_rows)
    _write_csv(output_dir / OUTPUT_NAMES["common_sources"], common_source_rows)
    _write_csv(output_dir / OUTPUT_NAMES["common_deltas"], common_delta_rows)
    _write_readme(output_dir, summary_rows, common_summary_rows)

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    published = {}
    for key in (
        "summary",
        "sources",
        "deltas",
        "common_summary",
        "common_sources",
        "common_deltas",
        "readme",
    ):
        path = output_dir / OUTPUT_NAMES[key]
        receipt: dict[str, object] = {"sha256": _sha256_file(path)}
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                receipt["row_count"] = sum(1 for _ in csv.DictReader(handle))
        published[path.name] = receipt
    manifest = {
        "schema_version": 1,
        "analysis": "natural-policy-axis-sensitivity-v1",
        "analysis_commit": head,
        "base_execution_commit": protocol["base_execution_commit"],
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "common_feasible_protocol_sha256": _sha256_file(COMMON_PROTOCOL_PATH),
        "route_bundle_sha256": dict(sorted(route_hashes.items())),
        "published_files": published,
        "query_count": sum(
            1 for row in records if row["axis_family"] == FAMILIES[0] and row["arm"] == ARMS[0]
        ),
        "common_feasible_query_count": sum(
            1
            for row in common_records
            if row["axis_family"] == FAMILIES[0] and row["arm"] == ARMS[0]
        ),
        "group_count": len({(row["source"], row["group"]) for row in records}),
        "contains_query_memory_or_group_ids": False,
        "contains_raw_text_embeddings_prompts_or_responses": False,
        "evaluation_retuning": False,
        "provider_calls": 0,
        "reader_calls": 0,
        "judge_calls": 0,
        "paid_calls": 0,
    }
    _write_json(output_dir / OUTPUT_NAMES["manifest"], manifest)
    print(json.dumps({"status": "complete", "query_count": manifest["query_count"]}))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "policy_axis_sensitivity"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _run(args.archive_root.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
