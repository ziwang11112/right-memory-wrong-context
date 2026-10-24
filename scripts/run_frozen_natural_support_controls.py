"""Run zero-call support controls and robustness checks on frozen natural rankings."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_frozen_natural_support_expansion import (
    _first_supported,
    _input_bindings,
    _namespace_support_at,
    _sha256_file,
    _write_csv,
    _write_json,
)
from verify_agent_memory.support_controls import (
    MatchedRecallScore,
    exact_sign_flip_pvalue,
    permute_labels,
    score_matched_recall,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT.parent / "bomi-codex-starter"
PROTOCOL_PATH = ROOT / "experiments" / "natural_support_controls_protocol.json"
PARETO_PATH = ROOT / "results" / "supplemental_natural" / "natural_top_k_pareto.csv"
OUTPUT_NAMES = {
    "seed_rows": "support_control_seed_rows.csv",
    "main": "support_control_main.csv",
    "recall": "target_recall_sensitivity.csv",
    "penalty": "infeasibility_cost_sensitivity.csv",
    "groups": "group_level_deltas.csv",
    "robustness": "few_cluster_robustness.csv",
    "manifest": "manifest.json",
    "readme": "README.md",
}


@dataclass(frozen=True)
class Protocol:
    arms: tuple[str, ...]
    route_limit: int
    post_filter_depths: tuple[int, ...]
    target_recalls: tuple[float, ...]
    infeasibility_costs: tuple[float, ...]
    random_seeds: tuple[int, ...]
    primary_target_recall: float
    primary_infeasibility_cost: float


@dataclass
class Aggregate:
    query_count: int = 0
    matched_count: int = 0
    evidence_recall: float = 0.0
    feasible: float = 0.0
    penalized_risk: float = 0.0
    penalized_residual_risk: float = 0.0
    conditional_risk: float = 0.0
    conditional_residual_risk: float = 0.0
    returned_wrong_scope: float = 0.0
    matched_wrong_scope: float = 0.0
    any_known_violation: float = 0.0
    known_violation_count: float = 0.0
    upper_violation_count: float = 0.0
    route_width: float = 0.0
    candidates_scored: float = 0.0

    def update(
        self,
        score: MatchedRecallScore,
        *,
        returned_wrong_scope: float,
        matched_wrong_scope: float | None,
        route_width: int,
        candidates_scored: int,
    ) -> None:
        self.query_count += 1
        self.evidence_recall += score.evidence_recall
        self.feasible += float(score.feasible)
        self.penalized_risk += score.penalized_admissibility_upper_risk
        self.penalized_residual_risk += score.penalized_residual_upper_risk
        self.returned_wrong_scope += returned_wrong_scope
        self.route_width += route_width
        self.candidates_scored += candidates_scored
        if score.feasible:
            self.matched_count += 1
            self.conditional_risk += float(score.admissibility_upper_risk)
            self.conditional_residual_risk += float(score.residual_upper_risk)
            self.matched_wrong_scope += float(matched_wrong_scope)
            self.any_known_violation += float(score.any_known_admissibility_violation)
            self.known_violation_count += float(score.known_admissibility_violation_count)
            self.upper_violation_count += float(score.upper_admissibility_violation_count)

    def means(self) -> dict[str, float | int | None]:
        if not self.query_count:
            raise RuntimeError("support-control aggregate is empty")
        denominator = float(self.query_count)
        matched = float(self.matched_count)
        return {
            "query_count": self.query_count,
            "matched_query_count": self.matched_count,
            "evidence_recall": self.evidence_recall / denominator,
            "feasible_rate": self.feasible / denominator,
            "penalized_admissibility_upper_risk": self.penalized_risk / denominator,
            "penalized_residual_upper_risk": self.penalized_residual_risk / denominator,
            "conditional_admissibility_upper_risk": (
                self.conditional_risk / matched if matched else None
            ),
            "conditional_residual_upper_risk": (
                self.conditional_residual_risk / matched if matched else None
            ),
            "returned_wrong_scope_rate": self.returned_wrong_scope / denominator,
            "matched_prefix_wrong_scope_rate": (
                self.matched_wrong_scope / matched if matched else None
            ),
            "matched_prefix_any_known_admissibility_violation_rate": (
                self.any_known_violation / matched if matched else None
            ),
            "matched_prefix_mean_known_admissibility_violation_count": (
                self.known_violation_count / matched if matched else None
            ),
            "matched_prefix_mean_upper_admissibility_violation_count": (
                self.upper_violation_count / matched if matched else None
            ),
            "mean_route_width": self.route_width / denominator,
            "mean_candidates_scored": self.candidates_scored / denominator,
        }

    def without(self, other: Aggregate) -> Aggregate:
        if other.query_count >= self.query_count:
            raise ValueError("leave-one-group-out would empty a source")
        return Aggregate(
            query_count=self.query_count - other.query_count,
            matched_count=self.matched_count - other.matched_count,
            evidence_recall=self.evidence_recall - other.evidence_recall,
            feasible=self.feasible - other.feasible,
            penalized_risk=self.penalized_risk - other.penalized_risk,
            penalized_residual_risk=(self.penalized_residual_risk - other.penalized_residual_risk),
            conditional_risk=self.conditional_risk - other.conditional_risk,
            conditional_residual_risk=(
                self.conditional_residual_risk - other.conditional_residual_risk
            ),
            returned_wrong_scope=self.returned_wrong_scope - other.returned_wrong_scope,
            matched_wrong_scope=self.matched_wrong_scope - other.matched_wrong_scope,
            any_known_violation=self.any_known_violation - other.any_known_violation,
            known_violation_count=self.known_violation_count - other.known_violation_count,
            upper_violation_count=self.upper_violation_count - other.upper_violation_count,
            route_width=self.route_width - other.route_width,
            candidates_scored=self.candidates_scored - other.candidates_scored,
        )


def _load_protocol() -> Protocol:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != ("natural-support-controls-v1"):
        raise ValueError("natural support-control protocol identity drifted")
    route_limit = int(raw["route_limit"])
    depths = tuple(int(value) for value in raw["post_filter_depths"])
    arms = tuple(str(value) for value in raw["arms"])
    expected_arms = (
        "global_dense",
        "size_matched_random_partition",
        "namespace_pre_filter",
        *(f"global_post_filter_{depth}" for depth in depths),
    )
    if route_limit < 1 or depths != tuple(sorted(set(depths))) or arms != expected_arms:
        raise ValueError("support-control arms or depths drifted")
    target_recalls = tuple(float(value) for value in raw["target_recall_sweep"])
    costs = tuple(float(value) for value in raw["infeasibility_cost_sweep"])
    seeds = tuple(int(value) for value in raw["random_partition_seeds"])
    primary_target = float(raw["primary_target_recall"])
    primary_cost = float(raw["primary_infeasibility_cost"])
    if primary_target not in target_recalls or primary_cost not in costs:
        raise ValueError("primary target or cost is absent from its sensitivity grid")
    return Protocol(
        arms=arms,
        route_limit=route_limit,
        post_filter_depths=depths,
        target_recalls=target_recalls,
        infeasibility_costs=costs,
        random_seeds=seeds,
        primary_target_recall=primary_target,
        primary_infeasibility_cost=primary_cost,
    )


def _source_macro(rows: Sequence[Mapping[str, float | int | None]]) -> dict[str, object]:
    output: dict[str, object] = {
        "query_count": sum(int(row["query_count"]) for row in rows),
        "matched_query_count": sum(int(row["matched_query_count"]) for row in rows),
    }
    for field in rows[0]:
        if field in output:
            continue
        values = [float(row[field]) for row in rows if row[field] is not None]
        output[field] = sum(values) / len(values) if values else None
    return output


def _full_status(query: Any, memory: Any) -> bool | None:
    if memory.namespace_id != query.namespace_id:
        return False
    return _residual_status(query, memory)


def _residual_status(query: Any, memory: Any) -> bool | None:
    if memory.memory_id in query.required_memory_ids:
        return True
    if memory.state == "uncertain":
        return None
    if memory.released_distractor:
        return False
    return not (query.query_intent == "current_state" and memory.state in {"stale", "superseded"})


def _score(
    query: Any,
    ranked: tuple[str, ...],
    memory_by_id: Mapping[str, Any],
    *,
    target_recall: float,
    infeasibility_cost: float,
    candidates_scored: int,
) -> tuple[MatchedRecallScore, dict[str, float | int | None]]:
    full = {memory_id: _full_status(query, memory_by_id[memory_id]) for memory_id in ranked}
    residual = {memory_id: _residual_status(query, memory_by_id[memory_id]) for memory_id in ranked}
    score = score_matched_recall(
        ranked,
        set(query.required_memory_ids),
        full,
        residual,
        target_recall=target_recall,
        infeasibility_cost=infeasibility_cost,
    )
    wrong_scope = {
        memory_id: memory_by_id[memory_id].namespace_id != query.namespace_id
        for memory_id in ranked
    }
    returned_rate = sum(wrong_scope.values()) / len(ranked) if ranked else 0.0
    prefix_rate = (
        sum(wrong_scope[memory_id] for memory_id in score.matched_prefix)
        / len(score.matched_prefix)
        if score.matched_prefix
        else None
    )
    return score, {
        "returned_wrong_scope": returned_rate,
        "matched_wrong_scope": prefix_rate,
        "route_width": len(ranked),
        "candidates_scored": candidates_scored,
    }


def _group_token(source: str, group: str) -> str:
    return hashlib.sha256(f"{source}\0{group}".encode()).hexdigest()[:12]


def _pareto_reference() -> dict[str, tuple[float, float, float]]:
    with PARETO_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["arm"]: (
            float(row["evidence_recall"]),
            float(row["feasible_rate"]),
            float(row["penalized_admissibility_upper_risk"]),
        )
        for row in rows
        if row["top_k"] == "20" and row["arm"] in {"global_dense", "namespace_dense"}
    }


def _metric_rows(
    accumulators: Mapping[tuple[str, int, str, float, float], Aggregate],
    *,
    protocol: Protocol,
    sources: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in protocol.arms:
        seeds = protocol.random_seeds if arm == "size_matched_random_partition" else (-1,)
        for seed in seeds:
            for target in protocol.target_recalls:
                for cost in protocol.infeasibility_costs:
                    per_source = [
                        accumulators[(arm, seed, source, target, cost)].means()
                        for source in sources
                    ]
                    rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "target_recall": target,
                            "infeasibility_cost": cost,
                            **_source_macro(per_source),
                        }
                    )
    return rows


def _summarize_seed_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    protocol: Protocol,
    target_recall: float | None = None,
    infeasibility_cost: float | None = None,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if (target_recall is None or float(row["target_recall"]) == target_recall)
        and (infeasibility_cost is None or float(row["infeasibility_cost"]) == infeasibility_cost)
    ]
    grouped: dict[tuple[str, float, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[
            (
                str(row["arm"]),
                float(row["target_recall"]),
                float(row["infeasibility_cost"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    excluded = {
        "arm",
        "seed",
        "target_recall",
        "infeasibility_cost",
        "query_count",
        "matched_query_count",
    }
    metric_fields = [field for field in selected[0] if field not in excluded]
    arm_order = {arm: index for index, arm in enumerate(protocol.arms)}
    for (arm, target, cost), group in sorted(
        grouped.items(), key=lambda item: (arm_order[item[0][0]], item[0][1], item[0][2])
    ):
        row: dict[str, object] = {
            "arm": arm,
            "seed_count": len(group),
            "target_recall": target,
            "infeasibility_cost": cost,
            "query_count": int(group[0]["query_count"]),
        }
        for field in metric_fields:
            values = np.asarray(
                [float(item[field]) for item in group if item[field] is not None],
                dtype=np.float64,
            )
            row[f"mean_{field}"] = float(values.mean()) if len(values) else None
            row[f"std_{field}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(row)
    return output


def _group_outputs(
    group_accumulators: Mapping[tuple[str, str, str], Aggregate],
    source_totals: Mapping[tuple[str, str], Aggregate],
    *,
    sources: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = (
        "evidence_recall",
        "feasible_rate",
        "penalized_admissibility_upper_risk",
        "penalized_residual_upper_risk",
    )
    group_rows: list[dict[str, object]] = []
    groups = sorted({(source, group) for _arm, source, group in group_accumulators})
    for source, group in groups:
        global_means = group_accumulators[("global_dense", source, group)].means()
        namespace_means = group_accumulators[("namespace_pre_filter", source, group)].means()
        for metric in metrics:
            group_rows.append(
                {
                    "source": source,
                    "group_token": _group_token(source, group),
                    "query_count": int(global_means["query_count"]),
                    "metric": metric,
                    "namespace_minus_global": (
                        float(namespace_means[metric]) - float(global_means[metric])
                    ),
                }
            )

    source_means = {
        (arm, source): source_totals[(arm, source)].means()
        for arm in ("global_dense", "namespace_pre_filter")
        for source in sources
    }
    robustness: list[dict[str, object]] = []
    for metric in metrics:
        estimate = sum(
            float(source_means[("namespace_pre_filter", source)][metric])
            - float(source_means[("global_dense", source)][metric])
            for source in sources
        ) / len(sources)
        loo_deltas = []
        for omitted_source, omitted_group in groups:
            source_deltas = []
            for source in sources:
                if source == omitted_source:
                    namespace = source_totals[("namespace_pre_filter", source)].without(
                        group_accumulators[("namespace_pre_filter", source, omitted_group)]
                    )
                    global_dense = source_totals[("global_dense", source)].without(
                        group_accumulators[("global_dense", source, omitted_group)]
                    )
                    source_deltas.append(
                        float(namespace.means()[metric]) - float(global_dense.means()[metric])
                    )
                else:
                    source_deltas.append(
                        float(source_means[("namespace_pre_filter", source)][metric])
                        - float(source_means[("global_dense", source)][metric])
                    )
            loo_deltas.append(sum(source_deltas) / len(source_deltas))

        rhelm = [
            float(row["namespace_minus_global"])
            for row in group_rows
            if row["source"] == "rhelm" and row["metric"] == metric
        ]
        all_deltas = [
            float(row["namespace_minus_global"]) for row in group_rows if row["metric"] == metric
        ]
        robustness.append(
            {
                "comparison": "namespace_pre_filter_minus_global_dense",
                "metric": metric,
                "equal_source_macro_estimate": estimate,
                "leave_one_group_out_min": min(loo_deltas),
                "leave_one_group_out_max": max(loo_deltas),
                "group_count": len(all_deltas),
                "groups_positive": sum(value > 0 for value in all_deltas),
                "groups_zero": sum(value == 0 for value in all_deltas),
                "groups_negative": sum(value < 0 for value in all_deltas),
                "rhelm_group_count": len(rhelm),
                "rhelm_exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(rhelm),
            }
        )
    return group_rows, robustness


def _write_readme(output_dir: Path, main_rows: Sequence[Mapping[str, object]]) -> None:
    by_arm = {str(row["arm"]): row for row in main_rows}
    global_row = by_arm["global_dense"]
    random_row = by_arm["size_matched_random_partition"]
    namespace_row = by_arm["namespace_pre_filter"]
    post20 = by_arm["global_post_filter_20"]

    def table_row(label: str, row: Mapping[str, object]) -> str:
        recall = float(row["mean_evidence_recall"])
        feasible = float(row["mean_feasible_rate"])
        risk = float(row["mean_penalized_admissibility_upper_risk"])
        residual = float(row["mean_penalized_residual_upper_risk"])
        any_violation = float(row["mean_matched_prefix_any_known_admissibility_violation_rate"])
        violation_count = float(row["mean_matched_prefix_mean_known_admissibility_violation_count"])
        candidates = float(row["mean_mean_candidates_scored"])
        return (
            f"| {label} | {recall:.4f} | {feasible:.4f} | {risk:.4f} | "
            f"{residual:.4f} | {any_violation:.4f} | {violation_count:.3f} | "
            f"{candidates:.1f} |"
        )

    text = f"""# Natural Support Controls

This zero-call post-hoc analysis uses the frozen natural-corpus embeddings and exact
dense ranker. It changes no embedding, label, query, or tuned setting. The primary
route limit is 20 and the primary matched-recall target is 0.8.

## Main comparison

| Arm | Recall | Feasible | Risk | Residual | Any violation | Mean violations | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table_row("Global dense", global_row)}
{table_row("Size-matched random partition", random_row)}
{table_row("Namespace pre-filter", namespace_row)}
{table_row("Global top-20 then namespace filter", post20)}

The random-partition arm preserves the source-level namespace label counts but breaks
their provenance assignment. It tests whether candidate-pool size alone explains the
namespace result. Post-filter arms test whether a shallow global retrieval can recover
the same support as filtering before ranking. Residual risk removes scope violations
from the numerator and retains released policy/lifecycle violations and unresolved
labels, so it is not mechanically reduced by the namespace check itself.
The two known-violation columns are calculated only on feasible matched-recall
prefixes. Unlike fractional risk, neither the any-violation indicator nor the count
can be reduced by inserting additional established-positive records before the
recall target.

Sensitivity files report target recall 0.5--1.0 and infeasibility cost 0--1. The
few-cluster file reports leave-one-namespace-out ranges and an exact two-sided sign-flip
test for the seven RHELM groups. This is a frozen diagnostic, not a new benchmark run.
It makes zero provider, reader, judge, or paid calls and publishes no raw text, IDs,
embeddings, prompts, responses, or private payloads.
"""
    (output_dir / OUTPUT_NAMES["readme"]).write_text(text, encoding="utf-8", newline="\n")


def _run(
    archive_root: Path,
    output_dir: Path,
    batch_size: int,
    *,
    source_root: Path | None = None,
    run_root: Path | None = None,
) -> None:
    protocol = _load_protocol()
    api = _input_bindings(archive_root, source_root=source_root, run_root=run_root)
    old_config_path = Path(api["config_path"])
    route_root = Path(api["route_root"])
    old_config = json.loads(old_config_path.read_text(encoding="utf-8"))
    selected = {row["arm"]: row for row in old_config["selected_settings"]}
    global_config_id = selected["flat_dense"]["ranking_config_id"]
    namespace_config_id = selected["namespace_dense"]["ranking_config_id"]
    route_files = {
        (
            path.name.split("-", maxsplit=2)[1],
            path.name.split("-", maxsplit=2)[2].split(".")[0],
        ): path
        for path in route_root.glob("*.jsonl.gz")
    }
    if len(route_files) != 18:
        raise RuntimeError("frozen route bundle count drifted")

    inputs = api["load_eval_inputs"](include_texts=False, include_model=False)
    checkpoint = api["load_store_checkpoint"](api["embedding_checkpoint_path"])
    table = api["load_validated_embedding_table"](
        inputs.population,
        checkpoint,
        shard_directory=api["embedding_shards"],
        expected_binding=checkpoint.binding,
        shard_size=api["shard_size"],
        dimension=api["dimension"],
    )

    accumulators: dict[tuple[str, int, str, float, float], Aggregate] = defaultdict(Aggregate)
    group_accumulators: dict[tuple[str, str, str], Aggregate] = defaultdict(Aggregate)
    source_totals: dict[tuple[str, str], Aggregate] = defaultdict(Aggregate)
    route_hashes: dict[str, str] = {}
    global_checks = 0
    namespace_checks = 0

    for source, corpus in inputs.corpora.items():
        global_path = route_files[(source, global_config_id)]
        namespace_path = route_files[(source, namespace_config_id)]
        route_hashes[global_path.name] = _sha256_file(global_path)
        route_hashes[namespace_path.name] = _sha256_file(namespace_path)
        global_routes = {row.query_id: row for row in api["load_route_bundle"](global_path)}
        namespace_routes = {row.query_id: row for row in api["load_route_bundle"](namespace_path)}
        memories = tuple(
            sorted(
                (memory for memory in corpus.memories if memory.split == "eval"),
                key=lambda memory: memory.memory_id,
            )
        )
        queries = tuple(
            sorted(
                (query for query in corpus.queries if query.split == "eval"),
                key=lambda query: query.query_id,
            )
        )
        memory_by_id = {memory.memory_id: memory for memory in memories}
        position_by_id = {memory.memory_id: index for index, memory in enumerate(memories)}
        memory_rows = [
            table.index_by_identity[("memory", memory.text_sha256)] for memory in memories
        ]
        query_rows = {
            query.query_id: table.index_by_identity[("query", query.query_text_sha256)]
            for query in queries
        }
        memory_matrix = np.ascontiguousarray(table.vectors[memory_rows], dtype=np.float32)
        namespace_values = tuple(sorted({memory.namespace_id for memory in memories}))
        namespace_index = {value: index for index, value in enumerate(namespace_values)}
        original_namespace = np.asarray(
            [namespace_index[memory.namespace_id] for memory in memories], dtype=np.int64
        )
        random_partitions = {
            seed: np.asarray(
                permute_labels(
                    original_namespace.tolist(),
                    seed=seed,
                    channel="size_matched_random_partition",
                    identity=source,
                ),
                dtype=np.int64,
            )
            for seed in protocol.random_seeds
        }

        for start in range(0, len(queries), batch_size):
            query_batch = queries[start : start + batch_size]
            query_matrix = np.ascontiguousarray(
                table.vectors[[query_rows[query.query_id] for query in query_batch]],
                dtype=np.float32,
            )
            score_matrix = np.asarray(memory_matrix @ query_matrix.T, dtype=np.float32)
            for column, query in enumerate(query_batch):
                candidate_ids = tuple(sorted(corpus.candidate_ids(query, namespace_only=False)))
                candidate_positions = np.asarray(
                    [position_by_id[memory_id] for memory_id in candidate_ids], dtype=np.int64
                )
                candidate_scores = score_matrix[candidate_positions, column]
                ranking = api["stable_rank_indices"](
                    candidate_scores,
                    api["lexical_tie_ranks"](candidate_ids),
                    limit=len(candidate_ids),
                )
                global_ranked = tuple(
                    candidate_ids[int(index)] for index in ranking[: protocol.route_limit]
                )
                frozen_global = global_routes[query.query_id].ranked_memory_ids
                if global_ranked != frozen_global[: protocol.route_limit]:
                    raise RuntimeError(
                        f"global ranking parity failed for {source}/{query.query_id}"
                    )
                global_checks += 1

                query_namespace = namespace_index[query.namespace_id]
                candidate_original = original_namespace[candidate_positions]
                own_namespace = candidate_original == query_namespace
                namespace_ranked = _first_supported(
                    candidate_ids,
                    ranking,
                    partial(_namespace_support_at, own_namespace=own_namespace),
                    limit=protocol.route_limit,
                )
                frozen_namespace = namespace_routes[query.query_id]
                if namespace_ranked != frozen_namespace.ranked_memory_ids[: protocol.route_limit]:
                    raise RuntimeError(
                        f"namespace ranking parity failed for {source}/{query.query_id}"
                    )
                namespace_checks += 1

                routes: dict[tuple[str, int], tuple[tuple[str, ...], int]] = {
                    ("global_dense", -1): (global_ranked, len(candidate_ids)),
                    ("namespace_pre_filter", -1): (
                        namespace_ranked,
                        int(np.count_nonzero(own_namespace)),
                    ),
                }
                for depth in protocol.post_filter_depths:
                    routes[(f"global_post_filter_{depth}", -1)] = (
                        _first_supported(
                            candidate_ids,
                            ranking[:depth],
                            partial(_namespace_support_at, own_namespace=own_namespace),
                            limit=protocol.route_limit,
                        ),
                        len(candidate_ids),
                    )
                for seed, partition in random_partitions.items():
                    random_support = partition[candidate_positions] == query_namespace
                    routes[("size_matched_random_partition", seed)] = (
                        _first_supported(
                            candidate_ids,
                            ranking,
                            partial(_namespace_support_at, own_namespace=random_support),
                            limit=protocol.route_limit,
                        ),
                        int(np.count_nonzero(random_support)),
                    )

                for (arm, seed), (route, candidates_scored) in routes.items():
                    for target in protocol.target_recalls:
                        for cost in protocol.infeasibility_costs:
                            score, diagnostics = _score(
                                query,
                                route,
                                memory_by_id,
                                target_recall=target,
                                infeasibility_cost=cost,
                                candidates_scored=candidates_scored,
                            )
                            accumulators[(arm, seed, source, target, cost)].update(
                                score, **diagnostics
                            )
                            if (
                                arm in {"global_dense", "namespace_pre_filter"}
                                and target == protocol.primary_target_recall
                                and cost == protocol.primary_infeasibility_cost
                            ):
                                group_accumulators[(arm, source, query.namespace_id)].update(
                                    score, **diagnostics
                                )
                                source_totals[(arm, source)].update(score, **diagnostics)
            del score_matrix, query_matrix
            gc.collect()
        del memory_matrix
        gc.collect()

    sources = tuple(sorted(inputs.corpora))
    metric_rows = _metric_rows(accumulators, protocol=protocol, sources=sources)
    main_rows = _summarize_seed_rows(
        metric_rows,
        protocol=protocol,
        target_recall=protocol.primary_target_recall,
        infeasibility_cost=protocol.primary_infeasibility_cost,
    )
    recall_rows = _summarize_seed_rows(
        [
            row
            for row in metric_rows
            if float(row["infeasibility_cost"]) == protocol.primary_infeasibility_cost
        ],
        protocol=protocol,
    )
    penalty_rows = _summarize_seed_rows(
        [
            row
            for row in metric_rows
            if float(row["target_recall"]) == protocol.primary_target_recall
        ],
        protocol=protocol,
    )
    group_rows, robustness_rows = _group_outputs(
        group_accumulators,
        source_totals,
        sources=sources,
    )

    if run_root is None:
        reference = _pareto_reference()
        primary = {str(row["arm"]): row for row in main_rows}
        for arm, old_arm in (
            ("global_dense", "global_dense"),
            ("namespace_pre_filter", "namespace_dense"),
        ):
            observed = primary[arm]
            expected = reference[old_arm]
            actual = (
                float(observed["mean_evidence_recall"]),
                float(observed["mean_feasible_rate"]),
                float(observed["mean_penalized_admissibility_upper_risk"]),
            )
            if any(abs(left - right) > 1e-12 for left, right in zip(actual, expected, strict=True)):
                raise RuntimeError(f"published top-20 parity failed for {arm}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        key: output_dir / name
        for key, name in OUTPUT_NAMES.items()
        if key not in {"manifest", "readme"}
    }
    _write_csv(paths["seed_rows"], metric_rows)
    _write_csv(paths["main"], main_rows)
    _write_csv(paths["recall"], recall_rows)
    _write_csv(paths["penalty"], penalty_rows)
    _write_csv(paths["groups"], group_rows)
    _write_csv(paths["robustness"], robustness_rows)
    _write_readme(output_dir, main_rows)

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    published = {
        path.name: {"sha256": _sha256_file(path), "row_count": sum(1 for _ in path.open()) - 1}
        for path in paths.values()
    }
    readme_path = output_dir / OUTPUT_NAMES["readme"]
    published[readme_path.name] = {"sha256": _sha256_file(readme_path)}
    manifest = {
        "schema_version": 1,
        "execution_kind": "archive_replay" if run_root is None else "public_source_reproduction",
        "input_run_manifest_sha256": None
        if run_root is None
        else _sha256_file(run_root / "manifest.json"),
        "analysis": "natural-support-controls-v1",
        "analysis_commit": head,
        "base_execution_commit": "8e34e3d41c56e1699696bc27be95cdac7c9528e5"
        if run_root is None
        else None,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "embedding_checkpoint_sha256": _sha256_file(api["embedding_checkpoint_path"]),
        "route_bundle_sha256": dict(sorted(route_hashes.items())),
        "published_files": published,
        "query_count": sum(
            source_totals[("global_dense", source)].query_count for source in sources
        ),
        "group_count": len({(source, group) for _arm, source, group in group_accumulators}),
        "global_ranking_checks": global_checks,
        "namespace_ranking_checks": namespace_checks,
        "contains_query_memory_or_group_ids": False,
        "contains_raw_text_embeddings_prompts_or_responses": False,
        "evaluation_retuning": False,
        "provider_calls": 0,
        "reader_calls": 0,
        "judge_calls": 0,
        "paid_calls": 0,
    }
    _write_json(output_dir / OUTPUT_NAMES["manifest"], manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "query_count": manifest["query_count"],
                "group_count": manifest["group_count"],
                "main_rows": len(main_rows),
            },
            sort_keys=True,
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            args.run_root / "support_controls"
            if args.run_root is not None
            else ROOT / "tmp" / "support_controls"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    _run(
        args.archive_root.resolve(),
        args.output_dir.resolve(),
        args.batch_size,
        source_root=args.source_root,
        run_root=args.run_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
