"""Run final zero-call partial-ID and support-attribution diagnostics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    AxisFamilyScore,
    admissibility_status,
    score_axis_family,
)
from verify_agent_memory.submission_diagnostics import (
    mask_established_statuses,
    sample_gold_preserving_positions,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT.parent / "bomi-codex-starter"
PROTOCOL_PATH = ROOT / "experiments" / "submission_zero_call_diagnostics_protocol.json"
SUPPORT_MAIN_PATH = ROOT / "results" / "support_controls" / "support_control_main.csv"
OUTPUT_NAMES = {
    "missing_seeds": "label_missingness_seed_rows.csv",
    "missing_summary": "label_missingness_summary.csv",
    "gold_seeds": "gold_preserving_seed_rows.csv",
    "gold_summary": "gold_preserving_summary.csv",
    "readme": "README.md",
    "manifest": "manifest.json",
}
MISSING_ARMS = ("global_dense", "namespace_dense")
SUPPORT_ARMS = (
    "global_dense",
    "size_matched_random_partition",
    "gold_preserving_same_size",
    "namespace_pre_filter",
)
METRICS = (
    "evidence_recall",
    "feasible_rate",
    "penalized_upper_risk",
    "conditional_upper_risk",
    "known_risk",
    "known_risk_evaluable_rate",
    "coverage",
    "lower_bound",
    "upper_bound",
    "bound_width",
    "any_known_violation",
    "known_violation_count",
    "mean_route_width",
    "mean_candidates_scored",
)


@dataclass(frozen=True)
class Protocol:
    route_limit: int
    target_recall: float
    infeasibility_cost: float
    seeds: tuple[int, ...]
    missingness_rates: tuple[float, ...]


@dataclass
class Aggregate:
    query_count: int = 0
    matched_count: int = 0
    known_risk_count: int = 0
    evidence_recall: float = 0.0
    feasible: float = 0.0
    penalized_upper_risk: float = 0.0
    conditional_upper_risk: float = 0.0
    known_risk: float = 0.0
    coverage: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    any_known_violation: float = 0.0
    known_violation_count: float = 0.0
    route_width: float = 0.0
    candidates_scored: float = 0.0

    def update(self, score: AxisFamilyScore, *, route_width: int, candidates_scored: int) -> None:
        self.query_count += 1
        self.evidence_recall += score.evidence_recall
        self.feasible += float(score.feasible)
        self.penalized_upper_risk += score.penalized_upper_risk
        self.route_width += route_width
        self.candidates_scored += candidates_scored
        if not score.feasible:
            return
        self.matched_count += 1
        self.conditional_upper_risk += float(score.upper_bound)
        self.coverage += float(score.coverage)
        self.lower_bound += float(score.lower_bound)
        self.upper_bound += float(score.upper_bound)
        self.any_known_violation += float(score.any_known_violation)
        self.known_violation_count += float(score.known_violation_count)
        if score.known_risk is not None:
            self.known_risk_count += 1
            self.known_risk += score.known_risk

    def means(self) -> dict[str, float | int | None]:
        if not self.query_count:
            raise RuntimeError("diagnostic aggregate is empty")
        query_denominator = float(self.query_count)
        matched_denominator = float(self.matched_count)
        if not matched_denominator:
            raise RuntimeError("diagnostic aggregate has no feasible matched prefixes")
        lower = self.lower_bound / matched_denominator
        upper = self.upper_bound / matched_denominator
        return {
            "query_count": self.query_count,
            "matched_query_count": self.matched_count,
            "evidence_recall": self.evidence_recall / query_denominator,
            "feasible_rate": self.feasible / query_denominator,
            "penalized_upper_risk": self.penalized_upper_risk / query_denominator,
            "conditional_upper_risk": self.conditional_upper_risk / matched_denominator,
            "known_risk": (
                self.known_risk / self.known_risk_count if self.known_risk_count else None
            ),
            "known_risk_evaluable_rate": self.known_risk_count / matched_denominator,
            "coverage": self.coverage / matched_denominator,
            "lower_bound": lower,
            "upper_bound": upper,
            "bound_width": upper - lower,
            "any_known_violation": self.any_known_violation / matched_denominator,
            "known_violation_count": self.known_violation_count / matched_denominator,
            "mean_route_width": self.route_width / query_denominator,
            "mean_candidates_scored": self.candidates_scored / query_denominator,
        }


def _load_protocol() -> Protocol:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("protocol_id") != (
        "submission-zero-call-diagnostics-v1"
    ):
        raise ValueError("submission diagnostic protocol identity drifted")
    if tuple(raw["missingness_arms"]) != MISSING_ARMS:
        raise ValueError("missingness arms drifted")
    if tuple(raw["support_control_arms"]) != SUPPORT_ARMS:
        raise ValueError("support-control arms drifted")
    route_limit = int(raw["route_limit"])
    target_recall = float(raw["target_recall"])
    infeasibility_cost = float(raw["infeasibility_cost"])
    seeds = tuple(int(value) for value in raw["seeds"])
    rates = tuple(float(value) for value in raw["evaluator_label_missingness_rates"])
    if route_limit != 20 or target_recall != 0.8 or infeasibility_cost != 1.0:
        raise ValueError("primary scoring contract drifted")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("diagnostic seeds must be non-empty and unique")
    if rates != tuple(sorted(set(rates))) or rates[0] != 0.0:
        raise ValueError("missingness rates must be unique, sorted, and include zero")
    if any(not 0.0 <= rate <= 1.0 for rate in rates):
        raise ValueError("missingness rate lies outside [0, 1]")
    return Protocol(route_limit, target_recall, infeasibility_cost, seeds, rates)


def _source_macro(rows: Sequence[Mapping[str, float | int | None]]) -> dict[str, object]:
    output: dict[str, object] = {
        "query_count": sum(int(row["query_count"]) for row in rows),
        "matched_query_count": sum(int(row["matched_query_count"]) for row in rows),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        output[metric] = sum(values) / len(values) if values else None
    return output


def _summarize_seeds(
    rows: Sequence[Mapping[str, object]],
    *,
    keys: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for identity, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        row: dict[str, object] = dict(zip(keys, identity, strict=True))
        row.update(
            {
                "seed_count": len(group),
                "query_count": int(group[0]["query_count"]),
                "matched_query_count": int(group[0]["matched_query_count"]),
            }
        )
        for metric in METRICS:
            values = np.asarray(
                [float(item[metric]) for item in group if item[metric] is not None],
                dtype=np.float64,
            )
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else None
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(row)
    return output


def _status(query: Any, memory: Any, *, required_anchor: bool) -> bool | None:
    return admissibility_status(
        same_namespace=(memory.namespace_id == query.namespace_id),
        required_anchor=required_anchor,
        memory_state=memory.state,
        query_intent=query.query_intent,
        policy_disallowed=bool(memory.released_distractor),
        include_policy=True,
    )


def _score(
    query: Any,
    ranked: Sequence[str],
    memory_by_id: Mapping[str, Any],
    *,
    target_recall: float,
    infeasibility_cost: float,
) -> AxisFamilyScore:
    anchors = frozenset(query.required_memory_ids)
    statuses = {
        memory_id: _status(
            query,
            memory_by_id[memory_id],
            required_anchor=(memory_id in anchors),
        )
        for memory_id in ranked
    }
    return score_axis_family(
        ranked,
        anchors,
        statuses,
        target_recall=target_recall,
        infeasibility_cost=infeasibility_cost,
    )


def _support_baselines() -> list[dict[str, str]]:
    with SUPPORT_MAIN_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_arm = {row["arm"]: row for row in rows}
    return [
        by_arm[arm]
        for arm in ("global_dense", "size_matched_random_partition", "namespace_pre_filter")
    ]


def _gold_summary(
    gold_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    gold = _summarize_seeds(gold_rows, keys=("arm",))[0]
    baselines = _support_baselines()
    output: list[dict[str, object]] = []
    for arm in SUPPORT_ARMS:
        if arm == "gold_preserving_same_size":
            output.append(
                {
                    "arm": arm,
                    "uses_released_gold": True,
                    "seed_count": gold["seed_count"],
                    "query_count": gold["query_count"],
                    "mean_evidence_recall": gold["mean_evidence_recall"],
                    "std_evidence_recall": gold["std_evidence_recall"],
                    "mean_feasible_rate": gold["mean_feasible_rate"],
                    "std_feasible_rate": gold["std_feasible_rate"],
                    "mean_penalized_upper_risk": gold["mean_penalized_upper_risk"],
                    "std_penalized_upper_risk": gold["std_penalized_upper_risk"],
                    "mean_conditional_upper_risk": gold["mean_conditional_upper_risk"],
                    "std_conditional_upper_risk": gold["std_conditional_upper_risk"],
                    "mean_any_known_violation": gold["mean_any_known_violation"],
                    "std_any_known_violation": gold["std_any_known_violation"],
                    "mean_known_violation_count": gold["mean_known_violation_count"],
                    "std_known_violation_count": gold["std_known_violation_count"],
                    "mean_candidates_scored": gold["mean_mean_candidates_scored"],
                    "std_candidates_scored": gold["std_mean_candidates_scored"],
                }
            )
            continue
        source_arm = arm
        baseline = next(row for row in baselines if row["arm"] == source_arm)
        output.append(
            {
                "arm": arm,
                "uses_released_gold": False,
                "seed_count": baseline["seed_count"],
                "query_count": baseline["query_count"],
                "mean_evidence_recall": baseline["mean_evidence_recall"],
                "std_evidence_recall": baseline["std_evidence_recall"],
                "mean_feasible_rate": baseline["mean_feasible_rate"],
                "std_feasible_rate": baseline["std_feasible_rate"],
                "mean_penalized_upper_risk": baseline["mean_penalized_admissibility_upper_risk"],
                "std_penalized_upper_risk": baseline["std_penalized_admissibility_upper_risk"],
                "mean_conditional_upper_risk": baseline[
                    "mean_conditional_admissibility_upper_risk"
                ],
                "std_conditional_upper_risk": baseline["std_conditional_admissibility_upper_risk"],
                "mean_any_known_violation": baseline[
                    "mean_matched_prefix_any_known_admissibility_violation_rate"
                ],
                "std_any_known_violation": baseline[
                    "std_matched_prefix_any_known_admissibility_violation_rate"
                ],
                "mean_known_violation_count": baseline[
                    "mean_matched_prefix_mean_known_admissibility_violation_count"
                ],
                "std_known_violation_count": baseline[
                    "std_matched_prefix_mean_known_admissibility_violation_count"
                ],
                "mean_candidates_scored": baseline["mean_mean_candidates_scored"],
                "std_candidates_scored": baseline["std_mean_candidates_scored"],
            }
        )
    return output


def _write_readme(
    output_dir: Path,
    missing_summary: Sequence[Mapping[str, object]],
    gold_summary: Sequence[Mapping[str, object]],
) -> None:
    missing_index = {(str(row["arm"]), float(row["missing_rate"])): row for row in missing_summary}
    gold_index = {str(row["arm"]): row for row in gold_summary}

    def missing_row(arm: str, rate: float) -> str:
        row = missing_index[(arm, rate)]
        return (
            f"| {arm.replace('_', ' ')} | {rate:.2f} | "
            f"{float(row['mean_coverage']):.3f} | "
            f"{float(row['mean_lower_bound']):.3f} | "
            f"{float(row['mean_upper_bound']):.3f} | "
            f"{float(row['mean_bound_width']):.3f} |"
        )

    def support_row(arm: str, label: str) -> str:
        row = gold_index[arm]
        return (
            f"| {label} | {str(row['uses_released_gold']).lower()} | "
            f"{float(row['mean_evidence_recall']):.3f} | "
            f"{float(row['mean_feasible_rate']):.3f} | "
            f"{float(row['mean_penalized_upper_risk']):.3f} | "
            f"{float(row['mean_candidates_scored']):.0f} |"
        )

    selected_rates = (0.0, 0.2, 0.5, 0.9)
    lines = [
        "# Submission Zero-Call Diagnostics",
        "",
        "These post-hoc diagnostics reuse the frozen top-20 natural rankings, exact",
        "dense scores, released required anchors, and primary 0.8 recall target. They",
        "make no provider, reader, judge, or paid calls and perform no retuning.",
        "",
        "## Evaluator-label missingness",
        "",
        "Established composite admissibility judgments are hidden deterministically",
        "after retrieval. Routes, anchors, recall, feasibility, and underlying labels",
        "remain fixed. The exercise isolates the empirical role of three-valued",
        "accounting; it is not a model of deployment-time metadata corruption.",
        "",
        "| Arm | Hidden fraction | Coverage | Lower risk | Upper risk | Bound width |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rate in selected_rates:
        for arm in MISSING_ARMS:
            lines.append(missing_row(arm, rate))
    lines.extend(
        [
            "",
            "## Gold-preserving same-size support",
            "",
            "The oracle control retains every released required anchor, then samples",
            "non-anchors without replacement until its candidate count exactly matches",
            "the query's namespace support. It separates anchor retention plus support",
            "size from the identity-aligned composition supplied by a namespace. Because",
            "it reads released gold, it is diagnostic and cannot be deployed.",
            "",
            "| Arm | Uses gold | Recall | Feasible | Penalized upper risk | Candidates |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            support_row("global_dense", "Global dense"),
            support_row("size_matched_random_partition", "Random same-size"),
            support_row("gold_preserving_same_size", "Gold-preserving same-size"),
            support_row("namespace_pre_filter", "Namespace pre-filter"),
            "",
            "All outputs are content-free aggregates. No query, memory, or namespace",
            "identifier; text; embedding; prompt; or response is written.",
            "",
        ]
    )
    (output_dir / OUTPUT_NAMES["readme"]).write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _run(archive_root: Path, output_dir: Path, batch_size: int) -> None:
    protocol = _load_protocol()
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

    missing_accumulators: dict[tuple[str, float, int, str], Aggregate] = defaultdict(Aggregate)
    gold_accumulators: dict[tuple[int, str], Aggregate] = defaultdict(Aggregate)
    route_hashes: dict[str, str] = {}
    global_checks = 0
    namespace_checks = 0

    for source, corpus in sorted(inputs.corpora.items()):
        paths = {arm: route_files[(source, config_ids[arm])] for arm in MISSING_ARMS}
        for path in paths.values():
            route_hashes[path.name] = _sha256_file(path)
        routes_by_arm = {
            arm: {row.query_id: row for row in api["load_route_bundle"](path)}
            for arm, path in paths.items()
        }
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

        for start in range(0, len(queries), batch_size):
            query_batch = queries[start : start + batch_size]
            query_matrix = np.ascontiguousarray(
                table.vectors[[query_rows[query.query_id] for query in query_batch]],
                dtype=np.float32,
            )
            score_matrix = np.asarray(memory_matrix @ query_matrix.T, dtype=np.float32)
            for column, query in enumerate(query_batch):
                anchors = frozenset(query.required_memory_ids)
                if not anchors:
                    raise RuntimeError(f"natural query lacks a required anchor: {source}")

                for arm in MISSING_ARMS:
                    ranked = tuple(
                        routes_by_arm[arm][query.query_id].ranked_memory_ids[: protocol.route_limit]
                    )
                    base_statuses = {
                        memory_id: _status(
                            query,
                            memory_by_id[memory_id],
                            required_anchor=(memory_id in anchors),
                        )
                        for memory_id in ranked
                    }
                    for rate in protocol.missingness_rates:
                        for seed in protocol.seeds:
                            statuses = mask_established_statuses(
                                base_statuses,
                                missing_rate=rate,
                                seed=seed,
                                identity=f"{source}\0{query.query_id}\0{arm}",
                            )
                            score = score_axis_family(
                                ranked,
                                anchors,
                                statuses,
                                target_recall=protocol.target_recall,
                                infeasibility_cost=protocol.infeasibility_cost,
                            )
                            missing_accumulators[(arm, rate, seed, source)].update(
                                score,
                                route_width=len(ranked),
                                candidates_scored=int(
                                    routes_by_arm[arm][query.query_id].candidates_scored
                                ),
                            )

                candidate_ids = tuple(sorted(corpus.candidate_ids(query, namespace_only=False)))
                candidate_positions = np.asarray(
                    [position_by_id[memory_id] for memory_id in candidate_ids], dtype=np.int64
                )
                candidate_scores = score_matrix[candidate_positions, column]
                global_order = api["stable_rank_indices"](
                    candidate_scores,
                    api["lexical_tie_ranks"](candidate_ids),
                    limit=protocol.route_limit,
                )
                global_ranked = tuple(candidate_ids[int(index)] for index in global_order)
                if global_ranked != tuple(
                    routes_by_arm["global_dense"][query.query_id].ranked_memory_ids[
                        : protocol.route_limit
                    ]
                ):
                    raise RuntimeError(f"global ranking parity failed for {source}")
                global_checks += 1

                own_namespace = np.asarray(
                    [
                        memory_by_id[memory_id].namespace_id == query.namespace_id
                        for memory_id in candidate_ids
                    ],
                    dtype=np.bool_,
                )
                namespace_positions = np.flatnonzero(own_namespace)
                namespace_ids = tuple(candidate_ids[int(index)] for index in namespace_positions)
                namespace_order = api["stable_rank_indices"](
                    candidate_scores[namespace_positions],
                    api["lexical_tie_ranks"](namespace_ids),
                    limit=min(protocol.route_limit, len(namespace_ids)),
                )
                namespace_ranked = tuple(namespace_ids[int(index)] for index in namespace_order)
                if namespace_ranked != tuple(
                    routes_by_arm["namespace_dense"][query.query_id].ranked_memory_ids[
                        : protocol.route_limit
                    ]
                ):
                    raise RuntimeError(f"namespace ranking parity failed for {source}")
                namespace_checks += 1

                anchor_positions = tuple(
                    index for index, memory_id in enumerate(candidate_ids) if memory_id in anchors
                )
                if len(anchor_positions) != len(anchors):
                    raise RuntimeError(
                        f"required anchor is absent from candidate support: {source}"
                    )
                non_anchor_positions = tuple(
                    index
                    for index, memory_id in enumerate(candidate_ids)
                    if memory_id not in anchors
                )
                target_size = int(np.count_nonzero(own_namespace))
                for seed in protocol.seeds:
                    selected_positions = sample_gold_preserving_positions(
                        anchor_positions,
                        non_anchor_positions,
                        target_size=target_size,
                        seed=seed,
                        identity=f"{source}\0{query.query_id}",
                    )
                    selected_index = np.asarray(selected_positions, dtype=np.int64)
                    selected_ids = tuple(candidate_ids[index] for index in selected_positions)
                    selected_order = api["stable_rank_indices"](
                        candidate_scores[selected_index],
                        api["lexical_tie_ranks"](selected_ids),
                        limit=min(protocol.route_limit, len(selected_ids)),
                    )
                    ranked = tuple(selected_ids[int(index)] for index in selected_order)
                    score = _score(
                        query,
                        ranked,
                        memory_by_id,
                        target_recall=protocol.target_recall,
                        infeasibility_cost=protocol.infeasibility_cost,
                    )
                    gold_accumulators[(seed, source)].update(
                        score,
                        route_width=len(ranked),
                        candidates_scored=target_size,
                    )
            del score_matrix, query_matrix
            gc.collect()
        del memory_matrix
        gc.collect()

    sources = tuple(sorted(inputs.corpora))
    missing_seed_rows: list[dict[str, object]] = []
    for arm in MISSING_ARMS:
        for rate in protocol.missingness_rates:
            for seed in protocol.seeds:
                missing_seed_rows.append(
                    {
                        "arm": arm,
                        "missing_rate": rate,
                        "seed": seed,
                        **_source_macro(
                            [
                                missing_accumulators[(arm, rate, seed, source)].means()
                                for source in sources
                            ]
                        ),
                    }
                )
    missing_summary = _summarize_seeds(
        missing_seed_rows,
        keys=("arm", "missing_rate"),
    )
    gold_seed_rows = [
        {
            "arm": "gold_preserving_same_size",
            "seed": seed,
            **_source_macro([gold_accumulators[(seed, source)].means() for source in sources]),
        }
        for seed in protocol.seeds
    ]
    gold_summary = _gold_summary(gold_seed_rows)

    zero_missing = {
        str(row["arm"]): row for row in missing_summary if float(row["missing_rate"]) == 0.0
    }
    baselines = {row["arm"]: row for row in _support_baselines()}
    for arm, support_arm in (
        ("global_dense", "global_dense"),
        ("namespace_dense", "namespace_pre_filter"),
    ):
        observed = zero_missing[arm]
        expected = baselines[support_arm]
        for observed_field, expected_field in (
            ("mean_evidence_recall", "mean_evidence_recall"),
            ("mean_feasible_rate", "mean_feasible_rate"),
            ("mean_penalized_upper_risk", "mean_penalized_admissibility_upper_risk"),
        ):
            if abs(float(observed[observed_field]) - float(expected[expected_field])) > 1e-12:
                raise RuntimeError(f"zero-missingness parity failed for {arm}/{observed_field}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / OUTPUT_NAMES["missing_seeds"], missing_seed_rows)
    _write_csv(output_dir / OUTPUT_NAMES["missing_summary"], missing_summary)
    _write_csv(output_dir / OUTPUT_NAMES["gold_seeds"], gold_seed_rows)
    _write_csv(output_dir / OUTPUT_NAMES["gold_summary"], gold_summary)
    _write_readme(output_dir, missing_summary, gold_summary)

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    published: dict[str, dict[str, object]] = {}
    for key in ("missing_seeds", "missing_summary", "gold_seeds", "gold_summary", "readme"):
        path = output_dir / OUTPUT_NAMES[key]
        receipt: dict[str, object] = {"sha256": _sha256_file(path)}
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                receipt["row_count"] = sum(1 for _ in csv.DictReader(handle))
        published[path.name] = receipt
    raw_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "analysis": raw_protocol["protocol_id"],
        "analysis_commit": head,
        "base_execution_commit": raw_protocol["base_execution_commit"],
        "embedding_checkpoint_sha256": raw_protocol["base_embedding_checkpoint_sha256"],
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "support_control_main_sha256": _sha256_file(SUPPORT_MAIN_PATH),
        "route_bundle_sha256": dict(sorted(route_hashes.items())),
        "published_files": published,
        "query_count": sum(
            missing_accumulators[("global_dense", 0.0, protocol.seeds[0], source)].query_count
            for source in sources
        ),
        "group_count": 87,
        "global_ranking_checks": global_checks,
        "namespace_ranking_checks": namespace_checks,
        "contains_query_memory_or_group_ids": False,
        "contains_raw_text_embeddings_prompts_or_responses": False,
        "gold_preserving_control_is_non_deployable": True,
        "missingness_changes_evaluator_labels_only": True,
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
                "global_ranking_checks": global_checks,
                "namespace_ranking_checks": namespace_checks,
            }
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "submission_zero_call_diagnostics",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    _run(args.archive_root.resolve(), args.output_dir.resolve(), args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
