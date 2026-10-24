"""Run the frozen natural namespace false-allow and label-swap diagnostic.

The public repository does not redistribute source payloads, embeddings, or route
bundles. This script reads a public reproduction run, or the historical provenance
archive in compatibility mode, and writes aggregate outputs without model calls.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT.parent / "bomi-codex-starter"
PROTOCOL_PATH = ROOT / "experiments" / "natural_namespace_support_expansion_protocol.json"
PARETO_PATH = ROOT / "results" / "supplemental_natural" / "natural_top_k_pareto.csv"
OUTPUT_NAMES = {
    "seed_rows": "natural_namespace_support_expansion_seed_rows.csv",
    "summary": "natural_namespace_support_expansion_summary.csv",
    "break_even": "natural_namespace_support_expansion_break_even.json",
    "manifest": "natural_namespace_support_expansion_manifest.json",
}


@dataclass
class Accumulator:
    query_count: int = 0
    matched_prefix_query_count: int = 0
    evidence_recall: float = 0.0
    feasible: float = 0.0
    penalized_admissibility_upper_risk: float = 0.0
    infeasibility_risk_component: float = 0.0
    admissibility_conditional_risk_component: float = 0.0
    matched_prefix_wrong_scope_exposure_rate: float = 0.0
    returned_wrong_scope_rate: float = 0.0
    known_relevant_inadmissible_rate: float = 0.0
    joint_label_coverage: float = 0.0
    candidates_scored: float = 0.0

    def update(self, row: Mapping[str, float | int | bool | None]) -> None:
        feasible = bool(row["feasible"])
        admissibility_upper = row["admissibility_upper"]
        risk = float(admissibility_upper) if feasible and admissibility_upper is not None else 1.0
        self.query_count += 1
        self.evidence_recall += float(row["evidence_recall"])
        self.feasible += float(feasible)
        self.penalized_admissibility_upper_risk += risk
        self.infeasibility_risk_component += float(not feasible)
        self.admissibility_conditional_risk_component += (
            float(admissibility_upper) if feasible and admissibility_upper is not None else 0.0
        )
        self.returned_wrong_scope_rate += float(row["returned_wrong_scope_rate"])
        self.candidates_scored += float(row["candidates_scored"])
        if feasible:
            self.matched_prefix_query_count += 1
            self.matched_prefix_wrong_scope_exposure_rate += float(
                row["matched_prefix_wrong_scope_exposure_rate"]
            )
            self.known_relevant_inadmissible_rate += float(row["known_relevant_inadmissible_rate"])
            self.joint_label_coverage += float(row["joint_label_coverage"])

    def means(self) -> dict[str, float | int]:
        if not self.query_count or not self.matched_prefix_query_count:
            raise RuntimeError("support-expansion accumulator lacks evaluable rows")
        query_denominator = float(self.query_count)
        prefix_denominator = float(self.matched_prefix_query_count)
        return {
            "query_count": self.query_count,
            "matched_prefix_query_count": self.matched_prefix_query_count,
            "evidence_recall": self.evidence_recall / query_denominator,
            "feasible_rate": self.feasible / query_denominator,
            "penalized_admissibility_upper_risk": (
                self.penalized_admissibility_upper_risk / query_denominator
            ),
            "infeasibility_risk_component": (self.infeasibility_risk_component / query_denominator),
            "admissibility_conditional_risk_component": (
                self.admissibility_conditional_risk_component / query_denominator
            ),
            "matched_prefix_wrong_scope_exposure_rate": (
                self.matched_prefix_wrong_scope_exposure_rate / prefix_denominator
            ),
            "returned_wrong_scope_rate": self.returned_wrong_scope_rate / query_denominator,
            "known_relevant_inadmissible_rate": (
                self.known_relevant_inadmissible_rate / prefix_denominator
            ),
            "joint_label_coverage": self.joint_label_coverage / prefix_denominator,
            "candidates_scored": self.candidates_scored / query_denominator,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _uniform_vector(seed: int, channel: str, *identity: str, count: int) -> np.ndarray:
    payload = "\0".join((str(seed), channel, *identity)).encode("utf-8")
    stream_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(stream_seed).random(count)


def _source_macro(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    output: dict[str, float | int] = {
        "query_count": sum(int(row["query_count"]) for row in rows),
        "matched_prefix_query_count": sum(int(row["matched_prefix_query_count"]) for row in rows),
    }
    for field in rows[0]:
        if field not in output:
            output[field] = sum(float(row[field]) for row in rows) / len(rows)
    return output


def _admissibility_status(query: Any, memory: Any) -> bool | None:
    if memory.namespace_id != query.namespace_id:
        return False
    if memory.memory_id in query.required_memory_ids:
        return True
    if memory.state == "uncertain":
        return None
    if memory.released_distractor:
        return False
    return not (query.query_intent == "current_state" and memory.state in {"stale", "superseded"})


def _score_ranking(
    query: Any,
    ranked: tuple[str, ...],
    memory_by_id: Mapping[str, Any],
    *,
    candidates_scored: int,
) -> dict[str, float | int | bool | None]:
    anchors = set(query.required_memory_ids)
    anchor_total = len(anchors)
    if not anchor_total:
        raise RuntimeError("natural query has no required evidence anchor")
    anchor_retrieved = len(anchors.intersection(ranked))
    recall = anchor_retrieved / anchor_total
    feasible = recall >= 0.8
    returned_wrong_scope = (
        sum(memory_by_id[memory_id].namespace_id != query.namespace_id for memory_id in ranked)
        / len(ranked)
        if ranked
        else 0.0
    )
    result: dict[str, float | int | bool | None] = {
        "evidence_recall": recall,
        "feasible": feasible,
        "admissibility_upper": None,
        "matched_prefix_wrong_scope_exposure_rate": None,
        "returned_wrong_scope_rate": returned_wrong_scope,
        "known_relevant_inadmissible_rate": None,
        "joint_label_coverage": None,
        "candidates_scored": candidates_scored,
    }
    if not feasible:
        return result

    hits = 0
    prefix: tuple[str, ...] = ()
    for index, memory_id in enumerate(ranked, start=1):
        hits += int(memory_id in anchors)
        if hits / anchor_total >= 0.8:
            prefix = ranked[:index]
            break
    if not prefix:
        raise RuntimeError("feasible ranking lacks a matched prefix")

    statuses = [_admissibility_status(query, memory_by_id[memory_id]) for memory_id in prefix]
    relevance = [memory_id in anchors for memory_id in prefix]
    result.update(
        {
            "admissibility_upper": sum(status is not True for status in statuses) / len(prefix),
            "matched_prefix_wrong_scope_exposure_rate": sum(
                memory_by_id[memory_id].namespace_id != query.namespace_id for memory_id in prefix
            )
            / len(prefix),
            "known_relevant_inadmissible_rate": sum(
                is_relevant and status is False
                for is_relevant, status in zip(relevance, statuses, strict=True)
            )
            / len(prefix),
            "joint_label_coverage": sum(
                is_relevant and status is not None
                for is_relevant, status in zip(relevance, statuses, strict=True)
            )
            / len(prefix),
        }
    )
    return result


def _first_supported(
    candidate_ids: Sequence[str],
    ranking: np.ndarray,
    support_at_positions: Any,
    *,
    limit: int,
    chunk_size: int = 4096,
) -> tuple[str, ...]:
    selected: list[str] = []
    for start in range(0, len(ranking), chunk_size):
        positions = ranking[start : start + chunk_size]
        support = np.asarray(support_at_positions(positions), dtype=bool)
        for position in positions[np.flatnonzero(support)]:
            selected.append(candidate_ids[int(position)])
            if len(selected) == limit:
                return tuple(selected)
    return tuple(selected)


def _namespace_support_at(positions: np.ndarray, *, own_namespace: np.ndarray) -> np.ndarray:
    return own_namespace[positions]


def _false_allow_support_at(
    positions: np.ndarray,
    *,
    own_namespace: np.ndarray,
    gate_values: np.ndarray,
    rate: float,
) -> np.ndarray:
    return own_namespace[positions] | (gate_values[positions] < rate)


def _swap_support_at(
    positions: np.ndarray,
    *,
    selected: np.ndarray,
    alternate: np.ndarray,
    candidate_original: np.ndarray,
    query_namespace: int,
    rate: float,
) -> np.ndarray:
    return (
        np.where(
            selected[positions] < rate,
            alternate[positions],
            candidate_original[positions],
        )
        == query_namespace
    )


def _alternate_namespace_indices(
    original: np.ndarray,
    namespace_count: int,
    random_values: np.ndarray,
) -> np.ndarray:
    if namespace_count < 2:
        return np.full_like(original, -1)
    alternatives = np.floor(random_values * (namespace_count - 1)).astype(np.int64)
    return alternatives + (alternatives >= original)


def _load_protocol() -> tuple[tuple[str, ...], tuple[float, ...], tuple[int, ...], int]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported support-expansion protocol")
    channels = tuple(str(item) for item in value["channels"])
    rates = tuple(float(item) for item in value["rates"])
    seeds = tuple(int(item) for item in value["seeds"])
    top_k = int(value["top_k"])
    if channels != ("namespace_false_allow", "namespace_swap"):
        raise ValueError("support-expansion channels drifted")
    return channels, rates, seeds, top_k


def _prepare_archive_imports(archive_root: Path) -> Mapping[str, object]:
    if not (archive_root / "bomi").is_dir():
        raise FileNotFoundError(f"provenance archive is unavailable: {archive_root}")
    sys.path.insert(0, str(archive_root))
    sys.path.insert(1, str(ROOT / "src"))

    from bomi.bench.stage3_natural_dev_runtime import load_validated_embedding_table
    from bomi.bench.stage3_natural_embedding_store import load_store_checkpoint
    from bomi.bench.stage3_natural_route_store import load_route_bundle
    from bomi.experiments.stage3_natural_dev_routing import (
        lexical_tie_ranks,
        stable_rank_indices,
    )

    from scripts.run_stage3_natural_corpus_public_eval import (
        DEFAULT_DIMENSION,
        DEFAULT_SHARD_SIZE,
        EMBEDDING_CHECKPOINT_PATH,
        EMBEDDING_SHARDS,
        _load_eval_inputs,
    )

    return {
        "load_validated_embedding_table": load_validated_embedding_table,
        "load_store_checkpoint": load_store_checkpoint,
        "load_route_bundle": load_route_bundle,
        "lexical_tie_ranks": lexical_tie_ranks,
        "stable_rank_indices": stable_rank_indices,
        "dimension": DEFAULT_DIMENSION,
        "shard_size": DEFAULT_SHARD_SIZE,
        "embedding_checkpoint_path": EMBEDDING_CHECKPOINT_PATH,
        "embedding_shards": EMBEDDING_SHARDS,
        "load_eval_inputs": _load_eval_inputs,
    }


def _input_bindings(
    archive_root: Path, *, source_root: Path | None = None, run_root: Path | None = None
) -> Mapping[str, object]:
    if (source_root is None) != (run_root is None):
        raise ValueError("public mode requires both --source-root and --run-root")
    if source_root is not None and run_root is not None:
        from verify_agent_memory.natural_storage import load_public_api

        return load_public_api(source_root.resolve(), run_root.resolve())
    import scripts as scripts_package

    archive_scripts = str(archive_root / "scripts")
    if archive_scripts not in scripts_package.__path__:
        scripts_package.__path__.append(archive_scripts)
    api = dict(_prepare_archive_imports(archive_root))
    api["config_path"] = (
        archive_root / "experiments" / "configs" / "stage3_natural_corpus_public_eval.json"
    )
    api["route_root"] = (
        archive_root
        / "data"
        / "private"
        / "stage3_natural_corpus_admissibility_eval"
        / "route_bundles"
    )
    return api


def _run_global_reference(
    inputs: Any,
    api: Mapping[str, object],
    route_files: Mapping[tuple[str, str], Path],
    global_config_id: str,
) -> tuple[float, float]:
    aggregates = []
    for source, corpus in inputs.corpora.items():
        memories = {m.memory_id: m for m in corpus.memories if m.split == "eval"}
        queries = {q.query_id: q for q in corpus.queries if q.split == "eval"}
        accumulator = Accumulator()
        routes = api["load_route_bundle"](route_files[(source, global_config_id)])
        for row in routes:
            accumulator.update(
                _score_ranking(
                    queries[row.query_id],
                    row.ranked_memory_ids,
                    memories,
                    candidates_scored=row.candidates_scored,
                )
            )
        aggregates.append(accumulator.means())
    macro = _source_macro(aggregates)
    return float(macro["feasible_rate"]), float(macro["penalized_admissibility_upper_risk"])


def _global_reference() -> tuple[float, float]:
    with PARETO_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["arm"] == "global_dense" and item["top_k"] == "100")
    return float(row["feasible_rate"]), float(row["penalized_admissibility_upper_risk"])


def _run(
    archive_root: Path,
    output_dir: Path,
    batch_size: int,
    *,
    source_root: Path | None = None,
    run_root: Path | None = None,
) -> None:
    channels, rates, seeds, top_k = _load_protocol()
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

    accumulators: dict[tuple[str, float, int, str], Accumulator] = defaultdict(Accumulator)
    route_hashes: dict[str, str] = {}
    global_ranking_checks = 0
    namespace_ranking_checks = 0

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
        memory_embedding_rows = [
            table.index_by_identity[("memory", memory.text_sha256)] for memory in memories
        ]
        query_embedding_rows = {
            query.query_id: table.index_by_identity[("query", query.query_text_sha256)]
            for query in queries
        }
        memory_matrix = np.ascontiguousarray(table.vectors[memory_embedding_rows], dtype=np.float32)

        namespace_values = tuple(sorted({memory.namespace_id for memory in memories}))
        namespace_index = {value: index for index, value in enumerate(namespace_values)}
        original_namespace = np.asarray(
            [namespace_index[memory.namespace_id] for memory in memories], dtype=np.int64
        )
        swap_selected = {
            seed: _uniform_vector(
                seed,
                "namespace_swap",
                source,
                "source_memory_selection",
                count=len(memories),
            )
            for seed in seeds
        }
        swap_alternate = {
            seed: _alternate_namespace_indices(
                original_namespace,
                len(namespace_values),
                _uniform_vector(
                    seed,
                    "namespace_swap",
                    source,
                    "source_memory_alternate",
                    count=len(memories),
                ),
            )
            for seed in seeds
        }
        full_source_swap_counts = {
            (seed, rate): np.bincount(
                np.where(
                    swap_selected[seed] < rate,
                    swap_alternate[seed],
                    original_namespace,
                ),
                minlength=len(namespace_values),
            )
            for seed in seeds
            for rate in rates
        }

        for query_start in range(0, len(queries), batch_size):
            query_batch = queries[query_start : query_start + batch_size]
            query_matrix = np.ascontiguousarray(
                table.vectors[[query_embedding_rows[query.query_id] for query in query_batch]],
                dtype=np.float32,
            )
            score_matrix = np.asarray(memory_matrix @ query_matrix.T, dtype=np.float32)
            for query_column, query in enumerate(query_batch):
                candidate_ids = tuple(sorted(corpus.candidate_ids(query, namespace_only=False)))
                candidate_positions = np.asarray(
                    [position_by_id[memory_id] for memory_id in candidate_ids], dtype=np.int64
                )
                candidate_scores = score_matrix[candidate_positions, query_column]
                ranking = api["stable_rank_indices"](
                    candidate_scores,
                    api["lexical_tie_ranks"](candidate_ids),
                    limit=len(candidate_ids),
                )
                global_ranked = tuple(candidate_ids[int(index)] for index in ranking[:top_k])
                if global_ranked != global_routes[query.query_id].ranked_memory_ids:
                    raise RuntimeError(
                        f"global ranking parity failed for {source}/{query.query_id}"
                    )
                global_ranking_checks += 1

                query_namespace = namespace_index[query.namespace_id]
                candidate_original = original_namespace[candidate_positions]
                own_namespace = candidate_original == query_namespace
                namespace_ranked = _first_supported(
                    candidate_ids,
                    ranking,
                    partial(_namespace_support_at, own_namespace=own_namespace),
                    limit=top_k,
                )
                frozen_namespace = namespace_routes[query.query_id]
                if (
                    namespace_ranked != frozen_namespace.ranked_memory_ids
                    or int(np.count_nonzero(own_namespace)) != frozen_namespace.candidates_scored
                ):
                    raise RuntimeError(
                        f"namespace ranking parity failed for {source}/{query.query_id}"
                    )
                namespace_ranking_checks += 1

                all_source_candidates = len(candidate_ids) == len(memories)
                for seed in seeds:
                    gate_values = _uniform_vector(
                        seed,
                        "namespace_false_allow",
                        source,
                        query.query_id,
                        "memory_id_ascending",
                        count=len(candidate_ids),
                    )
                    outside_histogram, _ = np.histogram(
                        gate_values[~own_namespace],
                        bins=np.asarray((*rates, 1.0), dtype=np.float64),
                    )
                    outside_cumulative = np.concatenate(
                        (np.asarray([0], dtype=np.int64), np.cumsum(outside_histogram[:-1]))
                    )
                    for rate_index, rate in enumerate(rates):
                        false_allow_ranked = _first_supported(
                            candidate_ids,
                            ranking,
                            partial(
                                _false_allow_support_at,
                                rate=rate,
                                own_namespace=own_namespace,
                                gate_values=gate_values,
                            ),
                            limit=top_k,
                        )
                        false_allow_count = int(np.count_nonzero(own_namespace)) + int(
                            outside_cumulative[rate_index]
                        )
                        accumulators[("namespace_false_allow", rate, seed, source)].update(
                            _score_ranking(
                                query,
                                false_allow_ranked,
                                memory_by_id,
                                candidates_scored=false_allow_count,
                            )
                        )

                        selected = swap_selected[seed][candidate_positions]
                        alternate = swap_alternate[seed][candidate_positions]
                        swap_ranked = _first_supported(
                            candidate_ids,
                            ranking,
                            partial(
                                _swap_support_at,
                                rate=rate,
                                selected=selected,
                                alternate=alternate,
                                candidate_original=candidate_original,
                                query_namespace=query_namespace,
                            ),
                            limit=top_k,
                        )
                        if all_source_candidates:
                            swap_count = int(full_source_swap_counts[(seed, rate)][query_namespace])
                        else:
                            swap_count = int(
                                np.count_nonzero(
                                    np.where(
                                        selected < rate,
                                        alternate,
                                        candidate_original,
                                    )
                                    == query_namespace
                                )
                            )
                        accumulators[("namespace_swap", rate, seed, source)].update(
                            _score_ranking(
                                query,
                                swap_ranked,
                                memory_by_id,
                                candidates_scored=swap_count,
                            )
                        )
            del score_matrix, query_matrix
            gc.collect()
        del memory_matrix
        gc.collect()

    seed_rows: list[dict[str, object]] = []
    for channel in channels:
        for rate in rates:
            for seed in seeds:
                per_source = [
                    accumulators[(channel, rate, seed, source)].means()
                    for source in sorted(inputs.corpora)
                ]
                seed_rows.append(
                    {"channel": channel, "rate": rate, "seed": seed, **_source_macro(per_source)}
                )

    summary_rows: list[dict[str, object]] = []
    metric_fields = tuple(
        field
        for field in seed_rows[0]
        if field not in {"channel", "rate", "seed", "query_count", "matched_prefix_query_count"}
    )
    for channel in channels:
        for rate in rates:
            rows = [
                row for row in seed_rows if row["channel"] == channel and float(row["rate"]) == rate
            ]
            summary: dict[str, object] = {
                "channel": channel,
                "rate": rate,
                "seeds": len(rows),
                "query_count": int(rows[0]["query_count"]),
            }
            for metric in metric_fields:
                values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                summary[f"mean_{metric}"] = float(values.mean())
                summary[f"std_{metric}"] = float(values.std(ddof=1))
            summary_rows.append(summary)

    global_feasible, global_risk = (
        _global_reference()
        if run_root is None
        else _run_global_reference(inputs, api, route_files, global_config_id)
    )
    break_even: dict[str, object] = {}
    for channel in channels:
        last_dominating = None
        first_non_dominating = None
        for row in [item for item in summary_rows if item["channel"] == channel]:
            dominates = (
                float(row["mean_feasible_rate"]) >= global_feasible
                and float(row["mean_penalized_admissibility_upper_risk"]) <= global_risk
            )
            if dominates and first_non_dominating is None:
                last_dominating = float(row["rate"])
            elif not dominates and first_non_dominating is None:
                first_non_dominating = float(row["rate"])
        break_even[channel] = {
            "last_observed_dominating_rate": last_dominating,
            "first_observed_non_dominating_rate": first_non_dominating,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / OUTPUT_NAMES["seed_rows"]
    summary_path = output_dir / OUTPUT_NAMES["summary"]
    break_even_path = output_dir / OUTPUT_NAMES["break_even"]
    manifest_path = output_dir / OUTPUT_NAMES["manifest"]
    _write_csv(seed_path, seed_rows)
    _write_csv(summary_path, summary_rows)
    _write_json(
        break_even_path,
        {
            "schema_version": 1,
            "reference_arm": "global_dense",
            "reference_feasible_rate": global_feasible,
            "reference_penalized_admissibility_upper_risk": global_risk,
            "dominance": "feasible_rate_gte_and_penalized_admissibility_upper_risk_lte",
            "break_even": break_even,
        },
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    manifest = {
        "schema_version": 1,
        "execution_kind": "archive_replay" if run_root is None else "public_source_reproduction",
        "input_run_manifest_sha256": None
        if run_root is None
        else _sha256_file(run_root / "manifest.json"),
        "analysis": "natural-namespace-support-expansion-v1",
        "new_repository_commit": head,
        "old_execution_commit": "8e34e3d41c56e1699696bc27be95cdac7c9528e5"
        if run_root is None
        else None,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "embedding_checkpoint_sha256": _sha256_file(api["embedding_checkpoint_path"]),
        "route_bundle_sha256": dict(sorted(route_hashes.items())),
        "seed_rows_sha256": _sha256_file(seed_path),
        "summary_sha256": _sha256_file(summary_path),
        "break_even_sha256": _sha256_file(break_even_path),
        "query_count": sum(
            accumulator.query_count
            for key, accumulator in accumulators.items()
            if key[0] == channels[0] and key[1] == rates[0] and key[2] == seeds[0]
        ),
        "global_ranking_checks": global_ranking_checks,
        "namespace_ranking_checks": namespace_ranking_checks,
        "channels": list(channels),
        "rates": list(rates),
        "seeds": list(seeds),
        "top_k": top_k,
        "evaluation_retuning": False,
        "contains_query_or_memory_ids": False,
        "contains_raw_text_embeddings_or_responses": False,
        "provider_calls": 0,
        "reader_calls": 0,
        "judge_calls": 0,
        "paid_calls": 0,
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "seed_rows": len(seed_rows),
                "summary_rows": len(summary_rows),
                "break_even": break_even,
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
            args.run_root / "support_expansion" if args.run_root is not None else ROOT / "tmp"
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
