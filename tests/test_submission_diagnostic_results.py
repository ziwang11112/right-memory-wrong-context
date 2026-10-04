from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "submission_zero_call_diagnostics"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_submission_diagnostic_bundle_is_hash_bound_and_content_free() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["analysis"] == "submission-zero-call-diagnostics-v1"
    assert manifest["analysis_commit"] == "616d84ec3a49eade0c6cef946f741ce0ea92565e"
    assert manifest["query_count"] == 3767
    assert manifest["group_count"] == 87
    assert manifest["global_ranking_checks"] == 3767
    assert manifest["namespace_ranking_checks"] == 3767
    assert manifest["contains_query_memory_or_group_ids"] is False
    assert manifest["contains_raw_text_embeddings_prompts_or_responses"] is False
    assert manifest["gold_preserving_control_is_non_deployable"] is True
    assert manifest["missingness_changes_evaluator_labels_only"] is True
    assert manifest["evaluation_retuning"] is False
    assert manifest["provider_calls"] == manifest["reader_calls"] == 0
    assert manifest["judge_calls"] == manifest["paid_calls"] == 0

    forbidden = {"query_id", "memory_id", "group_id", "query_text", "response", "embedding"}
    for name, receipt in manifest["published_files"].items():
        path = RESULTS / name
        assert _sha256(path) == receipt["sha256"]
        if path.suffix == ".csv":
            rows = _rows(name)
            assert len(rows) == receipt["row_count"]
            assert rows
            assert not forbidden.intersection(rows[0])


def test_missingness_changes_bounds_but_not_frozen_retrieval() -> None:
    rows = _rows("label_missingness_summary.csv")
    for arm in ("global_dense", "namespace_dense"):
        selected = sorted(
            (row for row in rows if row["arm"] == arm),
            key=lambda row: float(row["missing_rate"]),
        )
        assert len({row["mean_evidence_recall"] for row in selected}) == 1
        assert len({row["mean_feasible_rate"] for row in selected}) == 1
        coverage = [float(row["mean_coverage"]) for row in selected]
        widths = [float(row["mean_bound_width"]) for row in selected]
        assert coverage == sorted(coverage, reverse=True)
        assert widths == sorted(widths)
        assert widths[0] < 0.01
        assert widths[-1] > 0.89


def test_gold_preserving_oracle_matches_namespace_size_and_exposes_headroom() -> None:
    rows = {row["arm"]: row for row in _rows("gold_preserving_summary.csv")}
    oracle = rows["gold_preserving_same_size"]
    namespace = rows["namespace_pre_filter"]

    assert oracle["uses_released_gold"] == "True"
    assert float(oracle["mean_candidates_scored"]) == pytest.approx(
        float(namespace["mean_candidates_scored"])
    )
    assert float(oracle["mean_evidence_recall"]) == pytest.approx(0.8396360756421977)
    assert float(oracle["mean_feasible_rate"]) == pytest.approx(0.7466160206340636)
    assert float(oracle["mean_evidence_recall"]) > float(namespace["mean_evidence_recall"])
