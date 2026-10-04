from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from verify_agent_memory.policy_axis_sensitivity import (
    admissibility_status,
    score_axis_family,
    select_common_feasible_records,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_policy_omission_changes_only_policy_disallowance() -> None:
    common = {
        "same_namespace": True,
        "required_anchor": False,
        "memory_state": "current",
        "query_intent": "current_state",
        "policy_disallowed": True,
    }
    assert admissibility_status(**common, include_policy=True) is False
    assert admissibility_status(**common, include_policy=False) is True

    stale = {**common, "memory_state": "stale", "policy_disallowed": False}
    assert admissibility_status(**stale, include_policy=True) is False
    assert admissibility_status(**stale, include_policy=False) is False


def test_required_anchor_and_namespace_precedence_match_frozen_evaluator() -> None:
    assert (
        admissibility_status(
            same_namespace=True,
            required_anchor=True,
            memory_state="superseded",
            query_intent="current_state",
            policy_disallowed=True,
            include_policy=True,
        )
        is True
    )
    assert (
        admissibility_status(
            same_namespace=False,
            required_anchor=True,
            memory_state="current",
            query_intent="current_state",
            policy_disallowed=False,
            include_policy=False,
        )
        is False
    )


def test_axis_family_reports_known_coverage_and_sharp_bounds() -> None:
    score = score_axis_family(
        ("bad", "unknown", "good", "anchor"),
        {"anchor"},
        {"bad": False, "unknown": None, "good": True, "anchor": True},
        target_recall=0.8,
        infeasibility_cost=1.0,
    )

    assert score.feasible is True
    assert score.known_risk == pytest.approx(1 / 3)
    assert score.coverage == pytest.approx(3 / 4)
    assert score.lower_bound == pytest.approx(1 / 4)
    assert score.upper_bound == pytest.approx(2 / 4)
    assert score.any_known_violation == 1.0
    assert score.known_violation_count == 1.0


def test_infeasible_axis_family_uses_frozen_unit_penalty() -> None:
    score = score_axis_family(
        ("other",),
        {"anchor"},
        {"other": True},
        target_recall=0.8,
        infeasibility_cost=1.0,
    )

    assert score.feasible is False
    assert score.penalized_upper_risk == 1.0
    assert score.known_risk is None


def test_common_feasible_records_use_the_same_query_intersection() -> None:
    records = (
        {"source": "a", "query": "q1", "arm": "global", "feasible": True},
        {"source": "a", "query": "q1", "arm": "namespace", "feasible": True},
        {"source": "a", "query": "q2", "arm": "global", "feasible": False},
        {"source": "a", "query": "q2", "arm": "namespace", "feasible": True},
        {"source": "a", "query": "q3", "arm": "global", "feasible": True},
        {"source": "a", "query": "q3", "arm": "namespace", "feasible": False},
    )

    selected = select_common_feasible_records(
        records,
        arms=("global", "namespace"),
        identity_fields=("source", "query"),
    )

    assert len(selected) == 2
    assert {row["query"] for row in selected} == {"q1"}
    assert {row["arm"] for row in selected} == {"global", "namespace"}


def test_policy_axis_result_bundle_is_hash_bound_and_content_free() -> None:
    result_dir = ROOT / "results" / "policy_axis_sensitivity"
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["analysis"] == "natural-policy-axis-sensitivity-v1"
    assert manifest["query_count"] == 3767
    assert manifest["common_feasible_query_count"] > 0
    assert manifest["group_count"] == 87
    assert manifest["contains_query_memory_or_group_ids"] is False
    assert manifest["contains_raw_text_embeddings_prompts_or_responses"] is False
    assert manifest["evaluation_retuning"] is False
    assert manifest["provider_calls"] == manifest["reader_calls"] == 0
    assert manifest["judge_calls"] == manifest["paid_calls"] == 0

    forbidden = {"query_id", "memory_id", "group_id", "query_text", "response", "embedding"}
    for name, receipt in manifest["published_files"].items():
        path = result_dir / name
        assert _sha256(path) == receipt["sha256"]
        if path.suffix == ".csv":
            rows = _csv_rows(path)
            assert len(rows) == receipt["row_count"]
            assert rows
            assert not forbidden.intersection(rows[0])


def test_policy_omission_preserves_namespace_advantage() -> None:
    result_dir = ROOT / "results" / "policy_axis_sensitivity"
    summary = {
        (row["axis_family"], row["arm"]): row for row in _csv_rows(result_dir / "summary.csv")
    }
    deltas = {
        (row["axis_family"], row["metric"]): row
        for row in _csv_rows(result_dir / "paired_deltas.csv")
    }

    family = "scope_lifecycle_no_policy"
    assert float(summary[(family, "global_dense")]["penalized_upper_risk"]) == pytest.approx(
        0.7868183171784588
    )
    assert float(summary[(family, "namespace_dense")]["penalized_upper_risk"]) == pytest.approx(
        0.6934982422209819
    )
    delta = deltas[(family, "penalized_upper_risk")]
    assert float(delta["estimate"]) == pytest.approx(-0.09332007495747696)
    assert float(delta["ci_upper"]) < 0.0


def test_common_feasible_sensitivity_preserves_namespace_risk_direction() -> None:
    result_dir = ROOT / "results" / "policy_axis_sensitivity"
    summary = {
        (row["axis_family"], row["arm"]): row
        for row in _csv_rows(result_dir / "common_feasible_summary.csv")
    }
    deltas = {
        (row["axis_family"], row["metric"]): row
        for row in _csv_rows(result_dir / "common_feasible_paired_deltas.csv")
    }

    family = "full_scope_policy_lifecycle"
    assert (
        summary[(family, "global_dense")]["common_feasible_query_count"]
        == summary[(family, "namespace_dense")]["common_feasible_query_count"]
    )
    assert float(summary[(family, "namespace_dense")]["upper_bound"]) < float(
        summary[(family, "global_dense")]["upper_bound"]
    )
    assert float(deltas[(family, "upper_bound")]["ci_upper"]) < 0.0
