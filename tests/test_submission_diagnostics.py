from __future__ import annotations

import pytest

from verify_agent_memory.policy_axis_sensitivity import score_axis_family
from verify_agent_memory.submission_diagnostics import (
    deterministic_uniform,
    mask_established_statuses,
    sample_gold_preserving_positions,
)


def test_deterministic_uniform_is_stable_and_bounded() -> None:
    first = deterministic_uniform(seed=3, channel="test", identity="case")
    assert first == deterministic_uniform(seed=3, channel="test", identity="case")
    assert 0.0 <= first < 1.0
    assert first != deterministic_uniform(seed=4, channel="test", identity="case")


def test_status_missingness_changes_only_evaluator_knowledge() -> None:
    statuses = {"positive": True, "negative": False, "unknown": None}
    assert (
        mask_established_statuses(statuses, missing_rate=0.0, seed=0, identity="query") == statuses
    )
    assert mask_established_statuses(statuses, missing_rate=1.0, seed=0, identity="query") == {
        "positive": None,
        "negative": None,
        "unknown": None,
    }
    with pytest.raises(ValueError, match="missing_rate"):
        mask_established_statuses(statuses, missing_rate=1.01, seed=0, identity="query")


def test_missingness_widens_partial_identification_without_changing_recall() -> None:
    ranked = ("bad", "good", "anchor")
    anchors = {"anchor"}
    baseline = score_axis_family(
        ranked,
        anchors,
        {"bad": False, "good": True, "anchor": True},
        target_recall=0.8,
        infeasibility_cost=1.0,
    )
    masked = score_axis_family(
        ranked,
        anchors,
        {"bad": None, "good": None, "anchor": None},
        target_recall=0.8,
        infeasibility_cost=1.0,
    )

    assert masked.evidence_recall == baseline.evidence_recall == 1.0
    assert masked.feasible is baseline.feasible is True
    assert masked.coverage == 0.0
    assert masked.lower_bound == 0.0
    assert masked.upper_bound == 1.0
    assert baseline.upper_bound - baseline.lower_bound == 0.0


def test_gold_preserving_support_is_exact_size_deterministic_and_anchor_complete() -> None:
    anchors = (1, 8)
    non_anchors = (0, 2, 3, 4, 5, 6, 7, 9)
    first = sample_gold_preserving_positions(
        anchors, non_anchors, target_size=5, seed=2, identity="query"
    )
    second = sample_gold_preserving_positions(
        anchors, non_anchors, target_size=5, seed=2, identity="query"
    )

    assert first == second
    assert len(first) == 5
    assert set(anchors).issubset(first)
    assert first != sample_gold_preserving_positions(
        anchors, non_anchors, target_size=5, seed=3, identity="query"
    )


@pytest.mark.parametrize(
    ("anchors", "non_anchors", "target_size", "message"),
    [
        ((1, 1), (2,), 2, "unique"),
        ((1,), (1, 2), 2, "disjoint"),
        ((1, 2), (3,), 1, "too small"),
        ((1,), (2,), 3, "exceeds"),
    ],
)
def test_gold_preserving_support_rejects_invalid_partitions(
    anchors: tuple[int, ...],
    non_anchors: tuple[int, ...],
    target_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_gold_preserving_positions(
            anchors,
            non_anchors,
            target_size=target_size,
            seed=0,
            identity="query",
        )
