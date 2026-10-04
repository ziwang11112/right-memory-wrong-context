from __future__ import annotations

import pytest

from verify_agent_memory.support_controls import (
    exact_sign_flip_pvalue,
    filter_ranking,
    leave_one_out_means,
    permute_labels,
    scenario_selectivity,
    score_matched_recall,
)


def test_permuted_labels_are_deterministic_and_size_matched() -> None:
    labels = (0, 0, 0, 1, 1, 2)
    first = permute_labels(labels, seed=4, channel="random_support", identity="source")
    second = permute_labels(labels, seed=4, channel="random_support", identity="source")

    assert first == second
    assert sorted(first) == sorted(labels)
    assert first != permute_labels(labels, seed=5, channel="random_support", identity="source")


def test_filter_ranking_preserves_order_and_depth() -> None:
    ranked = ("m1", "m2", "m3", "m4", "m5")
    assert filter_ranking(ranked, {"m2", "m4", "m5"}, limit=2) == ("m2", "m4")
    assert filter_ranking(ranked[:3], {"m2", "m4", "m5"}, limit=5) == ("m2",)


def test_zero_depth_filter_exposes_no_memory() -> None:
    assert filter_ranking(("m1", "m2"), {"m1", "m2"}, limit=0) == ()


@pytest.mark.parametrize("numeric_label", [0, 1, 0.0, 1.0])
def test_matched_recall_rejects_numeric_statuses(numeric_label: object) -> None:
    with pytest.raises(ValueError, match="true, false, or unresolved"):
        score_matched_recall(
            ("m1",),
            {"m1"},
            {"m1": numeric_label},  # type: ignore[dict-item]
            {"m1": True},
            target_recall=0.8,
            infeasibility_cost=1.0,
        )


def test_matched_recall_scores_full_and_scope_excluded_risk() -> None:
    ranked = ("wrong-scope", "stale", "anchor", "extra")
    full = {
        "wrong-scope": False,
        "stale": False,
        "anchor": True,
        "extra": True,
    }
    residual = {
        "wrong-scope": True,
        "stale": False,
        "anchor": True,
        "extra": True,
    }

    score = score_matched_recall(
        ranked,
        {"anchor"},
        full,
        residual,
        target_recall=0.8,
        infeasibility_cost=1.0,
    )

    assert score.feasible is True
    assert score.matched_prefix == ranked[:3]
    assert score.admissibility_upper_risk == pytest.approx(2 / 3)
    assert score.residual_upper_risk == pytest.approx(1 / 3)
    assert score.known_admissibility_violation_count == 2
    assert score.any_known_admissibility_violation is True
    assert score.upper_admissibility_violation_count == 2


def test_non_dilutable_violation_companions_survive_admissible_padding() -> None:
    ranked = ("wrong-scope", "padding-1", "padding-2", "anchor")
    statuses = {
        "wrong-scope": False,
        "padding-1": True,
        "padding-2": True,
        "anchor": True,
    }

    score = score_matched_recall(
        ranked,
        {"anchor"},
        statuses,
        statuses,
        target_recall=0.8,
        infeasibility_cost=1.0,
    )

    assert score.admissibility_upper_risk == pytest.approx(0.25)
    assert score.known_admissibility_violation_count == 1
    assert score.any_known_admissibility_violation is True
    assert score.upper_admissibility_violation_count == 1


def test_infeasibility_cost_is_explicit_and_does_not_change_recall() -> None:
    statuses = {"other": True}
    low = score_matched_recall(
        ("other",),
        {"missing-anchor"},
        statuses,
        statuses,
        target_recall=0.8,
        infeasibility_cost=0.25,
    )
    high = score_matched_recall(
        ("other",),
        {"missing-anchor"},
        statuses,
        statuses,
        target_recall=0.8,
        infeasibility_cost=1.0,
    )

    assert low.evidence_recall == high.evidence_recall == 0.0
    assert low.penalized_admissibility_upper_risk == 0.25
    assert high.penalized_admissibility_upper_risk == 1.0
    assert low.known_admissibility_violation_count is None
    assert low.any_known_admissibility_violation is None
    assert low.upper_admissibility_violation_count is None


def test_leave_one_out_and_exact_sign_flip_are_deterministic() -> None:
    assert leave_one_out_means({"a": 1.0, "b": 2.0, "c": 3.0}) == {
        "a": 2.5,
        "b": 2.0,
        "c": 1.5,
    }
    assert exact_sign_flip_pvalue((1.0, 1.0, 1.0)) == pytest.approx(0.25)
    assert exact_sign_flip_pvalue((0.0, 0.0)) == 1.0


def test_scenario_selectivity_uses_paired_relevant_cells() -> None:
    rows = (
        {"scenario_id": "s1", "cell": "relevant_admissible", "exposure_effect": 1},
        {"scenario_id": "s1", "cell": "relevant_admissible", "exposure_effect": 0},
        {"scenario_id": "s1", "cell": "relevant_inadmissible", "exposure_effect": 0},
        {"scenario_id": "s1", "cell": "irrelevant_admissible", "exposure_effect": 1},
        {"scenario_id": "s2", "cell": "relevant_admissible", "exposure_effect": 1},
        {"scenario_id": "s2", "cell": "relevant_inadmissible", "exposure_effect": 1},
    )

    assert scenario_selectivity(rows) == {"s1": 0.5, "s2": 0.0}
