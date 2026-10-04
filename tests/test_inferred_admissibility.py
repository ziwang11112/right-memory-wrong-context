from __future__ import annotations

from dataclasses import replace

import pytest

from verify_agent_memory.inferred_admissibility import (
    CandidatePrediction,
    CasePrediction,
    FilterSetting,
    InferenceCandidate,
    InferenceCase,
    ProbabilityVector,
    aggregate_route_scores,
    binary_classification_metrics,
    case_from_mapping,
    filter_decision_metrics,
    intent_classification_metrics,
    prediction_from_mapping,
    prompt_payload,
    response_json_schema,
    route_keys,
    score_filtered_route,
    select_filter_setting,
    stratified_group_paired_bootstrap,
    validate_sample_contract,
)


def _case(*, role: str = "calibration", case_id: str = "case-1") -> InferenceCase:
    return InferenceCase(
        case_id=case_id,
        source="memops",
        group_id="hidden-group",
        role=role,
        query_text="What is the current destination?",
        query_visible_time=None,
        released_query_intent="current_state",
        anchor_total=1,
        candidates=(
            InferenceCandidate(
                candidate_key="c01",
                rank=1,
                text="The former destination was Rome.",
                visible_order="event-1",
                required_evidence=False,
                released_policy_allowed=False,
                released_lifecycle_compatible=True,
            ),
            InferenceCandidate(
                candidate_key="c02",
                rank=2,
                text="The current destination is Paris.",
                visible_order="event-2",
                required_evidence=True,
                released_policy_allowed=True,
                released_lifecycle_compatible=True,
            ),
        ),
    )


def _vector(labels: tuple[str, ...], values: tuple[float, ...]) -> ProbabilityVector:
    return ProbabilityVector(labels=labels, values=values)


@pytest.mark.parametrize("field", ["released_policy_allowed", "released_lifecycle_compatible"])
def test_inference_candidate_rejects_numeric_labels(field: str) -> None:
    with pytest.raises(TypeError, match="released candidate labels"):
        replace(_case().candidates[0], **{field: 0})

    row = {
        "case_id": "case-1",
        "source": "memops",
        "group_id": "group-1",
        "role": "analysis",
        "query": {
            "text": "Where is the parcel?",
            "visible_time": None,
            "released_query_intent": "current_state",
            "anchor_total": 1,
        },
        "candidates": [
            {
                "candidate_key": "c01",
                "rank": 1,
                "text": "It is in Paris.",
                "visible_order": None,
                "required_evidence": True,
                "released_policy_allowed": True,
                "released_lifecycle_compatible": True,
                field: 1,
            }
        ],
    }
    with pytest.raises(TypeError, match=field):
        case_from_mapping(row)


def _prediction(case: InferenceCase) -> CasePrediction:
    return CasePrediction(
        case_id=case.case_id,
        query_intent=_vector(
            ("current_state", "history", "unknown"),
            (0.9, 0.05, 0.05),
        ),
        candidates=(
            CandidatePrediction(
                candidate_key="c01",
                policy=_vector(("allowed", "disallowed", "unknown"), (0.05, 0.9, 0.05)),
                lifecycle=_vector(
                    ("compatible", "incompatible", "unknown"),
                    (0.9, 0.05, 0.05),
                ),
                admissibility=_vector(
                    ("admissible", "inadmissible", "unknown"),
                    (0.05, 0.9, 0.05),
                ),
            ),
            CandidatePrediction(
                candidate_key="c02",
                policy=_vector(("allowed", "disallowed", "unknown"), (0.9, 0.05, 0.05)),
                lifecycle=_vector(
                    ("compatible", "incompatible", "unknown"),
                    (0.9, 0.05, 0.05),
                ),
                admissibility=_vector(
                    ("admissible", "inadmissible", "unknown"),
                    (0.9, 0.05, 0.05),
                ),
            ),
        ),
    )


def _prediction_mapping() -> dict[str, object]:
    return {
        "query_intent": {"current_state": 0.9, "history": 0.05, "unknown": 0.05},
        "candidates": [
            {
                "candidate_key": "c01",
                "policy": {"allowed": 0.05, "disallowed": 0.9, "unknown": 0.05},
                "lifecycle": {"compatible": 0.9, "incompatible": 0.05, "unknown": 0.05},
                "admissibility": {"admissible": 0.05, "inadmissible": 0.9, "unknown": 0.05},
            },
            {
                "candidate_key": "c02",
                "policy": {"allowed": 0.9, "disallowed": 0.05, "unknown": 0.05},
                "lifecycle": {"compatible": 0.9, "incompatible": 0.05, "unknown": 0.05},
                "admissibility": {"admissible": 0.9, "inadmissible": 0.05, "unknown": 0.05},
            },
        ],
    }


def test_prompt_payload_excludes_scorer_only_fields() -> None:
    case = _case(case_id="secret-case-id")
    payload = prompt_payload(case)

    assert "secret-case-id" not in payload
    assert "hidden-group" not in payload
    assert "memops" not in payload
    assert "required_evidence" not in payload
    assert "released_policy_allowed" not in payload
    assert "released_lifecycle_compatible" not in payload
    assert "c01" in payload
    assert "former destination" in payload


def test_response_schema_binds_exact_candidate_count() -> None:
    schema = response_json_schema(20)
    candidates = schema["properties"]["candidates"]  # type: ignore[index]

    assert candidates["minItems"] == 20  # type: ignore[index]
    assert candidates["maxItems"] == 20  # type: ignore[index]


def test_prediction_rejects_probability_or_order_drift() -> None:
    case = _case()
    parsed = prediction_from_mapping(_prediction_mapping(), case)
    assert parsed.candidates[0].violation_probability == pytest.approx(0.9)

    bad_sum = _prediction_mapping()
    bad_sum["query_intent"] = {"current_state": 0.8, "history": 0.1, "unknown": 0.2}
    with pytest.raises(ValueError, match="sum to 1"):
        prediction_from_mapping(bad_sum, case)

    bad_order = _prediction_mapping()
    rows = bad_order["candidates"]
    assert isinstance(rows, list)
    rows.reverse()
    with pytest.raises(ValueError, match="request order"):
        prediction_from_mapping(bad_order, case)


def test_calibration_selection_preserves_required_anchor() -> None:
    case = _case()
    prediction = _prediction(case)

    setting = select_filter_setting(
        (case,),
        {case.case_id: prediction},
        violation_thresholds=(0.5, 0.8, 0.95),
        unknown_thresholds=None,
        maximum_required_anchor_false_deny_rate=0.01,
    )

    assert setting.violation_threshold == 0.8
    assert setting.known_violation_recall == 1
    assert setting.known_violation_precision == 1
    assert setting.required_anchor_false_deny_rate == 0


def test_route_arms_measure_oracle_and_inferred_gap() -> None:
    case = _case(role="analysis")
    prediction = _prediction(case)
    setting = FilterSetting(
        violation_threshold=0.8,
        unknown_threshold=None,
        known_violation_precision=1.0,
        known_violation_recall=1.0,
        required_anchor_false_deny_rate=0.0,
        dropped_unknown_gold=0,
    )

    baseline = score_filtered_route(
        case,
        route_keys(case, arm="namespace_dense"),
        target_recall=0.8,
    )
    oracle = score_filtered_route(
        case,
        route_keys(case, arm="released_oracle"),
        target_recall=0.8,
    )
    inferred = score_filtered_route(
        case,
        route_keys(case, arm="text_inferred", prediction=prediction, setting=setting),
        target_recall=0.8,
    )

    assert baseline.admissibility_upper_risk == 0.5
    assert oracle.admissibility_upper_risk == 0
    assert inferred == oracle
    aggregate = aggregate_route_scores((baseline,))
    assert aggregate["penalized_admissibility_upper_risk"] == 0.5

    decision = filter_decision_metrics((case,), {case.case_id: prediction}, setting)
    assert decision["violation_precision"] == 1
    assert decision["violation_recall"] == 1
    assert decision["required_anchor_false_deny_rate"] == 0

    bootstrap = stratified_group_paired_bootstrap(
        (case,),
        {case.case_id: inferred},
        {case.case_id: baseline},
        metric="penalized_admissibility_upper_risk",
        replicates=50,
        seed=7,
    )
    assert bootstrap["estimate"] == -0.5
    assert bootstrap["ci_lower"] == -0.5
    assert bootstrap["ci_upper"] == -0.5


def test_classification_metrics_report_discrimination_and_intent() -> None:
    calibration = _case()
    analysis = _case(role="analysis", case_id="case-2")
    predictions = {
        calibration.case_id: _prediction(calibration),
        analysis.case_id: _prediction(analysis),
    }

    classification = binary_classification_metrics(
        (calibration, analysis),
        predictions,
        axis="admissibility",
    )
    intent = intent_classification_metrics((calibration, analysis), predictions)

    assert classification["accuracy"] == 1
    assert classification["roc_auc"] == 1
    assert classification["pr_auc"] == 1
    assert intent["accuracy"] == 1


def test_balanced_sample_contract_rejects_split_drift() -> None:
    cases = (
        _case(role="calibration", case_id="c-cal"),
        _case(role="analysis", case_id="c-analysis"),
    )
    summary = validate_sample_contract(
        cases,
        strata=("memops/current_state",),
        queries_per_stratum=2,
        calibration_per_stratum=1,
        candidate_depth=2,
    )
    assert summary["case_count"] == 2

    with pytest.raises(ValueError, match="calibration count drifted"):
        validate_sample_contract(
            cases,
            strata=("memops/current_state",),
            queries_per_stratum=2,
            calibration_per_stratum=2,
            candidate_depth=2,
        )
