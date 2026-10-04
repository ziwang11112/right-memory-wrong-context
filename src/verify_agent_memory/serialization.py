"""Strict JSON adapters for the public experiment interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from verify_agent_memory.experiment import ExperimentCase, QueryRun, SettingSummary
from verify_agent_memory.retrieval import (
    MemoryRecord,
    PolicyDecision,
    PolicyPurpose,
    QueryRecord,
    RetrievalArm,
    RetrievalConfig,
)
from verify_agent_memory.schema import (
    LifecycleState,
    MemoryAssessment,
    QueryIntent,
    Relevance,
    Scope,
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TypeError(f"{label} must be a string")
    return value


def _optional_bool(value: object, label: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{label} must be true, false, or null")
    return value


def _vector(value: object, label: str) -> tuple[float, ...]:
    rows = _sequence(value, label)
    try:
        return tuple(float(item) for item in rows)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain numeric values") from error


def case_from_mapping(value: object) -> ExperimentCase:
    """Parse one query case; assessment labels remain scorer-only fields."""
    row = _mapping(value, "case")
    query_row = _mapping(row.get("query"), "query")
    query = QueryRecord(
        query_id=_string(query_row.get("query_id"), "query.query_id"),
        namespace=_string(query_row.get("namespace"), "query.namespace"),
        text=_string(query_row.get("text"), "query.text", allow_empty=True),
        embedding=_vector(query_row.get("embedding"), "query.embedding"),
        intent=QueryIntent(_string(query_row.get("intent", "unknown"), "query.intent")),
        policy_purpose=PolicyPurpose(
            _string(
                query_row.get("policy_purpose", "content_disclosure"),
                "query.policy_purpose",
            )
        ),
    )

    memories: list[MemoryRecord] = []
    assessments: list[MemoryAssessment] = []
    policy_decisions: list[PolicyDecision] = []
    for index, raw_memory in enumerate(_sequence(row.get("memories"), "memories")):
        memory_row = _mapping(raw_memory, f"memories[{index}]")
        if "policy_allowed" in memory_row:
            raise ValueError(
                f"memories[{index}].policy_allowed is deprecated; use released_policy "
                "for router-visible decisions and assessment.policy_allowed for scorer gold"
            )
        memory_id = _string(memory_row.get("memory_id"), f"memories[{index}].memory_id")
        lifecycle_state = LifecycleState(
            _string(
                memory_row.get("lifecycle_state", "unknown"),
                f"memories[{index}].lifecycle_state",
            )
        )
        try:
            released_order = float(memory_row.get("released_order"))
        except (TypeError, ValueError) as error:
            raise TypeError(f"memories[{index}].released_order must be numeric") from error
        memories.append(
            MemoryRecord(
                memory_id=memory_id,
                namespace=_string(memory_row.get("namespace"), f"memories[{index}].namespace"),
                text=_string(memory_row.get("text"), f"memories[{index}].text", allow_empty=True),
                embedding=_vector(memory_row.get("embedding"), f"memories[{index}].embedding"),
                released_order=released_order,
                lifecycle_state=lifecycle_state,
            )
        )
        released_policy = memory_row.get("released_policy")
        if released_policy is not None:
            policy_row = _mapping(released_policy, f"memories[{index}].released_policy")
            policy_decisions.append(
                PolicyDecision(
                    memory_id=memory_id,
                    content_disclosure_allowed=_optional_bool(
                        policy_row.get("content_disclosure_allowed"),
                        f"memories[{index}].released_policy.content_disclosure_allowed",
                    ),
                    operation_trace_allowed=_optional_bool(
                        policy_row.get("operation_trace_allowed"),
                        f"memories[{index}].released_policy.operation_trace_allowed",
                    ),
                )
            )
        assessment = _mapping(memory_row.get("assessment"), f"memories[{index}].assessment")
        assessments.append(
            MemoryAssessment(
                memory_id=memory_id,
                relevance=Relevance(
                    _string(
                        assessment.get("relevance", "unknown"),
                        f"memories[{index}].assessment.relevance",
                    )
                ),
                scope=Scope(
                    _string(
                        assessment.get("scope", "unknown"),
                        f"memories[{index}].assessment.scope",
                    )
                ),
                lifecycle_state=lifecycle_state,
                policy_allowed=_optional_bool(
                    assessment.get("policy_allowed"),
                    f"memories[{index}].assessment.policy_allowed",
                ),
                query_intent=query.intent,
            )
        )
    return ExperimentCase(
        source=_string(row.get("source"), "source"),
        group_id=_string(row.get("group_id"), "group_id"),
        query=query,
        memories=tuple(memories),
        assessments=tuple(assessments),
        policy_decisions=tuple(policy_decisions),
    )


_CONFIG_FIELDS = {
    "top_k",
    "bm25_k1",
    "bm25_b",
    "rrf_k",
    "recency_gamma",
    "theta",
    "top_l",
    "alpha_like",
    "cluster_beta",
    "cluster_gamma",
    "cluster_tau",
    "cluster_tau_new",
}


def config_from_mapping(value: object) -> RetrievalConfig:
    row = _mapping(value, "setting")
    unknown = set(row) - {"setting_id", "arm", *_CONFIG_FIELDS}
    if unknown:
        raise ValueError(f"unknown setting fields: {sorted(unknown)!r}")
    kwargs = {name: row[name] for name in _CONFIG_FIELDS if name in row}
    return RetrievalConfig(
        setting_id=_string(row.get("setting_id"), "setting_id"),
        arm=RetrievalArm(_string(row.get("arm"), "arm")),
        **kwargs,
    )


def query_run_to_mapping(run: QueryRun) -> dict[str, object]:
    score = run.score
    return {
        "metric_schema_version": 2,
        "source": run.source,
        "group_id": run.group_id,
        "query_id": run.query_id,
        "setting_id": run.setting_id,
        "arm": run.arm.value,
        "ranked_memory_ids": list(run.route.ranked_memory_ids),
        "selected_cluster_ids": list(run.route.selected_cluster_ids),
        "fallback_used": run.route.fallback_used,
        "route_width": run.route.route_width,
        "candidates_scored": run.route.candidates_scored,
        "anchor_total": score.anchor_total,
        "anchor_retrieved": score.anchor_retrieved,
        "evidence_recall": score.evidence_recall,
        "feasible": score.feasible,
        "matched_prefix_count": score.matched_prefix_count,
        "non_usable_known_rate": score.non_usable_known_rate,
        "non_usable_label_coverage": score.non_usable_label_coverage,
        "non_usable_lower_bound": score.non_usable_lower_bound,
        "non_usable_upper_bound": score.non_usable_upper_bound,
        "admissibility_violation_known_rate": score.admissibility_violation_known_rate,
        "admissibility_label_coverage": score.admissibility_label_coverage,
        "admissibility_violation_lower_bound": score.admissibility_violation_lower_bound,
        "admissibility_violation_upper_bound": score.admissibility_violation_upper_bound,
        "known_relevant_admissible_rate": score.known_relevant_admissible_rate,
        "known_relevant_inadmissible_rate": score.known_relevant_inadmissible_rate,
        "known_irrelevant_admissible_rate": score.known_irrelevant_admissible_rate,
        "known_irrelevant_inadmissible_rate": score.known_irrelevant_inadmissible_rate,
        "joint_label_coverage": score.joint_label_coverage,
        "wrong_scope_exposure_rate": score.wrong_scope_exposure_rate,
        "policy_disallowed_exposure_rate": score.policy_disallowed_exposure_rate,
        "lifecycle_incompatible_exposure_rate": score.lifecycle_incompatible_exposure_rate,
        "unresolved_admissibility_rate": score.unresolved_admissibility_rate,
    }


def setting_summary_to_mapping(summary: SettingSummary) -> dict[str, object]:
    return {
        "metric_schema_version": 2,
        "setting_id": summary.setting_id,
        "arm": summary.arm.value,
        "source_count": summary.source_count,
        "query_count": summary.query_count,
        "feasible_rate": summary.feasible_rate,
        "penalized_non_usable_upper_risk": summary.penalized_non_usable_upper_risk,
        "penalized_non_usable_known_risk": summary.penalized_non_usable_known_risk,
        "penalized_admissibility_upper_risk": summary.penalized_admissibility_upper_risk,
        "penalized_admissibility_known_risk": summary.penalized_admissibility_known_risk,
        "infeasibility_risk_component": summary.infeasibility_risk_component,
        "non_usable_conditional_risk_component": (summary.non_usable_conditional_risk_component),
        "admissibility_conditional_risk_component": (
            summary.admissibility_conditional_risk_component
        ),
        "conditional_non_usable_upper_risk": summary.conditional_non_usable_upper_risk,
        "conditional_admissibility_upper_risk": (summary.conditional_admissibility_upper_risk),
        "evidence_recall": summary.evidence_recall,
        "candidates_scored": summary.candidates_scored,
    }
