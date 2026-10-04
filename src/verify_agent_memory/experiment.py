"""Execution and development-only selection for retrieval experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from verify_agent_memory.metrics import RouteScore, score_route
from verify_agent_memory.retrieval import (
    MemoryRecord,
    PolicyDecision,
    QueryRecord,
    RetrievalArm,
    RetrievalConfig,
    RouteResult,
    route,
)
from verify_agent_memory.schema import MemoryAssessment


@dataclass(frozen=True)
class ExperimentCase:
    source: str
    group_id: str
    query: QueryRecord
    memories: tuple[MemoryRecord, ...]
    assessments: tuple[MemoryAssessment, ...]
    policy_decisions: tuple[PolicyDecision, ...] = ()

    def __post_init__(self) -> None:
        if not self.source or not self.group_id:
            raise ValueError("source and group_id must be nonempty")
        memory_ids = {memory.memory_id for memory in self.memories}
        if len(memory_ids) != len(self.memories):
            raise ValueError("case memory IDs must be unique")
        assessment_ids = {assessment.memory_id for assessment in self.assessments}
        if len(assessment_ids) != len(self.assessments):
            raise ValueError("case assessment IDs must be unique")
        if assessment_ids != memory_ids:
            raise ValueError("assessments must cover every case memory exactly once")
        policy_ids = [decision.memory_id for decision in self.policy_decisions]
        if len(set(policy_ids)) != len(policy_ids):
            raise ValueError("case policy decisions must be unique")
        unknown_policy_ids = set(policy_ids) - memory_ids
        if unknown_policy_ids:
            raise ValueError(
                f"policy decisions reference unknown memories: {sorted(unknown_policy_ids)!r}"
            )


@dataclass(frozen=True)
class QueryRun:
    source: str
    group_id: str
    query_id: str
    setting_id: str
    arm: RetrievalArm
    route: RouteResult
    score: RouteScore


class SelectionRisk(StrEnum):
    NON_USABLE_UPPER_BOUND = "non_usable_upper_bound"
    ADMISSIBILITY_UPPER_BOUND = "admissibility_upper_bound"


@dataclass(frozen=True)
class SettingSummary:
    setting_id: str
    arm: RetrievalArm
    source_count: int
    query_count: int
    feasible_rate: float
    penalized_non_usable_upper_risk: float
    penalized_non_usable_known_risk: float
    penalized_admissibility_upper_risk: float
    penalized_admissibility_known_risk: float
    infeasibility_risk_component: float
    non_usable_conditional_risk_component: float
    admissibility_conditional_risk_component: float
    conditional_non_usable_upper_risk: float | None
    conditional_admissibility_upper_risk: float | None
    evidence_recall: float
    candidates_scored: float

    def selection_key(self, risk_target: SelectionRisk) -> tuple[float | str, ...]:
        """Lexicographic key; smaller is better."""
        if risk_target is SelectionRisk.NON_USABLE_UPPER_BOUND:
            upper_risk = self.penalized_non_usable_upper_risk
            known_risk = self.penalized_non_usable_known_risk
        else:
            upper_risk = self.penalized_admissibility_upper_risk
            known_risk = self.penalized_admissibility_known_risk
        return (
            -self.feasible_rate,
            upper_risk,
            known_risk,
            -self.evidence_recall,
            self.candidates_scored,
            self.setting_id,
        )

    @property
    def penalized_conservative_risk(self) -> float:
        """Legacy alias for the frozen v1 non-usable selection quantity."""
        return self.penalized_non_usable_upper_risk

    @property
    def penalized_resolved_contamination(self) -> float:
        return self.penalized_non_usable_known_risk


def run_case(
    case: ExperimentCase,
    config: RetrievalConfig,
    *,
    target_recall: float = 0.8,
) -> QueryRun:
    """Route and score one query without mutating the case or configuration."""
    routed = route(
        case.memories,
        case.query,
        config,
        policy_decisions=case.policy_decisions,
    )
    score = score_route(
        routed.ranked_memory_ids,
        case.assessments,
        target_recall=target_recall,
        route_width=routed.route_width,
        candidates_scored=routed.candidates_scored,
    )
    return QueryRun(
        source=case.source,
        group_id=case.group_id,
        query_id=case.query.query_id,
        setting_id=config.setting_id,
        arm=config.arm,
        route=routed,
        score=score,
    )


def validate_experiment(
    cases: Sequence[ExperimentCase],
    configs: Sequence[RetrievalConfig],
    *,
    target_recall: float = 0.8,
) -> None:
    """Check experiment-wide inputs without running retrieval or scoring."""
    if not cases or not configs:
        raise ValueError("cases and configs must be nonempty")
    if isinstance(target_recall, bool) or not isinstance(target_recall, (int, float)):
        raise TypeError("target_recall must be numeric")
    if not math.isfinite(float(target_recall)) or not 0 <= target_recall <= 1:
        raise ValueError("target_recall must be finite and in [0, 1]")
    case_ids = [(case.source, case.query.query_id) for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("source/query identities must be unique")
    setting_ids = [config.setting_id for config in configs]
    if len(set(setting_ids)) != len(setting_ids):
        raise ValueError("setting IDs must be unique")
    for case in cases:
        if any(len(memory.embedding) != len(case.query.embedding) for memory in case.memories):
            raise ValueError("all memory and query embeddings must share one dimension")


def run_experiment(
    cases: Sequence[ExperimentCase],
    configs: Sequence[RetrievalConfig],
    *,
    target_recall: float = 0.8,
) -> tuple[QueryRun, ...]:
    """Run the complete Cartesian product of cases and frozen settings."""
    validate_experiment(cases, configs, target_recall=target_recall)
    return tuple(
        run_case(case, config, target_recall=target_recall) for config in configs for case in cases
    )


def _mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("macro means require finite, nonempty values")
    return sum(values) / len(values)


def _penalized(score: RouteScore, value: float | None) -> float | None:
    if score.feasible is None:
        return None
    if score.feasible and value is not None:
        return value
    return 1.0


def _available(values: Sequence[float | bool | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


@dataclass(frozen=True)
class _SourceMetrics:
    feasible_rate: float
    non_usable_upper: float
    non_usable_known: float
    admissibility_upper: float
    admissibility_known: float
    infeasibility_component: float
    non_usable_conditional_component: float
    admissibility_conditional_component: float
    conditional_non_usable_upper: float | None
    conditional_admissibility_upper: float | None
    recall: float
    candidates: float


def _risk_decomposition(
    source_runs: Sequence[QueryRun],
    *,
    value_name: str,
) -> tuple[float, float, float | None]:
    evaluable = [run for run in source_runs if run.score.feasible is not None]
    if not evaluable:
        raise ValueError("risk decomposition requires recall-evaluable queries")
    infeasibility = _mean([float(run.score.feasible is False) for run in evaluable])
    feasible_values = [
        (
            float(getattr(run.score, value_name))
            if getattr(run.score, value_name) is not None
            else 1.0
        )
        for run in evaluable
        if run.score.feasible is True
    ]
    conditional = _mean(feasible_values) if feasible_values else None
    conditional_component = _mean(
        [
            (
                float(getattr(run.score, value_name))
                if run.score.feasible is True and getattr(run.score, value_name) is not None
                else float(run.score.feasible is True)
            )
            for run in evaluable
        ]
    )
    return infeasibility, conditional_component, conditional


def _optional_source_mean(values: Sequence[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return _mean(available) if available else None


def summarize_setting(runs: Sequence[QueryRun]) -> SettingSummary:
    """Equal-weight sources after query-macro aggregation within each source."""
    if not runs:
        raise ValueError("at least one query run is required")
    setting_ids = {run.setting_id for run in runs}
    arms = {run.arm for run in runs}
    if len(setting_ids) != 1 or len(arms) != 1:
        raise ValueError("one setting summary cannot mix settings or arms")
    identities = [(run.source, run.query_id) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("setting rows contain duplicate source/query identities")

    by_source: defaultdict[str, list[QueryRun]] = defaultdict(list)
    for run in runs:
        by_source[run.source].append(run)

    source_metrics: list[_SourceMetrics] = []
    for source_runs in by_source.values():
        feasible = _available([run.score.feasible for run in source_runs])
        non_usable_upper = _available(
            [_penalized(run.score, run.score.non_usable_upper_bound) for run in source_runs]
        )
        non_usable_known = _available(
            [_penalized(run.score, run.score.non_usable_known_rate) for run in source_runs]
        )
        admissibility_upper = _available(
            [
                _penalized(run.score, run.score.admissibility_violation_upper_bound)
                for run in source_runs
            ]
        )
        admissibility_known = _available(
            [
                _penalized(run.score, run.score.admissibility_violation_known_rate)
                for run in source_runs
            ]
        )
        recall = _available([run.score.evidence_recall for run in source_runs])
        candidates = [float(run.route.candidates_scored) for run in source_runs]
        if not all(
            (
                feasible,
                non_usable_upper,
                non_usable_known,
                admissibility_upper,
                admissibility_known,
                recall,
            )
        ):
            raise ValueError("every source needs recall-evaluable development queries")
        infeasibility, non_usable_component, conditional_non_usable = _risk_decomposition(
            source_runs,
            value_name="non_usable_upper_bound",
        )
        (
            admissibility_infeasibility,
            admissibility_component,
            conditional_admissibility,
        ) = _risk_decomposition(
            source_runs,
            value_name="admissibility_violation_upper_bound",
        )
        if not math.isclose(infeasibility, admissibility_infeasibility, abs_tol=1e-15):
            raise RuntimeError("risk decompositions disagree on infeasibility")
        non_usable_upper_mean = _mean(non_usable_upper)
        admissibility_upper_mean = _mean(admissibility_upper)
        if not math.isclose(
            non_usable_upper_mean,
            infeasibility + non_usable_component,
            abs_tol=1e-15,
        ):
            raise RuntimeError("non-usable risk decomposition does not close")
        if not math.isclose(
            admissibility_upper_mean,
            infeasibility + admissibility_component,
            abs_tol=1e-15,
        ):
            raise RuntimeError("admissibility risk decomposition does not close")
        source_metrics.append(
            _SourceMetrics(
                feasible_rate=_mean(feasible),
                non_usable_upper=non_usable_upper_mean,
                non_usable_known=_mean(non_usable_known),
                admissibility_upper=admissibility_upper_mean,
                admissibility_known=_mean(admissibility_known),
                infeasibility_component=infeasibility,
                non_usable_conditional_component=non_usable_component,
                admissibility_conditional_component=admissibility_component,
                conditional_non_usable_upper=conditional_non_usable,
                conditional_admissibility_upper=conditional_admissibility,
                recall=_mean(recall),
                candidates=_mean(candidates),
            )
        )

    return SettingSummary(
        setting_id=next(iter(setting_ids)),
        arm=next(iter(arms)),
        source_count=len(by_source),
        query_count=len(runs),
        feasible_rate=_mean([row.feasible_rate for row in source_metrics]),
        penalized_non_usable_upper_risk=_mean([row.non_usable_upper for row in source_metrics]),
        penalized_non_usable_known_risk=_mean([row.non_usable_known for row in source_metrics]),
        penalized_admissibility_upper_risk=_mean(
            [row.admissibility_upper for row in source_metrics]
        ),
        penalized_admissibility_known_risk=_mean(
            [row.admissibility_known for row in source_metrics]
        ),
        infeasibility_risk_component=_mean([row.infeasibility_component for row in source_metrics]),
        non_usable_conditional_risk_component=_mean(
            [row.non_usable_conditional_component for row in source_metrics]
        ),
        admissibility_conditional_risk_component=_mean(
            [row.admissibility_conditional_component for row in source_metrics]
        ),
        conditional_non_usable_upper_risk=_optional_source_mean(
            [row.conditional_non_usable_upper for row in source_metrics]
        ),
        conditional_admissibility_upper_risk=_optional_source_mean(
            [row.conditional_admissibility_upper for row in source_metrics]
        ),
        evidence_recall=_mean([row.recall for row in source_metrics]),
        candidates_scored=_mean([row.candidates for row in source_metrics]),
    )


def select_dev_settings(
    runs: Sequence[QueryRun],
    *,
    risk_target: SelectionRisk = SelectionRisk.ADMISSIBILITY_UPPER_BOUND,
) -> Mapping[RetrievalArm, SettingSummary]:
    """Select one setting per arm after checking complete dev-grid coverage."""
    if not runs:
        raise ValueError("development rows are required")
    by_setting: defaultdict[tuple[RetrievalArm, str], list[QueryRun]] = defaultdict(list)
    cases_by_setting: dict[tuple[RetrievalArm, str], set[tuple[str, str]]] = {}
    for run in runs:
        key = (run.arm, run.setting_id)
        by_setting[key].append(run)
        cases_by_setting.setdefault(key, set()).add((run.source, run.query_id))

    reference_cases = next(iter(cases_by_setting.values()))
    if any(cases != reference_cases for cases in cases_by_setting.values()):
        raise ValueError("every development setting must cover the same source/query rows")

    summaries: defaultdict[RetrievalArm, list[SettingSummary]] = defaultdict(list)
    for (arm, _), setting_runs in by_setting.items():
        summaries[arm].append(summarize_setting(setting_runs))
    return {
        arm: min(
            arm_summaries,
            key=lambda summary: summary.selection_key(risk_target),
        )
        for arm, arm_summaries in sorted(summaries.items(), key=lambda row: row[0].value)
    }
