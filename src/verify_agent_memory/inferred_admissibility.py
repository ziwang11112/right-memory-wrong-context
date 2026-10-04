"""Pure contracts and scoring for text-inferred admissibility diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

INTENT_LABELS = ("current_state", "history", "unknown")
POLICY_LABELS = ("allowed", "disallowed", "unknown")
LIFECYCLE_LABELS = ("compatible", "incompatible", "unknown")
ADMISSIBILITY_LABELS = ("admissible", "inadmissible", "unknown")
ROLES = {"calibration", "analysis"}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array")
    return value


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TypeError(f"{label} must be a string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _optional_bool(value: object, label: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{label} must be true, false, or null")
    return value


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{label} must be an integer at least {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProbabilityVector:
    """A validated probability vector with a fixed label order."""

    labels: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.labels or len(self.labels) != len(self.values):
            raise ValueError("probability labels and values must be nonempty and aligned")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("probability labels must be unique")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.values):
            raise ValueError("probabilities must be finite and in [0, 1]")

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        labels: Sequence[str],
        tolerance: float,
        location: str,
    ) -> ProbabilityVector:
        row = _mapping(value, location)
        expected = tuple(labels)
        if set(row) != set(expected):
            raise ValueError(f"{location} labels differ from {expected!r}")
        vector = cls(
            labels=expected,
            values=tuple(_number(row[label], f"{location}.{label}") for label in expected),
        )
        if abs(sum(vector.values) - 1.0) > tolerance:
            raise ValueError(f"{location} probabilities must sum to 1")
        return vector

    def probability(self, label: str) -> float:
        try:
            return self.values[self.labels.index(label)]
        except ValueError as error:
            raise KeyError(label) from error

    def argmax(self) -> str:
        return self.labels[max(range(len(self.values)), key=self.values.__getitem__)]


@dataclass(frozen=True)
class InferenceCandidate:
    """One ranked candidate with visible text and scorer-only released labels."""

    candidate_key: str
    rank: int
    text: str
    visible_order: str | None
    required_evidence: bool
    released_policy_allowed: bool | None
    released_lifecycle_compatible: bool | None

    def __post_init__(self) -> None:
        if not self.candidate_key or not self.text:
            raise ValueError("candidate key and text must be nonempty")
        if self.rank < 1:
            raise ValueError("candidate rank must be positive")
        if not isinstance(self.required_evidence, bool):
            raise TypeError("required_evidence must be boolean")
        for value in (
            self.released_policy_allowed,
            self.released_lifecycle_compatible,
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError("released candidate labels must be true, false, or unknown")

    @property
    def released_admissible(self) -> bool | None:
        if self.released_policy_allowed is False or self.released_lifecycle_compatible is False:
            return False
        if self.released_policy_allowed is True and self.released_lifecycle_compatible is True:
            return True
        return None


@dataclass(frozen=True)
class InferenceCase:
    """One private dev case; prompt construction exposes only named visible fields."""

    case_id: str
    source: str
    group_id: str
    role: str
    query_text: str
    query_visible_time: str | None
    released_query_intent: str
    anchor_total: int
    candidates: tuple[InferenceCandidate, ...]

    def __post_init__(self) -> None:
        if not all((self.case_id, self.source, self.group_id, self.query_text)):
            raise ValueError("case identity and query text must be nonempty")
        if self.role not in ROLES:
            raise ValueError(f"unexpected case role {self.role!r}")
        if self.released_query_intent not in INTENT_LABELS[:2]:
            raise ValueError("released query intent must be current_state or history")
        if self.anchor_total < 1 or not self.candidates:
            raise ValueError("cases require anchors and ranked candidates")
        keys = [candidate.candidate_key for candidate in self.candidates]
        ranks = [candidate.rank for candidate in self.candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate keys must be unique within a case")
        if ranks != list(range(1, len(self.candidates) + 1)):
            raise ValueError("candidate ranks must be contiguous and ordered")


def case_from_mapping(value: object) -> InferenceCase:
    """Parse a private inference case without changing scorer labels."""
    row = _mapping(value, "case")
    query = _mapping(row.get("query"), "case.query")
    candidates = []
    for index, raw_candidate in enumerate(_sequence(row.get("candidates"), "case.candidates")):
        candidate = _mapping(raw_candidate, f"case.candidates[{index}]")
        candidates.append(
            InferenceCandidate(
                candidate_key=_string(
                    candidate.get("candidate_key"),
                    f"case.candidates[{index}].candidate_key",
                ),
                rank=_integer(
                    candidate.get("rank"),
                    f"case.candidates[{index}].rank",
                    minimum=1,
                ),
                text=_string(candidate.get("text"), f"case.candidates[{index}].text"),
                visible_order=_optional_string(
                    candidate.get("visible_order"),
                    f"case.candidates[{index}].visible_order",
                ),
                required_evidence=_bool(
                    candidate.get("required_evidence"),
                    f"case.candidates[{index}].required_evidence",
                ),
                released_policy_allowed=_optional_bool(
                    candidate.get("released_policy_allowed"),
                    f"case.candidates[{index}].released_policy_allowed",
                ),
                released_lifecycle_compatible=_optional_bool(
                    candidate.get("released_lifecycle_compatible"),
                    f"case.candidates[{index}].released_lifecycle_compatible",
                ),
            )
        )
    return InferenceCase(
        case_id=_string(row.get("case_id"), "case.case_id"),
        source=_string(row.get("source"), "case.source"),
        group_id=_string(row.get("group_id"), "case.group_id"),
        role=_string(row.get("role"), "case.role"),
        query_text=_string(query.get("text"), "case.query.text"),
        query_visible_time=_optional_string(
            query.get("visible_time"),
            "case.query.visible_time",
        ),
        released_query_intent=_string(
            query.get("released_query_intent"),
            "case.query.released_query_intent",
        ),
        anchor_total=_integer(query.get("anchor_total"), "case.query.anchor_total", minimum=1),
        candidates=tuple(candidates),
    )


def prompt_payload(case: InferenceCase) -> str:
    """Serialize only model-visible fields into a deterministic user payload."""
    visible = {
        "query": {
            "text": case.query_text,
            "visible_time": case.query_visible_time,
        },
        "candidates": [
            {
                "candidate_key": candidate.candidate_key,
                "rank": candidate.rank,
                "text": candidate.text,
                "visible_order": candidate.visible_order,
            }
            for candidate in case.candidates
        ],
    }
    return json.dumps(visible, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def response_json_schema(candidate_count: int) -> dict[str, object]:
    """Return the provider-neutral JSON Schema for one model response."""
    _integer(candidate_count, "candidate_count", minimum=1)

    def probabilities(labels: Sequence[str]) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                label: {"type": "number", "minimum": 0, "maximum": 1} for label in labels
            },
            "required": list(labels),
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query_intent": probabilities(INTENT_LABELS),
            "candidates": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_key": {"type": "string"},
                        "policy": probabilities(POLICY_LABELS),
                        "lifecycle": probabilities(LIFECYCLE_LABELS),
                        "admissibility": probabilities(ADMISSIBILITY_LABELS),
                    },
                    "required": ["candidate_key", "policy", "lifecycle", "admissibility"],
                },
            },
        },
        "required": ["query_intent", "candidates"],
    }


@dataclass(frozen=True)
class CandidatePrediction:
    candidate_key: str
    policy: ProbabilityVector
    lifecycle: ProbabilityVector
    admissibility: ProbabilityVector

    @property
    def violation_probability(self) -> float:
        return self.admissibility.probability("inadmissible")

    @property
    def unknown_probability(self) -> float:
        return self.admissibility.probability("unknown")


@dataclass(frozen=True)
class CasePrediction:
    case_id: str
    query_intent: ProbabilityVector
    candidates: tuple[CandidatePrediction, ...]


def prediction_from_mapping(
    value: object,
    case: InferenceCase,
    *,
    tolerance: float = 0.0001,
) -> CasePrediction:
    """Parse one strict model response and bind it to the requested candidate order."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("probability tolerance must be finite and nonnegative")
    row = _mapping(value, "prediction")
    if set(row) != {"query_intent", "candidates"}:
        raise ValueError("prediction has missing or unknown fields")
    raw_candidates = _sequence(row["candidates"], "prediction.candidates")
    if len(raw_candidates) != len(case.candidates):
        raise ValueError("prediction candidate count differs from request")
    candidates = []
    for index, (raw_prediction, expected) in enumerate(
        zip(raw_candidates, case.candidates, strict=True)
    ):
        candidate = _mapping(raw_prediction, f"prediction.candidates[{index}]")
        if set(candidate) != {"candidate_key", "policy", "lifecycle", "admissibility"}:
            raise ValueError(f"prediction.candidates[{index}] has missing or unknown fields")
        key = _string(
            candidate["candidate_key"],
            f"prediction.candidates[{index}].candidate_key",
        )
        if key != expected.candidate_key:
            raise ValueError("prediction candidate keys must reproduce request order")
        candidates.append(
            CandidatePrediction(
                candidate_key=key,
                policy=ProbabilityVector.from_mapping(
                    candidate["policy"],
                    labels=POLICY_LABELS,
                    tolerance=tolerance,
                    location=f"prediction.candidates[{index}].policy",
                ),
                lifecycle=ProbabilityVector.from_mapping(
                    candidate["lifecycle"],
                    labels=LIFECYCLE_LABELS,
                    tolerance=tolerance,
                    location=f"prediction.candidates[{index}].lifecycle",
                ),
                admissibility=ProbabilityVector.from_mapping(
                    candidate["admissibility"],
                    labels=ADMISSIBILITY_LABELS,
                    tolerance=tolerance,
                    location=f"prediction.candidates[{index}].admissibility",
                ),
            )
        )
    return CasePrediction(
        case_id=case.case_id,
        query_intent=ProbabilityVector.from_mapping(
            row["query_intent"],
            labels=INTENT_LABELS,
            tolerance=tolerance,
            location="prediction.query_intent",
        ),
        candidates=tuple(candidates),
    )


@dataclass(frozen=True)
class FilterSetting:
    violation_threshold: float | None
    unknown_threshold: float | None
    known_violation_precision: float | None
    known_violation_recall: float
    required_anchor_false_deny_rate: float
    dropped_unknown_gold: int

    @property
    def retains_all(self) -> bool:
        return self.violation_threshold is None


def _drop_candidate(
    prediction: CandidatePrediction,
    *,
    violation_threshold: float | None,
    unknown_threshold: float | None,
) -> bool:
    if violation_threshold is None:
        return False
    if prediction.violation_probability < violation_threshold:
        return False
    return unknown_threshold is None or prediction.unknown_probability <= unknown_threshold


def select_filter_setting(
    cases: Sequence[InferenceCase],
    predictions: Mapping[str, CasePrediction],
    *,
    violation_thresholds: Sequence[float],
    unknown_thresholds: Sequence[float] | None,
    maximum_required_anchor_false_deny_rate: float,
) -> FilterSetting:
    """Select a calibration-only filter under a required-anchor preservation constraint."""
    if not cases or any(case.role != "calibration" for case in cases):
        raise ValueError("selection requires nonempty calibration-only cases")
    if not 0 <= maximum_required_anchor_false_deny_rate <= 1:
        raise ValueError("maximum false-deny rate must be in [0, 1]")
    thresholds = tuple(float(value) for value in violation_thresholds)
    unknowns = (
        (None,) if unknown_thresholds is None else tuple(float(v) for v in unknown_thresholds)
    )
    if not thresholds or not unknowns:
        raise ValueError("selection grids must be nonempty")
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("violation thresholds must be in [0, 1]")
    if any(value is not None and not 0 <= value <= 1 for value in unknowns):
        raise ValueError("unknown thresholds must be in [0, 1]")

    required_total = sum(
        candidate.required_evidence for case in cases for candidate in case.candidates
    )
    known_violations = sum(
        candidate.released_admissible is False for case in cases for candidate in case.candidates
    )
    if required_total == 0 or known_violations == 0:
        raise ValueError("calibration cases require anchors and known violations")

    feasible: list[FilterSetting] = []
    for threshold in thresholds:
        for unknown_threshold in unknowns:
            dropped_known = 0
            dropped_violations = 0
            dropped_required = 0
            dropped_unknown = 0
            for case in cases:
                prediction = predictions.get(case.case_id)
                if prediction is None:
                    raise ValueError(f"missing prediction for {case.case_id!r}")
                if len(prediction.candidates) != len(case.candidates):
                    raise ValueError("prediction candidates differ from calibration case")
                for candidate, candidate_prediction in zip(
                    case.candidates,
                    prediction.candidates,
                    strict=True,
                ):
                    if not _drop_candidate(
                        candidate_prediction,
                        violation_threshold=threshold,
                        unknown_threshold=unknown_threshold,
                    ):
                        continue
                    dropped_required += int(candidate.required_evidence)
                    if candidate.released_admissible is None:
                        dropped_unknown += 1
                    else:
                        dropped_known += 1
                        dropped_violations += int(candidate.released_admissible is False)
            false_deny = dropped_required / required_total
            if false_deny > maximum_required_anchor_false_deny_rate:
                continue
            feasible.append(
                FilterSetting(
                    violation_threshold=threshold,
                    unknown_threshold=unknown_threshold,
                    known_violation_precision=(
                        dropped_violations / dropped_known if dropped_known else None
                    ),
                    known_violation_recall=dropped_violations / known_violations,
                    required_anchor_false_deny_rate=false_deny,
                    dropped_unknown_gold=dropped_unknown,
                )
            )
    if not feasible:
        return FilterSetting(
            violation_threshold=None,
            unknown_threshold=None,
            known_violation_precision=None,
            known_violation_recall=0.0,
            required_anchor_false_deny_rate=0.0,
            dropped_unknown_gold=0,
        )

    def objective(setting: FilterSetting) -> tuple[float, float, float, float, float]:
        precision = setting.known_violation_precision
        return (
            setting.known_violation_recall,
            -1.0 if precision is None else precision,
            -setting.required_anchor_false_deny_rate,
            float(setting.violation_threshold),
            -(1.0 if setting.unknown_threshold is None else setting.unknown_threshold),
        )

    return max(feasible, key=objective)


def filter_decision_metrics(
    cases: Sequence[InferenceCase],
    predictions: Mapping[str, CasePrediction],
    setting: FilterSetting,
) -> dict[str, float | int | None]:
    """Evaluate a selected verifier threshold without retuning it."""
    if not cases:
        raise ValueError("filter metrics require cases")
    candidate_count = 0
    known_count = 0
    known_violations = 0
    dropped_count = 0
    dropped_known = 0
    dropped_violations = 0
    dropped_unknown_gold = 0
    required_count = 0
    required_false_denies = 0
    covered_count = 0
    for case in cases:
        prediction = predictions.get(case.case_id)
        if prediction is None:
            raise ValueError(f"missing prediction for {case.case_id!r}")
        if len(prediction.candidates) != len(case.candidates):
            raise ValueError("prediction candidates differ from evaluation case")
        for candidate, candidate_prediction in zip(
            case.candidates,
            prediction.candidates,
            strict=True,
        ):
            candidate_count += 1
            required_count += int(candidate.required_evidence)
            known = candidate.released_admissible is not None
            violation = candidate.released_admissible is False
            known_count += int(known)
            known_violations += int(violation)
            covered = (
                setting.unknown_threshold is None
                or candidate_prediction.unknown_probability <= setting.unknown_threshold
            )
            covered_count += int(covered)
            dropped = _drop_candidate(
                candidate_prediction,
                violation_threshold=setting.violation_threshold,
                unknown_threshold=setting.unknown_threshold,
            )
            if not dropped:
                continue
            dropped_count += 1
            required_false_denies += int(candidate.required_evidence)
            if known:
                dropped_known += 1
                dropped_violations += int(violation)
            else:
                dropped_unknown_gold += 1
    return {
        "candidate_count": candidate_count,
        "known_count": known_count,
        "known_violation_count": known_violations,
        "dropped_count": dropped_count,
        "dropped_known_count": dropped_known,
        "dropped_unknown_gold_count": dropped_unknown_gold,
        "violation_precision": (dropped_violations / dropped_known if dropped_known else None),
        "violation_recall": (dropped_violations / known_violations if known_violations else None),
        "required_anchor_count": required_count,
        "required_anchor_false_denies": required_false_denies,
        "required_anchor_false_deny_rate": (
            required_false_denies / required_count if required_count else None
        ),
        "abstention_coverage": covered_count / candidate_count,
    }


@dataclass(frozen=True)
class FilteredRouteScore:
    evidence_recall: float
    feasible: bool
    matched_prefix_count: int | None
    admissibility_violation_known_rate: float | None
    admissibility_upper_risk: float | None
    penalized_admissibility_upper_risk: float
    route_width: int


def score_filtered_route(
    case: InferenceCase,
    retained_candidate_keys: Sequence[str],
    *,
    target_recall: float,
) -> FilteredRouteScore:
    """Score a filtered fixed ranking against released scorer-only labels."""
    if not 0 <= target_recall <= 1:
        raise ValueError("target recall must be in [0, 1]")
    by_key = {candidate.candidate_key: candidate for candidate in case.candidates}
    keys = tuple(retained_candidate_keys)
    if len(set(keys)) != len(keys) or any(key not in by_key for key in keys):
        raise ValueError("retained candidate keys must be unique members of the case")
    original_positions = [by_key[key].rank for key in keys]
    if original_positions != sorted(original_positions):
        raise ValueError("filtering must preserve the original ranking")
    anchor_retrieved = sum(by_key[key].required_evidence for key in keys)
    recall = anchor_retrieved / case.anchor_total
    feasible = recall >= target_recall
    prefix: tuple[str, ...] | None = None
    if feasible:
        if target_recall == 0:
            prefix = ()
        else:
            hits = 0
            for index, key in enumerate(keys, start=1):
                hits += int(by_key[key].required_evidence)
                if hits / case.anchor_total >= target_recall:
                    prefix = keys[:index]
                    break
        if prefix is None:
            raise RuntimeError("feasible filtered route has no matched-recall prefix")

    if prefix is None:
        known_rate = None
        upper = None
    elif not prefix:
        known_rate = 0.0
        upper = 0.0
    else:
        statuses = [by_key[key].released_admissible for key in prefix]
        known = sum(status is not None for status in statuses)
        violations = sum(status is False for status in statuses)
        unresolved = len(statuses) - known
        known_rate = violations / known if known else None
        upper = (violations + unresolved) / len(statuses)
    return FilteredRouteScore(
        evidence_recall=recall,
        feasible=feasible,
        matched_prefix_count=None if prefix is None else len(prefix),
        admissibility_violation_known_rate=known_rate,
        admissibility_upper_risk=upper,
        penalized_admissibility_upper_risk=1.0 if upper is None else upper,
        route_width=len(keys),
    )


def route_keys(
    case: InferenceCase,
    *,
    arm: str,
    prediction: CasePrediction | None = None,
    setting: FilterSetting | None = None,
) -> tuple[str, ...]:
    """Apply one frozen route arm while preserving candidate order."""
    if arm == "namespace_dense":
        return tuple(candidate.candidate_key for candidate in case.candidates)
    if arm == "released_oracle":
        return tuple(
            candidate.candidate_key
            for candidate in case.candidates
            if candidate.released_admissible is not False
        )
    if arm not in {"text_inferred", "abstaining_verifier"}:
        raise ValueError(f"unknown inferred-admissibility arm {arm!r}")
    if prediction is None or setting is None:
        raise ValueError("model-derived arms require a prediction and selected setting")
    if prediction.case_id != case.case_id or len(prediction.candidates) != len(case.candidates):
        raise ValueError("prediction differs from route case")
    return tuple(
        candidate.candidate_key
        for candidate, candidate_prediction in zip(
            case.candidates,
            prediction.candidates,
            strict=True,
        )
        if not _drop_candidate(
            candidate_prediction,
            violation_threshold=setting.violation_threshold,
            unknown_threshold=(setting.unknown_threshold if arm == "abstaining_verifier" else None),
        )
    )


def _mean(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


def aggregate_route_scores(scores: Sequence[FilteredRouteScore]) -> dict[str, float | int | None]:
    """Query-macro aggregate for one diagnostic arm and stratum."""
    if not scores:
        raise ValueError("at least one filtered route score is required")
    feasible_scores = [score for score in scores if score.admissibility_upper_risk is not None]
    return {
        "query_count": len(scores),
        "evidence_recall": float(sum(score.evidence_recall for score in scores) / len(scores)),
        "feasible_rate": float(sum(score.feasible for score in scores) / len(scores)),
        "penalized_admissibility_upper_risk": float(
            sum(score.penalized_admissibility_upper_risk for score in scores) / len(scores)
        ),
        "conditional_admissibility_upper_risk": _mean(
            score.admissibility_upper_risk for score in feasible_scores
        ),
        "known_admissibility_violation_rate": _mean(
            score.admissibility_violation_known_rate for score in feasible_scores
        ),
        "mean_route_width": float(sum(score.route_width for score in scores) / len(scores)),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("quantile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stratified_group_paired_bootstrap(
    cases: Sequence[InferenceCase],
    treatment: Mapping[str, FilteredRouteScore],
    comparator: Mapping[str, FilteredRouteScore],
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Bootstrap an equal-stratum paired delta by namespace group."""
    allowed_metrics = {
        "evidence_recall",
        "feasible",
        "penalized_admissibility_upper_risk",
        "route_width",
    }
    if metric not in allowed_metrics:
        raise ValueError(f"unsupported paired bootstrap metric {metric!r}")
    if not cases or replicates < 1:
        raise ValueError("paired bootstrap requires cases and positive replicates")
    case_ids = {case.case_id for case in cases}
    if set(treatment) != case_ids or set(comparator) != case_ids:
        raise ValueError("paired bootstrap scores must cover the cases exactly")

    by_stratum_group: dict[str, dict[str, list[InferenceCase]]] = {}
    for case in cases:
        stratum = f"{case.source}/{case.released_query_intent}"
        by_stratum_group.setdefault(stratum, {}).setdefault(case.group_id, []).append(case)

    def delta(case: InferenceCase) -> float:
        left = getattr(treatment[case.case_id], metric)
        right = getattr(comparator[case.case_id], metric)
        return float(left) - float(right)

    def equal_stratum_delta(sampled: Mapping[str, Sequence[InferenceCase]]) -> float:
        stratum_means = []
        for stratum in sorted(sampled):
            rows = sampled[stratum]
            if not rows:
                raise RuntimeError("bootstrap stratum unexpectedly empty")
            stratum_means.append(sum(delta(case) for case in rows) / len(rows))
        return sum(stratum_means) / len(stratum_means)

    observed = {
        stratum: [case for rows in groups.values() for case in rows]
        for stratum, groups in by_stratum_group.items()
    }
    estimate = equal_stratum_delta(observed)
    generator = random.Random(seed)
    draws = []
    for _index in range(replicates):
        sampled: dict[str, list[InferenceCase]] = {}
        for stratum, groups in sorted(by_stratum_group.items()):
            group_rows = sorted(groups.items())
            sampled[stratum] = []
            for _group_index in range(len(group_rows)):
                _group_id, rows = group_rows[generator.randrange(len(group_rows))]
                sampled[stratum].extend(rows)
        draws.append(equal_stratum_delta(sampled))
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_lower": _quantile(draws, 0.025),
        "ci_upper": _quantile(draws, 0.975),
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "namespace_group",
        "bootstrap_stratification": "source_query_intent",
        "analysis_queries": len(cases),
        "stratum_group_count": sum(len(groups) for groups in by_stratum_group.values()),
    }


def _binary_rows(
    cases: Sequence[InferenceCase],
    predictions: Mapping[str, CasePrediction],
    axis: str,
) -> tuple[tuple[int, float], ...]:
    rows = []
    for case in cases:
        prediction = predictions.get(case.case_id)
        if prediction is None:
            raise ValueError(f"missing prediction for {case.case_id!r}")
        for candidate, candidate_prediction in zip(
            case.candidates,
            prediction.candidates,
            strict=True,
        ):
            if axis == "policy":
                gold = candidate.released_policy_allowed
                score = candidate_prediction.policy.probability("disallowed")
            elif axis == "lifecycle":
                gold = candidate.released_lifecycle_compatible
                score = candidate_prediction.lifecycle.probability("incompatible")
            elif axis == "admissibility":
                gold = candidate.released_admissible
                score = candidate_prediction.violation_probability
            else:
                raise ValueError(f"unknown binary axis {axis!r}")
            if gold is not None:
                rows.append((int(gold is False), score))
    return tuple(rows)


def _roc_auc(rows: Sequence[tuple[int, float]]) -> float | None:
    positives = sum(label for label, _score in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(rows, key=lambda row: row[1])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        rank_sum += average_rank * sum(label for label, _score in ordered[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(rows: Sequence[tuple[int, float]]) -> float | None:
    positives = sum(label for label, _score in rows)
    if not positives:
        return None
    ordered = sorted(rows, key=lambda row: row[1], reverse=True)
    true_positive = 0
    seen = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        true_positive += sum(label for label, _score in ordered[start:end])
        seen += end - start
        recall = true_positive / positives
        precision = true_positive / seen
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return area


def _ece(rows: Sequence[tuple[int, float]], bins: int = 10) -> float:
    total = len(rows)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            (label, score)
            for label, score in rows
            if lower <= score <= upper and (index == bins - 1 or score < upper)
        ]
        if selected:
            accuracy = sum(label for label, _score in selected) / len(selected)
            confidence = sum(score for _label, score in selected) / len(selected)
            error += len(selected) / total * abs(accuracy - confidence)
    return error


def binary_classification_metrics(
    cases: Sequence[InferenceCase],
    predictions: Mapping[str, CasePrediction],
    *,
    axis: str,
) -> dict[str, float | int | None]:
    """Return deterministic binary discrimination and calibration metrics."""
    rows = _binary_rows(cases, predictions, axis)
    if not rows:
        raise ValueError(f"axis {axis!r} has no known labels")
    positives = sum(label for label, _score in rows)
    predicted = [int(score >= 0.5) for _label, score in rows]
    correct = sum(
        prediction == label for prediction, (label, _score) in zip(predicted, rows, strict=True)
    )
    positive_rows = [index for index, (label, _score) in enumerate(rows) if label]
    negative_rows = [index for index, (label, _score) in enumerate(rows) if not label]
    true_positive_rate = (
        sum(predicted[index] for index in positive_rows) / len(positive_rows)
        if positive_rows
        else None
    )
    true_negative_rate = (
        sum(1 - predicted[index] for index in negative_rows) / len(negative_rows)
        if negative_rows
        else None
    )
    balanced = (
        (true_positive_rate + true_negative_rate) / 2
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    return {
        "known_count": len(rows),
        "positive_count": positives,
        "positive_rate": positives / len(rows),
        "accuracy": correct / len(rows),
        "balanced_accuracy": balanced,
        "brier": sum((score - label) ** 2 for label, score in rows) / len(rows),
        "ece_10": _ece(rows),
        "roc_auc": _roc_auc(rows),
        "pr_auc": _average_precision(rows),
    }


def intent_classification_metrics(
    cases: Sequence[InferenceCase],
    predictions: Mapping[str, CasePrediction],
) -> dict[str, float | int]:
    """Return multiclass query-intent accuracy, Brier score, and confidence ECE."""
    if not cases:
        raise ValueError("intent metrics require cases")
    correct = 0
    brier = 0.0
    confidence_rows: list[tuple[int, float]] = []
    for case in cases:
        prediction = predictions.get(case.case_id)
        if prediction is None:
            raise ValueError(f"missing prediction for {case.case_id!r}")
        predicted = prediction.query_intent.argmax()
        correct += int(predicted == case.released_query_intent)
        confidence_rows.append(
            (int(predicted == case.released_query_intent), max(prediction.query_intent.values))
        )
        for label, probability in zip(
            prediction.query_intent.labels,
            prediction.query_intent.values,
            strict=True,
        ):
            brier += (probability - int(label == case.released_query_intent)) ** 2
    return {
        "known_count": len(cases),
        "accuracy": correct / len(cases),
        "multiclass_brier": brier / len(cases),
        "confidence_ece_10": _ece(confidence_rows),
    }


def load_cases(path: Path) -> tuple[InferenceCase, ...]:
    """Load a private JSONL case bundle with duplicate-ID rejection."""
    cases = tuple(
        case_from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case bundle must be nonempty with unique case IDs")
    return cases


def validate_sample_contract(
    cases: Sequence[InferenceCase],
    *,
    strata: Sequence[str],
    queries_per_stratum: int,
    calibration_per_stratum: int,
    candidate_depth: int,
    max_visible_utf8_bytes: int | None = None,
) -> dict[str, object]:
    """Validate the frozen balanced sample without exposing raw content."""
    expected_strata = tuple(strata)
    counts: dict[tuple[str, str], int] = {}
    for case in cases:
        stratum = f"{case.source}/{case.released_query_intent}"
        if stratum not in expected_strata:
            raise ValueError(f"case has unexpected stratum {stratum!r}")
        if len(case.candidates) != candidate_depth:
            raise ValueError("case candidate depth differs from protocol")
        if max_visible_utf8_bytes is not None:
            if max_visible_utf8_bytes < 1:
                raise ValueError("visible text byte limit must be positive")
            visible_texts = (case.query_text, *(candidate.text for candidate in case.candidates))
            if any(len(text.encode("utf-8")) > max_visible_utf8_bytes for text in visible_texts):
                raise ValueError("case visible text exceeds the protocol byte limit")
        key = stratum, case.role
        counts[key] = counts.get(key, 0) + 1
    for stratum in expected_strata:
        calibration = counts.get((stratum, "calibration"), 0)
        analysis = counts.get((stratum, "analysis"), 0)
        if calibration != calibration_per_stratum:
            raise ValueError(f"{stratum} calibration count drifted")
        if calibration + analysis != queries_per_stratum:
            raise ValueError(f"{stratum} total count drifted")
    return {
        "case_count": len(cases),
        "candidate_count": sum(len(case.candidates) for case in cases),
        "strata": {
            stratum: {
                "calibration": counts.get((stratum, "calibration"), 0),
                "analysis": counts.get((stratum, "analysis"), 0),
            }
            for stratum in expected_strata
        },
    }
