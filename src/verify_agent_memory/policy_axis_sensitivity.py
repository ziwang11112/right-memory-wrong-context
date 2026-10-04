"""Pure helpers for policy-axis admissibility sensitivity analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verify_agent_memory.support_controls import score_matched_recall


@dataclass(frozen=True)
class AxisFamilyScore:
    """Query-level matched-prefix metrics for one admissibility axis family."""

    evidence_recall: float
    feasible: bool
    penalized_upper_risk: float
    known_risk: float | None
    coverage: float | None
    lower_bound: float | None
    upper_bound: float | None
    any_known_violation: float | None
    known_violation_count: float | None
    matched_prefix_count: int | None


def select_common_feasible_records(
    records: Sequence[Mapping[str, object]],
    *,
    arms: Sequence[str],
    identity_fields: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    """Return records whose paired identity is feasible for every requested arm."""
    expected_arms = tuple(arms)
    if not expected_arms or len(set(expected_arms)) != len(expected_arms):
        raise ValueError("arms must be non-empty and unique")
    if not identity_fields or len(set(identity_fields)) != len(identity_fields):
        raise ValueError("identity fields must be non-empty and unique")

    grouped: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = {}
    for row in records:
        try:
            identity = tuple(row[field] for field in identity_fields)
            arm = str(row["arm"])
            feasible = row["feasible"]
        except KeyError as error:
            raise ValueError(f"common-feasible record lacks {error.args[0]!r}") from error
        if arm not in expected_arms:
            continue
        if feasible not in {True, False}:
            raise ValueError("common-feasible records require boolean feasibility")
        paired = grouped.setdefault(identity, {})
        if arm in paired:
            raise ValueError(f"duplicate arm {arm!r} for common-feasible identity")
        paired[arm] = row

    selected: list[Mapping[str, object]] = []
    expected = set(expected_arms)
    for paired in grouped.values():
        if set(paired) != expected:
            raise ValueError("common-feasible identity does not contain every requested arm")
        if all(bool(paired[arm]["feasible"]) for arm in expected_arms):
            selected.extend(paired[arm] for arm in expected_arms)
    return tuple(selected)


def admissibility_status(
    *,
    same_namespace: bool,
    required_anchor: bool,
    memory_state: str,
    query_intent: str,
    policy_disallowed: bool,
    include_policy: bool,
) -> bool | None:
    """Return the frozen admissibility status with or without the policy predicate.

    Source-required evidence remains an anchor after the namespace check. This matches
    the frozen evaluator and avoids redefining source gold when a coarse released field
    conflicts with a required reference.
    """
    if not same_namespace:
        return False
    if required_anchor:
        return True
    if memory_state == "uncertain":
        return None
    if include_policy and policy_disallowed:
        return False
    return not (query_intent == "current_state" and memory_state in {"stale", "superseded"})


def score_axis_family(
    ranked_memory_ids: Sequence[str],
    anchor_ids: set[str] | frozenset[str],
    statuses: Mapping[str, bool | None],
    *,
    target_recall: float,
    infeasibility_cost: float,
) -> AxisFamilyScore:
    """Score known risk, coverage, and sharp bounds at matched recall."""
    score = score_matched_recall(
        ranked_memory_ids,
        anchor_ids,
        statuses,
        statuses,
        target_recall=target_recall,
        infeasibility_cost=infeasibility_cost,
    )
    if not score.feasible:
        return AxisFamilyScore(
            evidence_recall=score.evidence_recall,
            feasible=False,
            penalized_upper_risk=score.penalized_admissibility_upper_risk,
            known_risk=None,
            coverage=None,
            lower_bound=None,
            upper_bound=None,
            any_known_violation=None,
            known_violation_count=None,
            matched_prefix_count=None,
        )

    prefix_statuses = [statuses[memory_id] for memory_id in score.matched_prefix]
    known_negative = sum(value is False for value in prefix_statuses)
    known_positive = sum(value is True for value in prefix_statuses)
    unresolved = sum(value is None for value in prefix_statuses)
    total = len(prefix_statuses)
    known = known_negative + known_positive
    known_risk = known_negative / known if known else None
    upper = (known_negative + unresolved) / total
    if score.admissibility_upper_risk != upper:
        raise RuntimeError("matched-prefix upper-risk accounting drifted")
    return AxisFamilyScore(
        evidence_recall=score.evidence_recall,
        feasible=True,
        penalized_upper_risk=score.penalized_admissibility_upper_risk,
        known_risk=known_risk,
        coverage=known / total,
        lower_bound=known_negative / total,
        upper_bound=upper,
        any_known_violation=float(known_negative > 0),
        known_violation_count=float(known_negative),
        matched_prefix_count=total,
    )
