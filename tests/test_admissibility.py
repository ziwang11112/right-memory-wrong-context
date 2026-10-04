from __future__ import annotations

import pytest

from verify_agent_memory.admissibility import (
    admissible_status,
    lifecycle_compatible,
    usable_status,
)
from verify_agent_memory.schema import (
    LifecycleState,
    MemoryAssessment,
    QueryIntent,
    Relevance,
    Scope,
)


def assessment(
    *,
    relevance: Relevance = Relevance.REQUIRED,
    scope: Scope = Scope.ALLOWED,
    lifecycle_state: LifecycleState = LifecycleState.CURRENT,
    policy_allowed: bool | None = True,
    query_intent: QueryIntent = QueryIntent.CURRENT_STATE,
) -> MemoryAssessment:
    return MemoryAssessment(
        memory_id="memory-1",
        relevance=relevance,
        scope=scope,
        lifecycle_state=lifecycle_state,
        policy_allowed=policy_allowed,
        query_intent=query_intent,
    )


@pytest.mark.parametrize(
    ("state", "intent", "expected"),
    [
        (LifecycleState.CURRENT, QueryIntent.CURRENT_STATE, True),
        (LifecycleState.CURRENT, QueryIntent.HISTORY, True),
        (LifecycleState.STALE, QueryIntent.CURRENT_STATE, False),
        (LifecycleState.SUPERSEDED, QueryIntent.CURRENT_STATE, False),
        (LifecycleState.STALE, QueryIntent.HISTORY, True),
        (LifecycleState.SUPERSEDED, QueryIntent.HISTORY, True),
        (LifecycleState.UNKNOWN, QueryIntent.CURRENT_STATE, None),
        (LifecycleState.CURRENT, QueryIntent.UNKNOWN, None),
    ],
)
def test_lifecycle_compatibility_is_query_conditioned(
    state: LifecycleState,
    intent: QueryIntent,
    expected: bool | None,
) -> None:
    assert lifecycle_compatible(state, intent) is expected


def test_admissibility_does_not_depend_on_relevance() -> None:
    irrelevant = assessment(relevance=Relevance.NOT_USEFUL)
    assert admissible_status(irrelevant) is True
    assert usable_status(irrelevant) is False


def test_definite_violation_dominates_unknown_fields() -> None:
    item = assessment(
        scope=Scope.UNKNOWN,
        lifecycle_state=LifecycleState.UNKNOWN,
        policy_allowed=False,
    )
    assert admissible_status(item) is False
    assert usable_status(item) is False


def test_unknown_is_not_imputed_as_allowed_or_disallowed() -> None:
    item = assessment(scope=Scope.UNKNOWN)
    assert admissible_status(item) is None
    assert usable_status(item) is None


@pytest.mark.parametrize("numeric_label", [0, 1, 0.0, 1.0])
def test_policy_label_rejects_numbers_that_compare_equal_to_booleans(numeric_label: object) -> None:
    with pytest.raises(TypeError, match="policy_allowed"):
        assessment(policy_allowed=numeric_label)  # type: ignore[arg-type]


def test_schema_rejects_empty_ids_and_untyped_enums() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        MemoryAssessment(
            memory_id="",
            relevance=Relevance.REQUIRED,
            scope=Scope.ALLOWED,
            lifecycle_state=LifecycleState.CURRENT,
            policy_allowed=True,
            query_intent=QueryIntent.CURRENT_STATE,
        )
    with pytest.raises(TypeError, match="relevance"):
        MemoryAssessment(
            memory_id="memory-1",
            relevance="required",  # type: ignore[arg-type]
            scope=Scope.ALLOWED,
            lifecycle_state=LifecycleState.CURRENT,
            policy_allowed=True,
            query_intent=QueryIntent.CURRENT_STATE,
        )
