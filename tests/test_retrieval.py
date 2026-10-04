from __future__ import annotations

import pytest

from verify_agent_memory.retrieval import (
    MemoryRecord,
    PolicyDecision,
    PolicyPurpose,
    QueryRecord,
    RetrievalArm,
    RetrievalConfig,
    route,
)
from verify_agent_memory.schema import LifecycleState, QueryIntent


def memory(
    memory_id: str,
    namespace: str,
    embedding: tuple[float, ...],
    *,
    text: str = "",
    order: float = 0,
    state: LifecycleState = LifecycleState.CURRENT,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        namespace=namespace,
        text=text,
        embedding=embedding,
        released_order=order,
        lifecycle_state=state,
    )


def query(
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
    text: str = "target",
    intent: QueryIntent = QueryIntent.CURRENT_STATE,
    policy_purpose: PolicyPurpose = PolicyPurpose.CONTENT_DISCLOSURE,
) -> QueryRecord:
    return QueryRecord(
        query_id="q",
        namespace="a",
        text=text,
        embedding=embedding,
        intent=intent,
        policy_purpose=policy_purpose,
    )


def config(arm: RetrievalArm, **kwargs: object) -> RetrievalConfig:
    return RetrievalConfig(setting_id=f"setting-{arm.value}", arm=arm, **kwargs)


@pytest.mark.parametrize("numeric_label", [0, 1, 0.0, 1.0])
@pytest.mark.parametrize("field", ["content_disclosure_allowed", "operation_trace_allowed"])
def test_policy_decisions_reject_numeric_labels(field: str, numeric_label: object) -> None:
    # Numeric zero must not pass validation then bypass the identity-based deny gate.
    with pytest.raises(TypeError, match=field):
        PolicyDecision("memory", **{field: numeric_label})  # type: ignore[arg-type]


def test_namespace_support_excludes_more_similar_wrong_namespace() -> None:
    memories = (
        memory("a-memory", "a", (0.8, 0.6)),
        memory("b-memory", "b", (1.0, 0.0)),
    )
    global_result = route(memories, query(), config(RetrievalArm.GLOBAL_DENSE))
    namespace_result = route(memories, query(), config(RetrievalArm.NAMESPACE_DENSE))

    assert global_result.ranked_memory_ids[0] == "b-memory"
    assert namespace_result.ranked_memory_ids == ("a-memory",)
    assert namespace_result.candidates_scored == 1


def test_historical_and_v2_released_governance_semantics_are_distinct() -> None:
    memories = (
        memory("current", "a", (1.0, 0.0)),
        memory("stale", "a", (0.9, 0.43589), state=LifecycleState.STALE),
        memory("blocked", "a", (0.8, 0.6), state=LifecycleState.CURRENT),
    )
    policies = (
        PolicyDecision("current", True, True),
        PolicyDecision("stale", True, True),
        PolicyDecision("blocked", False, True),
    )
    current = route(
        memories,
        query(intent=QueryIntent.CURRENT_STATE),
        config(RetrievalArm.RELEASED_INTENT_LIFECYCLE_UPPER_BOUND),
        policy_decisions=policies,
    )
    history = route(
        memories,
        query(intent=QueryIntent.HISTORY),
        config(RetrievalArm.RELEASED_INTENT_LIFECYCLE_UPPER_BOUND),
        policy_decisions=policies,
    )
    governance_v2 = route(
        memories,
        query(intent=QueryIntent.HISTORY),
        config(RetrievalArm.RELEASED_GOVERNANCE_ORACLE_V2),
        policy_decisions=policies,
    )
    operation_trace = route(
        memories,
        query(
            intent=QueryIntent.HISTORY,
            policy_purpose=PolicyPurpose.OPERATION_TRACE,
        ),
        config(RetrievalArm.RELEASED_INTENT_LIFECYCLE_UPPER_BOUND),
        policy_decisions=policies,
    )
    policy_only = route(
        memories,
        query(intent=QueryIntent.CURRENT_STATE),
        config(RetrievalArm.NAMESPACE_POLICY_ONLY),
        policy_decisions=policies,
    )
    lifecycle_only = route(
        memories,
        query(intent=QueryIntent.CURRENT_STATE),
        config(RetrievalArm.NAMESPACE_LIFECYCLE_ONLY),
        policy_decisions=policies,
    )
    current_only = route(
        memories,
        query(intent=QueryIntent.HISTORY),
        config(RetrievalArm.QUERY_AGNOSTIC_CURRENT_ONLY),
    )

    assert current.ranked_memory_ids == ("current",)
    assert set(history.ranked_memory_ids) == {"current", "stale", "blocked"}
    assert set(governance_v2.ranked_memory_ids) == {"current", "stale"}
    assert set(operation_trace.ranked_memory_ids) == {"current", "stale", "blocked"}
    assert set(policy_only.ranked_memory_ids) == {"current", "stale"}
    assert set(lifecycle_only.ranked_memory_ids) == {"current", "blocked"}
    assert set(current_only.ranked_memory_ids) == {"current", "blocked"}


def test_threshold_fallback_stays_namespace_local() -> None:
    memories = (
        memory("a-one", "a", (1.0, 0.0)),
        memory("a-two", "a", (0.99, 0.141067), order=1),
        memory("b-one", "b", (0.0, 1.0)),
    )
    result = route(
        memories,
        query(embedding=(0.0, 1.0)),
        config(RetrievalArm.THRESHOLD_ROUTER, theta=0.9, top_l=3),
    )

    assert result.fallback_used is True
    assert set(result.ranked_memory_ids) == {"a-one", "a-two"}
    assert "b-one" not in result.ranked_memory_ids


def test_cluster_router_is_deterministic_and_namespace_local() -> None:
    memories = (
        memory("a-one", "a", (1.0, 0.0), order=0),
        memory("a-two", "a", (0.99, 0.141067), order=1),
        memory("a-three", "a", (0.0, 1.0), order=2),
        memory("b-one", "b", (1.0, 0.0), order=0),
    )
    arm = config(
        RetrievalArm.CLUSTER_ROUTER,
        top_l=1,
        alpha_like=1.0,
        cluster_beta=1.0,
        cluster_gamma=0.5,
        cluster_tau=0.1,
        cluster_tau_new=0.4,
    )

    first = route(memories, query(), arm)
    second = route(tuple(reversed(memories)), query(), arm)

    assert first == second
    assert first.route_width == 1
    assert "b-one" not in first.ranked_memory_ids


def test_bm25_and_rrf_use_stable_rankings() -> None:
    memories = (
        memory("semantic", "a", (1.0, 0.0), text="unrelated words"),
        memory("lexical", "a", (0.0, 1.0), text="target target evidence"),
    )
    bm25 = route(
        memories,
        query(text="target evidence"),
        config(RetrievalArm.GLOBAL_BM25),
    )
    rrf = route(
        memories,
        query(text="target evidence"),
        config(RetrievalArm.GLOBAL_BM25_DENSE_RRF),
    )

    assert bm25.ranked_memory_ids[0] == "lexical"
    assert set(rrf.ranked_memory_ids) == {"semantic", "lexical"}


def test_recency_adjustment_can_change_dense_order() -> None:
    memories = (
        memory("older", "a", (1.0, 0.0), order=0),
        memory("newer", "a", (0.995, 0.099875), order=10),
    )
    dense = route(memories, query(), config(RetrievalArm.GLOBAL_DENSE))
    recency = route(
        memories,
        query(),
        config(RetrievalArm.GLOBAL_RECENCY_DENSE, recency_gamma=0.1),
    )

    assert dense.ranked_memory_ids[0] == "older"
    assert recency.ranked_memory_ids[0] == "newer"
