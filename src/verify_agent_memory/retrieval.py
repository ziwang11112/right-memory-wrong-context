"""Deterministic retrieval arms for admissibility experiments.

The module is deliberately dataset-independent. Callers provide already constructed
memory/query records and frozen embeddings; no benchmark payload, model, or provider
is accessed here.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from verify_agent_memory.schema import LifecycleState, QueryIntent

Vector = tuple[float, ...]
_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


class RetrievalArm(StrEnum):
    GLOBAL_BM25 = "global_bm25"
    GLOBAL_DENSE = "global_dense"
    GLOBAL_BM25_DENSE_RRF = "global_bm25_dense_rrf"
    GLOBAL_RECENCY_DENSE = "global_recency_dense"
    NAMESPACE_DENSE = "namespace_dense"
    QUERY_AGNOSTIC_CURRENT_ONLY = "query_agnostic_current_only"
    NAMESPACE_POLICY_ONLY = "namespace_policy_only"
    NAMESPACE_LIFECYCLE_ONLY = "namespace_lifecycle_only"
    RELEASED_INTENT_LIFECYCLE_UPPER_BOUND = "released_intent_lifecycle_upper_bound"
    RELEASED_GOVERNANCE_ORACLE_V2 = "released_governance_oracle_v2"
    THRESHOLD_ROUTER = "threshold_router"
    CLUSTER_ROUTER = "cluster_router"


FROZEN_PUBLIC_ARMS = frozenset(
    {
        RetrievalArm.GLOBAL_BM25,
        RetrievalArm.GLOBAL_DENSE,
        RetrievalArm.GLOBAL_BM25_DENSE_RRF,
        RetrievalArm.GLOBAL_RECENCY_DENSE,
        RetrievalArm.NAMESPACE_DENSE,
        RetrievalArm.QUERY_AGNOSTIC_CURRENT_ONLY,
        RetrievalArm.RELEASED_INTENT_LIFECYCLE_UPPER_BOUND,
        RetrievalArm.THRESHOLD_ROUTER,
        RetrievalArm.CLUSTER_ROUTER,
    }
)

ATTRIBUTION_DIAGNOSTIC_ARMS = frozenset(
    {
        RetrievalArm.NAMESPACE_POLICY_ONLY,
        RetrievalArm.NAMESPACE_LIFECYCLE_ONLY,
        RetrievalArm.RELEASED_GOVERNANCE_ORACLE_V2,
    }
)


def _validate_vector(vector: Vector, name: str) -> None:
    if not isinstance(vector, tuple) or not vector:
        raise ValueError(f"{name} must be a nonempty tuple")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in vector
    ):
        raise ValueError(f"{name} must contain only finite numeric values")
    if not any(float(value) != 0 for value in vector):
        raise ValueError(f"{name} must be nonzero")


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    namespace: str
    text: str
    embedding: Vector
    released_order: float
    lifecycle_state: LifecycleState = LifecycleState.UNKNOWN

    def __post_init__(self) -> None:
        if not self.memory_id or not self.namespace:
            raise ValueError("memory_id and namespace must be nonempty")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_vector(self.embedding, "memory embedding")
        if isinstance(self.released_order, bool) or not math.isfinite(float(self.released_order)):
            raise ValueError("released_order must be finite")
        if not isinstance(self.lifecycle_state, LifecycleState):
            raise TypeError("lifecycle_state must be LifecycleState")


class PolicyPurpose(StrEnum):
    CONTENT_DISCLOSURE = "content_disclosure"
    OPERATION_TRACE = "operation_trace"


@dataclass(frozen=True)
class PolicyDecision:
    """Released query-memory policy approximation visible to a routing arm."""

    memory_id: str
    content_disclosure_allowed: bool | None = None
    operation_trace_allowed: bool | None = None

    def __post_init__(self) -> None:
        if not self.memory_id:
            raise ValueError("policy memory_id must be nonempty")
        for name, value in (
            ("content_disclosure_allowed", self.content_disclosure_allowed),
            ("operation_trace_allowed", self.operation_trace_allowed),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be true, false, or unknown")

    def allowed_for(self, purpose: PolicyPurpose) -> bool | None:
        if purpose is PolicyPurpose.CONTENT_DISCLOSURE:
            return self.content_disclosure_allowed
        return self.operation_trace_allowed


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    namespace: str
    text: str
    embedding: Vector
    intent: QueryIntent = QueryIntent.UNKNOWN
    policy_purpose: PolicyPurpose = PolicyPurpose.CONTENT_DISCLOSURE

    def __post_init__(self) -> None:
        if not self.query_id or not self.namespace:
            raise ValueError("query_id and namespace must be nonempty")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_vector(self.embedding, "query embedding")
        if not isinstance(self.intent, QueryIntent):
            raise TypeError("intent must be QueryIntent")
        if not isinstance(self.policy_purpose, PolicyPurpose):
            raise TypeError("policy_purpose must be PolicyPurpose")


@dataclass(frozen=True)
class RetrievalConfig:
    setting_id: str
    arm: RetrievalArm
    top_k: int = 100
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    recency_gamma: float = 0.1
    theta: float = 0.8
    top_l: int = 1
    alpha_like: float = 1.0
    cluster_beta: float = 1.0
    cluster_gamma: float = 0.5
    cluster_tau: float = 0.1
    cluster_tau_new: float = 0.4

    def __post_init__(self) -> None:
        if not self.setting_id:
            raise ValueError("setting_id must be nonempty")
        if not isinstance(self.arm, RetrievalArm):
            raise TypeError("arm must be RetrievalArm")
        integer_fields = {
            "top_k": self.top_k,
            "rrf_k": self.rrf_k,
            "top_l": self.top_l,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        positive_fields = {
            "bm25_k1": self.bm25_k1,
            "alpha_like": self.alpha_like,
            "cluster_beta": self.cluster_beta,
            "cluster_tau": self.cluster_tau,
            "cluster_tau_new": self.cluster_tau_new,
        }
        for name, value in positive_fields.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        bounded_fields = {"bm25_b": self.bm25_b, "theta": self.theta}
        for name, value in bounded_fields.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        nonnegative_fields = {
            "recency_gamma": self.recency_gamma,
            "cluster_gamma": self.cluster_gamma,
        }
        for name, value in nonnegative_fields.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class RouteResult:
    query_id: str
    setting_id: str
    arm: RetrievalArm
    ranked_memory_ids: tuple[str, ...]
    candidates_scored: int
    route_width: int
    fallback_used: bool
    selected_cluster_ids: tuple[int, ...] = ()


def _dot(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("zero vectors cannot be normalized")
    return tuple(value / norm for value in vector)


def _cosine(left: Vector, right: Vector) -> float:
    return _dot(_normalize(left), _normalize(right))


def _stable_ranking(scored: Iterable[tuple[str, float]], top_k: int) -> tuple[str, ...]:
    rows = list(scored)
    if any(not math.isfinite(score) for _, score in rows):
        raise ValueError("ranking scores must be finite")
    return tuple(
        memory_id for memory_id, _ in sorted(rows, key=lambda row: (-row[1], row[0]))[:top_k]
    )


def _dense_rank(
    memories: Sequence[MemoryRecord],
    query: QueryRecord,
    *,
    top_k: int,
    adjustments: dict[str, float] | None = None,
) -> tuple[str, ...]:
    adjustments = adjustments or {}
    return _stable_ranking(
        (
            (
                memory.memory_id,
                _dot(memory.embedding, query.embedding) + adjustments.get(memory.memory_id, 0.0),
            )
            for memory in memories
        ),
        top_k,
    )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN.findall(text))


def _bm25_rank(
    memories: Sequence[MemoryRecord],
    query: QueryRecord,
    *,
    top_k: int,
    k1: float,
    b: float,
) -> tuple[str, ...]:
    if not memories:
        return ()
    documents = {memory.memory_id: Counter(_tokens(memory.text)) for memory in memories}
    lengths = {memory_id: sum(counts.values()) for memory_id, counts in documents.items()}
    average_length = sum(lengths.values()) / len(lengths)
    query_terms = set(_tokens(query.text))
    document_frequency = {
        term: sum(term in counts for counts in documents.values()) for term in query_terms
    }
    scores: list[tuple[str, float]] = []
    for memory in memories:
        counts = documents[memory.memory_id]
        length = lengths[memory.memory_id]
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (len(memories) - df + 0.5) / (df + 0.5))
            length_ratio = length / average_length if average_length else 0.0
            denominator = frequency + k1 * (1 - b + b * length_ratio)
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append((memory.memory_id, score))
    return _stable_ranking(scores, top_k)


def _rrf_rank(
    memories: Sequence[MemoryRecord],
    query: QueryRecord,
    *,
    top_k: int,
    k1: float,
    b: float,
    rrf_k: int,
) -> tuple[str, ...]:
    full_k = len(memories)
    bm25 = _bm25_rank(memories, query, top_k=full_k, k1=k1, b=b)
    dense = _dense_rank(memories, query, top_k=full_k)
    bm25_rank = {memory_id: rank for rank, memory_id in enumerate(bm25, start=1)}
    dense_rank = {memory_id: rank for rank, memory_id in enumerate(dense, start=1)}
    return _stable_ranking(
        (
            (
                memory.memory_id,
                1 / (rrf_k + bm25_rank[memory.memory_id])
                + 1 / (rrf_k + dense_rank[memory.memory_id]),
            )
            for memory in memories
        ),
        top_k,
    )


@dataclass
class _Cluster:
    cluster_id: int
    members: list[MemoryRecord] = field(default_factory=list)
    vector_sum: list[float] = field(default_factory=list)

    @classmethod
    def create(cls, cluster_id: int, memory: MemoryRecord) -> _Cluster:
        return cls(cluster_id, [memory], list(memory.embedding))

    @property
    def centroid(self) -> Vector:
        return _normalize(self.vector_sum)

    def add(self, memory: MemoryRecord) -> None:
        if len(memory.embedding) != len(self.vector_sum):
            raise ValueError("embedding dimensions do not match")
        self.members.append(memory)
        for index, value in enumerate(memory.embedding):
            self.vector_sum[index] += value


def _ordered_for_insertion(memories: Sequence[MemoryRecord]) -> tuple[MemoryRecord, ...]:
    return tuple(sorted(memories, key=lambda memory: (memory.released_order, memory.memory_id)))


def _threshold_clusters(memories: Sequence[MemoryRecord], theta: float) -> list[_Cluster]:
    clusters: list[_Cluster] = []
    for memory in _ordered_for_insertion(memories):
        if not clusters:
            clusters.append(_Cluster.create(0, memory))
            continue
        scores = [(_cosine(memory.embedding, cluster.centroid), cluster) for cluster in clusters]
        score, selected = max(scores, key=lambda row: (row[0], -row[1].cluster_id))
        if score >= theta:
            selected.add(memory)
        else:
            clusters.append(_Cluster.create(len(clusters), memory))
    return clusters


def _a5_clusters(memories: Sequence[MemoryRecord], config: RetrievalConfig) -> list[_Cluster]:
    clusters: list[_Cluster] = []
    for memory in _ordered_for_insertion(memories):
        if not clusters:
            clusters.append(_Cluster.create(0, memory))
            continue
        similarities = [
            (_cosine(memory.embedding, cluster.centroid), cluster) for cluster in clusters
        ]
        existing = [
            (
                config.cluster_gamma * math.log(len(cluster.members) + config.cluster_beta)
                + similarity / config.cluster_tau,
                cluster,
            )
            for similarity, cluster in similarities
        ]
        existing_score, selected = max(existing, key=lambda row: (row[0], -row[1].cluster_id))
        max_similarity = max(similarity for similarity, _ in similarities)
        new_score = math.log(config.alpha_like) + (1 - max_similarity) / config.cluster_tau_new
        if new_score > existing_score:
            clusters.append(_Cluster.create(len(clusters), memory))
        else:
            selected.add(memory)
    return clusters


def _rank_cluster_candidates(
    candidates: Sequence[MemoryRecord],
    query: QueryRecord,
    config: RetrievalConfig,
    *,
    selected: Sequence[_Cluster],
    fallback_used: bool,
) -> RouteResult:
    ranked = _dense_rank(candidates, query, top_k=config.top_k)
    return RouteResult(
        query_id=query.query_id,
        setting_id=config.setting_id,
        arm=config.arm,
        ranked_memory_ids=ranked,
        candidates_scored=len(candidates),
        route_width=(len(selected) if selected else (1 if candidates else 0)),
        fallback_used=fallback_used,
        selected_cluster_ids=tuple(cluster.cluster_id for cluster in selected),
    )


def _threshold_route(
    memories: Sequence[MemoryRecord], query: QueryRecord, config: RetrievalConfig
) -> RouteResult:
    clusters = _threshold_clusters(memories, config.theta)
    scored = sorted(
        ((_cosine(query.embedding, cluster.centroid), cluster) for cluster in clusters),
        key=lambda row: (-row[0], row[1].cluster_id),
    )
    selected = tuple(cluster for score, cluster in scored if score >= config.theta)[: config.top_l]
    fallback_used = not selected
    candidates = (
        tuple(member for cluster in selected for member in cluster.members)
        if selected
        else tuple(memories)
    )
    return _rank_cluster_candidates(
        candidates,
        query,
        config,
        selected=selected,
        fallback_used=fallback_used,
    )


def _a5_route(
    memories: Sequence[MemoryRecord], query: QueryRecord, config: RetrievalConfig
) -> RouteResult:
    clusters = _a5_clusters(memories, config)
    scored = sorted(
        (
            (
                config.cluster_gamma * math.log(len(cluster.members) + config.cluster_beta)
                + _cosine(query.embedding, cluster.centroid) / config.cluster_tau,
                cluster,
            )
            for cluster in clusters
        ),
        key=lambda row: (-row[0], row[1].cluster_id),
    )
    selected = tuple(cluster for _, cluster in scored[: config.top_l])
    fallback_used = not selected
    candidates = (
        tuple(member for cluster in selected for member in cluster.members)
        if selected
        else tuple(memories)
    )
    return _rank_cluster_candidates(
        candidates,
        query,
        config,
        selected=selected,
        fallback_used=fallback_used,
    )


def _namespace_support(
    memories: Sequence[MemoryRecord], query: QueryRecord
) -> tuple[MemoryRecord, ...]:
    return tuple(memory for memory in memories if memory.namespace == query.namespace)


def _policy_support(
    memories: Sequence[MemoryRecord],
    query: QueryRecord,
    policy_decisions: Sequence[PolicyDecision],
) -> tuple[MemoryRecord, ...]:
    policy_by_memory: dict[str, PolicyDecision] = {}
    for decision in policy_decisions:
        if decision.memory_id in policy_by_memory:
            raise ValueError(f"duplicate policy decision for {decision.memory_id!r}")
        policy_by_memory[decision.memory_id] = decision

    return tuple(
        memory
        for memory in memories
        if (
            policy_by_memory.get(memory.memory_id) is None
            or policy_by_memory[memory.memory_id].allowed_for(query.policy_purpose) is not False
        )
    )


def _lifecycle_support(
    memories: Sequence[MemoryRecord],
    query: QueryRecord,
) -> tuple[MemoryRecord, ...]:
    if query.intent is not QueryIntent.CURRENT_STATE:
        return tuple(memories)
    return tuple(
        memory
        for memory in memories
        if memory.lifecycle_state not in {LifecycleState.STALE, LifecycleState.SUPERSEDED}
    )


def _combined_admissibility_support(
    memories: Sequence[MemoryRecord],
    query: QueryRecord,
    policy_decisions: Sequence[PolicyDecision],
) -> tuple[MemoryRecord, ...]:
    return _lifecycle_support(
        _policy_support(memories, query, policy_decisions),
        query,
    )


def _historical_released_support(
    memories: Sequence[MemoryRecord],
    query: QueryRecord,
    policy_decisions: Sequence[PolicyDecision],
) -> tuple[MemoryRecord, ...]:
    """Reproduce the frozen v1 arm: history bypasses both released filters."""
    if query.intent is not QueryIntent.CURRENT_STATE:
        return tuple(memories)
    return _combined_admissibility_support(memories, query, policy_decisions)


def _direct_result(
    candidates: Sequence[MemoryRecord],
    query: QueryRecord,
    config: RetrievalConfig,
    ranked: tuple[str, ...],
) -> RouteResult:
    return RouteResult(
        query_id=query.query_id,
        setting_id=config.setting_id,
        arm=config.arm,
        ranked_memory_ids=ranked,
        candidates_scored=len(candidates),
        route_width=1 if candidates else 0,
        fallback_used=False,
    )


def route(
    memories: Sequence[MemoryRecord],
    query: QueryRecord,
    config: RetrievalConfig,
    *,
    policy_decisions: Sequence[PolicyDecision] = (),
) -> RouteResult:
    """Run one frozen retrieval arm with deterministic ID tie breaking."""
    memory_ids = [memory.memory_id for memory in memories]
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("memory IDs must be unique")
    policy_ids = [decision.memory_id for decision in policy_decisions]
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("policy decisions must have unique memory IDs")
    unknown_policy_ids = set(policy_ids) - set(memory_ids)
    if unknown_policy_ids:
        raise ValueError(
            f"policy decisions reference unknown memories: {sorted(unknown_policy_ids)!r}"
        )
    if memories and any(len(memory.embedding) != len(query.embedding) for memory in memories):
        raise ValueError("all memory and query embeddings must share one dimension")

    if config.arm is RetrievalArm.GLOBAL_BM25:
        candidates = tuple(memories)
        ranked = _bm25_rank(
            candidates,
            query,
            top_k=config.top_k,
            k1=config.bm25_k1,
            b=config.bm25_b,
        )
        return _direct_result(candidates, query, config, ranked)
    if config.arm is RetrievalArm.GLOBAL_DENSE:
        candidates = tuple(memories)
        return _direct_result(
            candidates,
            query,
            config,
            _dense_rank(candidates, query, top_k=config.top_k),
        )
    if config.arm is RetrievalArm.GLOBAL_BM25_DENSE_RRF:
        candidates = tuple(memories)
        ranked = _rrf_rank(
            candidates,
            query,
            top_k=config.top_k,
            k1=config.bm25_k1,
            b=config.bm25_b,
            rrf_k=config.rrf_k,
        )
        return _direct_result(candidates, query, config, ranked)
    if config.arm is RetrievalArm.GLOBAL_RECENCY_DENSE:
        candidates = tuple(memories)
        orders = [float(memory.released_order) for memory in candidates]
        minimum = min(orders, default=0.0)
        span = max(orders, default=minimum) - minimum
        adjustments = {
            memory.memory_id: config.recency_gamma
            * ((float(memory.released_order) - minimum) / span if span else 0.0)
            for memory in candidates
        }
        ranked = _dense_rank(
            candidates,
            query,
            top_k=config.top_k,
            adjustments=adjustments,
        )
        return _direct_result(candidates, query, config, ranked)

    namespace = _namespace_support(memories, query)
    if config.arm is RetrievalArm.NAMESPACE_DENSE:
        return _direct_result(
            namespace,
            query,
            config,
            _dense_rank(namespace, query, top_k=config.top_k),
        )
    if config.arm is RetrievalArm.QUERY_AGNOSTIC_CURRENT_ONLY:
        candidates = tuple(
            memory
            for memory in namespace
            if memory.lifecycle_state not in {LifecycleState.STALE, LifecycleState.SUPERSEDED}
        )
        return _direct_result(
            candidates,
            query,
            config,
            _dense_rank(candidates, query, top_k=config.top_k),
        )
    if config.arm is RetrievalArm.NAMESPACE_POLICY_ONLY:
        candidates = _policy_support(namespace, query, policy_decisions)
        return _direct_result(
            candidates,
            query,
            config,
            _dense_rank(candidates, query, top_k=config.top_k),
        )
    if config.arm is RetrievalArm.NAMESPACE_LIFECYCLE_ONLY:
        candidates = _lifecycle_support(namespace, query)
        return _direct_result(
            candidates,
            query,
            config,
            _dense_rank(candidates, query, top_k=config.top_k),
        )
    if config.arm is RetrievalArm.RELEASED_INTENT_LIFECYCLE_UPPER_BOUND:
        candidates = _historical_released_support(namespace, query, policy_decisions)
        return _direct_result(
            candidates,
            query,
            config,
            _dense_rank(candidates, query, top_k=config.top_k),
        )
    if config.arm is RetrievalArm.RELEASED_GOVERNANCE_ORACLE_V2:
        candidates = _combined_admissibility_support(namespace, query, policy_decisions)
        return _direct_result(
            candidates,
            query,
            config,
            _dense_rank(candidates, query, top_k=config.top_k),
        )
    if config.arm is RetrievalArm.THRESHOLD_ROUTER:
        return _threshold_route(namespace, query, config)
    if config.arm is RetrievalArm.CLUSTER_ROUTER:
        return _a5_route(namespace, query, config)
    raise AssertionError(f"unhandled retrieval arm: {config.arm}")
