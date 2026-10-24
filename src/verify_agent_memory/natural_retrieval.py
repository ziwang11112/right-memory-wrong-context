"""Portable frozen natural-corpus routing; no archive imports or model execution.

The exact float32, sparse BM25, chronology and router rules are retained from the
study implementation. Source construction and score labels are separate modules.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from verify_agent_memory.natural_corpus import NaturalMemory, NaturalQuery
from verify_agent_memory.natural_storage import RouteObservation, validate_vector_matrix

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.int64]
FloatMatrix = NDArray[np.float32]
IntVector = NDArray[np.int64]
EmbeddingIdentity = tuple[str, str]
FLAT_BM25 = "flat_bm25"
FLAT_DENSE = "flat_dense"
FLAT_RRF = "flat_bm25_dense_rrf"
RECENCY_DENSE = "recency_dense"
NAMESPACE_DENSE = "namespace_dense"
NAMESPACE_CURRENT_ONLY = "namespace_current_only"
NAMESPACE_SOURCE_INTENT_LIFECYCLE = "namespace_source_intent_lifecycle"
NCR_THRESHOLD = "ncr_threshold"
NCR_A5 = "ncr_a5"
ARMS = (
    FLAT_BM25,
    FLAT_DENSE,
    FLAT_RRF,
    RECENCY_DENSE,
    NAMESPACE_DENSE,
    NAMESPACE_CURRENT_ONLY,
    NAMESPACE_SOURCE_INTENT_LIFECYCLE,
    NCR_THRESHOLD,
    NCR_A5,
)
ROUTER_ARMS = frozenset({NCR_THRESHOLD, NCR_A5})
TOP_K_GRID = (5, 10, 20, 50, 100)
SOURCES = ("rhelm", "memops")
MAX_RANKED = 100
_GLOBAL = "global"
_NAMESPACE = "namespace"
_CURRENT_ONLY = "current_only"
_SOURCE_INTENT = "source_intent"
_CANDIDATE_POLICIES = {_GLOBAL, _NAMESPACE, _CURRENT_ONLY, _SOURCE_INTENT}


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"(?u)\b\w+\b", text)]


def _text_fingerprint(text: str) -> tuple[str, int]:
    payload = text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_id(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()[:12]


@dataclass(frozen=True)
class MethodMemory:
    """Method-visible memory fields shared by non-lifecycle arms."""

    source: str
    memory_id: str
    namespace_id: str
    chronology: str
    source_date: str | None
    text: str
    text_sha256: str
    vector_index: int

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError("method memory source is invalid")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.memory_id,
                self.namespace_id,
                self.chronology,
                self.text,
                self.text_sha256,
            )
        ):
            raise ValueError("method memory string fields must be nonempty")
        if self.vector_index < 0:
            raise ValueError("method memory vector index must be nonnegative")


@dataclass(frozen=True)
class LifecycleMemory:
    """Lifecycle-only view layered over an otherwise label-free memory."""

    memory: MethodMemory
    state: str
    released_distractor: bool

    def __post_init__(self) -> None:
        if self.state not in {"current", "stale", "superseded", "uncertain"}:
            raise ValueError("lifecycle memory state is invalid")


@dataclass(frozen=True)
class MethodQuery:
    """Method-visible query without intent, evidence IDs, labels, or answers."""

    source: str
    query_id: str
    namespace_id: str
    cutoff_date: str | None
    text: str
    text_sha256: str
    vector_index: int

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError("method query source is invalid")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.query_id,
                self.namespace_id,
                self.text,
                self.text_sha256,
            )
        ):
            raise ValueError("method query string fields must be nonempty")
        if self.vector_index < 0:
            raise ValueError("method query vector index must be nonnegative")


@dataclass(frozen=True)
class IntentQuery:
    """Intent-visible upper-bound query view."""

    query: MethodQuery
    query_intent: str

    def __post_init__(self) -> None:
        if self.query_intent not in {"current_state", "history"}:
            raise ValueError("query intent is invalid")


@dataclass(frozen=True)
class NaturalSetting:
    """One complete arm setting, including retrieval depth."""

    arm: str
    setting_id: str
    params: Mapping[str, float | int]

    @classmethod
    def create(
        cls,
        arm: str,
        params: Mapping[str, float | int],
    ) -> NaturalSetting:
        if arm not in ARMS:
            raise ValueError(f"unsupported natural routing arm {arm!r}")
        normalized: dict[str, float | int] = {}
        for key, value in params.items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("setting parameters must be finite numeric values")
            normalized[key] = value
        top_k = normalized.get("top_k")
        if not isinstance(top_k, int) or top_k not in TOP_K_GRID:
            raise ValueError("natural setting top_k is outside the frozen grid")
        setting_id = _stable_id({"arm": arm, "params": normalized})
        return cls(arm, setting_id, MappingProxyType(dict(normalized)))

    @property
    def ranking_params(self) -> Mapping[str, float | int]:
        return MappingProxyType(
            {key: value for key, value in self.params.items() if key != "top_k"}
        )

    @property
    def ranking_config_id(self) -> str:
        return _stable_id({"arm": self.arm, "params": dict(self.ranking_params)})


def stable_rank_indices(
    scores: FloatArray,
    id_tie_rank: IntArray,
    *,
    limit: int = MAX_RANKED,
) -> IntArray:
    """Rank by descending score, then stable lexical memory-ID rank."""
    values = np.asarray(scores)
    tie_rank = np.asarray(id_tie_rank, dtype=np.int64)
    if (
        values.ndim != 1
        or tie_rank.shape != values.shape
        or not np.isfinite(values).all()
        or len(set(int(value) for value in tie_rank)) != tie_rank.size
        or limit < 0
    ):
        raise ValueError("stable ranking inputs are invalid")
    order = np.lexsort((tie_rank, -values))
    return np.asarray(order[: min(limit, order.size)], dtype=np.int64)


def lexical_tie_ranks(memory_ids: Sequence[str]) -> IntArray:
    """Return each row's stable lexical rank."""
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("memory IDs must be unique")
    order = sorted(range(len(memory_ids)), key=lambda index: memory_ids[index])
    ranks = np.empty(len(memory_ids), dtype=np.int64)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def rank_rrf(
    first_order: IntArray,
    second_order: IntArray,
    *,
    tie_ranks: IntArray | None = None,
    limit: int = MAX_RANKED,
    rrf_k: int = 60,
) -> IntArray:
    """Fuse two complete candidate-index orders with an explicit stable tie."""
    first = tuple(int(value) for value in np.asarray(first_order, dtype=np.int64))
    second = tuple(int(value) for value in np.asarray(second_order, dtype=np.int64))
    if set(first) != set(second) or len(first) != len(second):
        raise ValueError("RRF orders must cover the same candidates exactly once")
    if rrf_k <= 0 or limit < 0:
        raise ValueError("RRF constants must be positive")
    if tie_ranks is None:
        stable_ties = np.arange(len(first), dtype=np.int64)
    else:
        stable_ties = np.asarray(tie_ranks, dtype=np.int64)
        if stable_ties.shape != (len(first),) or len(
            set(int(value) for value in stable_ties)
        ) != len(first):
            raise ValueError("RRF tie ranks are invalid")

    first_array = np.asarray(first, dtype=np.int64)
    second_array = np.asarray(second, dtype=np.int64)
    zero_based_ranks = np.arange(len(first), dtype=np.int64)
    reciprocal = 1.0 / (rrf_k + zero_based_ranks + 1)
    scores = np.zeros(len(first), dtype=np.float64)
    scores[first_array] += reciprocal
    scores[second_array] += reciprocal
    first_ranks = np.empty(len(first), dtype=np.int64)
    second_ranks = np.empty(len(first), dtype=np.int64)
    first_ranks[first_array] = zero_based_ranks
    second_ranks[second_array] = zero_based_ranks
    best_rank = np.minimum(first_ranks, second_ranks)
    fused = np.lexsort((stable_ties, best_rank, -scores))
    return np.asarray(fused[: min(limit, len(fused))], dtype=np.int64)


class OnlineRouter:
    """Deterministic float32 online threshold or A5 index."""

    def __init__(
        self,
        arm: str,
        params: Mapping[str, float | int],
        *,
        dimension: int,
    ) -> None:
        if arm not in ROUTER_ARMS or dimension < 1:
            raise ValueError("online router contract is invalid")
        self.arm = arm
        self.params = dict(params)
        self.dimension = dimension
        self.counts: list[int] = []
        self.members: list[list[int]] = []
        self._capacity = 0
        self._sums = np.empty((0, dimension), dtype=np.float32)
        self._centroids = np.empty((0, dimension), dtype=np.float32)

    @property
    def n_clusters(self) -> int:
        return len(self.counts)

    @property
    def centroids(self) -> NDArray[np.float32]:
        return self._centroids[: self.n_clusters]

    def _grow(self) -> None:
        capacity = 16 if self._capacity == 0 else self._capacity * 2
        sums = np.empty((capacity, self.dimension), dtype=np.float32)
        centroids = np.empty((capacity, self.dimension), dtype=np.float32)
        if self.n_clusters:
            sums[: self.n_clusters] = self._sums[: self.n_clusters]
            centroids[: self.n_clusters] = self._centroids[: self.n_clusters]
        self._capacity = capacity
        self._sums = sums
        self._centroids = centroids

    def _existing_scores(self, row: NDArray[np.float32]) -> NDArray[np.floating]:
        cosines = np.asarray(self.centroids @ row, dtype=np.float32)
        if self.arm == NCR_THRESHOLD:
            return cosines
        counts = np.asarray(self.counts, dtype=np.float64)
        return float(self.params["gamma"]) * np.log(
            counts + float(self.params["beta"])
        ) + cosines.astype(np.float64) / float(self.params["tau"])

    def insert(self, row: FloatArray, memory_index: int) -> int:
        """Insert one memory in frozen chronology order and return its cluster."""
        vector = np.asarray(row, dtype=np.float32)
        if (
            vector.shape != (self.dimension,)
            or memory_index < 0
            or not np.isfinite(vector).all()
            or not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-5)
        ):
            raise ValueError("online router insert is invalid")
        existing = self._existing_scores(vector)
        if self.n_clusters == 0:
            choice = 0
        elif self.arm == NCR_THRESHOLD:
            best = int(np.argmax(existing))
            choice = (
                best if float(existing[best]) >= float(self.params["theta"]) else self.n_clusters
            )
        else:
            cosines = np.asarray(self.centroids @ vector, dtype=np.float32)
            novelty = 1.0 - float(np.max(cosines))
            log_new = math.log(float(self.params["alpha_like"])) + novelty / float(
                self.params["tau_new"]
            )
            choice = int(np.argmax(np.concatenate([existing, np.asarray([log_new])])))

        if choice == self.n_clusters:
            if self.n_clusters == self._capacity:
                self._grow()
            self._sums[choice] = vector
            self._centroids[choice] = vector
            self.counts.append(1)
            self.members.append([memory_index])
        else:
            self.counts[choice] += 1
            self._sums[choice] = np.asarray(self._sums[choice] + vector, dtype=np.float32)
            norm = float(np.linalg.norm(self._sums[choice]))
            self._centroids[choice] = np.asarray(self._sums[choice] / norm, dtype=np.float32)
            self.members[choice].append(memory_index)
        return choice

    def selected_clusters(self, query: FloatArray, *, top_l: int) -> tuple[int, ...]:
        """Route over existing clusters using the frozen query rule."""
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (self.dimension,) or top_l < 1:
            raise ValueError("online router query is invalid")
        if not self.n_clusters:
            return ()
        scores = self._existing_scores(vector)
        order = np.lexsort((np.arange(self.n_clusters, dtype=np.int64), -scores))
        if self.arm == NCR_THRESHOLD:
            order = np.asarray(
                [index for index in order if float(scores[index]) >= float(self.params["theta"])],
                dtype=np.int64,
            )
        return tuple(int(index) for index in order[:top_l])

    def candidate_indices(self, selected: Sequence[int]) -> tuple[int, ...]:
        """Return stable unique members of selected clusters."""
        if any(index < 0 or index >= self.n_clusters for index in selected):
            raise ValueError("selected cluster index is invalid")
        return tuple(sorted({member for index in selected for member in self.members[index]}))


@dataclass(frozen=True)
class MethodViews:
    """Label-free method views for one source's evaluation partition."""

    source: str
    memories: tuple[MethodMemory, ...]
    lifecycle_memories: tuple[LifecycleMemory, ...]
    queries: tuple[MethodQuery, ...]
    intent_queries: tuple[IntentQuery, ...]

    def __post_init__(self) -> None:
        if self.source not in {"rhelm", "memops"}:
            raise ValueError("dev method-view source is invalid")
        if not self.memories or not self.queries:
            raise ValueError("dev method views require memories and queries")
        if tuple(row.memory for row in self.lifecycle_memories) != self.memories:
            raise ValueError("lifecycle memories do not align with base memories")
        if tuple(row.query for row in self.intent_queries) != self.queries:
            raise ValueError("intent queries do not align with base queries")
        if [row.memory_id for row in self.memories] != sorted(
            row.memory_id for row in self.memories
        ):
            raise ValueError("method memories must be sorted by ID")
        if [row.query_id for row in self.queries] != sorted(row.query_id for row in self.queries):
            raise ValueError("method queries must be sorted by ID")


@dataclass(frozen=True)
class ExactBM25Index:
    """Sparse scorer exactly matching the frozen rank-bm25 Okapi formula."""

    idf: Mapping[str, float]
    average_document_length: float
    document_lengths: NDArray[np.float64]
    postings: Mapping[str, tuple[IntVector, NDArray[np.float64]]]

    @classmethod
    def build(
        cls,
        tokenized_documents: Sequence[Sequence[str]],
    ) -> ExactBM25Index:
        if not tokenized_documents:
            raise ValueError("BM25 index requires at least one document")
        documents = [list(tokens) for tokens in tokenized_documents]
        model = BM25Okapi(
            documents,
            k1=1.5,
            b=0.75,
            epsilon=0.25,
        )
        if model.avgdl <= 0.0:
            raise ValueError("BM25 documents must contain at least one token")
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        for document_index, tokens in enumerate(documents):
            for term, frequency in Counter(tokens).items():
                postings[term].append((document_index, frequency))
        frozen_postings = {}
        for term, rows in postings.items():
            indices = np.asarray([row[0] for row in rows], dtype=np.int64)
            frequencies = np.asarray([row[1] for row in rows], dtype=np.float64)
            indices.setflags(write=False)
            frequencies.setflags(write=False)
            frozen_postings[term] = indices, frequencies
        document_lengths = np.asarray(model.doc_len, dtype=np.float64)
        document_lengths.setflags(write=False)
        return cls(
            idf=MappingProxyType(dict(model.idf)),
            average_document_length=float(model.avgdl),
            document_lengths=document_lengths,
            postings=MappingProxyType(frozen_postings),
        )

    def scores(
        self,
        query_tokens: Sequence[str],
        *,
        k1: float,
        b: float,
    ) -> NDArray[np.float64]:
        """Return full float64 scores in the same query-token addition order."""
        if k1 <= 0.0 or not 0.0 <= b <= 1.0:
            raise ValueError("BM25 parameters are invalid")
        scores = np.zeros(len(self.document_lengths), dtype=np.float64)
        length_normalizer = 1.0 - b + b * self.document_lengths / self.average_document_length
        for term in query_tokens:
            posting = self.postings.get(term)
            if posting is None:
                continue
            indices, frequencies = posting
            contribution = (
                float(self.idf.get(term, 0.0))
                * frequencies
                * (k1 + 1.0)
                / (frequencies + k1 * length_normalizer[indices])
            )
            scores[indices] += contribution
        return scores


class NaturalIndex:
    """Read-only exact index over one source's method-visible evaluation rows."""

    def __init__(self, views: MethodViews, vectors: FloatMatrix) -> None:
        matrix = validate_vector_matrix(vectors)
        memory_indices = np.asarray(
            [memory.vector_index for memory in views.memories],
            dtype=np.int64,
        )
        query_indices = np.asarray(
            [query.vector_index for query in views.queries],
            dtype=np.int64,
        )
        if (
            int(memory_indices.max()) >= matrix.shape[0]
            or int(query_indices.max()) >= matrix.shape[0]
        ):
            raise ValueError("method view references an absent embedding")
        self.views = views
        self.source = views.source
        self.memories = views.memories
        self.queries = views.queries
        self.memory_vectors = np.ascontiguousarray(matrix[memory_indices], dtype=np.float32)
        self.query_vectors = np.ascontiguousarray(matrix[query_indices], dtype=np.float32)
        validate_vector_matrix(self.memory_vectors)
        validate_vector_matrix(self.query_vectors)
        self.memory_ids = tuple(memory.memory_id for memory in self.memories)
        self.query_ids = tuple(query.query_id for query in self.queries)
        self.memory_index_by_id = MappingProxyType(
            {memory_id: index for index, memory_id in enumerate(self.memory_ids)}
        )
        self.query_index_by_id = MappingProxyType(
            {query_id: index for index, query_id in enumerate(self.query_ids)}
        )
        self.tie_ranks = lexical_tie_ranks(self.memory_ids)
        self._lifecycle = views.lifecycle_memories
        self._intents = tuple(row.query_intent for row in views.intent_queries)
        self._chronology_order = np.asarray(
            sorted(
                range(len(self.memories)),
                key=lambda index: (
                    self.memories[index].source_date or "",
                    self.memories[index].chronology,
                    self.memories[index].memory_id,
                ),
            ),
            dtype=np.int64,
        )
        by_namespace: defaultdict[str, list[int]] = defaultdict(list)
        for index in self._chronology_order:
            by_namespace[self.memories[int(index)].namespace_id].append(int(index))
        self._by_namespace = {
            namespace: np.asarray(indices, dtype=np.int64)
            for namespace, indices in by_namespace.items()
        }
        self._candidate_cache: dict[tuple[int, str], IntVector] = {}
        self._memory_tokens: tuple[tuple[str, ...], ...] | None = None
        self._global_bm25_index: ExactBM25Index | None = None
        self._bm25_order_cache_key: tuple[float, float] | None = None
        self._bm25_order_cache: tuple[IntVector, ...] | None = None
        self._bm25_latency_cache: tuple[float, ...] | None = None
        self._dense_order_cache: dict[int, IntVector] = {}
        self._router_route_cache: dict[
            tuple[str, tuple[tuple[str, float | int], ...]],
            Mapping[int, tuple[RouteObservation, ...]],
        ] = {}
        self._dense_scores: FloatMatrix | None = None
        self.dense_build_seconds = 0.0

    @property
    def dimension(self) -> int:
        return int(self.memory_vectors.shape[1])

    def dense_scores(self) -> FloatMatrix:
        """Return the exact float32 memory-by-query score matrix."""
        if self._dense_scores is None:
            started = time.perf_counter()
            scores = np.asarray(
                self.memory_vectors @ self.query_vectors.T,
                dtype=np.float32,
            )
            if scores.shape != (len(self.memories), len(self.queries)):
                raise RuntimeError("dense score matrix shape drifted")
            if not np.isfinite(scores).all():
                raise RuntimeError("dense score matrix contains non-finite values")
            self._dense_scores = scores
            self.dense_build_seconds = time.perf_counter() - started
        return self._dense_scores

    def memory_tokens(self) -> tuple[tuple[str, ...], ...]:
        """Tokenize each memory once under the frozen BM25 tokenizer."""
        if self._memory_tokens is None:
            self._memory_tokens = tuple(
                tuple(tokenize_for_bm25(memory.text)) for memory in self.memories
            )
        return self._memory_tokens

    def bm25_index(self, candidates: IntVector) -> ExactBM25Index:
        """Build source-appropriate sparse BM25 statistics."""
        if self.source == "memops":
            if not np.array_equal(candidates, self._chronology_order):
                raise ValueError("MemOps BM25 must use the complete evaluation corpus")
            if self._global_bm25_index is None:
                tokens = self.memory_tokens()
                self._global_bm25_index = ExactBM25Index.build(
                    [tokens[int(index)] for index in candidates]
                )
            return self._global_bm25_index
        tokens = self.memory_tokens()
        return ExactBM25Index.build([tokens[int(index)] for index in candidates])

    def candidate_indices(self, query_index: int, policy: str) -> IntVector:
        """Return one exact candidate set in immutable chronology order."""
        if not 0 <= query_index < len(self.queries) or policy not in _CANDIDATE_POLICIES:
            raise ValueError("candidate request is invalid")
        key = query_index, policy
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        query = self.queries[query_index]
        if policy == _GLOBAL:
            candidates = self._chronology_order
        else:
            candidates = self._by_namespace.get(
                query.namespace_id,
                np.empty(0, dtype=np.int64),
            )
        if self.source == "rhelm":
            if query.cutoff_date is None:
                raise ValueError("RHELM method query lacks a cutoff date")
            candidates = np.asarray(
                [
                    int(index)
                    for index in candidates
                    if (
                        self.memories[int(index)].source_date is not None
                        and self.memories[int(index)].source_date < query.cutoff_date
                    )
                ],
                dtype=np.int64,
            )
        if policy in {_CURRENT_ONLY, _SOURCE_INTENT}:
            intent = self._intents[query_index]
            if policy == _CURRENT_ONLY or intent == "current_state":
                candidates = np.asarray(
                    [
                        int(index)
                        for index in candidates
                        if self._lifecycle[int(index)].state not in {"stale", "superseded"}
                        and (
                            policy == _CURRENT_ONLY
                            or not self._lifecycle[int(index)].released_distractor
                        )
                    ],
                    dtype=np.int64,
                )
        if not candidates.size and policy in {_GLOBAL, _NAMESPACE}:
            raise ValueError("method-visible candidate policy produced an empty set")
        candidates.setflags(write=False)
        self._candidate_cache[key] = candidates
        return candidates


def _rank_dense_candidates(
    index: NaturalIndex,
    query_index: int,
    candidates: IntVector,
    *,
    gamma: float = 0.0,
    limit: int = MAX_RANKED,
) -> tuple[str, ...]:
    scores = np.asarray(index.dense_scores()[candidates, query_index], dtype=np.float32)
    if gamma:
        denominator = max(len(candidates) - 1, 1)
        recency = np.arange(len(candidates), dtype=np.float32) / np.float32(denominator)
        scores = np.asarray(scores + np.float32(gamma) * recency, dtype=np.float32)
    ranked_local = stable_rank_indices(
        scores,
        index.tie_ranks[candidates],
        limit=limit,
    )
    return tuple(index.memory_ids[int(candidates[position])] for position in ranked_local)


def _observation(
    index: NaturalIndex,
    query_index: int,
    ranked_ids: Sequence[str],
    *,
    route_width: int,
    candidates_scored: int,
    centroid_scores: int,
    fallback_used: bool,
    started: float,
) -> RouteObservation:
    return RouteObservation(
        query_id=index.query_ids[query_index],
        ranked_memory_ids=tuple(ranked_ids[:MAX_RANKED]),
        route_width=route_width,
        candidates_scored=candidates_scored,
        centroid_scores=centroid_scores,
        fallback_used=fallback_used,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def _route_dense_config(
    index: NaturalIndex,
    setting: NaturalSetting,
) -> tuple[RouteObservation, ...]:
    observations = []
    gamma = float(setting.ranking_params.get("gamma", 0.0))
    for query_index in range(len(index.queries)):
        started = time.perf_counter()
        if setting.arm in {FLAT_DENSE, RECENCY_DENSE}:
            policy = _GLOBAL
        elif setting.arm == NAMESPACE_DENSE:
            policy = _NAMESPACE
        elif setting.arm == NAMESPACE_CURRENT_ONLY:
            policy = _CURRENT_ONLY
        elif setting.arm == NAMESPACE_SOURCE_INTENT_LIFECYCLE:
            policy = _SOURCE_INTENT
        else:
            raise ValueError("dense route arm is invalid")
        candidates = index.candidate_indices(query_index, policy)
        ranked = _rank_dense_candidates(
            index,
            query_index,
            candidates,
            gamma=gamma,
        )
        observations.append(
            _observation(
                index,
                query_index,
                ranked,
                route_width=0,
                candidates_scored=len(candidates),
                centroid_scores=0,
                fallback_used=False,
                started=started,
            )
        )
    return tuple(sorted(observations, key=lambda row: row.query_id))


def _bm25_query_groups(index: NaturalIndex) -> tuple[tuple[int, ...], ...]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for query_index, query in enumerate(index.queries):
        key = query.cutoff_date if index.source == "rhelm" else "all"
        if key is None:
            raise ValueError("RHELM BM25 query lacks a cutoff date")
        grouped[key].append(query_index)
    return tuple(tuple(rows) for _key, rows in sorted(grouped.items()))


def _bm25_complete_orders(
    index: NaturalIndex,
    *,
    k1: float,
    b: float,
) -> tuple[tuple[IntVector, ...], tuple[float, ...]]:
    cache_key = k1, b
    if (
        index._bm25_order_cache_key == cache_key
        and index._bm25_order_cache is not None
        and index._bm25_latency_cache is not None
    ):
        return index._bm25_order_cache, index._bm25_latency_cache
    orders: list[IntVector | None] = [None] * len(index.queries)
    latencies = [0.0] * len(index.queries)
    for query_group in _bm25_query_groups(index):
        first_query = query_group[0]
        candidates = index.candidate_indices(first_query, _GLOBAL)
        if any(
            not np.array_equal(
                candidates,
                index.candidate_indices(query_index, _GLOBAL),
            )
            for query_index in query_group[1:]
        ):
            raise RuntimeError("BM25 query grouping changed the candidate set")
        build_started = time.perf_counter()
        model = index.bm25_index(candidates)
        shared_build_ms = (time.perf_counter() - build_started) * 1000.0 / len(query_group)
        for query_index in query_group:
            started = time.perf_counter()
            scores = model.scores(
                tokenize_for_bm25(index.queries[query_index].text),
                k1=k1,
                b=b,
            )
            order = stable_rank_indices(
                scores,
                index.tie_ranks[candidates],
                limit=len(candidates),
            )
            order = np.asarray(order, dtype=np.int32)
            order.setflags(write=False)
            orders[query_index] = order
            latencies[query_index] = (time.perf_counter() - started) * 1000.0 + shared_build_ms
    if any(order is None for order in orders):
        raise RuntimeError("BM25 order cache is incomplete")
    completed = tuple(order for order in orders if order is not None)
    index._bm25_order_cache_key = cache_key
    index._bm25_order_cache = completed
    index._bm25_latency_cache = tuple(latencies)
    return completed, tuple(latencies)


def _dense_complete_order(
    index: NaturalIndex,
    query_index: int,
    candidates: IntVector,
) -> IntVector:
    cached = index._dense_order_cache.get(query_index)
    if cached is not None:
        return cached
    scores = np.asarray(
        index.dense_scores()[candidates, query_index],
        dtype=np.float32,
    )
    order = stable_rank_indices(
        scores,
        index.tie_ranks[candidates],
        limit=len(candidates),
    )
    order = np.asarray(order, dtype=np.int32)
    order.setflags(write=False)
    index._dense_order_cache[query_index] = order
    return order


def _route_bm25_config(
    index: NaturalIndex,
    setting: NaturalSetting,
) -> tuple[RouteObservation, ...]:
    params = setting.ranking_params
    k1 = float(params["bm25_k1"])
    b = float(params["bm25_b"])
    rrf = setting.arm == FLAT_RRF
    bm25_orders, bm25_latencies = _bm25_complete_orders(index, k1=k1, b=b)
    observations = []
    for query_index in range(len(index.queries)):
        started = time.perf_counter()
        candidates = index.candidate_indices(query_index, _GLOBAL)
        bm25_order = bm25_orders[query_index]
        if rrf:
            dense_order = _dense_complete_order(
                index,
                query_index,
                candidates,
            )
            ranked_local = rank_rrf(
                bm25_order,
                dense_order,
                tie_ranks=index.tie_ranks[candidates],
                limit=MAX_RANKED,
                rrf_k=int(params["rrf_k"]),
            )
        else:
            ranked_local = bm25_order[:MAX_RANKED]
        ranked = tuple(
            index.memory_ids[int(candidates[int(position)])] for position in ranked_local
        )
        observations.append(
            RouteObservation(
                query_id=index.query_ids[query_index],
                ranked_memory_ids=ranked,
                route_width=0,
                candidates_scored=len(candidates),
                centroid_scores=0,
                fallback_used=False,
                latency_ms=(bm25_latencies[query_index] + (time.perf_counter() - started) * 1000.0),
            )
        )
    return tuple(sorted(observations, key=lambda row: row.query_id))


def _router_for_setting(
    setting: NaturalSetting,
    *,
    dimension: int,
) -> OnlineRouter:
    return OnlineRouter(
        setting.arm,
        setting.ranking_params,
        dimension=dimension,
    )


def _route_one_router_query(
    index: NaturalIndex,
    query_index: int,
    router: OnlineRouter,
    fallback_candidates: IntVector,
    *,
    top_l: int,
) -> RouteObservation:
    started = time.perf_counter()
    selected = router.selected_clusters(
        index.query_vectors[query_index],
        top_l=top_l,
    )
    fallback = not selected
    if fallback:
        candidates = fallback_candidates
    else:
        candidates = np.asarray(
            router.candidate_indices(selected),
            dtype=np.int64,
        )
    ranked = _rank_dense_candidates(index, query_index, candidates)
    return _observation(
        index,
        query_index,
        ranked,
        route_width=len(selected),
        candidates_scored=len(candidates),
        centroid_scores=router.n_clusters,
        fallback_used=fallback,
        started=started,
    )


def _route_memops_router(
    index: NaturalIndex,
    setting: NaturalSetting,
    *,
    top_ls: Sequence[int],
) -> Mapping[int, tuple[RouteObservation, ...]]:
    routers = {}
    for namespace, candidates in sorted(index._by_namespace.items()):
        router = _router_for_setting(setting, dimension=index.dimension)
        for memory_index in candidates:
            router.insert(index.memory_vectors[int(memory_index)], int(memory_index))
        routers[namespace] = router
    observations: dict[int, list[RouteObservation]] = {top_l: [] for top_l in top_ls}
    for query_index, query in enumerate(index.queries):
        candidates = index.candidate_indices(query_index, _NAMESPACE)
        for top_l in top_ls:
            observations[top_l].append(
                _route_one_router_query(
                    index,
                    query_index,
                    routers[query.namespace_id],
                    candidates,
                    top_l=top_l,
                )
            )
    return MappingProxyType(
        {
            top_l: tuple(sorted(rows, key=lambda row: row.query_id))
            for top_l, rows in observations.items()
        }
    )


def _route_rhelm_router(
    index: NaturalIndex,
    setting: NaturalSetting,
    *,
    top_ls: Sequence[int],
) -> Mapping[int, tuple[RouteObservation, ...]]:
    query_groups: defaultdict[str, list[int]] = defaultdict(list)
    for query_index, query in enumerate(index.queries):
        query_groups[query.namespace_id].append(query_index)
    observations: dict[int, list[RouteObservation]] = {top_l: [] for top_l in top_ls}
    for namespace, query_indices in sorted(query_groups.items()):
        candidates = index._by_namespace[namespace]
        ordered_queries = sorted(
            query_indices,
            key=lambda query_index: (
                index.queries[query_index].cutoff_date or "",
                index.queries[query_index].query_id,
            ),
        )
        router = _router_for_setting(setting, dimension=index.dimension)
        pointer = 0
        for query_index in ordered_queries:
            cutoff = index.queries[query_index].cutoff_date
            if cutoff is None:
                raise ValueError("RHELM router query lacks a cutoff date")
            while pointer < len(candidates):
                memory_index = int(candidates[pointer])
                source_date = index.memories[memory_index].source_date
                if source_date is None or source_date >= cutoff:
                    break
                router.insert(index.memory_vectors[memory_index], memory_index)
                pointer += 1
            eligible = index.candidate_indices(query_index, _NAMESPACE)
            if pointer != len(eligible) or not np.array_equal(
                candidates[:pointer],
                eligible,
            ):
                raise RuntimeError("incremental RHELM router prefix drifted")
            for top_l in top_ls:
                observations[top_l].append(
                    _route_one_router_query(
                        index,
                        query_index,
                        router,
                        eligible,
                        top_l=top_l,
                    )
                )
    return MappingProxyType(
        {
            top_l: tuple(sorted(rows, key=lambda row: row.query_id))
            for top_l, rows in observations.items()
        }
    )


def _route_router_config(
    index: NaturalIndex,
    setting: NaturalSetting,
) -> tuple[RouteObservation, ...]:
    params = tuple(
        sorted((key, value) for key, value in setting.ranking_params.items() if key != "top_l")
    )
    cache_key = setting.arm, params
    cached = index._router_route_cache.get(cache_key)
    if cached is None:
        top_ls = (1, 3)
        cached = (
            _route_rhelm_router(index, setting, top_ls=top_ls)
            if index.source == "rhelm"
            else _route_memops_router(index, setting, top_ls=top_ls)
        )
        index._router_route_cache[cache_key] = cached
    top_l = int(setting.ranking_params["top_l"])
    if top_l not in cached:
        raise ValueError("router top_l is outside the frozen paired cache")
    return cached[top_l]


def route_ranking_config(
    index: NaturalIndex,
    setting: NaturalSetting,
) -> tuple[RouteObservation, ...]:
    """Run one physical ranking configuration without scorer-only fields."""
    if setting.arm in {
        FLAT_DENSE,
        RECENCY_DENSE,
        NAMESPACE_DENSE,
        NAMESPACE_CURRENT_ONLY,
        NAMESPACE_SOURCE_INTENT_LIFECYCLE,
    }:
        observations = _route_dense_config(index, setting)
    elif setting.arm in {FLAT_BM25, FLAT_RRF}:
        observations = _route_bm25_config(index, setting)
    elif setting.arm in {NCR_THRESHOLD, NCR_A5}:
        observations = _route_router_config(index, setting)
    else:
        raise ValueError("physical route setting has an unsupported arm")
    if (
        len(observations) != len(index.queries)
        or tuple(row.query_id for row in observations) != index.query_ids
        or any(len(row.ranked_memory_ids) > MAX_RANKED for row in observations)
    ):
        raise RuntimeError("physical route output differs from the frozen query set")
    return observations


def build_eval_method_views(
    source: str,
    memories: Sequence[NaturalMemory],
    queries: Sequence[NaturalQuery],
    *,
    raw_texts: Mapping[tuple[str, str], str],
    vector_indices: Mapping[tuple[str, str], int],
) -> MethodViews:
    """Build label-free retrieval views after enforcing eval-only descriptors."""
    if (
        source not in SOURCES
        or not memories
        or not queries
        or any(memory.source != source or memory.split != "eval" for memory in memories)
        or any(query.source != source or query.split != "eval" for query in queries)
    ):
        raise ValueError("method views require one nonempty eval-only source partition")
    base_memories = []
    lifecycle_memories = []
    for memory in sorted(memories, key=lambda row: row.memory_id):
        identity = "memory", memory.text_sha256
        text = raw_texts.get(identity)
        vector_index = vector_indices.get(identity)
        if text is None or vector_index is None:
            raise ValueError("eval method memory lacks its raw text or embedding")
        if _text_fingerprint(text) != (memory.text_sha256, memory.text_utf8_bytes):
            raise ValueError("eval method memory text fingerprint drifted")
        view = MethodMemory(
            source=source,
            memory_id=memory.memory_id,
            namespace_id=memory.namespace_id,
            chronology=memory.chronology,
            source_date=memory.source_date,
            text=text,
            text_sha256=memory.text_sha256,
            vector_index=vector_index,
        )
        base_memories.append(view)
        lifecycle_memories.append(
            LifecycleMemory(
                memory=view,
                state=memory.state,
                released_distractor=memory.released_distractor,
            )
        )
    base_queries = []
    intent_queries = []
    for query in sorted(queries, key=lambda row: row.query_id):
        identity = "query", query.query_text_sha256
        text = raw_texts.get(identity)
        vector_index = vector_indices.get(identity)
        if text is None or vector_index is None:
            raise ValueError("eval method query lacks its raw text or embedding")
        if _text_fingerprint(text) != (
            query.query_text_sha256,
            query.query_text_utf8_bytes,
        ):
            raise ValueError("eval method query text fingerprint drifted")
        view = MethodQuery(
            source=source,
            query_id=query.query_id,
            namespace_id=query.namespace_id,
            cutoff_date=query.cutoff_date,
            text=text,
            text_sha256=query.query_text_sha256,
            vector_index=vector_index,
        )
        base_queries.append(view)
        intent_queries.append(IntentQuery(query=view, query_intent=query.query_intent))
    return MethodViews(
        source=source,
        memories=tuple(base_memories),
        lifecycle_memories=tuple(lifecycle_memories),
        queries=tuple(base_queries),
        intent_queries=tuple(intent_queries),
    )


PUBLIC_TO_FROZEN = {
    "global_bm25": FLAT_BM25,
    "global_dense": FLAT_DENSE,
    "global_bm25_dense_rrf": FLAT_RRF,
    "global_recency_dense": RECENCY_DENSE,
    "namespace_dense": NAMESPACE_DENSE,
    "query_agnostic_current_only": NAMESPACE_CURRENT_ONLY,
    "released_intent_lifecycle_upper_bound": NAMESPACE_SOURCE_INTENT_LIFECYCLE,
    "threshold_router": NCR_THRESHOLD,
    "cluster_router": NCR_A5,
}


def load_frozen_settings(protocol_path: Path) -> tuple[NaturalSetting, ...]:
    """Translate public names without changing frozen setting/config identities."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    settings = []
    aliases = {
        "recency_gamma": "gamma",
        "cluster_beta": "beta",
        "cluster_gamma": "gamma",
        "cluster_tau": "tau",
        "cluster_tau_new": "tau_new",
    }
    for row in protocol["settings"]:
        arm = PUBLIC_TO_FROZEN[row["arm"]]
        params = {aliases.get(k, k): v for k, v in row.items() if k not in {"arm", "setting_id"}}
        if arm in {FLAT_BM25, FLAT_RRF}:
            params.setdefault("rrf_k", 60)
        setting = NaturalSetting.create(arm, params)
        if setting.setting_id != row["setting_id"]:
            raise ValueError("frozen setting identity differs from public protocol")
        settings.append(setting)
    if tuple(s.arm for s in settings) != ARMS:
        raise ValueError("the public protocol must contain the nine frozen arms in order")
    return tuple(settings)
