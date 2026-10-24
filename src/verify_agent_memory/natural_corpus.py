"""Construct natural retrieval corpora with the frozen source mapping and split."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verify_agent_memory.natural_memops import (
    EvidenceState,
    QueryIntent,
    map_source_span,
    query_intent,
    replay_operation_states,
)
from verify_agent_memory.natural_rhelm import iter_rhelm_question_resolutions
from verify_agent_memory.natural_source_indices import (
    injected_message_ids,
    load_memops_message_index,
    load_rhelm_qa_index,
    load_rhelm_turn_index,
)

# Preserve the original mapping identifier because it is also the split hash salt.
MAPPING_VERSION = "stage3-admissibility-suite-v1"
SPLIT_SALT = MAPPING_VERSION
_SOURCES = {"rhelm", "memops"}
_SPLITS = {"dev", "eval"}
_INTENTS = {item.value for item in QueryIntent}
_STATES = {item.value for item in EvidenceState}
_LIFECYCLE_POLICIES = {
    "namespace_dense",
    "namespace_current_only",
    "namespace_source_intent_lifecycle",
}


def assign_group_splits(
    groups_by_source: Mapping[str, Iterable[str]],
) -> dict[str, dict[str, str]]:
    """Apply the preregistered group-hash split independently by source."""
    expected = {"rhelm", "memops"}
    if set(groups_by_source) != expected:
        raise ValueError("group universe must contain exactly rhelm and memops")
    output: dict[str, dict[str, str]] = {}
    for source in sorted(expected):
        groups = sorted(set(groups_by_source[source]))
        if not groups or any(not isinstance(group, str) or not group for group in groups):
            raise ValueError(f"{source} group universe is invalid")
        ranked = sorted(
            groups,
            key=lambda group: (
                hashlib.sha256(f"{SPLIT_SALT}|{source}|{group}".encode()).hexdigest(),
                group,
            ),
        )
        if source == "rhelm":
            if len(ranked) < 4:
                raise ValueError("RHELM split requires at least four groups")
            dev_count = 3
        else:
            dev_count = max(1, len(ranked) // 5)
        dev = set(ranked[:dev_count])
        output[source] = {group: ("dev" if group in dev else "eval") for group in groups}
    return output


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _text_fingerprint(text: str) -> tuple[str, int]:
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a nonempty string")
    encoded = text.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class NaturalMemory:
    """One content-free memory descriptor from a natural source corpus."""

    source: str
    memory_id: str
    namespace_id: str
    split: str
    chronology: str
    source_date: str | None
    text_sha256: str
    text_utf8_bytes: int
    state: str = EvidenceState.CURRENT.value
    released_distractor: bool = False

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError(f"unexpected source {self.source!r}")
        if self.split not in _SPLITS:
            raise ValueError(f"unexpected split {self.split!r}")
        if not all(
            isinstance(value, str) and value
            for value in (self.memory_id, self.namespace_id, self.chronology)
        ):
            raise ValueError("memory identity fields must be nonempty")
        if not _valid_sha256(self.text_sha256) or self.text_utf8_bytes < 1:
            raise ValueError("memory text fingerprint is invalid")
        if self.state not in _STATES:
            raise ValueError(f"unexpected lifecycle state {self.state!r}")
        if self.source == "rhelm" and not self.source_date:
            raise ValueError("RHELM memories require a source date")
        if self.source == "memops" and self.source_date is not None:
            raise ValueError("MemOps memories do not have a cross-case source date")

    @classmethod
    def from_text(
        cls,
        *,
        source: str,
        memory_id: str,
        namespace_id: str,
        split: str,
        chronology: str,
        text: str,
        source_date: str | None = None,
        state: str = EvidenceState.CURRENT.value,
        released_distractor: bool = False,
    ) -> NaturalMemory:
        """Create a descriptor while discarding source text after hashing."""
        text_sha256, text_utf8_bytes = _text_fingerprint(text)
        return cls(
            source=source,
            memory_id=memory_id,
            namespace_id=namespace_id,
            split=split,
            chronology=chronology,
            source_date=source_date,
            text_sha256=text_sha256,
            text_utf8_bytes=text_utf8_bytes,
            state=state,
            released_distractor=released_distractor,
        )


@dataclass(frozen=True)
class NaturalQuery:
    """One content-free query descriptor with scorer-only required evidence."""

    source: str
    query_id: str
    namespace_id: str
    split: str
    query_text_sha256: str
    query_text_utf8_bytes: int
    query_intent: str
    required_memory_ids: tuple[str, ...]
    cutoff_date: str | None = None
    evaluation_setting: str | None = None

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError(f"unexpected source {self.source!r}")
        if self.split not in _SPLITS:
            raise ValueError(f"unexpected split {self.split!r}")
        if not all(
            isinstance(value, str) and value for value in (self.query_id, self.namespace_id)
        ):
            raise ValueError("query identity fields must be nonempty")
        if not _valid_sha256(self.query_text_sha256) or self.query_text_utf8_bytes < 1:
            raise ValueError("query text fingerprint is invalid")
        if self.query_intent not in _INTENTS:
            raise ValueError(f"unexpected query intent {self.query_intent!r}")
        if not self.required_memory_ids or any(
            not isinstance(memory_id, str) or not memory_id
            for memory_id in self.required_memory_ids
        ):
            raise ValueError("query requires at least one evidence memory")
        if len(set(self.required_memory_ids)) != len(self.required_memory_ids):
            raise ValueError("query required-memory IDs must be unique")
        if self.source == "rhelm" and not self.cutoff_date:
            raise ValueError("RHELM queries require a strict cutoff date")
        if self.source == "memops" and self.cutoff_date is not None:
            raise ValueError("MemOps queries do not use a cross-case date cutoff")

    @classmethod
    def from_text(
        cls,
        *,
        source: str,
        query_id: str,
        namespace_id: str,
        split: str,
        query_text: str,
        query_intent: str,
        required_memory_ids: Iterable[str],
        cutoff_date: str | None = None,
        evaluation_setting: str | None = None,
    ) -> NaturalQuery:
        """Create a descriptor while discarding query text after hashing."""
        text_sha256, text_utf8_bytes = _text_fingerprint(query_text)
        return cls(
            source=source,
            query_id=query_id,
            namespace_id=namespace_id,
            split=split,
            query_text_sha256=text_sha256,
            query_text_utf8_bytes=text_utf8_bytes,
            query_intent=query_intent,
            required_memory_ids=tuple(required_memory_ids),
            cutoff_date=cutoff_date,
            evaluation_setting=evaluation_setting,
        )


class NaturalCorpus:
    """Validated content-free source corpus with exact candidate-set semantics."""

    def __init__(
        self,
        source: str,
        memories: Iterable[NaturalMemory],
        queries: Iterable[NaturalQuery],
    ) -> None:
        if source not in _SOURCES:
            raise ValueError(f"unexpected source {source!r}")
        self.source = source
        self.memories = tuple(sorted(memories, key=lambda memory: memory.memory_id))
        self.queries = tuple(sorted(queries, key=lambda query: query.query_id))
        if not self.memories or not self.queries:
            raise ValueError("natural corpus requires memories and queries")
        if any(memory.source != source for memory in self.memories):
            raise ValueError("memory source differs from corpus source")
        if any(query.source != source for query in self.queries):
            raise ValueError("query source differs from corpus source")

        memory_by_id = {memory.memory_id: memory for memory in self.memories}
        if len(memory_by_id) != len(self.memories):
            raise ValueError("natural corpus memory IDs must be unique")
        if len({query.query_id for query in self.queries}) != len(self.queries):
            raise ValueError("natural corpus query IDs must be unique")

        by_split: defaultdict[str, list[NaturalMemory]] = defaultdict(list)
        by_split_namespace: defaultdict[tuple[str, str], list[NaturalMemory]] = defaultdict(list)
        groups_by_split: defaultdict[str, set[str]] = defaultdict(set)
        for memory in self.memories:
            by_split[memory.split].append(memory)
            by_split_namespace[(memory.split, memory.namespace_id)].append(memory)
            groups_by_split[memory.split].add(memory.namespace_id)
        self._by_split = {
            split: tuple(sorted(rows, key=self._chronology_key)) for split, rows in by_split.items()
        }
        self._by_split_namespace = {
            key: tuple(sorted(rows, key=self._chronology_key))
            for key, rows in by_split_namespace.items()
        }
        self.groups_by_split = {
            split: tuple(sorted(groups)) for split, groups in groups_by_split.items()
        }
        self._dates_by_split = {
            split: tuple(memory.source_date or "" for memory in rows)
            for split, rows in self._by_split.items()
        }
        self._dates_by_split_namespace = {
            key: tuple(memory.source_date or "" for memory in rows)
            for key, rows in self._by_split_namespace.items()
        }

        for query in self.queries:
            for memory_id in query.required_memory_ids:
                memory = memory_by_id.get(memory_id)
                if (
                    memory is None
                    or memory.split != query.split
                    or memory.namespace_id != query.namespace_id
                    or (
                        source == "rhelm"
                        and (
                            memory.source_date is None
                            or query.cutoff_date is None
                            or memory.source_date >= query.cutoff_date
                        )
                    )
                ):
                    raise ValueError(
                        f"required evidence is absent from the query candidate set: "
                        f"{query.query_id!r}/{memory_id!r}"
                    )

    @staticmethod
    def _chronology_key(memory: NaturalMemory) -> tuple[str, str, str]:
        return memory.source_date or "", memory.chronology, memory.memory_id

    def _candidate_rows(
        self,
        query: NaturalQuery,
        *,
        namespace_only: bool,
    ) -> tuple[NaturalMemory, ...]:
        if query.source != self.source:
            raise ValueError("query source differs from corpus source")
        if namespace_only:
            key = (query.split, query.namespace_id)
            rows = self._by_split_namespace.get(key, ())
            dates = self._dates_by_split_namespace.get(key, ())
        else:
            rows = self._by_split.get(query.split, ())
            dates = self._dates_by_split.get(query.split, ())
        if self.source == "rhelm":
            if query.cutoff_date is None:
                raise ValueError("RHELM query lacks cutoff date")
            return rows[: bisect_left(dates, query.cutoff_date)]
        return rows

    def candidate_ids(
        self,
        query: NaturalQuery,
        *,
        namespace_only: bool,
    ) -> tuple[str, ...]:
        """Return the exact full candidate set in deterministic source order."""
        return tuple(
            memory.memory_id
            for memory in self._candidate_rows(query, namespace_only=namespace_only)
        )

    def candidate_count(
        self,
        query: NaturalQuery,
        *,
        namespace_only: bool,
    ) -> int:
        """Return candidate count without materializing per-query ID lists."""
        if query.source != self.source:
            raise ValueError("query source differs from corpus source")
        if namespace_only:
            key = (query.split, query.namespace_id)
            rows = self._by_split_namespace.get(key, ())
            dates = self._dates_by_split_namespace.get(key, ())
        else:
            rows = self._by_split.get(query.split, ())
            dates = self._dates_by_split.get(query.split, ())
        if self.source == "rhelm":
            if query.cutoff_date is None:
                raise ValueError("RHELM query lacks cutoff date")
            return bisect_left(dates, query.cutoff_date)
        return len(rows)


def candidate_label(query: NaturalQuery, memory: NaturalMemory) -> dict[str, object]:
    """Return scorer-only typed labels without treating source non-gold as negative."""
    if query.source != memory.source or query.split != memory.split:
        raise ValueError("query and memory must share a source split")
    relevance = "required" if memory.memory_id in query.required_memory_ids else "unresolved"
    same_namespace = memory.namespace_id == query.namespace_id
    scope = "allowed" if same_namespace else "disallowed"
    if relevance == "required" and same_namespace and not memory.released_distractor:
        prohibited: bool | str = False
    elif memory.state == EvidenceState.UNCERTAIN.value:
        prohibited = "uncertain"
    else:
        prohibited = (
            not same_namespace
            or memory.released_distractor
            or (
                query.query_intent == QueryIntent.CURRENT_STATE.value
                and memory.state
                in {
                    EvidenceState.STALE.value,
                    EvidenceState.SUPERSEDED.value,
                }
            )
        )
    return {
        "relevance": relevance,
        "scope": scope,
        "state": memory.state,
        "prohibited": prohibited,
        "usable_evidence": (relevance == "required" and scope == "allowed" and prohibited is False),
    }


def lifecycle_allows(
    query: NaturalQuery,
    memory: NaturalMemory,
    policy: str,
) -> bool:
    """Apply only the named method-visible lifecycle policy."""
    if policy not in _LIFECYCLE_POLICIES:
        raise ValueError(f"unexpected lifecycle policy {policy!r}")
    if query.source != memory.source or query.split != memory.split:
        raise ValueError("query and memory must share a source split")
    if memory.namespace_id != query.namespace_id:
        return False
    if policy == "namespace_dense":
        return True
    if policy == "namespace_current_only":
        return memory.state not in {
            EvidenceState.STALE.value,
            EvidenceState.SUPERSEDED.value,
        }
    if query.query_intent == QueryIntent.HISTORY.value:
        return True
    return (
        memory.state
        not in {
            EvidenceState.STALE.value,
            EvidenceState.SUPERSEDED.value,
        }
        and not memory.released_distractor
    )


def build_rhelm_natural_corpus(
    data_root: Path,
    split_by_namespace: Mapping[str, str],
) -> NaturalCorpus:
    """Build the full conversation-only RHELM content-free corpus descriptor."""
    qa_index = load_rhelm_qa_index(data_root)
    turn_index = load_rhelm_turn_index(data_root)
    memories = []
    for turn in turn_index.values():
        split = split_by_namespace.get(turn.namespace_id)
        if split not in _SPLITS:
            raise ValueError(f"RHELM namespace {turn.namespace_id!r} lacks a split")
        memories.append(
            NaturalMemory.from_text(
                source="rhelm",
                memory_id=turn.memory_id,
                namespace_id=turn.namespace_id,
                split=split,
                chronology=f"{turn.date}/turn-{turn.turn}",
                source_date=turn.date,
                text=turn.text,
            )
        )

    queries = []
    for resolution in iter_rhelm_question_resolutions(data_root):
        if not (resolution.conversation_only and resolution.all_references_unique):
            continue
        row = qa_index[(resolution.namespace_id, resolution.query_id)]
        required_ids = tuple(
            dict.fromkeys(
                canonical_id
                for evidence in resolution.evidence
                for canonical_id in evidence.canonical_ids
            )
        )
        queries.append(
            NaturalQuery.from_text(
                source="rhelm",
                query_id=f"{resolution.namespace_id}/{resolution.query_id}",
                namespace_id=resolution.namespace_id,
                split=split_by_namespace[resolution.namespace_id],
                query_text=_string(row.get("question"), "RHELM question"),
                query_intent=QueryIntent.CURRENT_STATE.value,
                required_memory_ids=required_ids,
                cutoff_date=resolution.question_date,
            )
        )
    return NaturalCorpus("rhelm", memories, queries)


def _memops_distractor_ids(
    case_id: str,
    injected_case: Mapping[str, object],
) -> set[str]:
    metadata = _mapping(injected_case.get("injection_metadata"), "injection metadata")
    rows = metadata.get("distractor_insertions", ())
    return injected_message_ids(
        case_id,
        _sequence(rows, "distractor insertions"),
        count_key="distractor_message_count",
    )


def build_memops_natural_corpus(
    memops_root: Path,
    split_by_namespace: Mapping[str, str],
) -> NaturalCorpus:
    """Build all exact-mapped MemOps probes and full split-level message corpora."""
    source_root = memops_root / "generated_result" / "2-evidence_conversation"
    injected_root = memops_root / "generated_result" / "4-inject_evidence_with_distractors"
    memories = []
    queries = []
    for injected_path in sorted(injected_root.glob("*.json")):
        source_path = source_root / injected_path.name
        if not source_path.is_file():
            raise ValueError(f"MemOps source case is absent for {injected_path.name!r}")
        case_id = injected_path.stem
        namespace_id = case_id.split("_", maxsplit=1)[0]
        split = split_by_namespace.get(namespace_id)
        if split not in _SPLITS:
            raise ValueError(f"MemOps namespace {namespace_id!r} lacks a split")
        source_case = _read_json(source_path)
        injected_case = _read_json(injected_path)
        states = replay_operation_states(case_id, source_case, injected_case)
        distractor_ids = _memops_distractor_ids(case_id, injected_case)
        messages = load_memops_message_index(case_id, injected_case)
        for memory_id, record in messages.items():
            memories.append(
                NaturalMemory.from_text(
                    source="memops",
                    memory_id=memory_id,
                    namespace_id=namespace_id,
                    split=split,
                    chronology=f"{case_id}/{record['chronology']}",
                    text=_string(record.get("text"), "MemOps memory text"),
                    state=states.get(memory_id, EvidenceState.CURRENT).value,
                    released_distractor=memory_id in distractor_ids,
                )
            )

        for raw_answer in _sequence(injected_case.get("answer"), "MemOps answers"):
            answer = _mapping(raw_answer, "MemOps answer")
            evaluation_setting = _string(
                answer.get("evaluation_setting"),
                "MemOps evaluation setting",
            )
            evaluation_type = _string(
                answer.get("evaluation_type"),
                "MemOps evaluation type",
            )
            question_pair_id = _string(
                answer.get("question_pair_id"),
                "MemOps question pair id",
            )
            required_ids = tuple(
                dict.fromkeys(
                    map_source_span(
                        case_id,
                        source_case,
                        injected_case,
                        _mapping(raw_span, "MemOps gold span"),
                    ).canonical_id
                    for raw_span in _sequence(
                        answer.get("gold_provenance"),
                        "MemOps gold provenance",
                    )
                )
            )
            queries.append(
                NaturalQuery.from_text(
                    source="memops",
                    query_id=f"{case_id}/{question_pair_id}/{evaluation_setting}",
                    namespace_id=namespace_id,
                    split=split,
                    query_text=_string(answer.get("question"), "MemOps question"),
                    query_intent=query_intent(evaluation_type).value,
                    required_memory_ids=required_ids,
                    evaluation_setting=evaluation_setting,
                )
            )
    return NaturalCorpus("memops", memories, queries)


def build_natural_corpora(
    *,
    rhelm_data_root: Path,
    memops_root: Path,
) -> tuple[dict[str, NaturalCorpus], dict[str, dict[str, str]]]:
    """Build both public/dev corpora under the frozen group-hash split."""
    rhelm_groups = {
        resolution.namespace_id for resolution in iter_rhelm_question_resolutions(rhelm_data_root)
    }
    injected_root = memops_root / "generated_result" / "4-inject_evidence_with_distractors"
    memops_groups = {path.stem.split("_", maxsplit=1)[0] for path in injected_root.glob("*.json")}
    split_map = assign_group_splits(
        {
            "rhelm": rhelm_groups,
            "memops": memops_groups,
        }
    )
    return (
        {
            "rhelm": build_rhelm_natural_corpus(
                rhelm_data_root,
                split_map["rhelm"],
            ),
            "memops": build_memops_natural_corpus(
                memops_root,
                split_map["memops"],
            ),
        },
        split_map,
    )
