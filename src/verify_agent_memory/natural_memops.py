"""Exact mapping of released MemOps evidence spans and operation states."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EvidenceState(StrEnum):
    """Lifecycle state of one mapped MemOps message."""

    CURRENT = "current"
    STALE = "stale"
    SUPERSEDED = "superseded"
    UNCERTAIN = "uncertain"


class QueryIntent(StrEnum):
    """Whether a query asks for current state or an explicit history."""

    CURRENT_STATE = "current_state"
    HISTORY = "history"


@dataclass(frozen=True)
class MappedMessage:
    """One source message mapped to its immutable injected location."""

    canonical_id: str
    namespace_id: str
    role: str
    content: str
    source_segment_index: int
    source_turn_index: int
    destination_segment_index: int
    destination_item_index: int


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _one_by_integer(
    rows: Sequence[object],
    key: str,
    value: int,
    label: str,
) -> Mapping[str, Any]:
    matches = [
        _mapping(row, label) for row in rows if isinstance(row, Mapping) and row.get(key) == value
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} lookup for {key}={value} found {len(matches)} rows")
    return matches[0]


def _normalized(value: str) -> str:
    return " ".join(value.split())


def _namespace(case_id: str) -> str:
    namespace_id = case_id.split("_", maxsplit=1)[0]
    if not re.fullmatch(r"[A-Za-z]\d+", namespace_id):
        raise ValueError(f"invalid MemOps case id {case_id!r}")
    return namespace_id


def _case_id(case_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z]\d+_[a-z_]+", case_id):
        raise ValueError(f"invalid MemOps case id {case_id!r}")
    return case_id


def map_source_span(
    case_id: str,
    source_case: Mapping[str, object],
    injected_case: Mapping[str, object],
    span: Mapping[str, object],
) -> MappedMessage:
    """Map one source span to the exact injected message and verify its content."""
    case_id = _case_id(case_id)
    namespace_id = _namespace(case_id)
    source_segment_index = _positive_integer(span.get("segment_index"), "span.segment_index")
    source_turn_index = _positive_integer(span.get("turn_index"), "span.turn_index")
    source_segment = _one_by_integer(
        _sequence(source_case.get("conversations"), "source conversations"),
        "segment_index",
        source_segment_index,
        "source segment",
    )
    source_dialogue = _sequence(source_segment.get("dialogue"), "source dialogue")
    if source_turn_index > len(source_dialogue):
        raise ValueError("source turn is absent")
    source_message = _mapping(source_dialogue[source_turn_index - 1], "source message")

    metadata = _mapping(injected_case.get("injection_metadata"), "injection metadata")
    insertion = _one_by_integer(
        _sequence(metadata.get("insertions"), "injection insertions"),
        "evidence_segment_index",
        source_segment_index,
        "evidence insertion",
    )
    destination_segment_index = _positive_integer(
        insertion.get("conversation_index"),
        "insertion.conversation_index",
    )
    insertion_index = _nonnegative_integer(
        insertion.get("insertion_index"),
        "insertion.insertion_index",
    )
    message_count = _positive_integer(
        insertion.get("evidence_message_count"),
        "insertion.evidence_message_count",
    )
    if message_count != len(source_dialogue):
        raise ValueError("evidence message count differs from source dialogue")

    destination_segment = _one_by_integer(
        _sequence(injected_case.get("conversations"), "injected conversations"),
        "segment_index",
        destination_segment_index,
        "injected segment",
    )
    if destination_segment.get("evidence_segment_index") != source_segment_index:
        raise ValueError("injected segment points to a different source segment")
    if destination_segment.get("insertion_index") != insertion_index:
        raise ValueError("injected segment and metadata insertion indices differ")
    destination_dialogue = _sequence(destination_segment.get("dialogue"), "injected dialogue")
    destination_offset = insertion_index + source_turn_index - 1
    if destination_offset >= len(destination_dialogue):
        raise ValueError("mapped destination message is absent")
    destination_message = _mapping(
        destination_dialogue[destination_offset],
        "injected message",
    )

    role = _nonempty_string(source_message.get("role"), "source message role")
    content = _nonempty_string(source_message.get("content"), "source message content")
    if destination_message.get("role") != role or destination_message.get("content") != content:
        raise ValueError("source and injected message content mismatch")
    quote = span.get("quote")
    if quote is not None:
        quote = _nonempty_string(quote, "span.quote")
        if _normalized(quote) not in _normalized(content):
            raise ValueError("span quote is absent from mapped message")

    destination_item_index = destination_offset + 1
    return MappedMessage(
        canonical_id=(
            f"{case_id}/message/segment-{destination_segment_index}/item-{destination_item_index}"
        ),
        namespace_id=namespace_id,
        role=role,
        content=content,
        source_segment_index=source_segment_index,
        source_turn_index=source_turn_index,
        destination_segment_index=destination_segment_index,
        destination_item_index=destination_item_index,
    )


def _operation_messages(
    case_id: str,
    source_case: Mapping[str, object],
    injected_case: Mapping[str, object],
    operation: Mapping[str, object],
) -> tuple[MappedMessage, ...]:
    spans = list(_sequence(operation.get("evidence_spans"), "operation.evidence_spans"))
    trigger = _mapping(operation.get("trigger_span"), "operation.trigger_span")
    if not spans or not any(
        isinstance(span, Mapping)
        and span.get("segment_index") == trigger.get("segment_index")
        and span.get("turn_index") == trigger.get("turn_index")
        for span in spans
    ):
        spans.append(trigger)
    messages: dict[str, MappedMessage] = {}
    for raw_span in spans:
        mapped = map_source_span(
            case_id,
            source_case,
            injected_case,
            _mapping(raw_span, "operation span"),
        )
        messages[mapped.canonical_id] = mapped
    return tuple(messages.values())


def _merge_state(
    states: dict[str, EvidenceState],
    canonical_id: str,
    state: EvidenceState,
) -> None:
    previous = states.get(canonical_id)
    if previous is None or previous is state:
        states[canonical_id] = state
    elif EvidenceState.UNCERTAIN in {previous, state}:
        states[canonical_id] = EvidenceState.UNCERTAIN
    else:
        states[canonical_id] = EvidenceState.UNCERTAIN


def replay_operation_states(
    case_id: str,
    source_case: Mapping[str, object],
    injected_case: Mapping[str, object],
) -> dict[str, EvidenceState]:
    """Replay confirmed operations and return conservative message-level states."""
    states: dict[str, EvidenceState] = {}
    active_by_target: dict[str, list[str]] = {}
    history_by_target: dict[str, list[str]] = {}

    for raw_operation in _sequence(source_case.get("operations"), "operations"):
        operation = _mapping(raw_operation, "operation")
        operation_type = _nonempty_string(operation.get("type"), "operation.type").lower()
        validity = _nonempty_string(operation.get("validity"), "operation.validity").lower()
        target = _mapping(operation.get("target"), "operation.target")
        target_id = _nonempty_string(target.get("target_id"), "operation.target.target_id")
        message_ids = [
            message.canonical_id
            for message in _operation_messages(case_id, source_case, injected_case, operation)
        ]

        if validity == "tentative":
            for canonical_id in message_ids:
                _merge_state(states, canonical_id, EvidenceState.UNCERTAIN)
            continue
        if validity == "retracted":
            for canonical_id in message_ids:
                _merge_state(states, canonical_id, EvidenceState.STALE)
            continue
        if validity != "confirmed":
            raise ValueError(f"unexpected operation validity {validity!r}")

        if operation_type == "forget":
            for canonical_id in history_by_target.get(target_id, ()):
                states[canonical_id] = EvidenceState.STALE
            for canonical_id in message_ids:
                _merge_state(states, canonical_id, EvidenceState.CURRENT)
            history_by_target.setdefault(target_id, []).extend(message_ids)
            active_by_target[target_id] = []
            continue
        if operation_type == "update":
            for canonical_id in active_by_target.get(target_id, ()):
                states[canonical_id] = EvidenceState.SUPERSEDED
            for canonical_id in message_ids:
                _merge_state(states, canonical_id, EvidenceState.CURRENT)
            history_by_target.setdefault(target_id, []).extend(message_ids)
            active_by_target[target_id] = list(message_ids)
            continue
        if operation_type not in {"remember", "reflect"}:
            raise ValueError(f"unexpected operation type {operation_type!r}")
        for canonical_id in message_ids:
            _merge_state(states, canonical_id, EvidenceState.CURRENT)
        history_by_target.setdefault(target_id, []).extend(message_ids)
        active_by_target.setdefault(target_id, []).extend(message_ids)

    return states


def query_intent(evaluation_type: str) -> QueryIntent:
    """Map released MemOps evaluation types to a frozen retrieval intent."""
    if evaluation_type in {"OperationTrace", "StateTrajectory"}:
        return QueryIntent.HISTORY
    if evaluation_type in {
        "TargetBinding",
        "StateTransition",
        "CandidateDisambiguation",
        "OperationApplication",
    }:
        return QueryIntent.CURRENT_STATE
    raise ValueError(f"unexpected MemOps evaluation type {evaluation_type!r}")


def label_memops_message(
    *,
    relevance: str,
    state: EvidenceState,
    intent: QueryIntent,
    same_namespace: bool,
    released_distractor: bool,
) -> dict[str, object]:
    """Apply query-conditioned admissibility to one mapped MemOps message."""
    if relevance not in {"required", "supportive", "not_useful", "uncertain"}:
        raise ValueError(f"unexpected relevance {relevance!r}")
    scope = "allowed" if same_namespace else "disallowed"
    source_required = relevance in {"required", "supportive"}
    if source_required and same_namespace and not released_distractor:
        prohibited: bool | str = False
    elif state is EvidenceState.UNCERTAIN:
        prohibited: bool | str = "uncertain"
    else:
        prohibited = (
            not same_namespace
            or released_distractor
            or (
                intent is QueryIntent.CURRENT_STATE
                and state in {EvidenceState.STALE, EvidenceState.SUPERSEDED}
            )
        )
    usable = source_required and scope == "allowed" and prohibited is False
    return {
        "relevance": relevance,
        "scope": scope,
        "state": state.value,
        "prohibited": prohibited,
        "usable_evidence": usable,
    }
