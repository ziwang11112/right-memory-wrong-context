"""Read canonical records from user-fetched public benchmark checkouts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RhelmTurn:
    memory_id: str
    namespace_id: str
    date: str
    turn: int
    text: str


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


def _json(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            rows.append(_mapping(json.loads(line), f"{path}:{line_number}"))
    return rows


def load_rhelm_qa_index(data_root: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Load released RHELM questions keyed by namespace and query ID."""
    index = {}
    for path in sorted((data_root / "QA_final").glob("*.jsonl")):
        stem = path.stem
        namespace_id = stem[len("low_score_qa_") : -len("_all_validated")]
        for row in _jsonl(path):
            query_id = _string(row.get("id"), "RHELM query id")
            key = (namespace_id, query_id)
            if key in index:
                raise ValueError(f"duplicate RHELM query {key!r}")
            index[key] = row
    return index


def load_rhelm_turn_index(data_root: Path) -> dict[str, RhelmTurn]:
    """Load canonical conversation-turn memories from a released RHELM tree."""
    turns: dict[str, RhelmTurn] = {}
    for namespace_dir in sorted((data_root / "conversations").iterdir()):
        if not namespace_dir.is_dir():
            continue
        namespace_id = namespace_dir.name
        for path in sorted(namespace_dir.glob("conversation_*.json")):
            conversation = _json(path)
            date = _string(conversation.get("date"), "conversation date")
            for raw_turn in _sequence(conversation.get("conversation"), "conversation turns"):
                turn = _mapping(raw_turn, "conversation turn")
                turn_number = turn.get("turn")
                if not isinstance(turn_number, int) or isinstance(turn_number, bool):
                    raise ValueError("RHELM turn number must be an integer")
                memory_id = f"{namespace_id}/conversation/{date}/turn-{turn_number}"
                if memory_id in turns:
                    raise ValueError(f"duplicate RHELM memory {memory_id!r}")
                user = _string(turn.get("user"), "RHELM user text")
                assistant = _string(turn.get("assistant"), "RHELM assistant text")
                turns[memory_id] = RhelmTurn(
                    memory_id,
                    namespace_id,
                    date,
                    turn_number,
                    f"User: {user}\nAssistant: {assistant}",
                )
    return turns


def load_memops_message_index(
    case_id: str,
    injected_case: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Load canonical dialogue-message memories from one injected MemOps case."""
    namespace_id = case_id.split("_", maxsplit=1)[0]
    messages = {}
    for raw_segment in _sequence(injected_case.get("conversations"), "MemOps conversations"):
        segment = _mapping(raw_segment, "MemOps segment")
        segment_index = segment.get("segment_index")
        if not isinstance(segment_index, int) or isinstance(segment_index, bool):
            raise ValueError("MemOps segment index must be an integer")
        for item_index, raw_message in enumerate(
            _sequence(segment.get("dialogue"), "MemOps dialogue"),
            start=1,
        ):
            message = _mapping(raw_message, "MemOps message")
            memory_id = f"{case_id}/message/segment-{segment_index}/item-{item_index}"
            if memory_id in messages:
                raise ValueError(f"duplicate MemOps memory {memory_id!r}")
            messages[memory_id] = {
                "memory_id": memory_id,
                "namespace_id": namespace_id,
                "chronology": f"segment-{segment_index}/item-{item_index}",
                "text": (
                    f"{_string(message.get('role'), 'MemOps message role').title()}: "
                    f"{_string(message.get('content'), 'MemOps message content')}"
                ),
            }
    return messages


def injected_message_ids(
    case_id: str,
    rows: Sequence[object],
    *,
    count_key: str,
) -> set[str]:
    """Expand released MemOps insertion metadata into canonical message IDs."""
    memory_ids = set()
    for raw_row in rows:
        row = _mapping(raw_row, "injection row")
        segment_index = row.get("conversation_index")
        insertion_index = row.get("insertion_index")
        count = row.get(count_key)
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (segment_index, insertion_index, count)
        ):
            raise ValueError("injection range fields must be integers")
        for offset in range(count):
            memory_ids.add(
                f"{case_id}/message/segment-{segment_index}/item-{insertion_index + offset + 1}"
            )
    return memory_ids
