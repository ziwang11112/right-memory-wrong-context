"""Resolve released RHELM evidence references to canonical source records."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

CONVERSATION_REFERENCE = re.compile(r"^(\d{4}-\d{2}-\d{2}):([1-9]\d*)$")
EMAIL_REFERENCE = re.compile(r"^Emails_(\d{4}-\d{2}-\d{2})(?::(.*))?$")
EMAIL_FILE = re.compile(r"^\d+_email_(\d{4})_(\d{2})_(\d{2})\.txt$")
NUMBERED_EMAIL = re.compile(r"^email\s+([1-9]\d*)$", re.IGNORECASE)
MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
QA_PREFIX = "low_score_qa_"
QA_SUFFIX = "_all_validated"


class ResolutionStatus(StrEnum):
    """Outcome of resolving one released RHELM evidence reference."""

    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    FUTURE = "future"
    INVALID = "invalid"


@dataclass(frozen=True)
class EvidenceResolution:
    """Content-free resolution result for one released evidence reference."""

    modality: str
    status: ResolutionStatus
    candidate_count: int
    canonical_ids: tuple[str, ...] = ()
    source_date: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class QuestionResolution:
    """Resolution metadata needed for deterministic audit selection."""

    namespace_id: str
    query_id: str
    question_date: str
    question_type: str
    evidence: tuple[EvidenceResolution, ...]

    @property
    def all_references_unique(self) -> bool:
        return bool(self.evidence) and all(
            item.status is ResolutionStatus.UNIQUE for item in self.evidence
        )

    @property
    def conversation_only(self) -> bool:
        return bool(self.evidence) and all(
            item.modality == "conversation" for item in self.evidence
        )


class _HeadingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture_depth = 0
        self._parts: list[str] = []
        self.headings: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"h1", "h2", "h3", "h4", "h5", "h6", "caption"}:
            if self._capture_depth == 0:
                self._parts = []
            self._capture_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() not in {"h1", "h2", "h3", "h4", "h5", "h6", "caption"}:
            return
        if self._capture_depth:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                heading = " ".join(self._parts).strip()
                if heading:
                    self.headings.append(heading)
                self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_depth:
            self._parts.append(data)


def _normalized(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", folded, flags=re.UNICODE).split())


def _safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe {label}")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _conversation_resolution(
    data_root: Path,
    namespace_id: str,
    date: str,
    turn_number: int,
) -> EvidenceResolution:
    path = (
        data_root / "conversations" / namespace_id / f"conversation_{date.replace('-', '_')}.json"
    )
    if not path.is_file():
        return EvidenceResolution("conversation", ResolutionStatus.MISSING, 0, source_date=date)
    conversation = _json(path).get("conversation")
    if not isinstance(conversation, list):
        return EvidenceResolution(
            "conversation",
            ResolutionStatus.INVALID,
            0,
            source_date=date,
            reason="conversation_not_list",
        )
    matches = [
        turn for turn in conversation if isinstance(turn, dict) and turn.get("turn") == turn_number
    ]
    canonical = f"{namespace_id}/conversation/{date}/turn-{turn_number}"
    if len(matches) == 1:
        return EvidenceResolution(
            "conversation",
            ResolutionStatus.UNIQUE,
            1,
            (canonical,),
            date,
        )
    status = ResolutionStatus.MISSING if not matches else ResolutionStatus.AMBIGUOUS
    return EvidenceResolution(
        "conversation",
        status,
        len(matches),
        (canonical,) * len(matches),
        date,
    )


def _email_blocks(raw_text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\r?\n=+\r?\n", raw_text) if block.strip()]


def _email_resolution(
    data_root: Path,
    namespace_id: str,
    date: str,
    qualifier: str,
) -> EvidenceResolution:
    directory = data_root / "emails" / namespace_id
    candidates: list[tuple[Path, list[str]]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.txt")):
            match = EMAIL_FILE.fullmatch(path.name)
            if match and "-".join(match.groups()) == date:
                candidates.append((path, _email_blocks(path.read_text(encoding="utf-8"))))
    if not candidates:
        return EvidenceResolution("email", ResolutionStatus.MISSING, 0, source_date=date)
    if len(candidates) != 1:
        count = sum(len(blocks) for _, blocks in candidates)
        return EvidenceResolution(
            "email",
            ResolutionStatus.AMBIGUOUS,
            count,
            source_date=date,
            reason="multiple_email_files_for_date",
        )

    _, blocks = candidates[0]
    normalized_qualifier = _normalized(qualifier)
    numbered = NUMBERED_EMAIL.fullmatch(qualifier.strip())
    if numbered:
        block_number = int(numbered.group(1))
        matches = [block_number] if block_number <= len(blocks) else []
    elif normalized_qualifier in {"", "email", "header", "body"}:
        matches = list(range(1, len(blocks) + 1))
    else:
        for prefix in ("header ", "body "):
            if normalized_qualifier.startswith(prefix):
                normalized_qualifier = normalized_qualifier[len(prefix) :]
                break
        matches = [
            index
            for index, block in enumerate(blocks, start=1)
            if normalized_qualifier and normalized_qualifier in _normalized(block)
        ]

    canonical_ids = tuple(f"{namespace_id}/email/{date}/block-{index}" for index in matches)
    if len(matches) == 1:
        return EvidenceResolution(
            "email",
            ResolutionStatus.UNIQUE,
            1,
            canonical_ids,
            date,
        )
    status = ResolutionStatus.MISSING if not matches else ResolutionStatus.AMBIGUOUS
    return EvidenceResolution("email", status, len(matches), canonical_ids, date)


def _attachment_date(data_root: Path, namespace_id: str, filename: str) -> str | None:
    match = re.match(r"^(\d+)_", filename)
    if not match:
        return None
    sequence = int(match.group(1))
    directory = data_root / "emails" / namespace_id
    if not directory.is_dir():
        return None
    dates = []
    for path in directory.glob("*.txt"):
        name_match = EMAIL_FILE.fullmatch(path.name)
        sequence_match = re.match(r"^(\d+)_", path.name)
        if name_match and sequence_match and int(sequence_match.group(1)) == sequence:
            dates.append("-".join(name_match.groups()))
    return dates[0] if len(set(dates)) == 1 else None


def _attachment_headings(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".md":
        return [
            match.group(1).strip()
            for line in text.splitlines()
            if (match := MARKDOWN_HEADING.match(line))
        ]
    if path.suffix.lower() == ".html":
        parser = _HeadingParser()
        parser.feed(text)
        parser.close()
        return parser.headings
    return []


def _attachment_resolution(
    data_root: Path,
    namespace_id: str,
    reference: str,
) -> EvidenceResolution:
    filename, separator, qualifier = reference.partition(":")
    _safe_component(filename, "attachment filename")
    path = data_root / "attachments" / namespace_id / filename
    if not path.is_file():
        return EvidenceResolution("attachment", ResolutionStatus.MISSING, 0)
    source_date = _attachment_date(data_root, namespace_id, filename)
    whole_id = f"{namespace_id}/attachment/{filename}"
    if not separator or not qualifier.strip():
        return EvidenceResolution(
            "attachment",
            ResolutionStatus.UNIQUE,
            1,
            (whole_id,),
            source_date,
        )

    normalized_qualifier = _normalized(qualifier)
    if normalized_qualifier.startswith("header "):
        normalized_qualifier = normalized_qualifier[len("header ") :]
    matches = []
    for index, heading in enumerate(_attachment_headings(path), start=1):
        normalized_heading = _normalized(heading)
        if normalized_heading == normalized_qualifier or normalized_heading.startswith(
            normalized_qualifier + " "
        ):
            matches.append(index)
    canonical_ids = tuple(f"{whole_id}/section-{index}" for index in matches)
    if len(matches) == 1:
        return EvidenceResolution(
            "attachment",
            ResolutionStatus.UNIQUE,
            1,
            canonical_ids,
            source_date,
        )
    status = ResolutionStatus.MISSING if not matches else ResolutionStatus.AMBIGUOUS
    return EvidenceResolution("attachment", status, len(matches), canonical_ids, source_date)


def resolve_rhelm_evidence(
    data_root: Path,
    namespace_id: str,
    reference: str,
    *,
    question_date: str | None = None,
) -> EvidenceResolution:
    """Resolve one public RHELM evidence label without returning source text."""
    namespace_id = _safe_component(namespace_id, "namespace")
    if not isinstance(reference, str) or not reference:
        raise ValueError("evidence reference must be a nonempty string")

    if conversation := CONVERSATION_REFERENCE.fullmatch(reference):
        result = _conversation_resolution(
            data_root,
            namespace_id,
            conversation.group(1),
            int(conversation.group(2)),
        )
    elif email := EMAIL_REFERENCE.fullmatch(reference):
        result = _email_resolution(
            data_root,
            namespace_id,
            email.group(1),
            email.group(2) or "",
        )
    else:
        result = _attachment_resolution(data_root, namespace_id, reference)

    if (
        result.status is ResolutionStatus.UNIQUE
        and question_date is not None
        and result.source_date is not None
        and result.source_date >= question_date
    ):
        return replace(result, status=ResolutionStatus.FUTURE, reason="not_before_question")
    return result


def _qa_namespace(path: Path) -> str:
    stem = path.stem
    if not stem.startswith(QA_PREFIX) or not stem.endswith(QA_SUFFIX):
        raise ValueError(f"unexpected RHELM QA filename {path.name!r}")
    return _safe_component(stem[len(QA_PREFIX) : -len(QA_SUFFIX)], "QA namespace")


def iter_rhelm_question_resolutions(data_root: Path) -> Iterable[QuestionResolution]:
    """Yield content-free per-query resolution records for selection and auditing."""
    for path in sorted((data_root / "QA_final").glob("*.jsonl")):
        namespace_id = _qa_namespace(path)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            query_id = row.get("id")
            question_date = row.get("question_date")
            question_type = row.get("question_type")
            references = row.get("supporting_evidence")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"{path}:{line_number} lacks query id")
            if not isinstance(question_date, str) or not question_date:
                raise ValueError(f"{path}:{line_number} lacks question date")
            if not isinstance(question_type, str) or not question_type:
                raise ValueError(f"{path}:{line_number} lacks question type")
            if not isinstance(references, list) or not references:
                raise ValueError(f"{path}:{line_number} lacks supporting evidence")
            evidence = tuple(
                resolve_rhelm_evidence(
                    data_root,
                    namespace_id,
                    reference,
                    question_date=question_date,
                )
                for reference in references
            )
            yield QuestionResolution(
                namespace_id,
                query_id,
                question_date,
                question_type,
                evidence,
            )
