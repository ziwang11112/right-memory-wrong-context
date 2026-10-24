"""Build evaluation inputs from pinned, locally fetched RHELM and MemOps sources.

Corpus descriptors contain hashes, not text. Source text is recovered in memory only
when requested; this module does not read historical run locks or private caches.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verify_agent_memory.natural_corpus import NaturalCorpus, build_natural_corpora
from verify_agent_memory.natural_rhelm import iter_rhelm_question_resolutions
from verify_agent_memory.natural_source_indices import (
    load_memops_message_index,
    load_rhelm_qa_index,
    load_rhelm_turn_index,
)

SOURCE_ORDER = ("rhelm", "memops")
KIND_ORDER = ("memory", "query")
SOURCES = SOURCE_ORDER
QUERY_INSTRUCTION = (
    "Given a question about a software project, retrieve the memory passage "
    "that answers the question"
)
EXPECTED_POPULATION = {
    "rhelm": {
        "namespace_groups": 7,
        "memory_rows": 7764,
        "query_rows": 523,
        "required_references": 1296,
    },
    "memops": {
        "namespace_groups": 80,
        "memory_rows": 175144,
        "query_rows": 3244,
        "required_references": 11118,
    },
    "total": {
        "namespace_groups": 87,
        "memory_rows": 182908,
        "query_rows": 3767,
        "required_references": 12414,
    },
}


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def byte_length_bin(length: int) -> str:
    """Map a positive raw UTF-8 byte length to the frozen pilot bin."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("byte length must be a positive integer")
    if length <= 256:
        return "1--256"
    if length <= 1024:
        return "257--1024"
    if length <= 4096:
        return "1025--4096"
    return "4097+"


@dataclass(frozen=True)
class EmbeddingSampleItem:
    """One content-free embedding identity."""

    source: str
    kind: str
    split: str
    text_sha256: str
    text_utf8_bytes: int
    length_bin: str

    def __post_init__(self) -> None:
        if self.source not in SOURCE_ORDER:
            raise ValueError(f"unexpected sample source {self.source!r}")
        if self.kind not in KIND_ORDER:
            raise ValueError(f"unexpected sample kind {self.kind!r}")
        if self.split not in {"dev", "eval"}:
            raise ValueError(f"unexpected sample split {self.split!r}")
        if not _valid_sha256(self.text_sha256):
            raise ValueError("sample text SHA-256 is invalid")
        if byte_length_bin(self.text_utf8_bytes) != self.length_bin:
            raise ValueError("sample byte length and length bin differ")

    @property
    def embedding_identity(self) -> tuple[str, str]:
        """Return the identity after applying the frozen query instruction rule."""
        return self.kind, self.text_sha256

    def as_dict(self) -> dict[str, object]:
        """Return the tracked content-free representation."""
        return {
            "source": self.source,
            "kind": self.kind,
            "split": self.split,
            "text_sha256": self.text_sha256,
            "text_utf8_bytes": self.text_utf8_bytes,
            "length_bin": self.length_bin,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EmbeddingSampleItem:
        """Restore a content-free sample identity from a checkpoint."""
        return cls(
            source=str(value.get("source") or ""),
            kind=str(value.get("kind") or ""),
            split=str(value.get("split") or ""),
            text_sha256=str(value.get("text_sha256") or ""),
            text_utf8_bytes=int(value.get("text_utf8_bytes") or 0),
            length_bin=str(value.get("length_bin") or ""),
        )


@dataclass(frozen=True)
class RuntimeText:
    """In-memory source text bound to one content-free sample identity."""

    item: EmbeddingSampleItem
    prepared_text: str


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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _text_fingerprint(text: str) -> tuple[str, int]:
    encoded = text.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _register_selected_text(
    found: dict[tuple[str, str], str],
    selected: Mapping[tuple[str, str], EmbeddingSampleItem],
    *,
    source: str,
    kind: str,
    text: str,
) -> None:
    text_sha256, text_utf8_bytes = _text_fingerprint(text)
    identity = (kind, text_sha256)
    item = selected.get(identity)
    if item is None or item.source != source:
        return
    if item.text_utf8_bytes != text_utf8_bytes:
        raise RuntimeError("selected raw text byte length differs from its descriptor")
    previous = found.setdefault(identity, text)
    if previous != text:
        raise RuntimeError("selected text identity maps to different raw bytes")


def _load_selected_texts(
    sample: Sequence[EmbeddingSampleItem],
    source_root: Path,
) -> tuple[RuntimeText, ...]:
    selected = {item.embedding_identity: item for item in sample}
    if len(selected) != len(sample):
        raise RuntimeError("selected sample identities are not unique")
    found: dict[tuple[str, str], str] = {}

    rhelm_root = source_root / "rhelm" / "data"
    for turn in load_rhelm_turn_index(rhelm_root).values():
        _register_selected_text(
            found,
            selected,
            source="rhelm",
            kind="memory",
            text=turn.text,
        )
    qa_index = load_rhelm_qa_index(rhelm_root)
    for resolution in iter_rhelm_question_resolutions(rhelm_root):
        if not (resolution.conversation_only and resolution.all_references_unique):
            continue
        row = qa_index[(resolution.namespace_id, resolution.query_id)]
        _register_selected_text(
            found,
            selected,
            source="rhelm",
            kind="query",
            text=_string(row.get("question"), "RHELM question"),
        )

    injected_root = (
        source_root / "memops" / "generated_result" / "4-inject_evidence_with_distractors"
    )
    for injected_path in sorted(injected_root.glob("*.json")):
        injected_case = _read_json(injected_path)
        for record in load_memops_message_index(injected_path.stem, injected_case).values():
            _register_selected_text(
                found,
                selected,
                source="memops",
                kind="memory",
                text=_string(record.get("text"), "MemOps memory text"),
            )
        for raw_answer in _sequence(injected_case.get("answer"), "MemOps answers"):
            answer = _mapping(raw_answer, "MemOps answer")
            _register_selected_text(
                found,
                selected,
                source="memops",
                kind="query",
                text=_string(answer.get("question"), "MemOps question"),
            )

    missing = [item.embedding_identity for item in sample if item.embedding_identity not in found]
    if missing:
        raise RuntimeError(f"failed to recover {len(missing)} selected raw inputs")
    runtime_rows = []
    for item in sample:
        raw_text = found[item.embedding_identity]
        prepared = (
            f"Instruct: {QUERY_INSTRUCTION}\nQuery:{raw_text}" if item.kind == "query" else raw_text
        )
        runtime_rows.append(RuntimeText(item=item, prepared_text=prepared))
    return tuple(runtime_rows)


def build_eval_embedding_population(
    corpora: Mapping[str, NaturalCorpus],
) -> tuple[EmbeddingSampleItem, ...]:
    """Build an eval-only store, including eval text duplicated on dev."""
    unexpected = set(corpora) - set(SOURCE_ORDER)
    if unexpected:
        raise ValueError(f"unexpected natural corpus sources: {sorted(unexpected)}")
    owned: dict[tuple[str, str], list[EmbeddingSampleItem]] = {
        (source, kind): [] for source in SOURCE_ORDER for kind in KIND_ORDER
    }
    owner_by_identity: dict[tuple[str, str], EmbeddingSampleItem] = {}
    for source in SOURCE_ORDER:
        corpus = corpora.get(source)
        if corpus is None:
            continue
        if corpus.source != source:
            raise ValueError("natural corpus source differs from mapping key")
        for kind in KIND_ORDER:
            if kind == "memory":
                descriptors = (
                    (
                        memory.text_sha256,
                        memory.text_utf8_bytes,
                    )
                    for memory in corpus.memories
                    if memory.split == "eval"
                )
            else:
                descriptors = (
                    (
                        query.query_text_sha256,
                        query.query_text_utf8_bytes,
                    )
                    for query in corpus.queries
                    if query.split == "eval"
                )
            source_seen: dict[str, int] = {}
            for text_sha256, text_utf8_bytes in descriptors:
                previous_length = source_seen.setdefault(text_sha256, text_utf8_bytes)
                if previous_length != text_utf8_bytes:
                    raise ValueError("duplicate eval text hash has inconsistent lengths")
            for text_sha256, text_utf8_bytes in sorted(source_seen.items()):
                item = EmbeddingSampleItem(
                    source=source,
                    kind=kind,
                    split="eval",
                    text_sha256=text_sha256,
                    text_utf8_bytes=text_utf8_bytes,
                    length_bin=byte_length_bin(text_utf8_bytes),
                )
                identity = item.embedding_identity
                previous = owner_by_identity.setdefault(identity, item)
                if previous.text_utf8_bytes != item.text_utf8_bytes:
                    raise ValueError("cross-source eval text hash has inconsistent lengths")
                if previous is item:
                    owned[(source, kind)].append(item)
    population = tuple(
        item for source in SOURCE_ORDER for kind in KIND_ORDER for item in owned[(source, kind)]
    )
    if not population or any(item.split != "eval" for item in population):
        raise ValueError("eval embedding population is empty or crossed the split")
    if len({item.embedding_identity for item in population}) != len(population):
        raise ValueError("eval embedding population contains duplicate identities")
    return population


def validate_eval_population_counts(corpora: Mapping[str, NaturalCorpus]) -> None:
    """Validate only the preregistered aggregate evaluation population."""
    if set(corpora) != set(SOURCES):
        raise ValueError("evaluation corpus must contain both frozen sources")
    total_memories = 0
    total_queries = 0
    total_groups: set[tuple[str, str]] = set()
    total_references = 0
    for source in SOURCES:
        corpus = corpora[source]
        memories = [memory for memory in corpus.memories if memory.split == "eval"]
        queries = [query for query in corpus.queries if query.split == "eval"]
        groups = {memory.namespace_id for memory in memories}
        expected = EXPECTED_POPULATION[source]
        if (
            len(memories) != expected["memory_rows"]
            or len(queries) != expected["query_rows"]
            or len(groups) != expected["namespace_groups"]
            or sum(len(query.required_memory_ids) for query in queries)
            != expected["required_references"]
        ):
            raise ValueError(f"{source} evaluation population drifted")
        total_memories += len(memories)
        total_queries += len(queries)
        total_groups.update((source, group) for group in groups)
        total_references += sum(len(query.required_memory_ids) for query in queries)
    expected_total = EXPECTED_POPULATION["total"]
    if (
        total_memories != expected_total["memory_rows"]
        or total_queries != expected_total["query_rows"]
        or len(total_groups) != expected_total["namespace_groups"]
        or total_references != expected_total["required_references"]
    ):
        raise ValueError("total evaluation population drifted")


@dataclass(frozen=True)
class EvalInputs:
    """Descriptors and optional texts needed by public retrieval and reader runners."""

    corpora: Mapping[str, NaturalCorpus]
    population: tuple[EmbeddingSampleItem, ...]
    runtime_rows: tuple[RuntimeText, ...]
    source_root: Path


def _source_git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def verify_source_revisions(source_root: Path, manifest_path: Path | None = None) -> None:
    """Verify the two public source checkouts against the published source manifest."""
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parents[2] / "data" / "upstream_sources.json"
    manifest = _read_json(manifest_path)
    sources = {
        str(row["name"]): row
        for row in _sequence(manifest.get("sources"), "sources")
        if isinstance(row, Mapping)
    }
    for name in SOURCE_ORDER:
        contract = sources.get(name)
        if contract is None:
            raise ValueError(f"source manifest lacks {name}")
        directory = source_root / name

        if _source_git(directory, "rev-parse", "HEAD") != contract.get("revision"):
            raise ValueError(f"{name} source revision differs from the published manifest")
        if _source_git(directory, "rev-parse", "HEAD^{tree}") != contract.get("tree"):
            raise ValueError(f"{name} source tree differs from the published manifest")
        if _source_git(directory, "status", "--porcelain"):
            raise ValueError(f"{name} source checkout contains modified or untracked files")


def load_eval_inputs(
    source_root: Path,
    include_texts: bool = False,
    validate_expected: bool = True,
) -> EvalInputs:
    """Reconstruct eval inputs without historical archives, caches, or unlock records.

    ``source_root`` contains the lowercase ``rhelm`` and ``memops`` checkouts made
    by ``fetch_public_sources``. Set ``validate_expected=False`` only for synthetic
    fixtures or an explicitly different source population.
    """
    source_root = source_root.resolve()
    if validate_expected:
        verify_source_revisions(source_root)
    corpora, _split_map = build_natural_corpora(
        rhelm_data_root=source_root / "rhelm" / "data",
        memops_root=source_root / "memops",
    )
    if validate_expected:
        validate_eval_population_counts(corpora)
    population = build_eval_embedding_population(corpora)
    runtime_rows = _load_selected_texts(population, source_root) if include_texts else ()
    if runtime_rows and tuple(row.item for row in runtime_rows) != population:
        raise RuntimeError("recovered eval texts differ from the population")
    return EvalInputs(corpora, population, runtime_rows, source_root)
