"""Portable integrity-checked natural embedding and route storage.

Only explicit local input paths are read. No model, provider, or archive code is imported.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from verify_agent_memory.natural_inputs import EmbeddingSampleItem

EMBEDDING_DIMENSION = 4096
NORM_TOLERANCE = 1e-5
FloatMatrix = NDArray[np.float32]
EmbeddingIdentity = tuple[str, str]


def validate_vector_matrix(matrix: np.ndarray) -> NDArray[np.float32]:
    """Validate persisted float32 unit vectors without changing their bytes."""
    array = np.asarray(matrix)
    if array.dtype != np.float32 or array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("routing vectors must be a nonempty float32 matrix")
    if not np.isfinite(array).all():
        raise ValueError("routing vectors must be finite")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("routing vectors must be unit normalized")
    return array


def sample_bundle_sha256(items: Sequence[EmbeddingSampleItem]) -> str:
    """Hash the ordered content-free sample identities."""
    return hashlib.sha256(
        b"\n".join(_canonical_json_bytes(item.as_dict()) for item in items)
    ).hexdigest()


def validate_embedding_vectors(
    vectors: np.ndarray,
    *,
    expected_count: int,
    dimension: int = EMBEDDING_DIMENSION,
    norm_tolerance: float = NORM_TOLERANCE,
) -> dict[str, float | int]:
    """Validate shape, finiteness, dtype-independent norms, and return diagnostics."""
    array = np.asarray(vectors)
    if array.shape != (expected_count, dimension):
        raise ValueError(
            f"embedding shape {array.shape!r} differs from {(expected_count, dimension)!r}"
        )
    if not np.isfinite(array).all():
        raise ValueError("embedding vectors must be finite")
    norms = np.linalg.norm(array.astype(np.float64, copy=False), axis=1)
    errors = np.abs(norms - 1.0)
    max_error = float(errors.max(initial=0.0))
    if max_error > norm_tolerance:
        raise ValueError(
            f"embedding norm error {max_error:.8g} exceeds tolerance {norm_tolerance:.8g}"
        )
    return {
        "count": expected_count,
        "dimension": dimension,
        "max_norm_error": max_error,
        "min_norm": float(norms.min(initial=1.0)),
        "max_norm": float(norms.max(initial=1.0)),
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class EmbeddingShard:
    """One validated private embedding shard."""

    items: tuple[EmbeddingSampleItem, ...]
    token_lengths: tuple[int, ...]
    vectors: np.ndarray


@dataclass(frozen=True)
class EmbeddingShardRecord:
    """Content-free integrity record for one completed shard."""

    shard_index: int
    filename: str
    row_count: int
    sha256: str
    identity_bundle_sha256: str
    first_identity: dict[str, object]
    last_identity: dict[str, object]
    vector_payload_bytes: int
    max_norm_error: float
    token_length_min: int
    token_length_max: int
    encoding_seconds: float = 0.0
    shard_write_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "shard_index": self.shard_index,
            "filename": self.filename,
            "row_count": self.row_count,
            "sha256": self.sha256,
            "identity_bundle_sha256": self.identity_bundle_sha256,
            "first_identity": self.first_identity,
            "last_identity": self.last_identity,
            "vector_payload_bytes": self.vector_payload_bytes,
            "max_norm_error": self.max_norm_error,
            "token_length_min": self.token_length_min,
            "token_length_max": self.token_length_max,
            "encoding_seconds": self.encoding_seconds,
            "shard_write_seconds": self.shard_write_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EmbeddingShardRecord:
        first = value.get("first_identity")
        last = value.get("last_identity")
        if not isinstance(first, Mapping) or not isinstance(last, Mapping):
            raise ValueError("shard record identities must be mappings")
        record = cls(
            shard_index=int(value.get("shard_index", -1)),
            filename=str(value.get("filename") or ""),
            row_count=int(value.get("row_count", 0)),
            sha256=str(value.get("sha256") or ""),
            identity_bundle_sha256=str(value.get("identity_bundle_sha256") or ""),
            first_identity=dict(first),
            last_identity=dict(last),
            vector_payload_bytes=int(value.get("vector_payload_bytes", 0)),
            max_norm_error=float(value.get("max_norm_error", float("nan"))),
            token_length_min=int(value.get("token_length_min", 0)),
            token_length_max=int(value.get("token_length_max", 0)),
            encoding_seconds=float(value.get("encoding_seconds", 0.0)),
            shard_write_seconds=float(value.get("shard_write_seconds", 0.0)),
        )
        if (
            record.shard_index < 0
            or not record.filename
            or Path(record.filename).name != record.filename
            or record.row_count < 1
            or not _valid_sha256(record.sha256)
            or not _valid_sha256(record.identity_bundle_sha256)
            or record.vector_payload_bytes < 1
            or not np.isfinite(record.max_norm_error)
            or record.token_length_min < 1
            or record.token_length_max < record.token_length_min
            or not math.isfinite(record.encoding_seconds)
            or record.encoding_seconds < 0
            or not math.isfinite(record.shard_write_seconds)
            or record.shard_write_seconds < 0
        ):
            raise ValueError("shard record is invalid")
        return record


@dataclass(frozen=True)
class EmbeddingStoreCheckpoint:
    """Atomic resume state for a prefix of the frozen population."""

    binding: dict[str, object]
    completed_count: int
    next_shard_index: int
    shards: tuple[EmbeddingShardRecord, ...]
    forced_reload_verified: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding,
            "completed_count": self.completed_count,
            "next_shard_index": self.next_shard_index,
            "shards": [record.as_dict() for record in self.shards],
            "forced_reload_verified": self.forced_reload_verified,
        }


def write_embedding_shard(
    path: Path,
    *,
    items: Sequence[EmbeddingSampleItem],
    token_lengths: Sequence[int],
    vectors: np.ndarray,
    dimension: int = EMBEDDING_DIMENSION,
) -> None:
    """Atomically write one no-pickle compressed shard."""
    if not items:
        raise ValueError("embedding shard cannot be empty")
    if len(items) != len(token_lengths):
        raise ValueError("embedding shard token lengths do not align with identities")
    if any(
        isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= 512
        for length in token_lengths
    ):
        raise ValueError("embedding shard token lengths must be integers in [1, 512]")
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    validate_embedding_vectors(
        array,
        expected_count=len(items),
        dimension=dimension,
    )
    identities_json = _canonical_json_bytes([item.as_dict() for item in items]).decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            identities_json=np.asarray(identities_json),
            token_lengths=np.asarray(token_lengths, dtype=np.int32),
            vectors=array,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_embedding_shard(
    path: Path,
    *,
    dimension: int = EMBEDDING_DIMENSION,
) -> EmbeddingShard:
    """Load and validate one no-pickle embedding shard."""
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"identities_json", "token_lengths", "vectors"}:
            raise ValueError("embedding shard fields differ from the frozen schema")
        raw_identities: Any = json.loads(str(payload["identities_json"].item()))
        token_array = np.asarray(payload["token_lengths"], dtype=np.int64)
        vectors = np.ascontiguousarray(payload["vectors"], dtype=np.float32)
    if not isinstance(raw_identities, list) or not raw_identities:
        raise ValueError("embedding shard identities must be a nonempty list")
    items = tuple(
        EmbeddingSampleItem.from_dict(value)
        for value in raw_identities
        if isinstance(value, Mapping)
    )
    if len(items) != len(raw_identities):
        raise ValueError("embedding shard contains a malformed identity")
    if token_array.shape != (len(items),):
        raise ValueError("embedding shard token lengths do not align with identities")
    token_lengths = tuple(int(value) for value in token_array)
    if any(not 1 <= length <= 512 for length in token_lengths):
        raise ValueError("embedding shard token length is outside [1, 512]")
    validate_embedding_vectors(
        vectors,
        expected_count=len(items),
        dimension=dimension,
    )
    return EmbeddingShard(items=items, token_lengths=token_lengths, vectors=vectors)


def build_shard_record(
    path: Path,
    *,
    shard_index: int,
    dimension: int = EMBEDDING_DIMENSION,
    encoding_seconds: float = 0.0,
    shard_write_seconds: float = 0.0,
) -> EmbeddingShardRecord:
    """Build an integrity record by reading the final atomic shard."""
    if shard_index < 0:
        raise ValueError("shard index must be nonnegative")
    shard = load_embedding_shard(path, dimension=dimension)
    diagnostics = validate_embedding_vectors(
        shard.vectors,
        expected_count=len(shard.items),
        dimension=dimension,
    )
    return EmbeddingShardRecord(
        shard_index=shard_index,
        filename=path.name,
        row_count=len(shard.items),
        sha256=_file_sha256(path),
        identity_bundle_sha256=sample_bundle_sha256(shard.items),
        first_identity=shard.items[0].as_dict(),
        last_identity=shard.items[-1].as_dict(),
        vector_payload_bytes=int(shard.vectors.nbytes),
        max_norm_error=float(diagnostics["max_norm_error"]),
        token_length_min=min(shard.token_lengths),
        token_length_max=max(shard.token_lengths),
        encoding_seconds=encoding_seconds,
        shard_write_seconds=shard_write_seconds,
    )


def save_store_checkpoint(path: Path, checkpoint: EmbeddingStoreCheckpoint) -> None:
    """Atomically persist content-free checkpoint metadata."""
    if checkpoint.completed_count < 0 or checkpoint.next_shard_index < 0:
        raise ValueError("store checkpoint counts must be nonnegative")
    if checkpoint.next_shard_index != len(checkpoint.shards):
        raise ValueError("next shard index must equal the number of shard records")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            checkpoint.as_dict(),
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_store_checkpoint(path: Path) -> EmbeddingStoreCheckpoint:
    """Load and structurally validate checkpoint metadata."""
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("store checkpoint must be a mapping")
    binding = value.get("binding")
    raw_shards = value.get("shards")
    forced = value.get("forced_reload_verified")
    if not isinstance(binding, Mapping):
        raise ValueError("store checkpoint binding must be a mapping")
    if not isinstance(raw_shards, list):
        raise ValueError("store checkpoint shards must be a list")
    if not isinstance(forced, bool):
        raise ValueError("forced reload flag must be boolean")
    checkpoint = EmbeddingStoreCheckpoint(
        binding=dict(binding),
        completed_count=int(value.get("completed_count", -1)),
        next_shard_index=int(value.get("next_shard_index", -1)),
        shards=tuple(
            EmbeddingShardRecord.from_dict(record)
            for record in raw_shards
            if isinstance(record, Mapping)
        ),
        forced_reload_verified=forced,
    )
    if len(checkpoint.shards) != len(raw_shards):
        raise ValueError("store checkpoint contains a malformed shard record")
    if checkpoint.completed_count < 0 or checkpoint.next_shard_index != len(checkpoint.shards):
        raise ValueError("store checkpoint counters are invalid")
    return checkpoint


def validate_store_resume(
    population: Sequence[EmbeddingSampleItem],
    checkpoint: EmbeddingStoreCheckpoint,
    *,
    shard_directory: Path,
    expected_binding: Mapping[str, object],
    shard_size: int,
    dimension: int = EMBEDDING_DIMENSION,
) -> dict[str, float | int]:
    """Rehash all shards and prove they form an exact stable population prefix."""
    if checkpoint.binding != dict(expected_binding):
        raise ValueError("store checkpoint binding drifted")
    if shard_size < 1:
        raise ValueError("shard size must be positive")
    if checkpoint.completed_count > len(population):
        raise ValueError("store checkpoint exceeds the frozen population")
    expected_files = {record.filename for record in checkpoint.shards}
    observed_files = {path.name for path in shard_directory.glob("shard-*.npz")}
    orphans = observed_files - expected_files
    missing = expected_files - observed_files
    if orphans:
        raise ValueError(f"embedding store contains orphan shards: {sorted(orphans)!r}")
    if missing:
        raise ValueError(f"embedding store is missing shards: {sorted(missing)!r}")

    completed = 0
    vector_payload_bytes = 0
    max_norm_error = 0.0
    for expected_index, record in enumerate(checkpoint.shards):
        expected_filename = f"shard-{expected_index:05d}.npz"
        if record.shard_index != expected_index or record.filename != expected_filename:
            raise ValueError("embedding shard sequence is not contiguous")
        path = shard_directory / record.filename
        if _file_sha256(path) != record.sha256:
            raise ValueError(f"embedding shard {record.filename} SHA-256 drifted")
        shard = load_embedding_shard(path, dimension=dimension)
        end = completed + len(shard.items)
        if shard.items != tuple(population[completed:end]):
            raise ValueError("embedding shard identities are not an exact population prefix")
        expected_rows = min(shard_size, len(population) - completed)
        if len(shard.items) != expected_rows:
            raise ValueError("embedding shard row count differs from the frozen size")
        observed_record = build_shard_record(
            path,
            shard_index=expected_index,
            dimension=dimension,
            encoding_seconds=record.encoding_seconds,
            shard_write_seconds=record.shard_write_seconds,
        )
        if observed_record != record:
            raise ValueError("embedding shard integrity record drifted")
        completed = end
        vector_payload_bytes += int(shard.vectors.nbytes)
        max_norm_error = max(max_norm_error, record.max_norm_error)

    if completed != checkpoint.completed_count:
        raise ValueError("checkpoint completed count differs from shard rows")
    if completed > 1024 and not checkpoint.forced_reload_verified:
        raise ValueError("store advanced beyond forced reload without verification")
    return {
        "completed_count": completed,
        "shard_count": len(checkpoint.shards),
        "vector_payload_bytes": vector_payload_bytes,
        "max_norm_error": max_norm_error,
    }


@dataclass(frozen=True)
class RouteObservation:
    """One private top-100 route with content-free work diagnostics."""

    query_id: str
    ranked_memory_ids: tuple[str, ...]
    route_width: int
    candidates_scored: int
    centroid_scores: int
    fallback_used: bool
    latency_ms: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.query_id, str)
            or not self.query_id
            or len(self.ranked_memory_ids) > 100
            or len(set(self.ranked_memory_ids)) != len(self.ranked_memory_ids)
            or any(
                not isinstance(memory_id, str) or not memory_id
                for memory_id in self.ranked_memory_ids
            )
            or self.route_width < 0
            or self.candidates_scored < 0
            or self.centroid_scores < 0
            or not isinstance(self.fallback_used, bool)
            or not math.isfinite(self.latency_ms)
            or self.latency_ms < 0.0
        ):
            raise ValueError("route observation is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "ranked_memory_ids": list(self.ranked_memory_ids),
            "route_width": self.route_width,
            "candidates_scored": self.candidates_scored,
            "centroid_scores": self.centroid_scores,
            "fallback_used": self.fallback_used,
            "latency_ms": self.latency_ms,
        }

    def deterministic_dict(self) -> dict[str, object]:
        value = self.as_dict()
        del value["latency_ms"]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RouteObservation:
        raw_ids = value.get("ranked_memory_ids")
        fallback = value.get("fallback_used")
        if not isinstance(raw_ids, list) or not isinstance(fallback, bool):
            raise ValueError("route observation fields are malformed")
        return cls(
            query_id=str(value.get("query_id") or ""),
            ranked_memory_ids=tuple(str(memory_id) for memory_id in raw_ids),
            route_width=int(value.get("route_width", -1)),
            candidates_scored=int(value.get("candidates_scored", -1)),
            centroid_scores=int(value.get("centroid_scores", -1)),
            fallback_used=fallback,
            latency_ms=float(value.get("latency_ms", float("nan"))),
        )


def write_route_bundle(
    path: Path,
    observations: Sequence[RouteObservation],
) -> None:
    """Atomically write one deterministic gzip JSONL route bundle."""
    if not observations:
        raise ValueError("route bundle cannot be empty")
    query_ids = [observation.query_id for observation in observations]
    if query_ids != sorted(query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("route bundle queries must be unique and sorted")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        ):
            for observation in observations:
                compressed.write(_canonical_json_bytes(observation.as_dict()))
                compressed.write(b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_route_bundle(path: Path) -> tuple[RouteObservation, ...]:
    """Load and structurally validate one private route bundle."""
    observations = []
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"route bundle line {line_number} is invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise ValueError("route bundle rows must be mappings")
            observations.append(RouteObservation.from_dict(value))
    if not observations:
        raise ValueError("route bundle cannot be empty")
    query_ids = [observation.query_id for observation in observations]
    if query_ids != sorted(query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("route bundle query order is invalid")
    return tuple(observations)


@dataclass(frozen=True)
class EmbeddingTable:
    """One validated in-memory view of the private float32 embedding shards."""

    items: tuple[EmbeddingSampleItem, ...]
    vectors: FloatMatrix
    index_by_identity: Mapping[EmbeddingIdentity, int]

    def __post_init__(self) -> None:
        validate_vector_matrix(self.vectors)
        if self.vectors.shape[0] != len(self.items):
            raise ValueError("embedding items and vectors do not align")
        expected = {item.embedding_identity: index for index, item in enumerate(self.items)}
        if len(expected) != len(self.items) or dict(self.index_by_identity) != expected:
            raise ValueError("embedding identity index is invalid")


def load_validated_embedding_table(
    population: Sequence[EmbeddingSampleItem],
    checkpoint: EmbeddingStoreCheckpoint,
    *,
    shard_directory: Path,
    expected_binding: Mapping[str, object],
    shard_size: int,
    dimension: int,
) -> EmbeddingTable:
    """Rehash the complete store and load its exact population-ordered matrix."""
    diagnostics = validate_store_resume(
        population,
        checkpoint,
        shard_directory=shard_directory,
        expected_binding=expected_binding,
        shard_size=shard_size,
        dimension=dimension,
    )
    if diagnostics["completed_count"] != len(population):
        raise ValueError("embedding store is not complete")
    vectors = np.empty((len(population), dimension), dtype=np.float32)
    offset = 0
    loaded_items: list[EmbeddingSampleItem] = []
    for record in checkpoint.shards:
        shard = load_embedding_shard(
            shard_directory / record.filename,
            dimension=dimension,
        )
        end = offset + len(shard.items)
        vectors[offset:end] = shard.vectors
        loaded_items.extend(shard.items)
        offset = end
    items = tuple(loaded_items)
    if items != tuple(population) or offset != len(population):
        raise ValueError("loaded embedding store differs from the frozen population")
    return EmbeddingTable(
        items=items,
        vectors=vectors,
        index_by_identity=MappingProxyType(
            {item.embedding_identity: index for index, item in enumerate(items)}
        ),
    )


def load_route_checkpoint(path: Path) -> dict[str, Any]:
    """Load the portable manifest; records bind filenames, tasks and bytes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("route checkpoint must contain a records list")
    tasks = set()
    for record in value["records"]:
        if not isinstance(record, dict):
            raise ValueError("route checkpoint record must be an object")
        task = record.get("task_id")
        filename = record.get("filename")
        digest = record.get("sha256")
        if (
            not isinstance(task, str)
            or not task
            or task in tasks
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(digest, str)
            or not _valid_sha256(digest)
        ):
            raise ValueError("route checkpoint identity, path or digest is invalid")
        tasks.add(task)
    if value.get("completed_tasks") != len(tasks):
        raise ValueError("route checkpoint task count differs from records")
    return value


def load_public_api(source_root: Path, run_root: Path) -> Mapping[str, object]:
    """Compatibility-shaped public adapter for existing frozen diagnostics."""
    from verify_agent_memory.natural_inputs import load_eval_inputs
    from verify_agent_memory.natural_retrieval import lexical_tie_ranks, stable_rank_indices

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "routes_ready":
        raise ValueError("public retrieval run is incomplete")
    route_checkpoint_path = run_root / "route_checkpoint.json"
    if _file_sha256(route_checkpoint_path) != manifest.get("route_checkpoint_sha256"):
        raise ValueError("route checkpoint differs from the public run manifest")
    route_checkpoint = load_route_checkpoint(route_checkpoint_path)
    if route_checkpoint["binding"] != manifest["embedding_binding"]:
        raise ValueError("route checkpoint binding differs from public run inputs")
    for record in route_checkpoint["records"]:
        if _file_sha256(run_root / "route_bundles" / record["filename"]) != record["sha256"]:
            raise ValueError("public route bundle hash drifted")
    if _file_sha256(run_root / "retrieval_config.json") != manifest.get("retrieval_config_sha256"):
        raise ValueError("public run retrieval configuration drifted")
    checkpoint_path = run_root / "embedding_checkpoint.json"
    if _file_sha256(checkpoint_path) != manifest.get("embedding_checkpoint_sha256"):
        raise ValueError("embedding checkpoint differs from the public run manifest")
    if dict(load_store_checkpoint(checkpoint_path).binding) != manifest["embedding_binding"]:
        raise ValueError("embedding checkpoint binding differs from public run inputs")

    def inputs_loader(**kwargs: object):
        inputs = load_eval_inputs(
            source_root, include_texts=bool(kwargs.get("include_texts", False))
        )
        if (
            sample_bundle_sha256(inputs.population)
            != manifest["embedding_binding"]["population_bundle_sha256"]
        ):
            raise ValueError("source population differs from the public run")
        return inputs

    return {
        "load_eval_inputs": inputs_loader,
        "load_route_bundle": load_route_bundle,
        "load_store_checkpoint": load_store_checkpoint,
        "load_validated_embedding_table": load_validated_embedding_table,
        "lexical_tie_ranks": lexical_tie_ranks,
        "stable_rank_indices": stable_rank_indices,
        "dimension": int(manifest["dimension"]),
        "shard_size": int(manifest["shard_size"]),
        "embedding_checkpoint_path": checkpoint_path,
        "embedding_shards": run_root / "embedding_shards",
        "config_path": run_root / "retrieval_config.json",
        "route_root": run_root / "route_bundles",
    }
