"""Build public natural inputs, local Qwen embeddings, and all nine frozen routes.

Build/validate never load a model. Embed explicitly loads a locally cached pinned
snapshot; no network download or provider call is made by this command.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from verify_agent_memory.natural_inputs import QUERY_INSTRUCTION, load_eval_inputs
from verify_agent_memory.natural_retrieval import (
    NaturalIndex,
    build_eval_method_views,
    load_frozen_settings,
    route_ranking_config,
)
from verify_agent_memory.natural_storage import (
    EmbeddingStoreCheckpoint,
    _file_sha256,
    build_shard_record,
    load_route_bundle,
    load_route_checkpoint,
    load_store_checkpoint,
    load_validated_embedding_table,
    sample_bundle_sha256,
    save_store_checkpoint,
    validate_store_resume,
    write_embedding_shard,
    write_route_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "frozen_natural_protocol.json"
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
DIMENSION = 4096
SHARD_SIZE = 512


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def prepare_run(
    inputs: object, run_root: Path, *, dimension: int = DIMENSION, shard_size: int = SHARD_SIZE
) -> dict[str, object]:
    """Bind a new execution to reconstructed inputs and the public frozen protocol."""
    settings = load_frozen_settings(PROTOCOL)
    binding = {
        "population_bundle_sha256": sample_bundle_sha256(inputs.population),
        "population_count": len(inputs.population),
        "protocol_sha256": _file_sha256(PROTOCOL),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dimension": dimension,
        "shard_size": shard_size,
        "max_length": 512,
        "storage_dtype": "float32",
        "pooling": "unit_normalized_last_token",
        "query_instruction_sha256": hashlib.sha256(QUERY_INSTRUCTION.encode()).hexdigest(),
    }
    manifest_path = run_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding_binding") != binding:
            raise ValueError("run directory is bound to different inputs or model settings")
        return manifest
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "execution_kind": "public_source_reproduction",
        "historical_paper_result": False,
        "source_root": str(inputs.source_root.resolve()),
        "dimension": dimension,
        "shard_size": shard_size,
        "embedding_binding": binding,
        "source_counts": {
            source: {
                "memories": sum(row.split == "eval" for row in corpus.memories),
                "queries": sum(row.split == "eval" for row in corpus.queries),
            }
            for source, corpus in inputs.corpora.items()
        },
        "provider_calls": 0,
        "status": "inputs_ready",
    }
    _write_json(run_root / "population.json", [item.as_dict() for item in inputs.population])
    _write_json(
        run_root / "retrieval_config.json",
        {
            "selected_settings": [
                {
                    "arm": setting.arm,
                    "setting_id": setting.setting_id,
                    "ranking_config_id": setting.ranking_config_id,
                    "params": dict(setting.params),
                }
                for setting in settings
            ]
        },
    )
    manifest["retrieval_config_sha256"] = _file_sha256(run_root / "retrieval_config.json")
    _write_json(manifest_path, manifest)
    return manifest


def _renormalize_float32(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RuntimeError("model returned a nonfinite or zero embedding")
    persisted = np.ascontiguousarray(values / norms[:, None], dtype=np.float32)
    second_norms = np.linalg.norm(persisted.astype(np.float64), axis=1)
    return np.ascontiguousarray(persisted / second_norms[:, None], dtype=np.float32)


class LocalQwenEncoder:
    """Lazy local-only encoder with the paper's pinned Qwen/last-token contract."""

    def __init__(self, model_path: Path, *, batch_size: int = 8, device: str = "auto") -> None:
        if model_path.name != MODEL_REVISION:
            raise ValueError(
                "model-path must be the local Hugging Face snapshot for the pinned revision"
            )
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self._runtime = None

    def snapshot_manifest(self) -> dict[str, str]:
        paths = sorted(path for path in self.model_path.iterdir() if path.is_file())
        if not any(path.suffix == ".safetensors" for path in paths):
            raise ValueError("pinned local snapshot contains no safetensors weights")
        if not (self.model_path / "config.json").is_file():
            raise ValueError("pinned local snapshot lacks config.json")
        return {path.name: _file_sha256(path) for path in paths}

    def __call__(self, texts: Sequence[str]) -> tuple[np.ndarray, tuple[int, ...]]:
        if self._runtime is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                padding_side="left",
            )
            model = AutoModel.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            if model.config.hidden_size != DIMENSION:
                raise ValueError("local model dimension differs from the frozen Qwen contract")
            self._runtime = torch, tokenizer, model
        torch, tokenizer, model = self._runtime
        vectors = []
        lengths = []
        for start in range(0, len(texts), self.batch_size):
            batch = tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            lengths.extend(int(x) for x in batch["attention_mask"].sum(dim=1).tolist())
            input_device = next(model.parameters()).device
            batch = {key: value.to(input_device) for key, value in batch.items()}
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].to(hidden.device)
                if bool(mask[:, -1].all()):
                    pooled = hidden[:, -1]
                else:
                    last = mask.sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                array = normalized.float().cpu().numpy()
            # Preserve the final float32 renormalization used by the frozen encoder.
            vectors.append(_renormalize_float32(array))
        return np.concatenate(vectors), tuple(lengths)


def build_embedding_store(
    inputs: object,
    run_root: Path,
    encoder: Callable[[Sequence[str]], tuple[np.ndarray, tuple[int, ...]]],
    *,
    dimension: int = DIMENSION,
    shard_size: int = SHARD_SIZE,
) -> None:
    manifest = prepare_run(inputs, run_root, dimension=dimension, shard_size=shard_size)
    if tuple(row.item for row in inputs.runtime_rows) != tuple(inputs.population):
        raise ValueError("embedding texts do not cover the exact ordered population")
    checkpoint_path = run_root / "embedding_checkpoint.json"
    shard_root = run_root / "embedding_shards"
    if isinstance(encoder, LocalQwenEncoder):
        snapshot = encoder.snapshot_manifest()
        if manifest.get("model_snapshot_files", snapshot) != snapshot:
            raise ValueError("local model snapshot changed since this run began")
        manifest["model_snapshot_files"] = snapshot
        _write_json(run_root / "manifest.json", manifest)
    shard_root.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        checkpoint = load_store_checkpoint(checkpoint_path)
        validate_store_resume(
            inputs.population,
            checkpoint,
            shard_directory=shard_root,
            expected_binding=manifest["embedding_binding"],
            shard_size=shard_size,
            dimension=dimension,
        )
        records = list(checkpoint.shards)
        completed = checkpoint.completed_count
    else:
        if any(shard_root.iterdir()):
            raise FileExistsError("uncommitted embedding shards exist; use a fresh run directory")
        records = []
        completed = 0
    for start in range(completed, len(inputs.population), shard_size):
        rows = inputs.runtime_rows[start : start + shard_size]
        vectors, lengths = encoder([row.prepared_text for row in rows])
        path = shard_root / f"shard-{len(records):05d}.npz"
        write_embedding_shard(
            path,
            items=[row.item for row in rows],
            token_lengths=lengths,
            vectors=vectors,
            dimension=dimension,
        )
        # This rereads and validates the saved shard, including exact identities.
        records.append(build_shard_record(path, shard_index=len(records), dimension=dimension))
        save_store_checkpoint(
            checkpoint_path,
            EmbeddingStoreCheckpoint(
                binding=manifest["embedding_binding"],
                completed_count=start + len(rows),
                next_shard_index=len(records),
                shards=tuple(records),
                forced_reload_verified=True,
            ),
        )
    checkpoint = EmbeddingStoreCheckpoint(
        binding=manifest["embedding_binding"],
        completed_count=len(inputs.population),
        next_shard_index=len(records),
        shards=tuple(records),
        forced_reload_verified=True,
    )
    save_store_checkpoint(checkpoint_path, checkpoint)
    manifest.update(
        status="embeddings_ready", embedding_checkpoint_sha256=_file_sha256(checkpoint_path)
    )
    _write_json(run_root / "manifest.json", manifest)


def _embedding_table(inputs: object, run_root: Path, manifest: dict[str, object]):
    path = run_root / "embedding_checkpoint.json"
    if _file_sha256(path) != manifest.get("embedding_checkpoint_sha256"):
        raise ValueError("embedding checkpoint hash differs from run manifest")
    return load_validated_embedding_table(
        inputs.population,
        load_store_checkpoint(path),
        shard_directory=run_root / "embedding_shards",
        expected_binding=manifest["embedding_binding"],
        shard_size=int(manifest["shard_size"]),
        dimension=int(manifest["dimension"]),
    )


def build_routes(inputs: object, run_root: Path) -> None:
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    prepare_run(
        inputs,
        run_root,
        dimension=int(manifest["dimension"]),
        shard_size=int(manifest["shard_size"]),
    )
    table = _embedding_table(inputs, run_root, manifest)
    route_root = run_root / "route_bundles"
    checkpoint_path = run_root / "route_checkpoint.json"
    records = []
    if checkpoint_path.exists():
        checkpoint = load_route_checkpoint(checkpoint_path)
        if checkpoint["binding"] != manifest["embedding_binding"]:
            raise ValueError("route checkpoint belongs to different inputs")
        records = checkpoint["records"]
        for record in records:
            if _file_sha256(route_root / record["filename"]) != record["sha256"]:
                raise ValueError("completed route bundle changed")
    expected_files = {record["filename"] for record in records}
    observed_files = {path.name for path in route_root.glob("*.jsonl.gz")}
    if observed_files != expected_files:
        raise ValueError(
            "route directory has missing or uncommitted bundles; use a fresh run directory"
        )
    raw_texts = {}
    prefix = f"Instruct: {QUERY_INSTRUCTION}\nQuery:"
    for row in inputs.runtime_rows:
        text = row.prepared_text
        if row.item.kind == "query":
            if not text.startswith(prefix):
                raise ValueError("query preparation differs from the frozen instruction")
            text = text[len(prefix) :]
        raw_texts[row.item.embedding_identity] = text
    settings = load_frozen_settings(PROTOCOL)
    expected_task_order = [
        f"{source}/{setting.arm}/{setting.ranking_config_id}"
        for source in inputs.corpora
        for setting in settings
    ]
    if [record["task_id"] for record in records] != expected_task_order[: len(records)]:
        raise ValueError("completed route tasks are not the frozen task prefix")
    completed_tasks = {record["task_id"] for record in records}
    for source, corpus in inputs.corpora.items():
        views = build_eval_method_views(
            source,
            [m for m in corpus.memories if m.split == "eval"],
            [q for q in corpus.queries if q.split == "eval"],
            raw_texts=raw_texts,
            vector_indices=table.index_by_identity,
        )
        index = NaturalIndex(views, table.vectors)
        for setting in settings:
            task_id = f"{source}/{setting.arm}/{setting.ranking_config_id}"
            if task_id in completed_tasks:
                continue
            observations = route_ranking_config(index, setting)
            filename = f"{len(records):02d}-{source}-{setting.ranking_config_id}.jsonl.gz"
            path = route_root / filename
            write_route_bundle(path, observations)
            records.append(
                {
                    "task_id": task_id,
                    "filename": filename,
                    "row_count": len(observations),
                    "sha256": _file_sha256(path),
                    "query_bundle_sha256": hashlib.sha256(
                        "\n".join(row.query_id for row in observations).encode()
                    ).hexdigest(),
                }
            )
            _write_json(
                checkpoint_path,
                {
                    "binding": manifest["embedding_binding"],
                    "completed_tasks": len(records),
                    "records": records,
                },
            )
        del index
        gc.collect()
    checkpoint = {
        "binding": manifest["embedding_binding"],
        "completed_tasks": len(records),
        "records": records,
    }
    path = run_root / "route_checkpoint.json"
    _write_json(path, checkpoint)
    manifest.update(
        status="routes_ready",
        route_checkpoint_sha256=_file_sha256(path),
        route_rows=sum(record["row_count"] for record in records),
    )
    _write_json(run_root / "manifest.json", manifest)


def validate_run(inputs: object, run_root: Path) -> dict[str, object]:
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        sample_bundle_sha256(inputs.population)
        != manifest["embedding_binding"]["population_bundle_sha256"]
    ):
        raise ValueError("reconstructed population differs from run binding")
    _embedding_table(inputs, run_root, manifest)
    checkpoint_path = run_root / "route_checkpoint.json"
    if _file_sha256(checkpoint_path) != manifest.get("route_checkpoint_sha256"):
        raise ValueError("route checkpoint hash differs from run manifest")
    checkpoint = load_route_checkpoint(checkpoint_path)
    if checkpoint["binding"] != manifest["embedding_binding"]:
        raise ValueError("route checkpoint is bound to different inputs")
    expected = {
        f"{source}/{setting.arm}/{setting.ranking_config_id}"
        for source in inputs.corpora
        for setting in load_frozen_settings(PROTOCOL)
    }
    if {row["task_id"] for row in checkpoint["records"]} != expected:
        raise ValueError("route checkpoint does not contain all nine arms per source")
    total = 0
    for record in checkpoint["records"]:
        path = run_root / "route_bundles" / record["filename"]
        if _file_sha256(path) != record["sha256"]:
            raise ValueError("route bundle hash drifted")
        rows = load_route_bundle(path)
        source = record["task_id"].split("/")[0]
        queries = {q.query_id: q for q in inputs.corpora[source].queries if q.split == "eval"}
        if set(queries) != {row.query_id for row in rows} or len(rows) != record["row_count"]:
            raise ValueError("route bundle query coverage differs from source inputs")
        for row in rows:
            allowed = set(
                inputs.corpora[source].candidate_ids(queries[row.query_id], namespace_only=False)
            )
            if not set(row.ranked_memory_ids) <= allowed:
                raise ValueError("route contains a memory outside the visible candidate set")
        total += len(rows)
    return {
        "status": "valid",
        "route_tasks": len(expected),
        "route_rows": total,
        "provider_calls": 0,
        "historical_paper_result": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "embed", "route", "validate"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    needs_text = args.command in {"embed", "route"}
    inputs = load_eval_inputs(args.source_root, include_texts=needs_text)
    if args.command == "build":
        result = prepare_run(inputs, args.run_root)
    elif args.command == "embed":
        if args.model_path is None:
            parser.error("embed requires --model-path pointing to the pinned local snapshot")
        encoder = LocalQwenEncoder(args.model_path, batch_size=args.batch_size, device=args.device)
        build_embedding_store(inputs, args.run_root, encoder)
        result = {"status": "embeddings_ready", "provider_calls": 0}
    elif args.command == "route":
        build_routes(inputs, args.run_root)
        result = {"status": "routes_ready", "provider_calls": 0}
    else:
        result = validate_run(inputs, args.run_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
