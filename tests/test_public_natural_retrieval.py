"""Provider-free checks of the portable nine-arm execution and integrity boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_public_natural_retrieval as run
from verify_agent_memory.natural_corpus import NaturalCorpus, NaturalMemory, NaturalQuery
from verify_agent_memory.natural_inputs import (
    QUERY_INSTRUCTION,
    EvalInputs,
    RuntimeText,
    build_eval_embedding_population,
)
from verify_agent_memory.natural_retrieval import (
    lexical_tie_ranks,
    load_frozen_settings,
    stable_rank_indices,
)
from verify_agent_memory.natural_storage import (
    load_public_api,
    load_route_bundle,
    load_route_checkpoint,
)


def _inputs(root: Path, *, two_groups: bool = False) -> EvalInputs:
    corpora = {}
    texts = {}
    for source in ("rhelm", "memops"):
        memories = []
        for index, (namespace, state, distractor) in enumerate(
            (
                ("a", "current", False),
                ("a", "stale", False),
                ("b", "current", False),
                ("a", "current", True),
            )
        ):
            text = f"{source} alpha fact {index}"
            memory = NaturalMemory.from_text(
                source=source,
                memory_id=f"{source}-m{index}",
                namespace_id=namespace,
                split="eval",
                chronology=f"{index:03d}",
                text=text,
                source_date=("2020-01-03" if index == 1 else "2020-01-01")
                if source == "rhelm"
                else None,
                state=state,
                released_distractor=distractor,
            )
            memories.append(memory)
            texts[("memory", memory.text_sha256)] = text
        queries = []
        intents = ("current_state", "history") * (2 if two_groups else 1)
        for index, intent in enumerate(intents):
            text = f"{source} alpha question {index}"
            query = NaturalQuery.from_text(
                source=source,
                query_id=f"{source}-q{index}",
                namespace_id="b" if index >= 2 else "a",
                split="eval",
                query_text=text,
                query_intent=intent,
                required_memory_ids=(f"{source}-m{2 if index >= 2 else 0}",),
                cutoff_date="2020-01-02" if source == "rhelm" else None,
            )
            queries.append(query)
            texts[("query", query.query_text_sha256)] = text
        corpora[source] = NaturalCorpus(source, memories, queries)
    population = build_eval_embedding_population(corpora)
    rows = tuple(
        RuntimeText(
            item,
            f"Instruct: {QUERY_INSTRUCTION}\nQuery:{texts[item.embedding_identity]}"
            if item.kind == "query"
            else texts[item.embedding_identity],
        )
        for item in population
    )
    return EvalInputs(corpora, population, rows, root)


def _encoder(texts):
    # Exact ties deliberately exercise lexical-ID ordering without a model.
    vectors = np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)
    return vectors, tuple(8 for _ in texts)


def _completed_fixture(tmp_path: Path):
    inputs = _inputs(tmp_path / "sources")
    root = tmp_path / "run"
    run.build_embedding_store(inputs, root, _encoder, dimension=2, shard_size=3)
    run.build_routes(inputs, root)
    return inputs, root


def test_all_frozen_settings_keep_historical_config_identities():
    settings = load_frozen_settings(run.PROTOCOL)
    assert [s.ranking_config_id for s in settings] == [
        "063d0254c05e",
        "25ea22a7b2be",
        "87ff6b7bb649",
        "c73a827984b3",
        "ed4a7e3ae49d",
        "8df8253cca8a",
        "e2c5a29f60be",
        "eddf06a5c1fe",
        "61f68fe5bfd1",
    ]


def test_nine_arms_reconstruct_routes_without_archive_or_model(tmp_path):
    inputs, root = _completed_fixture(tmp_path)
    result = run.validate_run(inputs, root)
    assert result["route_tasks"] == 18
    assert result["route_rows"] == 36
    assert result["historical_paper_result"] is False
    checkpoint = load_route_checkpoint(root / "route_checkpoint.json")
    routes = {}
    for record in checkpoint["records"]:
        source, arm, _ = record["task_id"].split("/")
        routes[(source, arm)] = load_route_bundle(root / "route_bundles" / record["filename"])
    assert routes[("memops", "flat_dense")][0].ranked_memory_ids == (
        "memops-m0",
        "memops-m1",
        "memops-m2",
        "memops-m3",
    )
    # Strictly future RHELM rows never enter any method's candidates.
    assert all(
        "rhelm-m1" not in row.ranked_memory_ids
        for (source, _), rows in routes.items()
        if source == "rhelm"
        for row in rows
    )
    assert "memops-m2" not in routes[("memops", "namespace_dense")][0].ranked_memory_ids
    assert "memops-m1" not in routes[("memops", "namespace_current_only")][1].ranked_memory_ids
    lifecycle = routes[("memops", "namespace_source_intent_lifecycle")]
    assert "memops-m3" not in lifecycle[0].ranked_memory_ids
    assert "memops-m1" in lifecycle[1].ranked_memory_ids
    assert "memops-m3" in lifecycle[1].ranked_memory_ids


def test_completed_stages_resume_without_reencoding_or_rerouting(tmp_path, monkeypatch):
    inputs, root = _completed_fixture(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("completed work must not be executed again")

    run.build_embedding_store(inputs, root, fail, dimension=2, shard_size=3)
    monkeypatch.setattr(run, "route_ranking_config", fail)
    run.build_routes(inputs, root)
    assert run.validate_run(inputs, root)["route_tasks"] == 18


def test_interrupted_embedding_resumes_exact_completed_shards(tmp_path):
    inputs = _inputs(tmp_path / "sources")
    root = tmp_path / "run"
    calls = 0

    def partial(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture interruption")
        return _encoder(texts)

    with pytest.raises(RuntimeError, match="fixture interruption"):
        run.build_embedding_store(inputs, root, partial, dimension=2, shard_size=3)
    observed = []

    def remaining(texts):
        observed.extend(texts)
        return _encoder(texts)

    run.build_embedding_store(inputs, root, remaining, dimension=2, shard_size=3)
    assert len(observed) == len(inputs.population) - 3
    run.build_routes(inputs, root)
    assert run.validate_run(inputs, root)["route_rows"] == 36


def test_interrupted_routes_resume_completed_task_prefix(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path / "sources")
    root = tmp_path / "run"
    run.build_embedding_store(inputs, root, _encoder, dimension=2, shard_size=3)
    real_route = run.route_ranking_config
    calls = 0

    def partial(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("fixture route interruption")
        return real_route(*args)

    monkeypatch.setattr(run, "route_ranking_config", partial)
    with pytest.raises(RuntimeError, match="fixture route interruption"):
        run.build_routes(inputs, root)
    calls = 0

    def remaining(*args):
        nonlocal calls
        calls += 1
        return real_route(*args)

    monkeypatch.setattr(run, "route_ranking_config", remaining)
    run.build_routes(inputs, root)
    assert calls == 16
    assert run.validate_run(inputs, root)["route_rows"] == 36


def test_public_run_rejects_tampered_shard_and_route(tmp_path):
    inputs, root = _completed_fixture(tmp_path)
    shard = root / "embedding_shards" / "shard-00000.npz"
    original = shard.read_bytes()
    shard.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        run.validate_run(inputs, root)
    shard.write_bytes(original)
    checkpoint = load_route_checkpoint(root / "route_checkpoint.json")
    path = root / "route_bundles" / checkpoint["records"][0]["filename"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="route bundle hash drifted"):
        run.validate_run(inputs, root)


def test_public_binding_rejects_replaced_embedding_checkpoint(tmp_path):
    inputs, root = _completed_fixture(tmp_path)
    path = root / "embedding_checkpoint.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="embedding checkpoint differs"):
        load_public_api(inputs.source_root, root)
    checkpoint = json.loads(path.read_text())
    checkpoint["binding"]["model_revision"] = "different revision"
    path.write_text(json.dumps(checkpoint), encoding="utf-8", newline="\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["embedding_checkpoint_sha256"] = run._file_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="embedding checkpoint binding differs"):
        load_public_api(inputs.source_root, root)


def test_stable_ties_and_nonfinite_input_rejection():
    ranks = lexical_tie_ranks(("z", "a", "b"))
    assert stable_rank_indices(np.asarray([1, 1, 0], dtype=np.float32), ranks).tolist() == [1, 0, 2]
    with pytest.raises(ValueError, match="stable ranking"):
        stable_rank_indices(np.asarray([np.nan, 1, 0]), ranks)


def test_float32_renormalization_rejects_zero_and_persists_unit_rows():
    rows = np.asarray([[1.5, 2.7, -0.001], [1e-8, 0, 3]], dtype=np.float32)
    normalized = run._renormalize_float32(rows)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalized.astype(np.float64), axis=1), 1, atol=1e-7)
    with pytest.raises(RuntimeError, match="zero embedding"):
        run._renormalize_float32(np.zeros((1, 2)))


@pytest.mark.parametrize(
    "name",
    [
        "run_frozen_natural_support_expansion",
        "run_frozen_natural_support_controls",
        "run_policy_axis_sensitivity",
        "run_submission_zero_call_diagnostics",
    ],
)
def test_diagnostic_public_binding_does_not_import_legacy_archive(tmp_path, monkeypatch, name):
    inputs, root = _completed_fixture(tmp_path)
    from verify_agent_memory import natural_inputs

    monkeypatch.setattr(natural_inputs, "load_eval_inputs", lambda *a, **kw: inputs)
    module = importlib.import_module(f"scripts.{name}")
    api = module._input_bindings(
        tmp_path / "absent-archive", source_root=inputs.source_root, run_root=root
    )
    assert api["load_eval_inputs"](include_model=False).population == inputs.population
    assert api["route_root"] == root / "route_bundles"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["execution_kind"] == "public_source_reproduction"


def test_public_diagnostics_use_current_run_baselines(tmp_path, monkeypatch):
    from scripts import run_frozen_natural_support_controls as controls
    from scripts import run_frozen_natural_support_expansion as expansion
    from scripts import run_policy_axis_sensitivity as policy
    from scripts import run_submission_zero_call_diagnostics as diagnostics
    from verify_agent_memory import natural_inputs

    inputs = _inputs(tmp_path / "sources", two_groups=True)
    root = tmp_path / "run"
    run.build_embedding_store(inputs, root, _encoder, dimension=2, shard_size=3)
    run.build_routes(inputs, root)
    monkeypatch.setattr(natural_inputs, "load_eval_inputs", lambda *a, **kw: inputs)
    missing = tmp_path / "absent-archive"
    kwargs = {"source_root": inputs.source_root, "run_root": root}
    controls._run(missing, root / "support_controls", 2, **kwargs)
    policy._run(missing, root / "policy_axis_sensitivity", **kwargs)
    # With only two groups, a full label swap removes every feasible query.
    # Exercise the unchanged zero-perturbation baseline and run binding here.
    channels, _rates, seeds, top_k = expansion._load_protocol()
    monkeypatch.setattr(expansion, "_load_protocol", lambda: (channels, (0.0,), seeds, top_k))
    expansion._run(missing, root / "support_expansion", 2, **kwargs)
    diagnostics._run(missing, root / "submission_diagnostics", 2, **kwargs)
    result = json.loads((root / "submission_diagnostics" / "manifest.json").read_text())
    assert result["query_count"] == 8
    assert result["group_count"] == 4
    assert result["support_control_main_sha256"] == run._file_sha256(
        root / "support_controls" / "support_control_main.csv"
    )
    baseline = root / "support_controls" / "support_control_main.csv"
    baseline.write_bytes(baseline.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="baseline hash"):
        diagnostics._support_input_path(root)
