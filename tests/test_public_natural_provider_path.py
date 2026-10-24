from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import materialize_natural_end_to_end_cases as materializer
from scripts import run_natural_end_to_end_experiment as run
from verify_agent_memory.natural_end_to_end import load_cases


def _forbidden(*_args, **_kwargs):
    raise AssertionError("this operation must not access an archive, credentials, or a provider")


def _inputs(tmp_path: Path):
    cases_path = tmp_path / "cases.jsonl.gz"
    manifest_path = tmp_path / "materialization_manifest.json"
    materializer._write_gzip_jsonl(cases_path, [run._fixture_case().as_dict()])
    manifest = {
        "schema_version": 1,
        "status": "public_source_reproduction_materialization",
        "provider_calls": 0,
        "paid_calls": 0,
        "case_bundle_sha256": run._sha256_file(cases_path),
        "route_checkpoint_sha256": "a" * 64,
        "source_manifest_sha256": run._sha256_file(run.ROOT / "data" / "upstream_sources.json"),
        "candidate_depth": 20,
        "reader_memory_max_utf8_bytes": 8192,
        "case_count": 1,
    }
    run._write_json(manifest_path, manifest)
    return cases_path, manifest_path


def _fresh(tmp_path: Path):
    cases_path, manifest_path = _inputs(tmp_path)
    source = run.load_protocol(run.TWO_READER_PROTOCOL)
    protocol = run.bind_reproduction_protocol(
        source, cases_path=cases_path, materialization_path=manifest_path
    )
    return protocol, cases_path, manifest_path


def test_public_materializer_joins_fresh_rankings_without_legacy_runtime(tmp_path, monkeypatch):
    source_root = tmp_path / "sources"
    run_root = tmp_path / "run"
    bundle_root = run_root / "route_bundles"
    bundle_root.mkdir(parents=True)
    corpora = {}
    runtime_rows = []
    records = []
    gold = {}
    for source, count in (("rhelm", 523), ("memops", 3244)):
        memory = SimpleNamespace(
            memory_id=f"{source}-memory",
            namespace_id=f"{source}-group",
            split="eval",
            text_sha256=f"{source}-text",
            source=source,
            chronology="2026-01-01",
            state="current",
            released_distractor=False,
        )
        queries = [
            SimpleNamespace(
                query_id=f"{source}-q{index:04d}",
                namespace_id=memory.namespace_id,
                split="eval",
                required_memory_ids=(memory.memory_id,),
                query_text_sha256=f"{source}-query",
                query_intent="current_state",
                cutoff_date=None,
            )
            for index in range(count)
        ]
        corpora[source] = SimpleNamespace(queries=queries, memories=(memory,))
        gold[source] = {
            query.query_id: {"answer": "Paris", "expected_answer": "Paris"} for query in queries
        }
        for kind, identity, text in (
            ("memory", memory.text_sha256, "Current location: Paris.\N{GRINNING FACE}"),
            ("query", f"{source}-query", "Instruct: retrieve\nQuery:Where now?"),
        ):
            runtime_rows.append(
                SimpleNamespace(
                    item=SimpleNamespace(embedding_identity=(kind, identity)), prepared_text=text
                )
            )
        for arm in ("flat_dense", "namespace_dense"):
            filename = f"{source}-{arm}.json"
            path = bundle_root / filename
            run._write_json(
                path,
                {
                    "rows": [
                        {"query_id": query.query_id, "ranked_memory_ids": [memory.memory_id]}
                        for query in queries
                    ]
                },
            )
            records.append(
                {
                    "task_id": f"{source}/{arm}/"
                    + ("25ea22a7b2be" if arm == "flat_dense" else "ed4a7e3ae49d"),
                    "filename": filename,
                    "sha256": run._sha256_file(path),
                }
            )
    checkpoint_path = run_root / "route_checkpoint.json"
    run._write_json(checkpoint_path, {"records": records})
    run._write_json(run_root / "manifest.json", {"status": "fixture"})
    run._write_json(run_root / "retrieval_config.json", {"status": "fixture"})
    calls = []

    def public_api(sources, root):
        calls.append((sources, root))
        return {
            "SOURCE_CACHE": sources,
            "ROUTE_BUNDLES": root / "route_bundles",
            "ROUTE_CHECKPOINT_PATH": root / "route_checkpoint.json",
            "load_eval_inputs": lambda **_: SimpleNamespace(
                corpora=corpora, runtime_rows=runtime_rows, population=()
            ),
            "load_route_checkpoint": run._read_json,
            "load_route_bundle": lambda path: tuple(
                SimpleNamespace(**row) for row in run._read_json(path)["rows"]
            ),
        }

    monkeypatch.setattr(materializer, "_public_api", public_api)
    monkeypatch.setattr(materializer, "_legacy_api", _forbidden)
    monkeypatch.setattr(materializer, "_rhelm_gold", lambda *_: gold["rhelm"])
    monkeypatch.setattr(materializer, "_memops_gold", lambda *_: gold["memops"])
    manifest = materializer.materialize(source_root=source_root, run_root=run_root)
    cases_path = run_root / "natural_end_to_end" / "cases.jsonl.gz"
    manifest_path = cases_path.with_name("materialization_manifest.json")
    cases = load_cases(cases_path)
    assert len(cases) == 3767
    assert calls == [(source_root.resolve(), run_root.resolve())]
    assert manifest["route_checkpoint_sha256"] == run._sha256_file(checkpoint_path)
    assert manifest["provider_calls"] == manifest["paid_calls"] == 0
    assert "legacy_runtime_commit" not in manifest
    assert all(
        case.query_text == "Where now?" and case.expected_answer == "Paris" for case in cases
    )
    assert all(case.namespace_candidates[0].scope_allowed is True for case in cases)
    source = run.load_protocol(run.TWO_READER_PROTOCOL)
    fresh = run.bind_reproduction_protocol(
        source, cases_path=cases_path, materialization_path=manifest_path
    )
    audit = run.validate_runtime(
        protocol=fresh,
        cases_path=cases_path,
        materialization_path=manifest_path,
        runtime=tmp_path / "runtime",
    )
    assert audit["verifier_request_count"] == 3767
    assert audit["source_counts"] == {"rhelm": 523, "memops": 3244}
    with pytest.raises(ValueError, match="checkpoint differs"):
        run.validate_runtime(
            protocol=source,
            cases_path=cases_path,
            materialization_path=manifest_path,
            runtime=tmp_path / "runtime",
        )
    with pytest.raises(FileExistsError):
        materializer.materialize(source_root=source_root, run_root=run_root)
    (bundle_root / records[0]["filename"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="route bundle hash drifted"):
        materializer.materialize(source_root=source_root, run_root=run_root, overwrite=True)


def test_public_route_selection_rejects_ambiguous_or_missing_arms():
    record = {"task_id": "rhelm/flat_dense/25ea22a7b2be"}
    assert materializer._public_route_record([record], "rhelm", "global") is record
    with pytest.raises(ValueError, match="exactly one"):
        materializer._public_route_record([record, record], "rhelm", "global")
    with pytest.raises(ValueError, match="exactly one"):
        materializer._public_route_record([record], "rhelm", "namespace")
    with pytest.raises(ValueError, match="configuration differs"):
        materializer._public_route_record(
            [{"task_id": "rhelm/flat_dense/another-setting"}], "rhelm", "global"
        )


def test_real_public_api_checks_completed_run_and_population(tmp_path, monkeypatch):
    from test_public_natural_retrieval import _completed_fixture

    from verify_agent_memory import natural_inputs

    inputs, root = _completed_fixture(tmp_path)
    monkeypatch.setattr(natural_inputs, "load_eval_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(materializer, "_legacy_api", _forbidden)
    api = materializer._public_api(inputs.source_root, root)
    loaded = api["load_eval_inputs"](include_texts=True, include_model=False)
    assert loaded is inputs
    assert materializer._source_directory(api, "rhelm") == inputs.source_root / "rhelm"
    checkpoint = api["load_route_checkpoint"](api["ROUTE_CHECKPOINT_PATH"])
    for source in ("rhelm", "memops"):
        for arm in ("global", "namespace"):
            record = materializer._public_route_record(checkpoint["records"], source, arm)
            observations = api["load_route_bundle"](api["ROUTE_BUNDLES"] / record["filename"])
            assert {row.query_id for row in observations} == {
                query.query_id for query in inputs.corpora[source].queries
            }
    monkeypatch.setattr(
        natural_inputs,
        "load_eval_inputs",
        lambda *_args, **_kwargs: replace(inputs, population=inputs.population[:-1]),
    )
    with pytest.raises(ValueError, match="source population differs"):
        materializer._public_api(inputs.source_root, root)["load_eval_inputs"](include_texts=True)
    (root / "retrieval_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="retrieval configuration drifted"):
        materializer._public_api(inputs.source_root, root)


def test_reproduction_binding_preserves_science_and_rejects_input_drift(tmp_path):
    fresh, cases_path, manifest_path = _fresh(tmp_path)
    source = run.load_protocol(run.TWO_READER_PROTOCOL)
    assert fresh.mode == "reproduction"
    assert fresh.protocol_sha256 != source.protocol_sha256
    assert (fresh.target_recall, fresh.verifier_threshold) == (0.8, 0.95)
    assert fresh.readers == source.readers
    assert fresh.reader_incremental_hard_caps == source.reader_incremental_hard_caps
    assert fresh.reader_prompt == source.reader_prompt
    assert fresh.judge.model == "claude-haiku-4-5-20251001"
    assert fresh.judge.controls == {"thinking": "disabled", "effort": "omitted"}
    assert fresh.maximum_output_tokens_judge == 512
    assert fresh.raw["reproduction_binding"]["final_judge_protocol_sha256"]
    manifest = dict(run._read_json(manifest_path))
    manifest["route_checkpoint_sha256"] = "b" * 64
    run._write_json(manifest_path, manifest)
    changed = run.bind_reproduction_protocol(
        source, cases_path=cases_path, materialization_path=manifest_path
    )
    assert changed.protocol_sha256 != fresh.protocol_sha256
    with pytest.raises(ValueError, match="materialization binding drifted"):
        run.validate_runtime(
            protocol=fresh,
            cases_path=cases_path,
            materialization_path=manifest_path,
            runtime=tmp_path / "runtime",
        )
    cases_path.write_bytes(b"changed cases")
    with pytest.raises(ValueError, match="bundle hash drifted"):
        run.bind_reproduction_protocol(
            source, cases_path=cases_path, materialization_path=manifest_path
        )


def test_fixture_planning_is_zero_call_and_retains_strict_schemas(tmp_path, monkeypatch):
    fresh, _, _ = _fresh(tmp_path)
    monkeypatch.setattr(run.provider_runtime, "_load_dotenv", _forbidden)
    monkeypatch.setattr(run, "_call_provider", _forbidden)
    for stage, binding in (
        ("verifier", fresh.verifier),
        ("reader", fresh.readers[0]),
        ("judge", fresh.judge),
    ):
        planned = run.plan_fixture(fresh, stage, binding)
        assert planned["provider_calls_made"] == 0
        assert planned["response_schema"]["additionalProperties"] is False
        assert planned["synthetic_fixture"] is True
        assert planned["conservative_cost_bound_usd"] > 0


def test_new_run_requires_its_own_complete_verifier_bundle(tmp_path, monkeypatch):
    fresh, cases_path, _ = _fresh(tmp_path)
    cases = load_cases(cases_path)
    with pytest.raises(FileNotFoundError):
        run._reader_plan(cases, fresh, tmp_path / "empty")
    specs, _ = run._verifier_specs(cases, fresh)
    with pytest.raises(RuntimeError, match="partial provider stage"):
        run._complete_stage(fresh, tmp_path, "verifier", fresh.verifier, specs, {})
    run._write_json(
        run._completion_path(tmp_path, "verifier", fresh.verifier),
        {
            "schema_version": 1,
            "protocol_sha256": "old-archive-hash",
            "stage": "verifier",
            "provider": fresh.verifier.provider,
            "model": fresh.verifier.model,
            "complete_bundle": True,
            "implementation_commit": "old-commit",
        },
    )
    with pytest.raises(ValueError, match="completed stage binding drifted"):
        run._reader_plan(cases, fresh, tmp_path)
    monkeypatch.setattr(run, "_execution_commit", lambda *_args, **_kwargs: "test-commit")
    records = {}
    for case in cases:
        spec = specs[case.case_id]
        record = {
            "schema_version": 1,
            "protocol_sha256": fresh.protocol_sha256,
            "implementation_commit": "test-commit",
            "stage": "verifier",
            "provider": fresh.verifier.provider,
            "model": fresh.verifier.model,
            "request_id": spec.request_id,
            "payload_sha256": hashlib.sha256(spec.payload.encode()).hexdigest(),
            "response": {
                "query_intent": {"current_state": 1, "history": 0, "unknown": 0},
                "candidates": [
                    {
                        "candidate_key": f"c{index:02d}",
                        "policy": {"allowed": 1, "disallowed": 0, "unknown": 0},
                        "lifecycle": {"compatible": 1, "incompatible": 0, "unknown": 0},
                        "admissibility": {"admissible": 1, "inadmissible": 0, "unknown": 0},
                    }
                    for index, _ in enumerate(case.namespace_candidates, start=1)
                ],
            },
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "cost_usd": 0.0,
        }
        run._write_json(
            run._response_path(tmp_path, "verifier", fresh.verifier, spec.request_id), record
        )
        records[spec.request_id] = record
    run._complete_stage(fresh, tmp_path, "verifier", fresh.verifier, specs, records)
    reader_specs, assignments, _ = run._reader_plan(cases, fresh, tmp_path)
    assert reader_specs
    assert len(assignments) == len(cases) * 5


def test_fresh_haiku_fixture_uses_final_adapter_without_network(tmp_path, monkeypatch):
    fresh, _, _ = _fresh(tmp_path)
    body_seen = []
    response = {
        "answer_correct": True,
        "answer_quality": 10,
        "protected_disclosure": False,
        "stale_disclosure": False,
        "reason": "fixture",
    }

    def fake_http(**kwargs):
        body_seen.append(kwargs["body"])
        return (
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": json.dumps(response)}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            1,
            1.0,
        )

    monkeypatch.setattr(run, "_require_clean_contract", lambda: "test-commit")
    monkeypatch.setattr(
        run.provider_runtime,
        "_load_dotenv",
        lambda _: {run.provider_runtime._credential_key("Anthropic"): "synthetic-test-key"},
    )
    monkeypatch.setattr(run.provider_runtime, "_http_json", fake_http)
    receipt = run.run_fixture(
        protocol=fresh,
        runtime=tmp_path / "runtime",
        stage="judge",
        binding=fresh.judge,
        dotenv=tmp_path / "does-not-exist.env",
    )
    assert len(body_seen) == 1
    assert body_seen[0]["model"] == "claude-haiku-4-5-20251001"
    assert body_seen[0]["max_tokens"] == 512
    assert "effort" not in body_seen[0]["output_config"]
    assert receipt["response"] is None
    assert receipt["fixture_hard_cap_usd"] == 0.01


def test_implementation_identity_works_without_git_and_tracks_source_changes(tmp_path, monkeypatch):
    module = tmp_path / "src" / "verify_agent_memory" / "example.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    first = materializer.implementation_provenance(tmp_path)
    assert "implementation_commit" not in first
    assert len(first["implementation_source_sha256"]) == 64
    fresh, _, _ = _fresh(tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "_require_clean_contract", _forbidden)
    identity = run._execution_commit(fresh, require_clean=True)
    assert identity == f"source-sha256:{first['implementation_source_sha256']}"
    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert run._execution_commit(fresh) != identity


def test_semantic_sample_is_outcome_independent_and_preserves_paper_counts(tmp_path):
    fresh, _, _ = _fresh(tmp_path)
    fixture = run._fixture_case()
    population = tuple(
        replace(fixture, source=source, case_id=f"{source}-{index:04d}")
        for source, count in (("rhelm", 523), ("memops", 3244))
        for index in range(count)
    )
    selected = run.reproduction_analysis_cases(population, fresh)
    assert len(selected) == 1523
    assert sum(case.source == "rhelm" for case in selected) == 523
    assert sum(case.source == "memops" for case in selected) == 1000
    changed_answers = tuple(
        replace(case, expected_answer="different") for case in reversed(population)
    )
    assert [case.case_id for case in run.reproduction_analysis_cases(changed_answers, fresh)] == [
        case.case_id for case in selected
    ]
