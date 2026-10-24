from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from verify_agent_memory.natural_corpus import candidate_label, lifecycle_allows
from verify_agent_memory.natural_inputs import (
    QUERY_INSTRUCTION,
    EmbeddingSampleItem,
    load_eval_inputs,
    validate_eval_population_counts,
)
from verify_agent_memory.natural_memops import (
    EvidenceState,
    map_source_span,
    query_intent,
    replay_operation_states,
)
from verify_agent_memory.natural_rhelm import (
    ResolutionStatus,
    iter_rhelm_question_resolutions,
    resolve_rhelm_evidence,
)
from verify_agent_memory.natural_source_indices import load_rhelm_qa_index


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _span(turn: int) -> dict[str, int]:
    return {"segment_index": 1, "turn_index": turn}


def _memops_case() -> tuple[dict, dict]:
    dialogue = [
        {"role": "user", "content": "old setting"},
        {"role": "assistant", "content": "updated setting"},
        {"role": "user", "content": "forget the setting"},
        {"role": "assistant", "content": "tentative setting"},
    ]
    operations = [
        {
            "type": kind,
            "validity": validity,
            "target": {"target_id": "setting"},
            "evidence_spans": [_span(turn)],
            "trigger_span": _span(turn),
        }
        for turn, kind, validity in [
            (1, "remember", "confirmed"),
            (2, "update", "confirmed"),
            (3, "forget", "confirmed"),
            (4, "remember", "tentative"),
        ]
    ]
    source = {
        "conversations": [{"segment_index": 1, "dialogue": dialogue}],
        "operations": operations,
    }
    injected = {
        "conversations": [
            {
                "segment_index": 10,
                "evidence_segment_index": 1,
                "insertion_index": 1,
                "dialogue": [{"role": "user", "content": "distractor"}, *dialogue],
            }
        ],
        "injection_metadata": {
            "insertions": [
                {
                    "evidence_segment_index": 1,
                    "conversation_index": 10,
                    "insertion_index": 1,
                    "evidence_message_count": 4,
                }
            ],
            "distractor_insertions": [
                {
                    "conversation_index": 10,
                    "insertion_index": 0,
                    "distractor_message_count": 1,
                }
            ],
        },
        "answer": [
            {
                "question_pair_id": "q1",
                "evaluation_setting": "full",
                "evaluation_type": "StateTrajectory",
                "question": "Where is the record?",
                "gold_provenance": [_span(3), _span(3)],
            }
        ],
    }
    return source, injected


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    for group in range(1, 5):
        namespace = f"person-{group}"
        root = tmp_path / "rhelm" / "data"
        for day, content in [(1, "old"), (2, "same day"), (3, "future")]:
            _json(
                root / "conversations" / namespace / f"conversation_2020_01_0{day}.json",
                {
                    "date": f"2020-01-0{day}",
                    "conversation": [{"turn": 1, "user": content, "assistant": "reply"}],
                },
            )
        qa = root / "QA_final" / f"low_score_qa_{namespace}_all_validated.jsonl"
        qa.parent.mkdir(parents=True, exist_ok=True)
        base = {
            "question_date": "2020-01-02",
            "question_type": "factual",
            "question": "Where is the record?",
            "answer": "here",
        }
        rows = [
            {**base, "id": "kept", "supporting_evidence": ["2020-01-01:1"] * 2},
            {**base, "id": "future", "supporting_evidence": ["2020-01-02:1"]},
            {
                **base,
                "id": "mixed",
                "supporting_evidence": ["2020-01-01:1", "Emails_2020-01-01"],
            },
        ]
        qa.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    for group in range(1, 6):
        source, injected = _memops_case()
        filename = f"P{group}_remember.json"
        _json(tmp_path / "memops/generated_result/2-evidence_conversation" / filename, source)
        _json(
            tmp_path / "memops/generated_result/4-inject_evidence_with_distractors" / filename,
            injected,
        )
    return tmp_path


def test_reconstruct_sources_and_candidate_boundaries(sources: Path) -> None:
    inputs = load_eval_inputs(sources, validate_expected=False)
    assert inputs.source_root == sources
    assert inputs.runtime_rows == ()
    rhelm = inputs.corpora["rhelm"]
    memops = inputs.corpora["memops"]
    assert len(rhelm.memories) == 12
    assert len(rhelm.queries) == 4
    assert len(memops.memories) == 25
    assert len(memops.queries) == 5
    assert len(rhelm.groups_by_split["dev"]) == 3
    assert len(rhelm.groups_by_split["eval"]) == 1
    assert len(memops.groups_by_split["dev"]) == 1
    assert len(memops.groups_by_split["eval"]) == 4
    for query in rhelm.queries:
        assert query.query_id.endswith("/kept")
        assert query.cutoff_date == "2020-01-02"
        assert len(query.required_memory_ids) == 1
        assert rhelm.candidate_ids(query, namespace_only=True) == query.required_memory_ids
        assert rhelm.candidate_count(query, namespace_only=True) == 1
        by_id = {memory.memory_id: memory for memory in rhelm.memories}
        for memory_id in rhelm.candidate_ids(query, namespace_only=False):
            assert by_id[memory_id].source_date < query.cutoff_date
            assert by_id[memory_id].split == query.split
    for query in memops.queries:
        assert query.query_intent == "history"
        assert query.cutoff_date is None
        assert query.required_memory_ids == (
            f"{query.namespace_id}_remember/message/segment-10/item-4",
        )
        assert memops.candidate_count(query, namespace_only=True) == 5
    with pytest.raises(ValueError, match="population drifted"):
        validate_eval_population_counts(inputs.corpora)


def test_eval_population_deduplication_and_exact_text_preparation(sources: Path) -> None:
    inputs = load_eval_inputs(sources, include_texts=True, validate_expected=False)
    population = inputs.population
    assert all(item.split == "eval" for item in population)
    assert len({item.embedding_identity for item in population}) == len(population)
    assert tuple(row.item for row in inputs.runtime_rows) == population
    queries = [item for item in population if item.kind == "query"]
    # The same query occurs on both sources and both splits. Eval keeps it once,
    # with the first source owning the identity, rather than removing dev duplicates.
    assert len(queries) == 1
    assert queries[0].source == "rhelm"
    assert queries[0].text_sha256 == hashlib.sha256(b"Where is the record?").hexdigest()
    query_row = next(row for row in inputs.runtime_rows if row.item.kind == "query")
    assert query_row.prepared_text == f"Instruct: {QUERY_INSTRUCTION}\nQuery:Where is the record?"
    raw_memory = "User: old\nAssistant: reply"
    identity = ("memory", hashlib.sha256(raw_memory.encode()).hexdigest())
    memory_row = next(row for row in inputs.runtime_rows if row.item.embedding_identity == identity)
    assert memory_row.prepared_text == raw_memory
    assert memory_row.item.text_utf8_bytes == len(raw_memory.encode())
    for item in population:
        assert EmbeddingSampleItem.from_dict(item.as_dict()) == item


def test_memops_exact_span_and_operation_replay() -> None:
    source, injected = _memops_case()
    mapped = map_source_span("P1_remember", source, injected, {**_span(1), "quote": "old setting"})
    assert mapped.canonical_id == "P1_remember/message/segment-10/item-2"
    assert mapped.namespace_id == "P1"
    assert mapped.destination_item_index == 2
    before_forget = {**source, "operations": source["operations"][:2]}
    updated = replay_operation_states("P1_remember", before_forget, injected)
    assert updated[mapped.canonical_id] is EvidenceState.SUPERSEDED
    states = replay_operation_states("P1_remember", source, injected)
    assert states[mapped.canonical_id] is EvidenceState.STALE
    assert states["P1_remember/message/segment-10/item-3"] is EvidenceState.STALE
    assert states["P1_remember/message/segment-10/item-4"] is EvidenceState.CURRENT
    assert states["P1_remember/message/segment-10/item-5"] is EvidenceState.UNCERTAIN
    with pytest.raises(ValueError, match="quote is absent"):
        map_source_span("P1_remember", source, injected, {**_span(1), "quote": "different"})
    injected["conversations"][0]["dialogue"][1] = {"role": "user", "content": "changed"}
    with pytest.raises(ValueError, match="content mismatch"):
        map_source_span("P1_remember", source, injected, _span(1))
    assert query_intent("OperationTrace").value == "history"
    assert query_intent("StateTransition").value == "current_state"
    with pytest.raises(ValueError, match="unexpected MemOps evaluation type"):
        query_intent("new type")


def test_rhelm_excludes_nonconversation_and_same_day_evidence(sources: Path) -> None:
    root = sources / "rhelm" / "data"
    rows = list(iter_rhelm_question_resolutions(root))
    assert len(rows) == 12
    assert sum(row.conversation_only and row.all_references_unique for row in rows) == 4
    result = resolve_rhelm_evidence(root, "person-1", "2020-01-02:1", question_date="2020-01-02")
    assert result.status is ResolutionStatus.FUTURE
    assert load_rhelm_qa_index(root)[("person-1", "kept")]["answer"] == "here"
    with pytest.raises(ValueError, match="unsafe namespace"):
        resolve_rhelm_evidence(root, "../person-1", "2020-01-01:1")


def test_frozen_label_and_method_lifecycle_rules_remain_distinct(sources: Path) -> None:
    corpus = load_eval_inputs(sources, validate_expected=False).corpora["memops"]
    query = corpus.queries[0]
    stale_id = f"{query.namespace_id}_remember/message/segment-10/item-2"
    stale = next(memory for memory in corpus.memories if memory.memory_id == stale_id)
    query = replace(query, required_memory_ids=(stale_id,), query_intent="current_state")
    assert candidate_label(query, stale)["prohibited"] is False
    assert lifecycle_allows(query, stale, "namespace_dense") is True
    assert lifecycle_allows(query, stale, "namespace_current_only") is False
    assert (
        lifecycle_allows(
            replace(query, query_intent="history"), stale, "namespace_source_intent_lifecycle"
        )
        is True
    )
