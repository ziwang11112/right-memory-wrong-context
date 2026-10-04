from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.run_retrieval_experiment import main
from verify_agent_memory.experiment import (
    ExperimentCase,
    run_experiment,
    select_dev_settings,
    summarize_setting,
)
from verify_agent_memory.retrieval import (
    ATTRIBUTION_DIAGNOSTIC_ARMS,
    FROZEN_PUBLIC_ARMS,
    MemoryRecord,
    QueryRecord,
    RetrievalArm,
    RetrievalConfig,
)
from verify_agent_memory.schema import (
    LifecycleState,
    MemoryAssessment,
    QueryIntent,
    Relevance,
    Scope,
)
from verify_agent_memory.serialization import case_from_mapping

ROOT = Path(__file__).resolve().parents[1]


def selection_case() -> ExperimentCase:
    query = QueryRecord(
        query_id="q",
        namespace="a",
        text="query",
        embedding=(1.0, 0.0),
        intent=QueryIntent.CURRENT_STATE,
    )
    memories = (
        MemoryRecord(
            memory_id="contaminant",
            namespace="a",
            text="",
            embedding=(1.0, 0.0),
            released_order=0,
            lifecycle_state=LifecycleState.CURRENT,
        ),
        MemoryRecord(
            memory_id="anchor",
            namespace="a",
            text="",
            embedding=(0.9, 0.43589),
            released_order=1,
            lifecycle_state=LifecycleState.CURRENT,
        ),
    )
    assessments = (
        MemoryAssessment(
            memory_id="contaminant",
            relevance=Relevance.NOT_USEFUL,
            scope=Scope.ALLOWED,
            lifecycle_state=LifecycleState.CURRENT,
            policy_allowed=True,
            query_intent=QueryIntent.CURRENT_STATE,
        ),
        MemoryAssessment(
            memory_id="anchor",
            relevance=Relevance.REQUIRED,
            scope=Scope.ALLOWED,
            lifecycle_state=LifecycleState.CURRENT,
            policy_allowed=True,
            query_intent=QueryIntent.CURRENT_STATE,
        ),
    )
    return ExperimentCase("source", "group", query, memories, assessments)


def test_dev_selection_prioritizes_feasibility_before_contamination() -> None:
    configs = (
        RetrievalConfig("short", RetrievalArm.GLOBAL_DENSE, top_k=1),
        RetrievalConfig("long", RetrievalArm.GLOBAL_DENSE, top_k=2),
    )
    runs = run_experiment((selection_case(),), configs)
    selected = select_dev_settings(runs)

    assert selected[RetrievalArm.GLOBAL_DENSE].setting_id == "long"
    assert selected[RetrievalArm.GLOBAL_DENSE].feasible_rate == 1.0


def test_penalized_risk_decomposes_and_separates_irrelevance() -> None:
    config = RetrievalConfig("long", RetrievalArm.GLOBAL_DENSE, top_k=2)
    summary = summarize_setting(run_experiment((selection_case(),), (config,)))

    assert summary.infeasibility_risk_component == 0.0
    assert summary.penalized_non_usable_upper_risk == 0.5
    assert summary.non_usable_conditional_risk_component == 0.5
    assert summary.penalized_admissibility_upper_risk == 0.0
    assert summary.admissibility_conditional_risk_component == 0.0
    assert summary.conditional_admissibility_upper_risk == 0.0
    assert summary.penalized_non_usable_upper_risk == (
        summary.infeasibility_risk_component + summary.non_usable_conditional_risk_component
    )
    assert summary.penalized_admissibility_upper_risk == (
        summary.infeasibility_risk_component + summary.admissibility_conditional_risk_component
    )


def test_dev_selection_rejects_incomplete_setting_coverage() -> None:
    configs = (
        RetrievalConfig("short", RetrievalArm.GLOBAL_DENSE, top_k=1),
        RetrievalConfig("long", RetrievalArm.GLOBAL_DENSE, top_k=2),
    )
    runs = run_experiment((selection_case(),), configs)

    with pytest.raises(ValueError, match="same source/query rows"):
        select_dev_settings((runs[0], replace(runs[1], query_id="different-query")))


def test_protocol_contains_exact_nine_public_arms() -> None:
    protocol = json.loads(
        (ROOT / "experiments" / "frozen_natural_protocol.json").read_text(encoding="utf-8")
    )
    arms = {setting["arm"] for setting in protocol["settings"]}

    assert arms == {arm.value for arm in FROZEN_PUBLIC_ARMS}
    assert FROZEN_PUBLIC_ARMS.isdisjoint(ATTRIBUTION_DIAGNOSTIC_ARMS)
    assert frozenset(RetrievalArm) == FROZEN_PUBLIC_ARMS | ATTRIBUTION_DIAGNOSTIC_ARMS
    assert len(protocol["settings"]) == 9
    assert protocol["development_selection"]["evaluation_retuning"] is False


@pytest.mark.parametrize("field", ["assessment", "released_policy"])
@pytest.mark.parametrize("numeric_label", [0, 1, 0.0, 1.0])
def test_case_parser_rejects_numeric_policy_labels(field: str, numeric_label: object) -> None:
    row = json.loads((ROOT / "tests/fixtures/retrieval_cases.jsonl").read_text().splitlines()[0])
    label = "policy_allowed" if field == "assessment" else "content_disclosure_allowed"
    row["memories"][0].setdefault(field, {})[label] = numeric_label
    with pytest.raises(TypeError, match=label):
        case_from_mapping(row)


@pytest.mark.parametrize(
    "invalid_input",
    ["out_of_range_recall", "boolean_recall", "duplicate_setting", "duplicate_query", "dimension"],
)
def test_validate_rejects_inputs_that_cannot_run(
    invalid_input: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_rows = [
        json.loads(line)
        for line in (ROOT / "tests/fixtures/retrieval_cases.jsonl").read_text().splitlines()
    ]
    protocol = json.loads((ROOT / "experiments/frozen_natural_protocol.json").read_text())
    if invalid_input == "out_of_range_recall":
        protocol["target_recall"] = 1.1
    elif invalid_input == "boolean_recall":
        protocol["target_recall"] = True
    elif invalid_input == "duplicate_setting":
        protocol["settings"].append(protocol["settings"][0])
    elif invalid_input == "duplicate_query":
        case_rows.append(case_rows[0])
    else:
        case_rows[0]["query"]["embedding"].append(0.0)
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("\n".join(json.dumps(row) for row in case_rows), encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        main(["validate", "--cases", str(cases_path), "--protocol", str(protocol_path)])
    assert '"status": "valid"' not in capsys.readouterr().out


def test_cli_smoke_validates_runs_and_selects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = ROOT / "tests" / "fixtures" / "retrieval_cases.jsonl"
    protocol = ROOT / "experiments" / "frozen_natural_protocol.json"
    routes = tmp_path / "routes.jsonl"
    selected = tmp_path / "selected.jsonl"

    assert main(["validate", "--cases", str(cases), "--protocol", str(protocol)]) == 0
    assert '"status": "valid"' in capsys.readouterr().out
    assert (
        main(
            [
                "run",
                "--cases",
                str(cases),
                "--protocol",
                str(protocol),
                "--output",
                str(routes),
            ]
        )
        == 0
    )
    assert len(routes.read_text(encoding="utf-8").splitlines()) == 18
    capsys.readouterr()
    assert (
        main(
            [
                "select",
                "--cases",
                str(cases),
                "--protocol",
                str(protocol),
                "--output",
                str(selected),
            ]
        )
        == 0
    )
    assert len(selected.read_text(encoding="utf-8").splitlines()) == 9
