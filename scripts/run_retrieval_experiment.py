"""Validate, execute, or select settings for a normalized retrieval experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from verify_agent_memory.experiment import (
    ExperimentCase,
    SelectionRisk,
    run_experiment,
    select_dev_settings,
    validate_experiment,
)
from verify_agent_memory.retrieval import RetrievalConfig
from verify_agent_memory.serialization import (
    case_from_mapping,
    config_from_mapping,
    query_run_to_mapping,
    setting_summary_to_mapping,
)


def _json_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_cases(path: Path) -> tuple[ExperimentCase, ...]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(case_from_mapping(json.loads(line)))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid case at {path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError("case file is empty")
    return tuple(rows)


def load_protocol(path: Path) -> tuple[float, SelectionRisk, tuple[RetrievalConfig, ...]]:
    row = _json_object(path)
    if row.get("schema_version") != 1:
        raise ValueError("protocol schema_version must be 1")
    settings = row.get("settings")
    if isinstance(settings, (str, bytes)) or not isinstance(settings, Sequence):
        raise TypeError("protocol settings must be an array")
    configs = tuple(config_from_mapping(setting) for setting in settings)
    if not configs:
        raise ValueError("protocol must contain at least one setting")
    target_recall = row.get("target_recall")
    if isinstance(target_recall, bool) or not isinstance(target_recall, (int, float)):
        raise TypeError("target_recall must be numeric")
    risk_target = SelectionRisk(str(row.get("selection_risk", "admissibility_upper_bound")))
    return target_recall, risk_target, configs


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "run", "select"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--cases", type=Path, required=True)
        subparser.add_argument("--protocol", type=Path, required=True)
        if command != "validate":
            subparser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.cases)
    target_recall, risk_target, configs = load_protocol(args.protocol)
    if args.command == "validate":
        validate_experiment(cases, configs, target_recall=target_recall)
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "settings": len(configs),
                    "selection_risk": risk_target.value,
                    "target_recall": target_recall,
                    "status": "valid",
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0

    runs = run_experiment(cases, configs, target_recall=target_recall)
    if args.command == "run":
        _write_jsonl(args.output, [query_run_to_mapping(run) for run in runs])
        print(json.dumps({"rows": len(runs), "status": "complete"}, sort_keys=True))
        return 0

    selected = select_dev_settings(runs, risk_target=risk_target)
    _write_jsonl(
        args.output,
        [setting_summary_to_mapping(summary) for summary in selected.values()],
    )
    print(json.dumps({"arms": len(selected), "status": "selected"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
