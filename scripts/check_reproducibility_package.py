"""Validate the public code, data, protocol, evidence, and result package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from scripts.fetch_public_sources import load_manifest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = {
    "README.md",
    "REPRODUCIBILITY.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    "data/README.md",
    "data/upstream_sources.json",
    "experiments/README.md",
    "results/README.md",
    "results/natural_end_to_end_case_audit/README.md",
    "results/natural_end_to_end_case_audit/case_scores.csv",
    "results/natural_end_to_end_case_audit/case_weighted_sensitivity.csv",
    "results/natural_end_to_end_case_audit/manifest.json",
    "results/natural_end_to_end_case_audit/population_summary.csv",
    "results/natural_end_to_end_case_audit/source_specific_deltas.csv",
    "evidence/README.md",
    "tests/fixtures/retrieval_cases.jsonl",
}
FORBIDDEN_PREFIXES = (
    "paper/",
    "data/raw/",
    "data/private/",
    "evidence/raw/",
    "private/",
    "tmp/",
    "model_responses/",
    "responses/",
)
FORBIDDEN_EXACT = {
    ".env",
    "PAPER_REVIEW_GUIDE.md",
    "scripts/build_paper_artifacts.py",
    "scripts/compile_paper.py",
    "scripts/export_overleaf.py",
    "scripts/verify_paper.py",
}
RESULT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".pdf", ".png", ".svg"}
SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "Anthropic API key": re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    "Groq API key": re.compile(rb"\bgsk_[A-Za-z0-9]{16,}\b"),
    "Google API key": re.compile(rb"\bAIza[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_paths(root: Path) -> tuple[str, ...]:
    if (root / ".git").exists():
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return tuple(part.decode("utf-8") for part in completed.stdout.split(b"\0") if part)

    report_path = root / "ANONYMIZATION_REPORT.json"
    if not report_path.is_file():
        raise FileNotFoundError("repository has neither Git metadata nor an anonymization report")
    report = _strict_json(report_path)
    if not isinstance(report, dict) or report.get("history_included") is not False:
        raise ValueError("anonymization report is malformed")
    hashes = report.get("file_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("anonymization report has no file inventory")
    paths: list[str] = []
    for raw_path, expected_hash in hashes.items():
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise ValueError("anonymization report has malformed file receipts")
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"anonymization report has unsafe path: {raw_path!r}")
        path = root / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"anonymous artifact failed its receipt: {raw_path}")
        paths.append(raw_path)
    return tuple(sorted(paths))


def _strict_json(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def validate_package(root: Path = ROOT) -> tuple[list[str], dict[str, int]]:
    """Return release-blocking errors and a content inventory."""
    root = root.resolve()
    try:
        tracked = _tracked_paths(root)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [f"public file inventory could not be loaded: {exc}"], {}
    tracked_set = set(tracked)
    errors: list[str] = []

    missing = sorted(REQUIRED_PATHS - tracked_set)
    if missing:
        errors.append(f"required tracked paths are missing: {missing!r}")
    for value in tracked:
        relative = PurePosixPath(value)
        if value in FORBIDDEN_EXACT or any(
            value.startswith(prefix) for prefix in FORBIDDEN_PREFIXES
        ):
            errors.append(f"forbidden public path is tracked: {value}")
        if relative.name == ".env" or relative.name.startswith(".env."):
            errors.append(f"credential file is tracked: {value}")
        path = root / relative
        if not path.is_file():
            errors.append(f"tracked file is missing: {value}")
            continue
        data = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                errors.append(f"{value} contains a {label}")

    result_files = sorted(
        value for value in tracked if value.startswith("results/") and value != "results/README.md"
    )
    for value in result_files:
        if Path(value).suffix.lower() not in RESULT_SUFFIXES:
            errors.append(f"unsupported result-file type: {value}")
    result_root = root / "results"
    result_families = (
        sorted(path.name for path in result_root.iterdir() if path.is_dir())
        if result_root.is_dir()
        else []
    )
    for family in result_families:
        manifest = f"results/{family}/manifest.json"
        if manifest not in tracked_set:
            errors.append(f"result family lacks a tracked manifest: {family}")

    json_files = [value for value in tracked if Path(value).suffix.lower() == ".json"]
    for value in json_files:
        try:
            _strict_json(root / value)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid strict JSON in {value}: {exc}")
    csv_files = [value for value in tracked if Path(value).suffix.lower() == ".csv"]
    for value in csv_files:
        try:
            with (root / value).open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
            if not header or any(not field for field in header):
                errors.append(f"CSV has an empty or missing header: {value}")
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"invalid CSV in {value}: {exc}")

    try:
        public_sources = load_manifest(root / "data" / "upstream_sources.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid public-source manifest: {exc}")
        source_count = 0
    else:
        source_count = len(public_sources.sources)

    inventory = {
        "tracked_files": len(tracked),
        "experiment_protocols": sum(
            1 for value in tracked if value.startswith("experiments/") and value.endswith(".json")
        ),
        "result_files": len(result_files),
        "result_families": len(result_families),
        "normalized_evidence_files": sum(
            1 for value in tracked if value.startswith("evidence/normalized/")
        ),
        "public_sources": source_count,
        "tests": sum(1 for value in tracked if value.startswith("tests/test_")),
    }
    return errors, inventory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors, inventory = validate_package(args.repository_root)
    if errors:
        print("reproducibility package validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "valid", **inventory}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
