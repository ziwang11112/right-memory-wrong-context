from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_reproducibility_package
from scripts.check_reproducibility_package import _tracked_paths, validate_package

ROOT = Path(__file__).resolve().parents[1]


def test_repository_is_a_complete_reproducibility_package() -> None:
    errors, inventory = validate_package(ROOT)

    assert errors == []
    assert inventory["public_sources"] == 3
    assert inventory["result_families"] >= 9
    assert inventory["experiment_protocols"] >= 20
    assert inventory["normalized_evidence_files"] >= 10


def test_anonymous_inventory_is_hash_verified_without_git(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("release\n", encoding="utf-8")
    digest = hashlib.sha256(readme.read_bytes()).hexdigest()
    (tmp_path / "ANONYMIZATION_REPORT.json").write_text(
        json.dumps(
            {
                "history_included": False,
                "file_sha256": {"README.md": digest},
            }
        ),
        encoding="utf-8",
    )

    assert _tracked_paths(tmp_path) == ("README.md",)
    readme.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed its receipt"):
        _tracked_paths(tmp_path)


def test_missing_tracked_code_and_result_directory_report_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        check_reproducibility_package,
        "_tracked_paths",
        lambda root: ("src/missing.py",),
    )

    errors, inventory = validate_package(tmp_path)

    assert "tracked file is missing: src/missing.py" in errors
    assert inventory["result_families"] == 0
    assert any("uv.lock" in error for error in errors)
