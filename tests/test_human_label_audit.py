from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest

from scripts import publish_human_label_audit as human_audit
from scripts.publish_human_label_audit import ROOT, publish, verify


def _copy_package(tmp_path: Path) -> Path:
    for relative in ("results/human_label_audit", "evidence/normalized"):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    return tmp_path


def _synthetic_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    with (ROOT / "results/human_label_audit/summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Include fields that must never enter the public aggregate release.
    payload = {
        "reviewer_ids": ["synthetic-private-reviewer"],
        "agreement_rows": [*rows, {"axis": "usable_evidence", "private": "excluded"}],
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        human_audit, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    return source


def test_human_summary_publish_roundtrip_creates_receipts_and_omits_private_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _synthetic_source(tmp_path, monkeypatch)
    root = tmp_path / "release"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/publish_human_label_audit.py", root / "scripts")

    publish(source, root)

    assert verify(root)["status"] == "valid"
    assert (root / human_audit.SUMMARY_PATH).read_bytes() == (
        ROOT / human_audit.SUMMARY_PATH
    ).read_bytes()
    receipt = json.loads((root / "evidence/manifests/human_label_audit.json").read_text())
    assert (
        receipt["transformation_script_sha256"]
        == hashlib.sha256((root / "scripts/publish_human_label_audit.py").read_bytes()).hexdigest()
    )
    for path in (root / "results").rglob("*"):
        if path.is_file():
            assert "synthetic-private-reviewer" not in path.read_text()
    assert "usable_evidence" not in (root / human_audit.SUMMARY_PATH).read_text()


def test_human_summary_missing_script_does_not_write_partial_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _synthetic_source(tmp_path, monkeypatch)
    root = tmp_path / "release"
    with pytest.raises(FileNotFoundError):
        publish(source, root)
    assert not root.exists()


def test_human_summary_invalid_output_does_not_replace_existing_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _synthetic_source(tmp_path, monkeypatch)
    root = tmp_path / "release"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/publish_human_label_audit.py", root / "scripts")
    summary = root / human_audit.SUMMARY_PATH
    summary.parent.mkdir(parents=True)
    summary.write_bytes(b"existing summary")
    (root / "evidence/manifests/human_label_audit.json").mkdir(parents=True)
    with pytest.raises(IsADirectoryError):
        publish(source, root)
    assert summary.read_bytes() == b"existing summary"
    assert not (root / human_audit.EVIDENCE_PATH).exists()


def test_released_human_summary_matches_normalized_evidence() -> None:
    assert verify() == {
        "status": "valid",
        "records": 207,
        "axes": 4,
        "raw_label_recomputation_supported": False,
    }
    with (ROOT / "results/human_label_audit/summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["axis"] for row in rows} == {"relevance", "scope", "state", "prohibited"}
    state = next(row for row in rows if row["axis"] == "state")
    assert int(state["alpha_evaluable_records"]) == 184
    assert float(state["uncertainty_rate"]) == pytest.approx(23 / 414)


def test_human_summary_rejects_mismatched_source_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"agreement_rows": []}')
    with pytest.raises(ValueError, match="frozen source hash"):
        publish(source, tmp_path)
    assert not (tmp_path / "results").exists()


def test_human_summary_rejects_incorrect_rate_even_with_updated_receipt(tmp_path: Path) -> None:
    root = _copy_package(tmp_path)
    path = root / "results/human_label_audit/summary.csv"
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    rows[0]["exact_agreement"] = "0.5"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = root / "results/human_label_audit/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"]["summary.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="count and denominator"):
        verify(root)


def test_human_summary_rejects_claiming_raw_label_recomputation(tmp_path: Path) -> None:
    root = _copy_package(tmp_path)
    manifest_path = root / "results/human_label_audit/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_label_recomputation_supported"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="release boundary"):
        verify(root)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("schema_version", 2),
        ("analysis", "different-audit"),
        ("record_count", 208),
        ("packet_count", 21),
        ("independent_annotators", 3),
        ("adjudication_applied", True),
        ("model_labels_counted_as_human", True),
        ("provider_calls", 1),
        ("paid_calls", 1),
        ("provider_calls", False),
    ],
)
def test_human_summary_rejects_design_metadata_drift(
    field: str, changed: object, tmp_path: Path
) -> None:
    root = _copy_package(tmp_path)
    manifest_path = root / "results/human_label_audit/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = changed
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="design metadata"):
        verify(root)
