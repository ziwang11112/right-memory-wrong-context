"""Publish or verify the paper's content-free human-label agreement summary.

This imports reported aggregates; it does not recreate independent human labels or
recompute Krippendorff alpha from those unavailable labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = "reports/stage3_admissibility_label_suite/human_agreement_summary.json"
SOURCE_COMMIT = "a28093110325968c26906223e9eb0f1e078f6aad"
SOURCE_SHA256 = "2ecca3d86789f5061c362a17acef9ca39163ecff647470f2c3deeed18fe0a90f"
AXES = ("relevance", "scope", "state", "prohibited")
FIELDS = (
    "axis",
    "records",
    "exact_agreements",
    "exact_agreement",
    "uncertain_labels",
    "uncertainty_rate",
    "alpha_evaluable_records",
    "krippendorff_alpha_nominal",
)
METRICS = ("exact_agreement", "uncertainty_rate", "krippendorff_alpha_nominal")
EVIDENCE_FIELDS = (
    "claim_id",
    "family",
    "population",
    "source",
    "contrast",
    "metric",
    "estimate",
    "ci95_lower",
    "ci95_upper",
    "n",
    "notes",
)
SUMMARY_PATH = "results/human_label_audit/summary.csv"
EVIDENCE_PATH = "evidence/normalized/human_label_audit.csv"
AUDIT_DESIGN = {
    "schema_version": 1,
    "analysis": "human-label-agreement-reported-summary-v1",
    "record_count": 207,
    "packet_count": 20,
    "independent_annotators": 2,
    "adjudication_applied": False,
    "model_labels_counted_as_human": False,
    "provider_calls": 0,
    "paid_calls": 0,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_bytes(fields: tuple[str, ...], rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _validate_rows(rows: list[dict[str, object]]) -> None:
    if tuple(row.get("axis") for row in rows) != AXES:
        raise ValueError("human audit must contain the four independent label axes in order")
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("human audit has unexpected columns")
        integers: dict[str, int] = {}
        for key in ("records", "exact_agreements", "uncertain_labels", "alpha_evaluable_records"):
            value = row[key]
            if isinstance(value, bool) or str(value) != str(int(value)):
                raise ValueError(f"{key} must be an integer")
            integers[key] = int(value)
        n = integers["records"]
        if n != 207 or not 0 <= integers["exact_agreements"] <= n:
            raise ValueError("human audit record/agreement counts differ from the paper")
        uncertain = integers["uncertain_labels"]
        eligible = integers["alpha_evaluable_records"]
        if not 0 <= uncertain <= 2 * n or not 0 <= eligible <= n:
            raise ValueError("human audit uncertainty counts are invalid")
        if not math.ceil(uncertain / 2) <= n - eligible <= min(uncertain, n):
            raise ValueError("alpha eligibility is inconsistent with uncertain labels")
        for key in METRICS:
            number = float(row[key])
            if not math.isfinite(number) or number > 1:
                raise ValueError(f"{key} must be finite and at most 1")
            if key != "krippendorff_alpha_nominal" and number < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key, expected in (
            ("exact_agreement", integers["exact_agreements"] / n),
            ("uncertainty_rate", uncertain / (2 * n)),
        ):
            if not math.isclose(float(row[key]), expected, rel_tol=0, abs_tol=1e-14):
                raise ValueError(f"{key} disagrees with its count and denominator")


def _evidence_bytes(rows: list[dict[str, object]]) -> bytes:
    evidence: list[dict[str, object]] = []
    for row in rows:
        for metric in METRICS:
            evidence.append(
                {
                    "claim_id": "C15",
                    "family": "human_label_agreement",
                    "population": "207 record-query pairs in 20 blinded packets",
                    "source": "two_independent_human_annotators",
                    "contrast": row["axis"],
                    "metric": metric,
                    "estimate": row[metric],
                    "ci95_lower": "",
                    "ci95_upper": "",
                    "n": row["alpha_evaluable_records"]
                    if metric == METRICS[-1]
                    else row["records"],
                    "notes": "reported pre-adjudication aggregates; raw labels not released; "
                    "uncertainty denominator is twice the record count; not full-population gold",
                }
            )
    return _csv_bytes(EVIDENCE_FIELDS, evidence)


def publish(source: Path, root: Path = ROOT) -> None:
    """Select only four content-free axes from the hash-bound historical summary."""
    data = source.read_bytes()
    if _sha256(data) != SOURCE_SHA256:
        raise ValueError("human agreement source differs from the frozen source hash")
    payload = json.loads(data)
    rows = [
        {key: row[key] for key in FIELDS}
        for row in payload["agreement_rows"]
        if row["axis"] in AXES
    ]
    _validate_rows(rows)
    summary = _csv_bytes(FIELDS, rows)
    evidence = _evidence_bytes(rows)
    script_hash = _sha256((root / "scripts/publish_human_label_audit.py").read_bytes())
    manifest = {
        **AUDIT_DESIGN,
        "contains_raw_labels_text_or_reviewer_identifiers": False,
        "raw_label_recomputation_supported": False,
        "source_path": SOURCE_PATH,
        "source_snapshot_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "file_sha256": {"summary.csv": _sha256(summary)},
        "limitations": [
            "Reported aggregates are imported without new annotation or adjudication.",
            "Independent human labels are unavailable in this public package; "
            "alpha is not recomputed.",
            "The legacy usable_evidence composite excludes lifecycle "
            "and is not exported as an axis.",
            "Low prohibited-status alpha does not validate policy labels as gold.",
        ],
    }
    evidence_manifest = {
        "schema_version": 1,
        "normalized_file": EVIDENCE_PATH,
        "normalized_sha256": _sha256(evidence),
        "row_count": len(AXES) * len(METRICS),
        "contains_raw_text_or_private_content": False,
        "source_repository": "ziwang11112/bomi",
        "source_snapshot_commit": SOURCE_COMMIT,
        "source_artifacts": [{"path": SOURCE_PATH, "sha256": SOURCE_SHA256}],
        "transformation_script": "scripts/publish_human_label_audit.py",
        "transformation_script_sha256": script_hash,
    }
    outputs = {SUMMARY_PATH: summary, EVIDENCE_PATH: evidence}
    for relative, value in (
        ("results/human_label_audit/manifest.json", manifest),
        ("evidence/manifests/human_label_audit.json", evidence_manifest),
    ):
        outputs[relative] = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Resolve predictable publication failures before replacing any released file.
    destinations = [root / relative for relative in outputs]
    for destination in destinations:
        if destination.is_dir():
            raise IsADirectoryError(f"human audit output is a directory: {destination}")
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    for relative, content in outputs.items():
        (root / relative).write_bytes(content)


def verify(root: Path = ROOT) -> dict[str, object]:
    """Check receipts, count arithmetic, and normalized-evidence parity offline."""
    manifest = json.loads((root / "results/human_label_audit/manifest.json").read_text())
    summary = (root / SUMMARY_PATH).read_bytes()
    if manifest["file_sha256"] != {"summary.csv": _sha256(summary)}:
        raise ValueError("human audit summary hash mismatch")
    if (
        manifest.get("source_sha256") != SOURCE_SHA256
        or manifest.get("source_snapshot_commit") != SOURCE_COMMIT
        or manifest.get("source_path") != SOURCE_PATH
    ):
        raise ValueError("human audit source identity mismatch")
    if (
        manifest.get("raw_label_recomputation_supported") is not False
        or manifest.get("contains_raw_labels_text_or_reviewer_identifiers") is not False
    ):
        raise ValueError("human audit release boundary changed")
    if any(
        type(manifest.get(key)) is not type(expected) or manifest[key] != expected
        for key, expected in AUDIT_DESIGN.items()
    ):
        raise ValueError("human audit design metadata changed")
    rows: list[dict[str, object]] = list(csv.DictReader(io.StringIO(summary.decode("utf-8"))))
    _validate_rows(rows)
    if (root / EVIDENCE_PATH).read_bytes() != _evidence_bytes(rows):
        raise ValueError("human audit normalized evidence differs from the public summary")
    return {
        "status": "valid",
        "records": 207,
        "axes": len(rows),
        "raw_label_recomputation_supported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish", "verify"))
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.command == "publish":
        if args.source is None:
            parser.error("publish requires --source with the frozen aggregate JSON")
        publish(args.source)
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
