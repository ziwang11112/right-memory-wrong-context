"""Create a no-history, identity-redacted review artifact from tracked files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

EXCLUDED_PREFIXES = (
    ".git/",
    ".github/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
    "data/private/",
    "data/raw/",
    "evidence/raw/",
    "private/",
    "tmp/",
)
EXCLUDED_NAMES = {".env", ".DS_Store", "Thumbs.db", "CITATION.cff"}
EXCLUDED_PATHS = {
    "scripts/export_anonymous_artifact.py",
    "tests/test_export_anonymous_artifact.py",
}
ANONYMOUS_RELEASE_NOTE = """# Anonymous Artifact

This reviewer artifact contains no Git history or public citation metadata. Its
release exporter and exporter tests are omitted because their redaction rules
contain author identifiers. Reproduction code and its tests remain available.

Run these commands from the artifact root:

```shell
uv sync --locked --extra dev --extra plots
uv run --no-sync python -m pytest
uv run --no-sync python -m ruff check .
uv run --no-sync python -m ruff format --check .
uv run --no-sync python -m scripts.check_claim_contract
uv run --no-sync python -m scripts.verify_evidence
uv run --no-sync python -m scripts.check_reproducibility_package
```

`ANONYMIZATION_REPORT.json` records the included files and their SHA-256 hashes.
The package check verifies those receipts without requiring Git. See
`REPRODUCIBILITY.md` for the evidence and historical-execution boundaries.
"""
EXCLUDED_SUFFIXES = {".key", ".pem", ".pyc"}
PRIVATE_HANDLE = "zi" + "wang11112"
PRIVATE_NAME_PATTERN = r"zi" + r"\s+" + "wang"
PAPER_AUTHOR_PATTERN = r"\b(?:Xingqiao\s+Wang|Emmanuel\s+Addai|Devika\s+Ambekar|Xiaowei\s+Xu)\b"
PRIVATE_USERNAME_PATTERN = r"\bzi" + r"wan\b"
PRIVATE_EMAIL_PATTERN = r"\bzw" + r"ang@ua" + r"lr\.edu\b"
PRIVATE_INSTITUTION_PATTERN = r"\bUniversity\s+of\s+Arkansas\s+at\s+Little\s+Rock\b"
PRIVATE_INSTITUTION_SHORT_PATTERN = r"\bUA" + r"LR\b"
PRIVATE_WORKSPACE_PATTERN = r"\bD:[\\/]agent-mem\b"
CITATION_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((?:\.\.?/)?CITATION\.cff\)", re.IGNORECASE)
IDENTITY_REPLACEMENTS = (
    ("private repository handle", re.compile(PRIVATE_HANDLE, re.IGNORECASE), "anonymous"),
    ("private owner name", re.compile(PRIVATE_NAME_PATTERN, re.IGNORECASE), "Anonymous Owner"),
    ("paper author", re.compile(PAPER_AUTHOR_PATTERN, re.IGNORECASE), "Anonymous Author"),
    (
        "private workstation username",
        re.compile(PRIVATE_USERNAME_PATTERN, re.IGNORECASE),
        "anonymous",
    ),
    (
        "private owner email",
        re.compile(PRIVATE_EMAIL_PATTERN, re.IGNORECASE),
        "anonymous@example.invalid",
    ),
    (
        "private institution",
        re.compile(PRIVATE_INSTITUTION_PATTERN, re.IGNORECASE),
        "Anonymous Institution",
    ),
    (
        "private institution abbreviation",
        re.compile(PRIVATE_INSTITUTION_SHORT_PATTERN, re.IGNORECASE),
        "Anonymous Institution",
    ),
    (
        "private local workspace",
        re.compile(PRIVATE_WORKSPACE_PATTERN, re.IGNORECASE),
        "D:/anonymous-workspace",
    ),
)
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


def _safe_relative(path: str | Path) -> PurePosixPath:
    relative = PurePosixPath(str(path).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe tracked path: {path}")
    return relative


def _excluded(relative: PurePosixPath) -> bool:
    value = relative.as_posix()
    return (
        value in EXCLUDED_PATHS
        or relative.name in EXCLUDED_NAMES
        or relative.suffix.lower() in EXCLUDED_SUFFIXES
        or any(
            value == prefix.rstrip("/") or value.startswith(prefix) for prefix in EXCLUDED_PREFIXES
        )
        or relative.name.startswith(".env.")
    )


def _tracked_paths(repository_root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return tuple(part.decode("utf-8") for part in result.stdout.split(b"\0") if part)


def _require_clean(repository_root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.stdout.strip():
        raise RuntimeError("anonymous export requires a clean Git worktree")


def _redact(data: bytes) -> tuple[bytes, bool]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, False
    original = text
    for _, pattern, replacement in IDENTITY_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = CITATION_LINK_PATTERN.sub(r"\1 (omitted from the anonymous artifact)", text)
    return text.encode("utf-8"), text != original


def _repair_evidence_manifest_hashes(output_root: Path) -> tuple[str, ...]:
    repaired: list[str] = []
    manifest_root = output_root / "evidence" / "manifests"
    if not manifest_root.is_dir():
        return ()
    for path in sorted(manifest_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        script_name = payload.get("transformation_script")
        if not isinstance(script_name, str):
            continue
        script_path = output_root / _safe_relative(script_name)
        if not script_path.is_file():
            raise FileNotFoundError(f"manifest transformation script is missing: {script_name}")
        expected = _sha256(script_path)
        if payload.get("transformation_script_sha256") == expected:
            continue
        payload["transformation_script_sha256"] = expected
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        repaired.append(path.relative_to(output_root).as_posix())
    return tuple(repaired)


def _scan_export(output_root: Path) -> None:
    findings: list[str] = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        relative = path.relative_to(output_root).as_posix()
        data = path.read_bytes()
        scan_text = data.decode("utf-8", errors="ignore")
        for label, pattern, _ in IDENTITY_REPLACEMENTS:
            if pattern.search(scan_text):
                findings.append(f"{relative}: {label}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                findings.append(f"{relative}: {label}")
    if findings:
        raise RuntimeError("anonymous export scan failed:\n" + "\n".join(findings))


def export_anonymous_artifact(
    repository_root: Path,
    output_root: Path,
    *,
    tracked_paths: Iterable[str | Path] | None = None,
    require_clean: bool = True,
) -> dict[str, object]:
    repository_root = repository_root.resolve()
    output_root = output_root.resolve()
    if output_root == repository_root or output_root.is_relative_to(repository_root / ".git"):
        raise ValueError("output must not replace the repository or enter .git")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    if require_clean:
        _require_clean(repository_root)

    selected = (
        tuple(tracked_paths) if tracked_paths is not None else _tracked_paths(repository_root)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    excluded: list[str] = []
    redacted: list[str] = []
    for raw_relative in selected:
        relative = _safe_relative(raw_relative)
        if _excluded(relative):
            excluded.append(relative.as_posix())
            continue
        source = (repository_root / Path(*relative.parts)).resolve()
        if not source.is_relative_to(repository_root) or not source.is_file():
            raise FileNotFoundError(f"tracked file is unavailable: {relative.as_posix()}")
        destination = output_root / Path(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == "docs/ANONYMOUS_RELEASE.md":
            data, changed = ANONYMOUS_RELEASE_NOTE.encode("utf-8"), True
        else:
            data, changed = _redact(source.read_bytes())
        destination.write_bytes(data)
        shutil.copystat(source, destination)
        copied.append(relative.as_posix())
        if changed:
            redacted.append(relative.as_posix())

    repaired = _repair_evidence_manifest_hashes(output_root)
    _scan_export(output_root)
    hashes = {
        path.relative_to(output_root).as_posix(): _sha256(path)
        for path in sorted(item for item in output_root.rglob("*") if item.is_file())
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "history_included": False,
        "copied_file_count": len(copied),
        "excluded_files": sorted(excluded),
        "redacted_files": sorted(redacted),
        "repaired_evidence_manifests": list(repaired),
        "file_sha256": hashes,
    }
    report_path = output_root / "ANONYMIZATION_REPORT.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _scan_export(output_root)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "tmp" / "anonymous_artifact",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = export_anonymous_artifact(args.repository_root, args.output)
    print(
        f"exported {report['copied_file_count']} tracked files to {args.output.resolve()} "
        f"with {len(report['redacted_files'])} identity-redacted files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
