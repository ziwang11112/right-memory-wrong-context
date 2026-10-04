from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from scripts.export_anonymous_artifact import export_anonymous_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("repository_slug", ["verify-agent-memory", "right-memory-wrong-context"])
def test_anonymous_export_redacts_identity_and_repairs_manifest(
    tmp_path: Path, repository_slug: str
) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "artifact"
    script = repository / "scripts" / "import_evidence.py"
    manifest = repository / "evidence" / "manifests" / "sample.json"
    readme = repository / "README.md"
    workflow = repository / ".github" / "workflows" / "ci.yml"
    env_file = repository / ".env"
    for path in (script, manifest, readme, workflow, env_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    private_handle = "zi" + "wang11112"
    private_name = "zi" + " wang"
    private_username = "zi" + "wan"
    private_email = "zw" + "ang@" + "ua" + "lr.edu"
    private_institution = "University of " + "Arkansas at Little Rock"
    private_institution_short = "UA" + "LR"
    private_workspace = "D:\\" + "agent" + "-mem"
    script.write_text(f'SOURCE = "{private_handle}/{repository_slug}"\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "source_repository": f"{private_handle}/{repository_slug}",
                "transformation_script": "scripts/import_evidence.py",
                "transformation_script_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    readme.write_text(
        "\n".join(
            (
                f"Approved-by: {private_name}",
                f"Contact: {private_email}",
                f"Institution: {private_institution} ({private_institution_short})",
                f"Home: C:\\Users\\{private_username}",
                f"Workspace: {private_workspace}\\verify-agent-memory",
                f"Repository: https://github.com/{private_handle}/{repository_slug}",
                f"Checkout: cd {repository_slug}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    workflow.write_text("name: ci\n", encoding="utf-8")
    env_file.write_text("TOKEN=not-exported\n", encoding="utf-8")

    report = export_anonymous_artifact(
        repository,
        output,
        tracked_paths=(
            "scripts/import_evidence.py",
            "evidence/manifests/sample.json",
            "README.md",
            ".github/workflows/ci.yml",
            ".env",
        ),
        require_clean=False,
    )

    assert private_handle not in (output / "scripts" / "import_evidence.py").read_text()
    exported_readme = (output / "README.md").read_text(encoding="utf-8")
    assert private_name not in exported_readme.lower()
    assert private_username not in exported_readme.lower()
    assert private_email not in exported_readme.lower()
    assert private_institution not in exported_readme
    assert private_institution_short.lower() not in exported_readme.lower()
    assert private_workspace not in exported_readme
    assert "Repository: https://github.com/anonymous/verify-agent-memory" in exported_readme
    assert "Checkout: cd verify-agent-memory" in exported_readme
    assert "right-memory-wrong-context" not in exported_readme
    assert not (output / ".github").exists()
    assert not (output / ".env").exists()
    exported_manifest = json.loads(
        (output / "evidence" / "manifests" / "sample.json").read_text(encoding="utf-8")
    )
    assert exported_manifest["source_repository"] == "anonymous/verify-agent-memory"
    assert exported_manifest["transformation_script_sha256"] == _sha256(
        output / "scripts" / "import_evidence.py"
    )
    assert report["history_included"] is False
    assert report["repaired_evidence_manifests"] == ["evidence/manifests/sample.json"]
    assert (output / "ANONYMIZATION_REPORT.json").is_file()


def test_anonymous_export_rejects_secret_shaped_content(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "artifact"
    source = repository / "README.md"
    source.parent.mkdir(parents=True)
    source.write_text("token=gsk_" + "X" * 24, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Groq API key"):
        export_anonymous_artifact(
            repository,
            output,
            tracked_paths=("README.md",),
            require_clean=False,
        )


@pytest.mark.parametrize("repository_slug", ["verify-agent-memory", "right-memory-wrong-context"])
def test_anonymous_readme_uses_downloaded_zip_without_changing_public_instructions(
    tmp_path: Path, repository_slug: str
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "artifact"
    public_readme = (
        "# Paper title\n\n## Quick Start\n\n"
        f"```sh\ngit clone https://github.com/ziwang11112/{repository_slug}.git\n"
        f"cd {repository_slug}\n"
        "uv sync --locked --extra dev --extra plots\n"
        "uv run --no-sync python -m scripts.run_retrieval_experiment --help\n```\n"
    )
    readme = repository / "README.md"
    readme.write_text(public_readme, encoding="utf-8")

    report = export_anonymous_artifact(
        repository, output, tracked_paths=("README.md",), require_clean=False
    )

    exported = (output / "README.md").read_text(encoding="utf-8")
    assert "**Full repo** ZIP from this artifact's hosting page" in exported
    assert "Open a terminal in the extracted repository root" in exported
    assert "git clone " not in exported
    assert "\ncd " not in exported
    assert exported.startswith("# Paper title\n\n## Quick Start\n\n")
    assert "uv sync --locked --extra dev --extra plots\n" in exported
    assert "python -m scripts.run_retrieval_experiment --help\n```\n" in exported
    assert readme.read_text(encoding="utf-8") == public_readme
    assert report["file_sha256"]["README.md"] == _sha256(output / "README.md")


def test_anonymous_export_removes_public_citation_and_redacts_all_authors(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "artifact"
    names = (
        ("Zi", "Wang"),
        ("Xingqiao", "Wang"),
        ("Emmanuel", "Addai"),
        ("Devika", "Ambekar"),
        ("Xiaowei", "Xu"),
    )
    citation = "authors:\n" + "".join(
        f"  - family-names: {last}\n    given-names: {first}\n" for first, last in names
    )
    (repository / "CITATION.cff").write_text(citation, encoding="utf-8")
    citation_name = "CITATION.cff"
    (repository / "README.md").write_text(f"See [citation]({citation_name}).\n", encoding="utf-8")
    metadata = "[project]\nname = 'verify-agent-memory'\nauthors = [\n"
    metadata += "".join(f"  {{name = '{first} {last}'}},\n" for first, last in names)
    (repository / "pyproject.toml").write_text(metadata + "]\n", encoding="utf-8")

    report = export_anonymous_artifact(
        repository,
        output,
        tracked_paths=("CITATION.cff", "pyproject.toml", "README.md"),
        require_clean=False,
    )

    assert not (output / "CITATION.cff").exists()
    assert report["excluded_files"] == ["CITATION.cff"]
    assert (output / "README.md").read_text(encoding="utf-8") == (
        "See citation (omitted from the anonymous artifact).\n"
    )
    exported_metadata = (output / "pyproject.toml").read_text(encoding="utf-8")
    assert all(f"{first} {last}" not in exported_metadata for first, last in names)
    exported_authors = tomllib.loads(exported_metadata)["project"]["authors"]
    assert tomllib.loads(exported_metadata)["project"]["name"] == "verify-agent-memory"
    assert len(exported_authors) == len(names)
    assert all(author["name"].startswith("Anonymous ") for author in exported_authors)


def test_anonymous_export_rejects_identity_in_non_utf8_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = tmp_path / "artifact"
    source = repository / "fixture.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"\xffprivate=" + ("zi" + "wang11112").encode("ascii"))

    with pytest.raises(RuntimeError, match="private repository handle"):
        export_anonymous_artifact(
            repository,
            output,
            tracked_paths=("fixture.bin",),
            require_clean=False,
        )


def test_anonymous_export_omits_identity_bearing_release_tools(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "artifact"
    release_tools = (
        "scripts/export_anonymous_artifact.py",
        "tests/test_export_anonymous_artifact.py",
    )
    report = export_anonymous_artifact(
        repository,
        output,
        tracked_paths=(*release_tools, "docs/ANONYMOUS_RELEASE.md"),
        require_clean=False,
    )

    assert report["excluded_files"] == sorted(release_tools)
    assert report["copied_file_count"] == 1
    assert all(not (output / relative).exists() for relative in release_tools)
    note = (output / "docs/ANONYMOUS_RELEASE.md").read_text(encoding="utf-8")
    assert "python -m scripts.export_anonymous_artifact" not in note
    assert "python -m scripts.check_reproducibility_package" in note
