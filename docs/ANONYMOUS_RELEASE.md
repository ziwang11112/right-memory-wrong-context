# Anonymous Release

The anonymous release artifact must not be a fork and must not retain the private
repository's Git history. It may contain only tracked, reviewable files from a clean
commit. Raw benchmark text, provider prompts and responses, credentials, embeddings,
checkpoints, and private data remain excluded.

## Build and Check

Run the full repository checks first, commit the intended release, and require a
clean worktree. Then create the export:

```powershell
uv run --extra dev python -m scripts.export_anonymous_artifact `
  --output tmp/anonymous_artifact
```

The exporter:

1. copies only `git ls-files` entries;
2. excludes Git history, CI metadata, caches, local configuration, raw/private data,
   provider material, public `CITATION.cff` author metadata, and the exporter, its
   identity-bearing tests; these release instructions become a reviewer-facing
   verification note;
3. redacts the paper's author names, owner email, institution, repository,
   workstation-user, and local-workspace identifiers in copied UTF-8 text, and
   replaces local citation links with a notice that citation metadata was omitted;
4. repairs evidence-manifest hashes for redacted transformation scripts;
5. fails on residual identity tokens, including tokens in non-UTF-8 files, and on
   common API-key and private-key shapes; and
6. writes `ANONYMIZATION_REPORT.json` with output hashes and no source commit ID.

Verify the exported package independently before publishing it:

```powershell
Push-Location tmp/anonymous_artifact
uv sync --locked --extra dev --extra plots
uv run --extra dev --extra plots python -m pytest -q
uv run --extra dev python -m ruff check .
uv run --extra dev python -m ruff format --check .
uv run --extra dev python -m scripts.check_claim_contract
uv run --extra dev python -m scripts.verify_evidence
uv run --extra dev python -m scripts.check_reproducibility_package
Pop-Location
```

CI also builds this export and runs the remaining reproduction tests and validation
gates in its own locked environment. The exporter and its tests contain the original
identities in redaction rules, so they must stay in the public source repository
along with its citation metadata. They are not included in the anonymous artifact.

Only after these checks pass should this directory be initialized as a new anonymous
Git repository with a separate anonymous identity. The public repository remains the
canonical source for the paper's title, authors, and citation.
