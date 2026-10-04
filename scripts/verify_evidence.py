"""Verify normalized evidence hashes, schemas, and claim boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from verify_agent_memory.provenance import sha256_file

REQUIRED_COLUMNS = {
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
}
REQUIRED_CLAIMS = {f"C{number}" for number in range(2, 15)}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PROHIBITED_OUTPUT_LABELS = {"ncr_threshold", "ncr_a5"}
EXPECTED_METRIC_COUNTS = {
    "C2": 6,
    "C3": 3,
    "C4": 6,
    "C5": 48,
    "C6": 6,
    "C7": 4,
    "C8": 15,
    "C9": 83,
    "C10": 37,
    "C11": 29,
    "C12": 5,
    "C13": 34,
    "C14": 16,
}


def _repository_path(
    repository_root: Path,
    relative_name: object,
    *,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(relative_name, str) or not relative_name:
        errors.append(f"{label} must be a nonempty relative path")
        return None
    root = repository_root.resolve()
    path = (root / relative_name).resolve()
    if not path.is_relative_to(root):
        errors.append(f"{label} escapes the repository")
        return None
    return path


def _source_artifact_index(
    repository_root: Path,
    errors: list[str],
) -> dict[str, Mapping[str, object]]:
    index_path = repository_root / "SOURCE_ARTIFACTS.yaml"
    try:
        index: Any = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"SOURCE_ARTIFACTS.yaml could not be loaded: {exc}")
        return {}
    if not isinstance(index, Mapping) or not isinstance(index.get("artifacts"), list):
        errors.append("SOURCE_ARTIFACTS.yaml must contain an artifacts list")
        return {}
    artifacts: dict[str, Mapping[str, object]] = {}
    for position, artifact in enumerate(index["artifacts"]):
        if not isinstance(artifact, Mapping):
            errors.append(f"SOURCE_ARTIFACTS.yaml artifact {position} is malformed")
            continue
        source_path = artifact.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            errors.append(f"SOURCE_ARTIFACTS.yaml artifact {position} lacks source_path")
            continue
        if source_path in artifacts:
            errors.append(f"SOURCE_ARTIFACTS.yaml has duplicate source_path: {source_path}")
            continue
        artifacts[source_path] = artifact
    return artifacts


def _claim_index(
    repository_root: Path,
    errors: list[str],
) -> dict[str, Mapping[str, object]]:
    claims_path = repository_root / "claims" / "claims.yaml"
    try:
        contract: Any = yaml.safe_load(claims_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"claims/claims.yaml could not be loaded: {exc}")
        return {}
    if not isinstance(contract, Mapping) or not isinstance(contract.get("claims"), list):
        errors.append("claims/claims.yaml must contain a claims list")
        return {}
    claims: dict[str, Mapping[str, object]] = {}
    for position, claim in enumerate(contract["claims"]):
        if not isinstance(claim, Mapping) or not isinstance(claim.get("id"), str):
            errors.append(f"claims/claims.yaml claim {position} is malformed")
            continue
        claim_id = claim["id"]
        if claim_id in claims:
            errors.append(f"claims/claims.yaml has duplicate claim id: {claim_id}")
            continue
        claims[claim_id] = claim
    return claims


def _contract_value_path(row: Mapping[str, str]) -> tuple[str, ...] | None:
    claim_id = row["claim_id"]
    contrast = row["contrast"]
    metric = row["metric"]
    if claim_id == "C2":
        reader = {
            "gpt-4o-mini-2024-07-18": "reader_a",
            "gpt-4o-2024-08-06": "reader_b",
        }.get(row["source"])
        metric_key = {
            "answer_leakage_risk_difference": "risk_difference",
            "brier_delta": "brier_delta",
            "log_loss_delta": "log_loss_delta",
        }.get(metric)
        return (reader, metric_key) if reader and metric_key else None
    if claim_id in {"C3", "C7"}:
        return (metric,)
    if claim_id == "C4":
        suffix = {
            "evidence_recall": "recall",
            "wrong_scope_leakage": "wrong_namespace_leakage",
            "measured_non_usable_rate": "measured_non_usable_rate",
        }.get(metric)
        return (f"{contrast}_{suffix}",) if suffix else None
    if claim_id == "C5":
        absolute = {
            ("global_dense", "evidence_recall"): "global_dense_recall",
            ("namespace_dense", "evidence_recall"): "namespace_dense_recall",
            (
                "namespace_dense",
                "wrong_scope_leakage",
            ): "namespace_wrong_scope_leakage",
        }.get((contrast, metric))
        if absolute or "_delta" in metric:
            return (absolute or metric,)
        if contrast.endswith("_matched_prefix") and metric in {
            "known_non_usable_rate",
            "non_usable_label_coverage",
            "non_usable_lower_bound",
            "non_usable_upper_bound",
        }:
            arm = contrast.removesuffix("_matched_prefix")
            if arm in {"global_dense", "namespace_dense"}:
                return ("matched_prefix_diagnostics", arm, metric)
        if contrast in {
            "global_bm25",
            "global_dense",
            "global_bm25_dense_rrf",
            "global_recency_dense",
            "namespace_dense",
            "namespace_current_only",
            "released_intent_lifecycle_upper_bound",
            "threshold_router",
            "cluster_router",
        } and metric in {
            "evidence_recall",
            "feasible_rate",
            "penalized_non_usable_upper_risk",
            "mean_candidates_scored",
        }:
            return ("arm_summary", contrast, metric)
        return None
    if claim_id == "C6":
        method = contrast.removesuffix("_minus_namespace_dense")
        if method not in {"threshold_router", "cluster_router"}:
            return None
        return (method, metric)
    if claim_id == "C8":
        reader = {
            "gpt-5.6-sol": "openai",
            "gemini-3.6-flash": "gemini",
            "deepseek-v4-pro": "deepseek",
        }.get(row["source"])
        if reader and metric in {
            "relevant_admissible_effect",
            "relevant_inadmissible_effect",
            "irrelevant_admissible_effect",
            "irrelevant_inadmissible_effect",
            "selectivity_gap",
        }:
            return (reader, metric)
    if (
        claim_id == "C12"
        and row["source"] == "claude-opus-5"
        and metric
        in {
            "relevant_admissible_effect",
            "relevant_inadmissible_effect",
            "irrelevant_admissible_effect",
            "irrelevant_inadmissible_effect",
            "selectivity_gap",
        }
    ):
        return ("claude", metric)
    if claim_id == "C13":
        reader = {
            "deepseek-v4-pro": "deepseek_v4_pro",
            "gemini-3.6-flash": "gemini_3_6_flash",
            "gpt-5.6-luna": "gpt_5_6_luna",
        }.get(row["source"])
        return (reader, contrast, metric) if reader else None
    if claim_id == "C14":
        return (contrast, metric)
    if claim_id in {"C9", "C10", "C11"}:
        return (contrast, metric)
    return None


def _contract_interval_metric(row: Mapping[str, str]) -> str | None:
    claim_id = row["claim_id"]
    metric = row["metric"]
    if claim_id == "C2":
        reader = {
            "gpt-4o-mini-2024-07-18": "reader_a",
            "gpt-4o-2024-08-06": "reader_b",
        }.get(row["source"])
        metric_key = {
            "answer_leakage_risk_difference": "risk_difference",
            "brier_delta": "brier_delta",
            "log_loss_delta": "log_loss_delta",
        }.get(metric)
        return f"{reader}_{metric_key}" if reader and metric_key else None
    if claim_id in {"C3", "C5", "C7"}:
        return metric
    if claim_id == "C6":
        method = row["contrast"].removesuffix("_minus_namespace_dense")
        return f"{method}_{metric}"
    if claim_id == "C8":
        reader = {
            "gpt-5.6-sol": "openai",
            "gemini-3.6-flash": "gemini",
            "deepseek-v4-pro": "deepseek",
        }.get(row["source"])
        return f"{reader}_{metric}" if reader else None
    if claim_id == "C12" and row["source"] == "claude-opus-5":
        return f"claude_{metric}"
    if claim_id == "C13":
        reader = {
            "deepseek-v4-pro": "deepseek_v4_pro",
            "gemini-3.6-flash": "gemini_3_6_flash",
            "gpt-5.6-luna": "gpt_5_6_luna",
        }.get(row["source"])
        return f"{reader}__{row['contrast']}__{metric}" if reader else None
    if claim_id == "C14":
        return f"{row['contrast']}_{metric}"
    if claim_id in {"C9", "C10", "C11"}:
        return f"{row['contrast']}_{metric}"
    return None


def _nested_number(
    value: object,
    path: tuple[str, ...],
    *,
    label: str,
    errors: list[str],
) -> float | None:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            errors.append(f"{label} is missing contract path {'.'.join(path)}")
            return None
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        errors.append(f"{label} contract value must be numeric")
        return None
    number = float(current)
    if not math.isfinite(number):
        errors.append(f"{label} contract value must be finite")
        return None
    return number


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)


def _finite(value: str, label: str, errors: list[str]) -> float | None:
    try:
        number = float(value)
    except ValueError:
        errors.append(f"{label} must be numeric")
        return None
    if not math.isfinite(number):
        errors.append(f"{label} must be finite")
        return None
    return number


def validate_evidence(repository_root: Path) -> list[str]:
    errors: list[str] = []
    source_index = _source_artifact_index(repository_root, errors)
    claims = _claim_index(repository_root, errors)
    manifest_root = repository_root / "evidence" / "manifests"
    manifests = sorted(manifest_root.glob("*.json"))
    if not manifests:
        return ["no evidence manifests found"]

    observed_claims: set[str] = set()
    observed_contract_paths: dict[str, set[tuple[str, ...]]] = {
        claim_id: set() for claim_id in REQUIRED_CLAIMS
    }
    observed_intervals: dict[str, set[str]] = {claim_id: set() for claim_id in REQUIRED_CLAIMS}
    for manifest_path in manifests:
        try:
            manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{manifest_path.name} could not be loaded: {exc}")
            continue
        if not isinstance(manifest, Mapping):
            errors.append(f"{manifest_path.name} must contain an object")
            continue
        if manifest.get("schema_version") != 1:
            errors.append(f"{manifest_path.name} has unsupported schema_version")
        if manifest.get("contains_raw_text_or_private_content") is not False:
            errors.append(f"{manifest_path.name} is not marked content-free")

        normalized_name = manifest.get("normalized_file")
        normalized_path = _repository_path(
            repository_root,
            normalized_name,
            label=f"{manifest_path.name}.normalized_file",
            errors=errors,
        )
        if normalized_path is None:
            continue
        normalized_root = (repository_root / "evidence" / "normalized").resolve()
        if normalized_path.parent != normalized_root:
            errors.append(f"{manifest_path.name}.normalized_file is outside evidence/normalized")
        if not normalized_path.is_file():
            errors.append(f"{normalized_name} is missing")
            continue
        expected_hash = manifest.get("normalized_sha256")
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            errors.append(f"{manifest_path.name} has invalid normalized_sha256")
        elif sha256_file(normalized_path) != expected_hash:
            errors.append(f"{normalized_name} SHA-256 mismatch")

        source_artifacts = manifest.get("source_artifacts")
        if not isinstance(source_artifacts, list) or not source_artifacts:
            errors.append(f"{manifest_path.name} has no source artifacts")
        else:
            for source in source_artifacts:
                if not isinstance(source, Mapping):
                    errors.append(f"{manifest_path.name} has malformed source metadata")
                    continue
                source_path = source.get("path")
                digest = source.get("sha256")
                if not isinstance(source_path, str) or not source_path:
                    errors.append(f"{manifest_path.name} has invalid source path")
                    continue
                if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
                    errors.append(f"{manifest_path.name} has invalid source SHA-256")
                    continue
                indexed = source_index.get(source_path)
                if indexed is None:
                    errors.append(f"{manifest_path.name} source is not indexed: {source_path}")
                    continue
                if indexed.get("source_sha256") != digest:
                    errors.append(
                        f"{manifest_path.name} source SHA-256 disagrees with index: {source_path}"
                    )
                if indexed.get("old_repository") != manifest.get("source_repository"):
                    errors.append(
                        f"{manifest_path.name} source repository disagrees with index: "
                        f"{source_path}"
                    )
                source_commits = {manifest.get("source_snapshot_commit")}
                additional_commits = manifest.get("additional_source_snapshot_commits", [])
                if isinstance(additional_commits, list):
                    source_commits.update(additional_commits)
                if indexed.get("frozen_commit") not in source_commits:
                    errors.append(
                        f"{manifest_path.name} source commit disagrees with index: {source_path}"
                    )

        transformation_path = _repository_path(
            repository_root,
            manifest.get("transformation_script"),
            label=f"{manifest_path.name}.transformation_script",
            errors=errors,
        )
        transformation_hash = manifest.get("transformation_script_sha256")
        if not isinstance(transformation_hash, str) or not SHA256_PATTERN.fullmatch(
            transformation_hash
        ):
            errors.append(f"{manifest_path.name} has invalid transformation script SHA-256")
        elif transformation_path is None or not transformation_path.is_file():
            errors.append(f"{manifest_path.name} transformation script is missing")
        elif sha256_file(transformation_path) != transformation_hash:
            errors.append(f"{manifest_path.name} transformation script SHA-256 mismatch")

        with normalized_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            if columns != REQUIRED_COLUMNS:
                errors.append(f"{normalized_name} has unexpected columns")
                continue
            rows = list(reader)
        if len(rows) != manifest.get("row_count"):
            errors.append(f"{normalized_name} row count does not match manifest")

        for index, row in enumerate(rows, start=2):
            prefix = f"{normalized_name}:{index}"
            claim_id = row["claim_id"]
            observed_claims.add(claim_id)
            estimate = _finite(row["estimate"], f"{prefix}.estimate", errors)
            lower = (
                None
                if row["ci95_lower"] == ""
                else _finite(row["ci95_lower"], f"{prefix}.ci95_lower", errors)
            )
            upper = (
                None
                if row["ci95_upper"] == ""
                else _finite(row["ci95_upper"], f"{prefix}.ci95_upper", errors)
            )
            if (lower is None) != (upper is None):
                errors.append(f"{prefix} must provide both confidence bounds")
            if (
                estimate is not None
                and lower is not None
                and upper is not None
                and not lower <= estimate <= upper
            ):
                errors.append(f"{prefix} confidence interval excludes estimate")
            if row["n"]:
                count = _finite(row["n"], f"{prefix}.n", errors)
                if count is not None and (count < 0 or not count.is_integer()):
                    errors.append(f"{prefix}.n must be a non-negative integer")
            if row["contrast"] in PROHIBITED_OUTPUT_LABELS:
                errors.append(f"{prefix} uses a retired method label")
            if row["claim_id"] == "C2":
                notes = row["notes"]
                if "association_only" not in notes or "same_provider" not in notes:
                    errors.append(f"{prefix} lacks GateMem association boundaries")
            if row["claim_id"] == "C4" and "mechanism_smoke_only" not in row["notes"]:
                errors.append(f"{prefix} lacks mechanism-smoke boundary")
            if row["claim_id"] == "C7" and "released_field_upper_bound" not in row["notes"]:
                errors.append(f"{prefix} lacks lifecycle upper-bound boundary")
            if row["claim_id"] == "C8":
                notes = row["notes"]
                if (
                    "controlled_prompt_intervention" not in notes
                    or "providers_not_pooled" not in notes
                ):
                    errors.append(f"{prefix} lacks controlled-exposure boundaries")
            if row["claim_id"] == "C9":
                notes = row["notes"]
                if "post_hoc_v2" not in notes or "no_retuning" not in notes:
                    errors.append(f"{prefix} lacks post-hoc fixed-ranking boundaries")
            if row["claim_id"] == "C10":
                notes = row["notes"]
                if row["contrast"] in {"policy_only", "lifecycle_only", "governance_v2"}:
                    if "post_hoc_v2" not in notes or "corrected_governance_semantics" not in notes:
                        errors.append(f"{prefix} lacks corrected-v2 attribution boundaries")
                elif row["metric"] in {
                    "last_observed_dominating_rate",
                    "first_observed_non_dominating_rate",
                }:
                    if (
                        "observed_grid_bracket" not in notes
                        or "not_population_threshold" not in notes
                    ):
                        errors.append(f"{prefix} lacks observed-grid boundaries")
                elif "full_natural_reranking" not in notes or "no_retuning" not in notes:
                    errors.append(f"{prefix} lacks full-reranking boundaries")
            if row["claim_id"] == "C11":
                notes = row["notes"]
                if "public_development" not in notes or "no_model_pooling" not in notes:
                    errors.append(f"{prefix} lacks public-development no-pooling boundaries")
            if row["claim_id"] == "C12":
                notes = row["notes"]
                if (
                    "controlled_prompt_intervention" not in notes
                    or "separate_fourth_reader_replication" not in notes
                    or "providers_not_pooled" not in notes
                ):
                    errors.append(f"{prefix} lacks separate reader-replication boundaries")
            if row["claim_id"] == "C13":
                notes = row["notes"]
                required_notes = {
                    "natural_same_population",
                    "reader_estimates_not_pooled",
                    "shared_claude_haiku_judge",
                    "nonofficial_sample",
                    "no_general_disclosure_gain",
                }
                missing_notes = sorted(note for note in required_notes if note not in notes)
                if missing_notes:
                    errors.append(
                        f"{prefix} lacks natural end-to-end boundaries: {missing_notes!r}"
                    )
                if row["source"] == "gpt-5.6-luna" and "sequential_reader_replication" not in notes:
                    errors.append(f"{prefix} lacks sequential reader-replication boundary")
            if row["claim_id"] == "C14":
                notes = row["notes"]
                required_notes = {
                    "post_hoc_outcome_independent",
                    "not_independently_preregistered",
                    "full_population_not_rescored",
                    "one_alternate_judge",
                    "reader_effects_not_reestimated",
                    "nonofficial_sample",
                }
                missing_notes = sorted(note for note in required_notes if note not in notes)
                if missing_notes:
                    errors.append(f"{prefix} lacks cross-judge audit boundaries: {missing_notes!r}")

            claim = claims.get(claim_id)
            contract_path = _contract_value_path(row)
            if claim is None:
                errors.append(f"{prefix} has no claim contract")
            elif contract_path is None:
                errors.append(f"{prefix} has no exact-value binding")
            else:
                if contract_path in observed_contract_paths.setdefault(claim_id, set()):
                    errors.append(f"{prefix} duplicates contract path {'.'.join(contract_path)}")
                observed_contract_paths[claim_id].add(contract_path)
                expected = _nested_number(
                    claim.get("exact_values"),
                    contract_path,
                    label=prefix,
                    errors=errors,
                )
                if (
                    estimate is not None
                    and expected is not None
                    and not _same_number(estimate, expected)
                ):
                    errors.append(f"{prefix}.estimate disagrees with claims/claims.yaml")

            if lower is not None and upper is not None and claim is not None:
                interval_metric = _contract_interval_metric(row)
                if interval_metric is None:
                    errors.append(f"{prefix} has no confidence-interval binding")
                else:
                    if interval_metric in observed_intervals.setdefault(claim_id, set()):
                        errors.append(f"{prefix} duplicates confidence interval {interval_metric}")
                    observed_intervals[claim_id].add(interval_metric)
                    intervals = claim.get("confidence_intervals")
                    matching = (
                        [
                            interval
                            for interval in intervals
                            if isinstance(interval, Mapping)
                            and interval.get("metric") == interval_metric
                        ]
                        if isinstance(intervals, list)
                        else []
                    )
                    if len(matching) != 1:
                        errors.append(f"{prefix} has no unique contract interval {interval_metric}")
                    else:
                        for field, actual in (
                            ("estimate", estimate),
                            ("lower", lower),
                            ("upper", upper),
                        ):
                            expected = _nested_number(
                                matching[0],
                                (field,),
                                label=f"{prefix}.{field}",
                                errors=errors,
                            )
                            if (
                                actual is not None
                                and expected is not None
                                and not _same_number(actual, expected)
                            ):
                                errors.append(f"{prefix}.{field} disagrees with claims/claims.yaml")

    missing_claims = REQUIRED_CLAIMS - observed_claims
    if missing_claims:
        errors.append(f"normalized evidence is missing claims: {sorted(missing_claims)!r}")
    for claim_id, expected_count in EXPECTED_METRIC_COUNTS.items():
        observed_count = len(observed_contract_paths.get(claim_id, set()))
        if observed_count != expected_count:
            errors.append(
                f"{claim_id} binds {observed_count} exact metrics; expected {expected_count}"
            )
        claim = claims.get(claim_id)
        intervals = claim.get("confidence_intervals") if claim is not None else None
        expected_intervals = len(intervals) if isinstance(intervals, list) else 0
        observed_interval_count = len(observed_intervals.get(claim_id, set()))
        if observed_interval_count != expected_intervals:
            errors.append(
                f"{claim_id} binds {observed_interval_count} confidence intervals; "
                f"expected {expected_intervals}"
            )
    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=root)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors = validate_evidence(args.repository_root.resolve())
    if errors:
        print("evidence verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("normalized evidence valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
