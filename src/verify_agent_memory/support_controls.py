"""Pure helpers for frozen support-control and robustness analyses."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchedRecallScore:
    """Query-level matched-recall score before aggregate publication."""

    evidence_recall: float
    feasible: bool
    matched_prefix: tuple[str, ...]
    admissibility_upper_risk: float | None
    residual_upper_risk: float | None
    penalized_admissibility_upper_risk: float
    penalized_residual_upper_risk: float
    known_admissibility_violation_count: int | None
    any_known_admissibility_violation: bool | None
    upper_admissibility_violation_count: int | None


def identity_seed(seed: int, channel: str, *identity: str) -> int:
    """Derive a stable PRNG seed without depending on process hash randomization."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not channel or any(not value for value in identity):
        raise ValueError("channel and identity components must be nonempty")
    payload = "\0".join((str(seed), channel, *identity)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def permute_labels(
    labels: Sequence[int],
    *,
    seed: int,
    channel: str,
    identity: str,
) -> tuple[int, ...]:
    """Randomly permute labels while preserving their exact marginal counts."""
    output = list(labels)
    random.Random(identity_seed(seed, channel, identity)).shuffle(output)
    return tuple(output)


def filter_ranking(
    ranked_memory_ids: Sequence[str],
    allowed_memory_ids: set[str] | frozenset[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    """Preserve ranking order while applying a fixed support mask."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    if limit == 0:
        return ()
    selected: list[str] = []
    for memory_id in ranked_memory_ids:
        if memory_id in allowed_memory_ids:
            selected.append(memory_id)
            if len(selected) == limit:
                break
    return tuple(selected)


def _validate_statuses(
    ranked_memory_ids: Sequence[str],
    statuses: Mapping[str, bool | None],
    label: str,
) -> None:
    missing = set(ranked_memory_ids) - statuses.keys()
    if missing:
        raise ValueError(f"{label} lacks statuses for ranked memories")
    if any(value is not None and not isinstance(value, bool) for value in statuses.values()):
        raise ValueError(f"{label} values must be true, false, or unresolved")


def score_matched_recall(
    ranked_memory_ids: Sequence[str],
    anchor_ids: set[str] | frozenset[str],
    admissibility_statuses: Mapping[str, bool | None],
    residual_statuses: Mapping[str, bool | None],
    *,
    target_recall: float,
    infeasibility_cost: float,
) -> MatchedRecallScore:
    """Score full and scope-excluded upper risk at a fixed recall target."""
    if not 0 < target_recall <= 1 or not math.isfinite(target_recall):
        raise ValueError("target_recall must be finite and in (0, 1]")
    if not 0 <= infeasibility_cost <= 1 or not math.isfinite(infeasibility_cost):
        raise ValueError("infeasibility_cost must be finite and in [0, 1]")
    if not anchor_ids:
        raise ValueError("at least one required anchor is required")
    if len(set(ranked_memory_ids)) != len(ranked_memory_ids):
        raise ValueError("ranked memory IDs must be unique")
    _validate_statuses(ranked_memory_ids, admissibility_statuses, "admissibility")
    _validate_statuses(ranked_memory_ids, residual_statuses, "residual")

    retrieved = len(anchor_ids.intersection(ranked_memory_ids))
    evidence_recall = retrieved / len(anchor_ids)
    feasible = evidence_recall >= target_recall
    if not feasible:
        return MatchedRecallScore(
            evidence_recall=evidence_recall,
            feasible=False,
            matched_prefix=(),
            admissibility_upper_risk=None,
            residual_upper_risk=None,
            penalized_admissibility_upper_risk=infeasibility_cost,
            penalized_residual_upper_risk=infeasibility_cost,
            known_admissibility_violation_count=None,
            any_known_admissibility_violation=None,
            upper_admissibility_violation_count=None,
        )

    hits = 0
    prefix: tuple[str, ...] = ()
    for index, memory_id in enumerate(ranked_memory_ids, start=1):
        hits += int(memory_id in anchor_ids)
        if hits / len(anchor_ids) >= target_recall:
            prefix = tuple(ranked_memory_ids[:index])
            break
    if not prefix:
        raise RuntimeError("feasible ranking lacks a matched-recall prefix")

    known_violation_count = sum(admissibility_statuses[memory_id] is False for memory_id in prefix)
    upper_violation_count = sum(
        admissibility_statuses[memory_id] is not True for memory_id in prefix
    )
    full_upper = upper_violation_count / len(prefix)
    residual_upper = sum(residual_statuses[memory_id] is not True for memory_id in prefix) / len(
        prefix
    )
    return MatchedRecallScore(
        evidence_recall=evidence_recall,
        feasible=True,
        matched_prefix=prefix,
        admissibility_upper_risk=full_upper,
        residual_upper_risk=residual_upper,
        penalized_admissibility_upper_risk=full_upper,
        penalized_residual_upper_risk=residual_upper,
        known_admissibility_violation_count=known_violation_count,
        any_known_admissibility_violation=known_violation_count > 0,
        upper_admissibility_violation_count=upper_violation_count,
    )


def leave_one_out_means(values: Mapping[str, float]) -> dict[str, float]:
    """Return the mean after omitting each named unit once."""
    if len(values) < 2:
        raise ValueError("leave-one-out analysis requires at least two units")
    numeric = {key: float(value) for key, value in values.items()}
    invalid_name = any(not key for key in numeric)
    invalid_value = any(not math.isfinite(value) for value in numeric.values())
    if invalid_name or invalid_value:
        raise ValueError("leave-one-out inputs must have names and finite values")
    total = sum(numeric.values())
    return {key: (total - value) / (len(numeric) - 1) for key, value in sorted(numeric.items())}


def scenario_selectivity(
    rows: Sequence[Mapping[str, object]],
    *,
    scenario_field: str = "scenario_id",
    cell_field: str = "cell",
    effect_field: str = "exposure_effect",
) -> dict[str, float]:
    """Compute per-scenario admissible-minus-inadmissible exposure effects."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        scenario = str(row.get(scenario_field, ""))
        cell = str(row.get(cell_field, ""))
        if not scenario or cell not in {"relevant_admissible", "relevant_inadmissible"}:
            continue
        raw_effect = row.get(effect_field)
        if isinstance(raw_effect, bool) or not isinstance(raw_effect, (int, float, str)):
            raise TypeError("exposure effects must be numeric")
        effect = float(raw_effect)
        if not math.isfinite(effect):
            raise ValueError("exposure effects must be finite")
        grouped.setdefault(scenario, {}).setdefault(cell, []).append(effect)
    output: dict[str, float] = {}
    for scenario, cells in sorted(grouped.items()):
        if set(cells) != {"relevant_admissible", "relevant_inadmissible"}:
            raise ValueError("each scenario requires both relevant selectivity cells")
        admissible = sum(cells["relevant_admissible"]) / len(cells["relevant_admissible"])
        inadmissible = sum(cells["relevant_inadmissible"]) / len(cells["relevant_inadmissible"])
        output[scenario] = admissible - inadmissible
    if not output:
        raise ValueError("no scenario selectivity rows were found")
    return output


def exact_sign_flip_pvalue(deltas: Sequence[float]) -> float:
    """Two-sided exact sign-flip p-value for at most 20 paired units."""
    numeric = tuple(float(value) for value in deltas if float(value) != 0.0)
    if not numeric:
        return 1.0
    if len(numeric) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 nonzero units")
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("sign-flip deltas must be finite")
    observed = abs(sum(numeric) / len(numeric))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(numeric)):
        signed_sum = sum(sign * value for sign, value in zip(signs, numeric, strict=True))
        statistic = abs(signed_sum) / len(numeric)
        extreme += int(statistic >= observed - 1e-15)
        total += 1
    return extreme / total
