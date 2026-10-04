"""Pure helpers for the final zero-call submission diagnostics."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence


def deterministic_uniform(*, seed: int, channel: str, identity: str) -> float:
    """Map an identity to a deterministic value in the half-open interval [0, 1)."""
    payload = f"{seed}\0{channel}\0{identity}".encode()
    numerator = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return numerator / 2**64


def mask_established_statuses(
    statuses: Mapping[str, bool | None],
    *,
    missing_rate: float,
    seed: int,
    identity: str,
) -> dict[str, bool | None]:
    """Hide established item judgments without changing pre-existing unknowns.

    The diagnostic masks evaluator knowledge after retrieval. It therefore changes
    neither ranking nor recall and makes no assumption about the hidden label's value.
    """
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be between zero and one")
    output: dict[str, bool | None] = {}
    for item_id, status in statuses.items():
        hidden = (
            deterministic_uniform(
                seed=seed,
                channel="evaluator_label_missingness",
                identity=f"{identity}\0{item_id}",
            )
            < missing_rate
        )
        output[item_id] = None if status is not None and hidden else status
    return output


def sample_gold_preserving_positions(
    anchor_positions: Sequence[int],
    non_anchor_positions: Sequence[int],
    *,
    target_size: int,
    seed: int,
    identity: str,
) -> tuple[int, ...]:
    """Sample a fixed-size support while retaining every required anchor.

    This is an oracle diagnostic, not a deployable routing rule. The returned support
    is sorted only to make its representation independent of sampling order.
    """
    anchors = tuple(anchor_positions)
    non_anchors = tuple(non_anchor_positions)
    if len(set(anchors)) != len(anchors) or len(set(non_anchors)) != len(non_anchors):
        raise ValueError("support positions must be unique within each partition")
    if set(anchors).intersection(non_anchors):
        raise ValueError("anchor and non-anchor positions must be disjoint")
    if target_size < len(anchors):
        raise ValueError("target support is too small to retain every anchor")
    if target_size > len(anchors) + len(non_anchors):
        raise ValueError("target support exceeds the candidate pool")

    sample_size = target_size - len(anchors)
    stream_seed = int.from_bytes(
        hashlib.sha256(f"{seed}\0gold_preserving_same_size\0{identity}".encode()).digest()[:8],
        "big",
    )
    sampled = random.Random(stream_seed).sample(non_anchors, sample_size)
    return tuple(sorted((*anchors, *sampled)))
