"""Typed schema for query-conditioned memory admissibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Relevance(StrEnum):
    REQUIRED = "required"
    SUPPORTIVE = "supportive"
    NOT_USEFUL = "not_useful"
    UNKNOWN = "unknown"


class Scope(StrEnum):
    ALLOWED = "allowed"
    DISALLOWED = "disallowed"
    UNKNOWN = "unknown"


class LifecycleState(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class QueryIntent(StrEnum):
    CURRENT_STATE = "current_state"
    HISTORY = "history"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MemoryAssessment:
    """Typed assessment of one memory relative to one query."""

    memory_id: str
    relevance: Relevance
    scope: Scope
    lifecycle_state: LifecycleState
    policy_allowed: bool | None
    query_intent: QueryIntent

    def __post_init__(self) -> None:
        if not self.memory_id:
            raise ValueError("memory_id must be nonempty")
        enum_fields = {
            "relevance": (self.relevance, Relevance),
            "scope": (self.scope, Scope),
            "lifecycle_state": (self.lifecycle_state, LifecycleState),
            "query_intent": (self.query_intent, QueryIntent),
        }
        for name, (value, enum_type) in enum_fields.items():
            if not isinstance(value, enum_type):
                raise TypeError(f"{name} must be {enum_type.__name__}")
        if self.policy_allowed is not None and not isinstance(self.policy_allowed, bool):
            raise TypeError("policy_allowed must be true, false, or unknown")


@dataclass(frozen=True)
class ExposureUseRecord:
    """One fixed-checkpoint exposure and downstream-use observation."""

    checkpoint_id: str
    exposed: bool
    used_in_answer: bool

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be nonempty")
        if not isinstance(self.exposed, bool) or not isinstance(self.used_in_answer, bool):
            raise TypeError("exposed and used_in_answer must be booleans")
