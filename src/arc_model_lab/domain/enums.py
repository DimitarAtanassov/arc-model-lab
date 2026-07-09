from __future__ import annotations

from enum import StrEnum


class Provider(StrEnum):
    HUGGINGFACE = "huggingface"


class ModelStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"


class EvaluationStatus(StrEnum):
    """Outcome of an evaluation attempt for one inference."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
