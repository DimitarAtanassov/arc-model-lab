from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from arc_model_lab.domain.enums import EvaluationStatus


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    inference_id: UUID
    metric_name: str
    score: float
    evaluator_name: str
    reasoning: str | None = None
    evaluator_version: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """The result of one evaluation attempt for one inference.

    results is empty unless status is COMPLETED. A FAILED outcome
    means the eval service was unreachable or returned something unparseable and
    the caller chose to fail open; SKIPPED means evaluation was not wired for
    this environment.
    """

    status: EvaluationStatus
    results: tuple[EvaluationResult, ...] = ()
