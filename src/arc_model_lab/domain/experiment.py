from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from arc_model_lab.domain.generation import GenerationConfig


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    model_id: UUID
    generation_config: GenerationConfig
    description: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Associates one inference with the experiment that produced it.

    The link lives here, not on Inference,
    so inference stays orthogonal to experiments: an inference never references an
    experiment, and this association is written only when a run executes under the
    experiment endpoint.
    """

    experiment_id: UUID
    inference_id: UUID
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class ExperimentMetricAggregate:
    """One metric's aggregate score across an experiment's evaluated inferences."""

    metric_name: str
    average_score: float
    evaluated_count: int


@dataclass(frozen=True, slots=True)
class ExperimentResults:
    """An experiment's aggregated metric scores, tagged with its id.

    The unit of comparison: each experiment id paired with its per-metric
    aggregates.
    """

    experiment_id: UUID
    metrics: list[ExperimentMetricAggregate]
