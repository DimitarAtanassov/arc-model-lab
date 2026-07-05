"""The Experiment domain entity: a named, reproducible run configuration.

An experiment pins a catalog model and a :class:`GenerationConfig` (and, once
phase 03 lands, a prompt version). Running it produces inference and evaluation
records tagged with the experiment id, so results are comparable in plain SQL.
The caller of ``/inference`` never chooses a model; an experiment is an
engineer-owned construct that runs its own configuration server-side.
"""

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
    prompt_version_id: UUID | None = None  # reserved for phase 03; nullable now
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

    The unit of comparison: pairing each id with its aggregates keeps two
    experiments distinct even when an experiment is compared with itself, which a
    map keyed by id would silently collapse to one entry.
    """

    experiment_id: UUID
    metrics: list[ExperimentMetricAggregate]
