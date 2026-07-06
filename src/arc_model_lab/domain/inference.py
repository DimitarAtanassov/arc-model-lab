"""The Inference domain entity: one recorded execution of a model.

An inference is standalone: it holds no experiment reference, so inference stays
orthogonal to experiments. The experiment that produced an inference is recorded
separately (see :class:`~arc_model_lab.domain.experiment.ExperimentRun`), written
only when a run executes under the experiment endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Inference:
    model_id: UUID
    input_text: str
    prompt: str
    output_text: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
