"""The run context shared across the inference use case.

An experiment run passes this to the inference service in place of the deployed
model and default decoding; ``/inference`` passes none and uses the defaults.

It lives in its own module so the three collaborators that produce and consume it
(``experiment_service``, ``inference_workflow``, ``inference_service``) share one
definition without importing each other just to reach it.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from arc_model_lab.domain import GenerationConfig, Model


@dataclass(frozen=True, slots=True)
class RunContext:
    """The model, decoding config, and experiment that pin one non-default run."""

    model: Model
    config: GenerationConfig
    experiment_id: UUID
