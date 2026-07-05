"""Request/response contracts for the inference endpoint."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.domain import GenerationConfig, Inference
from arc_model_lab.domain.generation import DEFAULT_TEMPERATURE


class InferenceRequest(BaseModel):
    # The caller names the model and the sampling temperature. ``extra="forbid"``
    # rejects an unknown field (including ``max_output_tokens``, a server default
    # here) with 422 rather than silently ignoring it.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(min_length=1, description="Catalog model to run.")
    input_text: str = Field(min_length=1, description="Text to summarize.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature: 0 is greedy/deterministic, higher is more random.",
    )

    def to_config(self) -> GenerationConfig:
        # max_output_tokens is not caller-controlled on /inference; use the default.
        return GenerationConfig(temperature=self.temperature)


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)

    id: UUID  # noqa: A003 - mirrors the domain primary key
    model_id: UUID
    input_text: str
    prompt: str
    output_text: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    created_at: datetime

    @classmethod
    def from_inference(cls, inference: Inference) -> InferenceResponse:
        """Shape the pure-inference response.

        No experiment id and no evaluation: ``/inference`` neither runs under an
        experiment nor scores its output. Those belong to the experiment flow.
        """
        return cls.model_validate(inference)
