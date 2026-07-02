"""Request/response contracts for the inference endpoint."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.api.schemas.evaluations import EvaluationEnvelope


class InferenceRequest(BaseModel):
    input_text: str = Field(min_length=1, description="Text to summarize.")
    model_name: str | None = Field(
        default=None,
        description="Catalog model name to use; defaults to the configured model.",
    )
    metrics: list[str] | None = Field(
        default=None,
        description=(
            "Metrics to evaluate the output against. When omitted, the output is "
            "not evaluated. An unknown metric name is rejected with 404."
        ),
    )


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
    evaluation: EvaluationEnvelope | None = None
