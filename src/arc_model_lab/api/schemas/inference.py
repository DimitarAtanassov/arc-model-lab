"""Request/response contracts for the inference endpoint."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.api.schemas.evaluations import EvaluationEnvelope
from arc_model_lab.domain import EvaluationOutcome, Inference


class InferenceRequest(BaseModel):
    # The caller does not choose the model: every request runs on the deployed
    # model. ``extra="forbid"`` rejects a stale ``model_name`` with 422 rather
    # than silently ignoring it.
    model_config = ConfigDict(extra="forbid")

    input_text: str = Field(min_length=1, description="Text to summarize.")
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
    experiment_id: UUID | None = None
    created_at: datetime
    evaluation: EvaluationEnvelope | None = None

    @classmethod
    def from_inference(cls, inference: Inference, evaluation: EvaluationOutcome | None = None) -> InferenceResponse:
        """Assemble the response from a persisted inference and its optional scores.

        One factory builds this shape for both ``/inference`` and experiment runs,
        so the two transports cannot drift. The evaluation envelope is attached via
        an immutable copy rather than a post-validation field mutation.
        """
        response = cls.model_validate(inference)
        if evaluation is None:
            return response
        return response.model_copy(update={"evaluation": EvaluationEnvelope.from_outcome(evaluation)})
