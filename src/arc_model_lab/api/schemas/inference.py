"""Request/response contracts for the inference endpoint."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.domain import EvaluationResult, Inference
from arc_model_lab.domain.generation import MAX_TEMPERATURE

_PREVIEW_CHARS = 160


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """Collapse whitespace and truncate to a single-line table preview."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "\u2026"


class InferenceRequest(BaseModel):
    # The caller names the model and may set the sampling temperature; when it is
    # omitted the server default (``ARC_TEMPERATURE``) applies. ``extra="forbid"``
    # rejects an unknown field (including ``max_output_tokens``, a server-only
    # knob) with 422 rather than silently ignoring it.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(min_length=1, description="Catalog model to run.")
    input_text: str = Field(min_length=1, description="Text to summarize.")
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=MAX_TEMPERATURE,
        description=(
            "Sampling temperature: 0 is greedy/deterministic, higher is more random. "
            "Omit to use the server default (ARC_TEMPERATURE)."
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

    @classmethod
    def from_inference(cls, inference: Inference) -> InferenceResponse:
        """Shape the pure-inference response.

        No experiment id and no evaluation: ``/inference`` neither runs under an
        experiment nor scores its output. Those belong to the experiment flow.
        """
        return cls.model_validate(inference)


class InferenceListItem(BaseModel):
    """A compact inference row for the history table (previews, not full text)."""

    model_config = ConfigDict(protected_namespaces=())

    id: UUID  # noqa: A003 - mirrors the domain primary key
    model_id: UUID
    input_preview: str
    output_preview: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    created_at: datetime

    @classmethod
    def from_inference(cls, inference: Inference) -> InferenceListItem:
        return cls(
            id=inference.id,
            model_id=inference.model_id,
            input_preview=_preview(inference.input_text),
            output_preview=_preview(inference.output_text),
            latency_ms=inference.latency_ms,
            prompt_tokens=inference.prompt_tokens,
            completion_tokens=inference.completion_tokens,
            created_at=inference.created_at,
        )


class InferenceEvaluationOut(BaseModel):
    """One persisted metric score attached to an inference detail."""

    metric_name: str
    score: float
    reasoning: str | None
    evaluator_name: str
    evaluator_version: str | None
    created_at: datetime

    @classmethod
    def from_domain(cls, result: EvaluationResult) -> InferenceEvaluationOut:
        return cls(
            metric_name=result.metric_name,
            score=result.score,
            reasoning=result.reasoning,
            evaluator_name=result.evaluator_name,
            evaluator_version=result.evaluator_version,
            created_at=result.created_at,
        )


class InferenceDetailResponse(InferenceResponse):
    """A full inference plus every evaluation score recorded against it."""

    evaluations: list[InferenceEvaluationOut] = []

    @classmethod
    def from_inference_and_evaluations(
        cls, inference: Inference, evaluations: list[EvaluationResult]
    ) -> InferenceDetailResponse:
        return cls(
            id=inference.id,
            model_id=inference.model_id,
            input_text=inference.input_text,
            prompt=inference.prompt,
            output_text=inference.output_text,
            latency_ms=inference.latency_ms,
            prompt_tokens=inference.prompt_tokens,
            completion_tokens=inference.completion_tokens,
            created_at=inference.created_at,
            evaluations=[InferenceEvaluationOut.from_domain(result) for result in evaluations],
        )
