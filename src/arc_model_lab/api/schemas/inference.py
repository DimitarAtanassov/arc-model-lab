from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.domain import Inference
from arc_model_lab.domain.generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_TEMPERATURE,
    GenerationConfig,
)

_PREVIEW_CHARS = 160


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """Collapse whitespace and truncate to a single-line table preview."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "\u2026"


class InferenceRequest(BaseModel):
    # The caller names the model and may set the sampling temperature; when it is
    # omitted the server default (ARC_TEMPERATURE) applies. extra="forbid"
    # rejects an unknown field (including max_output_tokens, a server-only
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


class GenerationConfigSchema(BaseModel):
    """The decoding config a service-to-service caller sends explicitly."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=MAX_TEMPERATURE)
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=1)

    def to_domain(self) -> GenerationConfig:
        return GenerationConfig(temperature=self.temperature, max_output_tokens=self.max_output_tokens)


class InferenceRunRequest(BaseModel):
    # Service-to-service body for POST /v1/inference:run. Unlike /inference it
    # carries a full generation config (temperature and max_output_tokens) and may
    # run an inactive candidate model.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(min_length=1, description="Catalog model to run.")
    input_text: str = Field(min_length=1, description="Text to summarize.")
    generation_config: GenerationConfigSchema = Field(default_factory=GenerationConfigSchema)
    allow_inactive: bool = Field(
        default=False,
        description="Run the model even if it is not active. Off by default so the endpoint fails closed.",
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
