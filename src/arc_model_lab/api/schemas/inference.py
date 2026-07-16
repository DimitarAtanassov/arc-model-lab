from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.api.schemas.generation import GenerationParams
from arc_model_lab.domain import Inference

_PREVIEW_CHARS = 160


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """Collapse whitespace and truncate to a single-line table preview."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "\u2026"


class InferenceRequest(BaseModel):
    # The caller names the model and may inform decoding two ways, in precedence
    # order: an ad-hoc `model_params` override wins over a stored `preset_id`, and
    # both win over the server defaults (ARC_TEMPERATURE, ARC_MAX_OUTPUT_TOKENS).
    # `temperature` is not a top-level field; it is a key inside `model_params`.
    # extra="forbid" rejects an unknown field (including a legacy top-level
    # `temperature` or `max_output_tokens`) with 422 rather than silently ignoring it.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(min_length=1, description="Catalog model to run.")
    input_text: str = Field(min_length=1, description="Text to run through the model.")
    preset_id: UUID | None = Field(
        default=None,
        description="Optional stored preset to seed decoding; unknown or archived is 404.",
    )
    model_params: GenerationParams | None = Field(
        default=None,
        description=(
            "Optional ad-hoc decoding overrides, validated against the parameter "
            "registry allow-list. Wins over the preset; an out-of-range or "
            "contradictory value is 422."
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
    # The resolved config the row actually ran with (the to_dict payload), so the
    # response alone shows exactly what ran and can seed a "save as preset" action.
    generation_config: dict[str, Any]
    # The preset that informed this row, if any (lineage, not reproducibility).
    preset_id: UUID | None
    created_at: datetime

    @classmethod
    def from_inference(cls, inference: Inference) -> InferenceResponse:
        return cls(
            id=inference.id,
            model_id=inference.model_id,
            input_text=inference.input_text,
            prompt=inference.prompt,
            output_text=inference.output_text,
            latency_ms=inference.latency_ms,
            prompt_tokens=inference.prompt_tokens,
            completion_tokens=inference.completion_tokens,
            generation_config=inference.generation_config.to_dict(),
            preset_id=inference.preset_id,
            created_at=inference.created_at,
        )


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
