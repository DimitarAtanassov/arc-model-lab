"""Request/response contracts for the summarize endpoint."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SummarizeRequest(BaseModel):
    input_text: str = Field(min_length=1, description="Text to summarize.")
    model_name: str | None = Field(
        default=None,
        description="Catalog model name to use; defaults to the configured model.",
    )


class SummarizeResponse(BaseModel):
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
