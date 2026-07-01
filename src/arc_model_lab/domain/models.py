"""Pure domain entities. No framework, persistence, or I/O concerns live here."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _Entity(BaseModel):
    """Shared configuration for immutable domain entities."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())


class Model(_Entity):
    """A loadable inference model and the coordinates needed to load it."""

    id: UUID = Field(default_factory=uuid4)  # noqa: A003 - domain primary key
    name: str
    provider: str
    model_id: str
    tokenizer_id: str
    adapter_path: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Inference(_Entity):
    """A single, fully recorded model execution."""

    id: UUID = Field(default_factory=uuid4)  # noqa: A003 - domain primary key
    model_id: UUID
    input_text: str
    prompt: str
    output_text: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime = Field(default_factory=_utcnow)
