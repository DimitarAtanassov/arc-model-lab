from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from arc_model_lab.domain.enums import ModelStatus, Provider


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    provider: Provider
    model_id: str
    tokenizer_id: str
    revision: str | None = None
    adapter_path: str | None = None
    status: ModelStatus = ModelStatus.ACTIVE
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
