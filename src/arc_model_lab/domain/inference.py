from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from arc_model_lab.domain.generation import GenerationConfig


@dataclass(frozen=True, slots=True)
class Inference:
    model_id: UUID
    input_text: str
    prompt: str
    output_text: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # The resolved decoding config used for this generation (after server
    # defaults), so the row alone reproduces the call.
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
