from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from arc_model_lab.domain.generation import GenerationConfig


class PresetStatus(StrEnum):
    """A preset is active (usable, name-unique) or archived (soft-deleted, kept for lineage)."""

    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class GenerationPreset:
    """A named, reusable, model-agnostic bundle of decoding parameters.

    The preset is never the reproducibility source: an inference row copies the
    fully resolved ``generation_config`` it ran with, so a historical row still
    reproduces after the preset is edited or archived. Archiving is a soft delete
    that frees the name for reuse while keeping the row for the lineage link.
    """

    name: str
    config: GenerationConfig
    description: str | None = None
    status: PresetStatus = PresetStatus.ACTIVE
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
