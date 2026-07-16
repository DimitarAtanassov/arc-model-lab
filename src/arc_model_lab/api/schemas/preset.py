from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.api.schemas.generation import GenerationParams
from arc_model_lab.domain import GenerationPreset

_MAX_NAME_CHARS = 255
_MAX_DESCRIPTION_CHARS = 2000


class PresetCreateRequest(BaseModel):
    """Create body: a name, an optional note, and the decoding config to bundle.

    ``config`` is the registry allow-list, so an unknown knob is a 422 here; the
    numeric bounds and cross-field mode checks run when the service builds the
    ``GenerationConfig``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1, max_length=_MAX_NAME_CHARS, description="Stable, unique handle among active presets."
    )
    description: str | None = Field(default=None, max_length=_MAX_DESCRIPTION_CHARS)
    config: GenerationParams


class PresetUpdateRequest(BaseModel):
    """Patch body: change the description and/or replace the config bundle.

    A field left out is unchanged; ``description: null`` clears the note. ``config``
    when present replaces the whole bundle (a preset config is not merged).
    """

    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(default=None, max_length=_MAX_DESCRIPTION_CHARS)
    config: GenerationParams | None = None


class PresetResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: UUID  # noqa: A003 - mirrors the domain primary key
    name: str
    description: str | None
    # The resolved GenerationConfig payload the preset carries (the to_dict shape).
    config: dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, preset: GenerationPreset) -> PresetResponse:
        return cls(
            id=preset.id,
            name=preset.name,
            description=preset.description,
            config=preset.config.to_dict(),
            status=preset.status.value,
            created_at=preset.created_at,
            updated_at=preset.updated_at,
        )
