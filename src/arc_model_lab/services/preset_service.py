from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import PresetRepository
from arc_model_lab.domain import (
    GenerationConfig,
    GenerationPreset,
    PresetNameConflictError,
    PresetNotFoundError,
    PresetStatus,
)
from arc_model_lab.domain.generation import enforce_output_cap


class PresetService:
    """Owns the preset workflow: validate the config, persist, and commit.

    A preset's config is validated through the one ``GenerationConfig`` constructor
    (registry bounds and cross-field mode checks) plus the server output-token cap,
    so an out-of-range or contradictory config is a 422 before persistence. The
    duplicate-active-name 409 is enforced by the database partial unique index; the
    pre-check here is a fast path only.
    """

    def __init__(self, max_output_tokens_cap: int) -> None:
        self._max_output_tokens_cap = max_output_tokens_cap

    async def create(
        self,
        session: AsyncSession,
        *,
        name: str,
        description: str | None,
        config_params: Mapping[str, Any],
    ) -> GenerationPreset:
        """Create an active preset, or raise on an invalid config (422) or taken name (409)."""
        config = self._build_config(config_params)
        repo = PresetRepository(session)
        # Fast-path pre-check; the partial unique index is the authority and catches
        # the concurrent-create race that a check-then-insert cannot (see PresetRepository.add).
        if await repo.get_active_by_name(name) is not None:
            raise PresetNameConflictError(f"An active preset already uses the name: {name}")
        preset = GenerationPreset(name=name, description=description, config=config)
        persisted = await repo.add(preset)
        await session.commit()
        return persisted

    async def get(self, session: AsyncSession, preset_id: UUID) -> GenerationPreset:
        """Return one active preset, or raise PresetNotFoundError (404).

        An archived preset is hidden from use: it 404s here even though its row and
        lineage link remain, because it is no longer usable for new inferences.
        """
        preset = await PresetRepository(session).get(preset_id)
        if preset is None or preset.status is not PresetStatus.ACTIVE:
            raise PresetNotFoundError(f"Preset not found: {preset_id}")
        return preset

    async def list_active(self, session: AsyncSession) -> list[GenerationPreset]:
        """Return active presets, newest first (archived presets stay hidden)."""
        return await PresetRepository(session).list_active()

    async def update(
        self,
        session: AsyncSession,
        preset_id: UUID,
        *,
        config_params: Mapping[str, Any] | None,
        description: str | None,
        description_set: bool,
    ) -> GenerationPreset:
        """Edit an active preset's description and/or config, or raise 404/422.

        ``config_params`` is the full new bundle when present (a preset config is
        replaced, not merged); omit it to keep the current config. ``description_set``
        distinguishes "clear the description" (null) from "leave it unchanged".
        """
        preset = await self.get(session, preset_id)
        config = self._build_config(config_params) if config_params is not None else preset.config
        new_description = description if description_set else preset.description
        edited = replace(preset, description=new_description, config=config)
        persisted = await PresetRepository(session).update(edited)
        await session.commit()
        return persisted

    async def archive(self, session: AsyncSession, preset_id: UUID) -> None:
        """Soft-delete an active preset, freeing its name for reuse, or raise 404."""
        preset = await self.get(session, preset_id)
        await PresetRepository(session).update(replace(preset, status=PresetStatus.ARCHIVED))
        await session.commit()

    def _build_config(self, params: Mapping[str, Any]) -> GenerationConfig:
        config = GenerationConfig.from_dict(params)
        enforce_output_cap(config, self._max_output_tokens_cap)
        return config
