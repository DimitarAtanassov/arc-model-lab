from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.api.dependencies import get_preset_service, get_session
from arc_model_lab.api.schemas.preset import PresetCreateRequest, PresetResponse, PresetUpdateRequest
from arc_model_lab.services.preset_service import PresetService

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ServiceDep = Annotated[PresetService, Depends(get_preset_service)]

router = APIRouter(prefix="/presets", tags=["presets"])


@router.post("", response_model=PresetResponse, status_code=status.HTTP_201_CREATED)
async def create_preset(payload: PresetCreateRequest, session: SessionDep, service: ServiceDep) -> PresetResponse:
    """Create a preset; 422 on an invalid config, 409 on a duplicate active name."""
    preset = await service.create(
        session,
        name=payload.name,
        description=payload.description,
        config_params=payload.config.to_config_dict(),
    )
    return PresetResponse.from_domain(preset)


@router.get("", response_model=list[PresetResponse])
async def list_presets(session: SessionDep, service: ServiceDep) -> list[PresetResponse]:
    """Return active presets, newest first."""
    return [PresetResponse.from_domain(preset) for preset in await service.list_active(session)]


@router.get("/{preset_id}", response_model=PresetResponse)
async def get_preset(preset_id: UUID, session: SessionDep, service: ServiceDep) -> PresetResponse:
    """Return one active preset by id, or 404 when unknown or archived."""
    return PresetResponse.from_domain(await service.get(session, preset_id))


@router.patch("/{preset_id}", response_model=PresetResponse)
async def update_preset(
    preset_id: UUID, payload: PresetUpdateRequest, session: SessionDep, service: ServiceDep
) -> PresetResponse:
    """Update a preset's description and/or config; 404 unknown, 422 invalid config."""
    config_params = payload.config.to_config_dict() if payload.config is not None else None
    preset = await service.update(
        session,
        preset_id,
        config_params=config_params,
        description=payload.description,
        description_set="description" in payload.model_fields_set,
    )
    return PresetResponse.from_domain(preset)


@router.delete("/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_preset(preset_id: UUID, session: SessionDep, service: ServiceDep) -> Response:
    """Archive (soft-delete) a preset, freeing its name for reuse; 404 when unknown."""
    await service.archive(session, preset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
