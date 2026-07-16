from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from arc_model_lab.api.dependencies import get_settings
from arc_model_lab.api.schemas.generation import GenerationParamsResponse
from arc_model_lab.config import Settings

SettingsDep = Annotated[Settings, Depends(get_settings)]

router = APIRouter(tags=["generation"])


@router.get("/generation/params", response_model=GenerationParamsResponse)
async def get_generation_params(settings: SettingsDep) -> GenerationParamsResponse:
    """Return the decoding parameter registry and the effective output-token cap.

    The registry is the single source of bounds the UI renders controls from; the
    cap is the server-authoritative runtime ceiling for max_output_tokens.
    """
    return GenerationParamsResponse.build(settings.max_output_tokens_cap)
