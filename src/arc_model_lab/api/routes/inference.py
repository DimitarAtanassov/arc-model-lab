from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.api.dependencies import get_inference_service, get_session
from arc_model_lab.api.schemas import InferenceRequest, InferenceResponse
from arc_model_lab.api.schemas.inference import InferenceDetailResponse, InferenceListItem
from arc_model_lab.services.inference_service import InferenceService

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ServiceDep = Annotated[InferenceService, Depends(get_inference_service)]

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200

router = APIRouter(tags=["inference"])


@router.post("/inference", response_model=InferenceResponse, status_code=status.HTTP_201_CREATED)
async def infer(
    payload: InferenceRequest,
    session: SessionDep,
    service: ServiceDep,
) -> InferenceResponse:
    inference = await service.summarize(
        session,
        model_name=payload.model_name,
        input_text=payload.input_text,
        temperature=payload.temperature,
    )
    return InferenceResponse.from_inference(inference)


@router.get("/inference", response_model=list[InferenceListItem])
async def list_inferences(
    session: SessionDep,
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=_MAX_LIMIT)] = _DEFAULT_LIMIT,
) -> list[InferenceListItem]:
    """Return recent inferences, newest first (bounded page size)."""
    return [InferenceListItem.from_inference(inference) for inference in await service.list_recent(session, limit)]


@router.get("/inference/{inference_id}", response_model=InferenceDetailResponse)
async def get_inference(inference_id: UUID, session: SessionDep, service: ServiceDep) -> InferenceDetailResponse:
    """Return one inference with its evaluation scores, or 404 when absent."""
    view = await service.get_detail(session, inference_id)
    return InferenceDetailResponse.from_inference_and_evaluations(view.inference, view.evaluations)
