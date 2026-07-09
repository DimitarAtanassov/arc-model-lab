from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.api.dependencies import get_inference_service, get_session
from arc_model_lab.api.schemas import InferenceRequest, InferenceResponse
from arc_model_lab.api.schemas.inference import InferenceListItem, InferenceRunRequest
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


@router.post("/v1/inference:run", response_model=InferenceResponse, status_code=status.HTTP_201_CREATED)
async def run_inference(
    payload: InferenceRunRequest,
    session: SessionDep,
    service: ServiceDep,
) -> InferenceResponse:
    """Service-to-service inference: run a named model with an explicit config.

    Used by arc-eval-service to run an experiment's model, which may be inactive.
    Returns 404 for an unknown model and 409 for an inactive model when
    allow_inactive is false.

    Path-versioned (``/v1/...:run``) to mark the versioned service-to-service seam,
    distinct from the unversioned public ``/inference`` surface, so the contract
    with arc-eval-service can evolve independently.
    """
    inference = await service.run_named(
        session,
        model_name=payload.model_name,
        input_text=payload.input_text,
        config=payload.generation_config.to_domain(),
        allow_inactive=payload.allow_inactive,
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


@router.get("/inference/{inference_id}", response_model=InferenceResponse)
async def get_inference(inference_id: UUID, session: SessionDep, service: ServiceDep) -> InferenceResponse:
    """Return one inference by id, or 404 when absent."""
    return InferenceResponse.from_inference(await service.get(session, inference_id))
