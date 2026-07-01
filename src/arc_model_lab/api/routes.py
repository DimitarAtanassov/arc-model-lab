"""HTTP routing layer. Thin by design: validate input, delegate, shape output."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_inference_service, get_session
from arc_model_lab.api.schemas import SummarizeRequest, SummarizeResponse
from arc_model_lab.services.inference_service import InferenceService

SessionDep = Annotated[Session, Depends(get_session)]
InferenceServiceDep = Annotated[InferenceService, Depends(get_inference_service)]

router = APIRouter()


@router.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["inference"],
)
def summarize(
    payload: SummarizeRequest,
    session: SessionDep,
    service: InferenceServiceDep,
) -> SummarizeResponse:
    inference = service.summarize(session, payload.input_text)
    return SummarizeResponse.model_validate(inference)
