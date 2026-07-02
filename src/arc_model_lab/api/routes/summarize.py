"""The summarize endpoint: validate input, delegate to the service, shape output."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import (
    get_evaluation_service,
    get_inference_service,
    get_session,
)
from arc_model_lab.api.schemas import SummarizeRequest, SummarizeResponse
from arc_model_lab.api.schemas.evaluations import EvaluationEnvelope
from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.inference_service import InferenceService

SessionDep = Annotated[Session, Depends(get_session)]
InferenceServiceDep = Annotated[InferenceService, Depends(get_inference_service)]
EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]

router = APIRouter(tags=["inference"])


@router.post("/summarize", response_model=SummarizeResponse, status_code=status.HTTP_201_CREATED)
def summarize(
    payload: SummarizeRequest,
    session: SessionDep,
    inference_service: InferenceServiceDep,
    evaluation_service: EvaluationServiceDep,
) -> SummarizeResponse:
    inference = inference_service.summarize(session, payload.input_text, payload.model_name)
    response = SummarizeResponse.model_validate(inference)
    if payload.evaluate:
        outcome = evaluation_service.evaluate_inference(session, inference)
        response.evaluation = EvaluationEnvelope.from_outcome(outcome)
    return response
