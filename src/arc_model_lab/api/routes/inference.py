"""The inference endpoint: run the model, optionally evaluate, shape the output.

Evaluation is opt-in per request: a caller that names one or more ``metrics`` gets
its output scored against them; a caller that omits ``metrics`` gets inference
only. An unknown metric name is a client error (404), surfaced from arc-eval.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import (
    get_evaluation_service,
    get_inference_service,
    get_session,
)
from arc_model_lab.api.schemas import InferenceRequest, InferenceResponse
from arc_model_lab.api.schemas.evaluations import EvaluationEnvelope
from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.inference_service import InferenceService

SessionDep = Annotated[Session, Depends(get_session)]
InferenceServiceDep = Annotated[InferenceService, Depends(get_inference_service)]
EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]

router = APIRouter(tags=["inference"])


@router.post("/inference", response_model=InferenceResponse, status_code=status.HTTP_201_CREATED)
def infer(
    payload: InferenceRequest,
    session: SessionDep,
    inference_service: InferenceServiceDep,
    evaluation_service: EvaluationServiceDep,
) -> InferenceResponse:
    inference = inference_service.summarize(session, payload.input_text, payload.model_name)
    response = InferenceResponse.model_validate(inference)
    if payload.metrics:
        outcome = evaluation_service.evaluate_inference(session, inference, payload.metrics)
        response.evaluation = EvaluationEnvelope.from_outcome(outcome)
    return response
