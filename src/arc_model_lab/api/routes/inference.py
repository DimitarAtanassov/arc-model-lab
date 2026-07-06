"""The inference endpoint: run one model, shape the output.

Inference is standalone: it never evaluates and never runs under an experiment,
so the response carries neither scores nor an experiment id. Evaluation lives in
the experiment flow. The caller names the model and may set the sampling
temperature; an omitted temperature and the output length fall back to the
server default.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_inference_service, get_session
from arc_model_lab.api.schemas import InferenceRequest, InferenceResponse
from arc_model_lab.services.inference_service import InferenceService

SessionDep = Annotated[Session, Depends(get_session)]
ServiceDep = Annotated[InferenceService, Depends(get_inference_service)]

router = APIRouter(tags=["inference"])


@router.post("/inference", response_model=InferenceResponse, status_code=status.HTTP_201_CREATED)
def infer(
    payload: InferenceRequest,
    session: SessionDep,
    service: ServiceDep,
) -> InferenceResponse:
    inference = service.summarize(
        session,
        model_name=payload.model_name,
        input_text=payload.input_text,
        temperature=payload.temperature,
    )
    return InferenceResponse.from_inference(inference)
