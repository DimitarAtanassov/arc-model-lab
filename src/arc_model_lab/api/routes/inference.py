"""The inference endpoint: run the model, optionally evaluate, shape the output.

Evaluation is opt-in per request: a caller that names one or more ``metrics`` gets
its output scored against them; a caller that omits ``metrics`` gets inference
only. An unknown metric name is a client error (404), surfaced from arc-eval.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_inference_workflow, get_session
from arc_model_lab.api.schemas import InferenceRequest, InferenceResponse
from arc_model_lab.services.inference_workflow import InferenceWorkflow

SessionDep = Annotated[Session, Depends(get_session)]
WorkflowDep = Annotated[InferenceWorkflow, Depends(get_inference_workflow)]

router = APIRouter(tags=["inference"])


@router.post("/inference", response_model=InferenceResponse, status_code=status.HTTP_201_CREATED)
def infer(
    payload: InferenceRequest,
    session: SessionDep,
    workflow: WorkflowDep,
) -> InferenceResponse:
    result = workflow.run(
        session,
        input_text=payload.input_text,
        metrics=payload.metrics,
    )
    return InferenceResponse.from_inference(result.inference, result.evaluation)
