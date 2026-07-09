from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.api.dependencies import get_evaluation_service, get_session
from arc_model_lab.api.schemas import EvaluationEnvelope, EvaluationRequest
from arc_model_lab.services.evaluation_service import EvaluationService

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]

router = APIRouter(tags=["evaluations"])


@router.post(
    "/inference/{inference_id}/evaluate",
    response_model=EvaluationEnvelope,
    status_code=status.HTTP_200_OK,
)
async def evaluate_inference(
    inference_id: UUID,
    payload: EvaluationRequest,
    session: SessionDep,
    service: ServiceDep,
) -> EvaluationEnvelope:
    """Score an existing inference against the named metrics.

    Returns the outcome envelope: completed with a score per metric,
    skipped when no evaluator is configured for this environment, or
    failed when the evaluator was unreachable (the inference is left
    untouched). An unknown inference_id is a 404, as is a metric the evaluator
    does not define. Re-evaluating is safe: scores upsert on the metric key rather
    than duplicate.
    """
    outcome = await service.evaluate_inference_by_id(session, inference_id, payload.metrics)
    return EvaluationEnvelope.from_outcome(outcome)
