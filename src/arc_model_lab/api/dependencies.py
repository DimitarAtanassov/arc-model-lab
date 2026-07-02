"""FastAPI dependency wiring. Shared singletons are read from application state."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session, sessionmaker

from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.inference_workflow import InferenceWorkflow


def get_session(request: Request) -> Iterator[Session]:
    """Yield a request-scoped session; the ``with`` block rolls back on error.

    The service layer owns the commit, so a row is guaranteed to be persisted
    before any success response is returned.
    """
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as session:
        yield session


def get_inference_service(request: Request) -> InferenceService:
    service: InferenceService = request.app.state.inference_service
    return service


def get_evaluation_service(request: Request) -> EvaluationService:
    service: EvaluationService = request.app.state.evaluation_service
    return service


def get_inference_workflow(
    inference_service: Annotated[InferenceService, Depends(get_inference_service)],
    evaluation_service: Annotated[EvaluationService, Depends(get_evaluation_service)],
) -> InferenceWorkflow:
    """Compose the inference/evaluation use case from the shared services."""
    return InferenceWorkflow(inference_service, evaluation_service)
