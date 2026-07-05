"""The experiments endpoints: create, fetch, run, and compare run configurations.

A run reuses the deployed-model inference response shape; the difference is that
the model and decoding come from the experiment, not the deployed default. All
persistence and orchestration live in :class:`ExperimentService`; this module is
a thin transport adapter.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_experiment_service, get_session
from arc_model_lab.api.schemas import InferenceResponse
from arc_model_lab.api.schemas.experiments import (
    ExperimentComparisonResponse,
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentResultsResponse,
    ExperimentRunRequest,
)
from arc_model_lab.services.experiment_service import ExperimentService

SessionDep = Annotated[Session, Depends(get_session)]
ServiceDep = Annotated[ExperimentService, Depends(get_experiment_service)]

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreateRequest, session: SessionDep, service: ServiceDep) -> ExperimentResponse:
    experiment = service.create(session, payload.to_domain())
    return ExperimentResponse.from_domain(experiment)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: UUID, session: SessionDep, service: ServiceDep) -> ExperimentResponse:
    return ExperimentResponse.from_domain(service.get(session, experiment_id))


@router.post("/{experiment_id}/run", response_model=InferenceResponse, status_code=status.HTTP_201_CREATED)
def run_experiment(
    experiment_id: UUID, payload: ExperimentRunRequest, session: SessionDep, service: ServiceDep
) -> InferenceResponse:
    result = service.run(session, experiment_id, payload.input_text, metrics=payload.metrics)
    return InferenceResponse.from_inference(result.inference, result.evaluation)


@router.get("/{experiment_id}/results", response_model=ExperimentResultsResponse)
def get_results(experiment_id: UUID, session: SessionDep, service: ServiceDep) -> ExperimentResultsResponse:
    return ExperimentResultsResponse.from_domain(service.results(session, experiment_id))


@router.get("/{experiment_id}/compare/{other_id}", response_model=ExperimentComparisonResponse)
def compare_experiments(
    experiment_id: UUID, other_id: UUID, session: SessionDep, service: ServiceDep
) -> ExperimentComparisonResponse:
    return ExperimentComparisonResponse.from_domain(service.compare(session, experiment_id, other_id))
