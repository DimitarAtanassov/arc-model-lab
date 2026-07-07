"""The experiments endpoints: create, fetch, run, and compare run configurations.

A run infers under the experiment's model and decoding config, stores the
inference, then scores it via arc-eval when the run names metrics. All
persistence and orchestration live in :class:`ExperimentService`; this module is
a thin transport adapter.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_experiment_service, get_session
from arc_model_lab.api.schemas.experiments import (
    ExperimentComparisonResponse,
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentResultsResponse,
    ExperimentRunRequest,
    ExperimentRunResponse,
)
from arc_model_lab.services.experiment_service import ExperimentService

SessionDep = Annotated[Session, Depends(get_session)]
ServiceDep = Annotated[ExperimentService, Depends(get_experiment_service)]

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    session: SessionDep,
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=_MAX_LIMIT)] = _DEFAULT_LIMIT,
) -> list[ExperimentResponse]:
    """Return recent experiments, newest first (bounded page size)."""
    return [
        ExperimentResponse.from_domain(view.experiment, view.model_name) for view in service.list_recent(session, limit)
    ]


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreateRequest, session: SessionDep, service: ServiceDep) -> ExperimentResponse:
    view = service.create(
        session,
        name=payload.name,
        model_name=payload.model_name,
        generation_config=payload.generation_config.to_domain(),
        description=payload.description,
    )
    return ExperimentResponse.from_domain(view.experiment, view.model_name)


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: UUID, session: SessionDep, service: ServiceDep) -> ExperimentResponse:
    view = service.get(session, experiment_id)
    return ExperimentResponse.from_domain(view.experiment, view.model_name)


@router.post("/{experiment_id}/run", response_model=ExperimentRunResponse, status_code=status.HTTP_201_CREATED)
def run_experiment(
    experiment_id: UUID, payload: ExperimentRunRequest, session: SessionDep, service: ServiceDep
) -> ExperimentRunResponse:
    result = service.run(session, experiment_id, payload.input_text, metrics=payload.metrics)
    return ExperimentRunResponse.from_run(experiment_id, result.inference, result.evaluation)


@router.get("/{experiment_id}/results", response_model=ExperimentResultsResponse)
def get_results(experiment_id: UUID, session: SessionDep, service: ServiceDep) -> ExperimentResultsResponse:
    return ExperimentResultsResponse.from_domain(service.results(session, experiment_id))


@router.get("/{experiment_id}/compare/{other_id}", response_model=ExperimentComparisonResponse)
def compare_experiments(
    experiment_id: UUID, other_id: UUID, session: SessionDep, service: ServiceDep
) -> ExperimentComparisonResponse:
    return ExperimentComparisonResponse.from_domain(service.compare(session, experiment_id, other_id))
