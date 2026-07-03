"""The experiments endpoints: create, fetch, run, and compare run configurations.

A run reuses the deployed-model inference response shape; the difference is that
the model and decoding come from the experiment, not the deployed default. All
persistence and orchestration live in :class:`ExperimentService`; this module is
a thin transport adapter.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_experiment_service, get_session
from arc_model_lab.api.schemas import InferenceResponse
from arc_model_lab.api.schemas.evaluations import EvaluationEnvelope
from arc_model_lab.api.schemas.experiments import (
    ExperimentComparisonResponse,
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentResultsResponse,
    ExperimentRunRequest,
    MetricAggregateOut,
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
    experiment = service.get(session, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return ExperimentResponse.from_domain(experiment)


@router.post("/{experiment_id}/run", response_model=InferenceResponse, status_code=status.HTTP_201_CREATED)
def run_experiment(
    experiment_id: UUID, payload: ExperimentRunRequest, session: SessionDep, service: ServiceDep
) -> InferenceResponse:
    result = service.run(session, experiment_id, payload.input_text, metrics=payload.metrics)
    response = InferenceResponse.model_validate(result.inference)
    if result.evaluation is not None:
        response.evaluation = EvaluationEnvelope.from_outcome(result.evaluation)
    return response


@router.get("/{experiment_id}/results", response_model=ExperimentResultsResponse)
def get_results(experiment_id: UUID, session: SessionDep, service: ServiceDep) -> ExperimentResultsResponse:
    aggregates = service.results(session, experiment_id)
    return ExperimentResultsResponse(
        experiment_id=experiment_id,
        metrics=[MetricAggregateOut.from_domain(aggregate) for aggregate in aggregates],
    )


@router.get("/{experiment_id}/compare/{other_id}", response_model=ExperimentComparisonResponse)
def compare_experiments(
    experiment_id: UUID, other_id: UUID, session: SessionDep, service: ServiceDep
) -> ExperimentComparisonResponse:
    comparison = service.compare(session, experiment_id, other_id)
    return ExperimentComparisonResponse(
        experiments=[
            ExperimentResultsResponse(
                experiment_id=identifier,
                metrics=[MetricAggregateOut.from_domain(aggregate) for aggregate in aggregates],
            )
            for identifier, aggregates in comparison.items()
        ]
    )
