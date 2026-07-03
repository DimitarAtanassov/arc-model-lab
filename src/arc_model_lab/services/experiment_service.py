"""Experiment application service: create, run, and compare run configurations.

An experiment run composes the existing :class:`InferenceWorkflow` with the
experiment's model and generation config, so inference and evaluation keep a
single orchestration path. Comparison is plain SQL aggregation over the tagged
inference and evaluation rows, not an in-memory join.

The experiment's model is resolved by id and only required to exist, not to be
active: an experiment deliberately targets a chosen model (possibly a candidate),
unlike the deployed-model ``/inference`` path.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import ExperimentRepository, ModelRepository
from arc_model_lab.domain import (
    Experiment,
    ExperimentMetricAggregate,
    ExperimentNameConflictError,
    ExperimentNotFoundError,
    Model,
    ModelNotFoundError,
)
from arc_model_lab.services.inference_service import RunContext
from arc_model_lab.services.inference_workflow import InferenceResult, InferenceWorkflow


class ExperimentService:
    """Creates experiments and runs them through the shared inference workflow."""

    def __init__(self, workflow: InferenceWorkflow) -> None:
        self._workflow = workflow

    def create(self, session: Session, experiment: Experiment) -> Experiment:
        self._require_model(session, experiment.model_id)
        repository = ExperimentRepository(session)
        if repository.get_by_name(experiment.name) is not None:
            raise ExperimentNameConflictError(f"Experiment name already exists: {experiment.name}")
        saved = repository.add(experiment)
        session.commit()
        return saved

    def get(self, session: Session, experiment_id: UUID) -> Experiment | None:
        return ExperimentRepository(session).get(experiment_id)

    def run(
        self,
        session: Session,
        experiment_id: UUID,
        input_text: str,
        *,
        metrics: list[str] | None = None,
    ) -> InferenceResult:
        """Run the experiment's config once and return the inference (and scores).

        Raises :class:`ExperimentNotFoundError` for an unknown experiment and
        :class:`ModelNotFoundError` if its model has since been removed.
        """
        experiment = ExperimentRepository(session).get(experiment_id)
        if experiment is None:
            raise ExperimentNotFoundError(f"Experiment not found: {experiment_id}")
        model = self._require_model(session, experiment.model_id)
        return self._workflow.run(
            session,
            input_text=input_text,
            context=RunContext(
                model=model,
                config=experiment.generation_config,
                experiment_id=experiment.id,
            ),
            metrics=metrics,
        )

    def results(self, session: Session, experiment_id: UUID) -> list[ExperimentMetricAggregate]:
        self._require_experiment(session, experiment_id)
        return ExperimentRepository(session).aggregate_scores(experiment_id)

    def compare(
        self,
        session: Session,
        experiment_id_a: UUID,
        experiment_id_b: UUID,
    ) -> dict[UUID, list[ExperimentMetricAggregate]]:
        self._require_experiment(session, experiment_id_a)
        self._require_experiment(session, experiment_id_b)
        repository = ExperimentRepository(session)
        return {
            experiment_id_a: repository.aggregate_scores(experiment_id_a),
            experiment_id_b: repository.aggregate_scores(experiment_id_b),
        }

    def _require_model(self, session: Session, model_id: UUID) -> Model:
        model = ModelRepository(session).get_by_id(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {model_id}")
        return model

    def _require_experiment(self, session: Session, experiment_id: UUID) -> Experiment:
        experiment = ExperimentRepository(session).get(experiment_id)
        if experiment is None:
            raise ExperimentNotFoundError(f"Experiment not found: {experiment_id}")
        return experiment
