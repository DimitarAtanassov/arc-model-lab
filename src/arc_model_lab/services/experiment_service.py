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
    ExperimentNotFoundError,
    ExperimentResults,
    Model,
    ModelNotFoundError,
)
from arc_model_lab.services.inference_workflow import InferenceResult, InferenceWorkflow
from arc_model_lab.services.run_context import RunContext


class ExperimentService:
    """Creates experiments and runs them through the shared inference workflow."""

    def __init__(self, workflow: InferenceWorkflow) -> None:
        self._workflow = workflow

    def create(self, session: Session, experiment: Experiment) -> Experiment:
        # The unique name constraint is the single guard: ExperimentRepository.add
        # raises ExperimentNameConflictError on a duplicate, so a pre-check query is
        # redundant (and would not close the concurrent-create race anyway).
        self._require_model(session, experiment.model_id)
        saved = ExperimentRepository(session).add(experiment)
        session.commit()
        return saved

    def get(self, session: Session, experiment_id: UUID) -> Experiment:
        """Return the experiment or raise :class:`ExperimentNotFoundError` (404).

        Existence checks live here so routes and other methods share one lookup and
        stay free of transport-level status decisions.
        """
        experiment = ExperimentRepository(session).get(experiment_id)
        if experiment is None:
            raise ExperimentNotFoundError(f"Experiment not found: {experiment_id}")
        return experiment

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
        experiment = self.get(session, experiment_id)
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

    def results(self, session: Session, experiment_id: UUID) -> ExperimentResults:
        self.get(session, experiment_id)
        aggregates = ExperimentRepository(session).aggregate_scores(experiment_id)
        return ExperimentResults(experiment_id=experiment_id, metrics=aggregates)

    def compare(
        self,
        session: Session,
        experiment_id_a: UUID,
        experiment_id_b: UUID,
    ) -> list[ExperimentResults]:
        """Aggregate scores for both experiments, preserving order and identity.

        Returns a list, not a map: comparing an experiment with itself yields two
        entries rather than silently collapsing to one.
        """
        return [
            self.results(session, experiment_id_a),
            self.results(session, experiment_id_b),
        ]

    def _require_model(self, session: Session, model_id: UUID) -> Model:
        model = ModelRepository(session).get_by_id(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {model_id}")
        return model
