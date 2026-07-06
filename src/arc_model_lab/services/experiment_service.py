"""Experiment application service: create, run, and compare run configurations.

An experiment run composes inference and evaluation directly: it runs the
experiment's model and generation config, persists the inference, then (when the
run names metrics) scores it via arc-eval and persists the scores, and finally
records the experiment-inference association. Inference itself carries no
experiment reference; comparison is plain SQL aggregation over that association
and the evaluation rows, not an in-memory join.

The experiment's model is resolved by name on create and only required to exist,
not to be active: an experiment deliberately targets a chosen model (possibly a
candidate), unlike the deployed-model default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import ExperimentRepository, ExperimentRunRepository, ModelRepository
from arc_model_lab.domain import (
    EvaluationOutcome,
    Experiment,
    ExperimentNotFoundError,
    ExperimentResults,
    ExperimentRun,
    GenerationConfig,
    Inference,
)
from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.inference_service import InferenceService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentView:
    """An experiment paired with its model's name, for the API response.

    The experiment stores a model id (the foreign key); the API speaks model
    names, so the service resolves the name here rather than leaking the id.
    """

    experiment: Experiment
    model_name: str


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    """One experiment run: the inference and, when scored, its evaluation."""

    inference: Inference
    evaluation: EvaluationOutcome | None = None


class ExperimentService:
    """Creates experiments and runs them through inference and evaluation."""

    def __init__(self, inference_service: InferenceService, evaluation_service: EvaluationService) -> None:
        self._inference = inference_service
        self._evaluation = evaluation_service

    def create(
        self,
        session: Session,
        *,
        name: str,
        model_name: str,
        generation_config: GenerationConfig,
        description: str | None = None,
    ) -> ExperimentView:
        """Create an experiment, resolving its model by name.

        The unique name constraint is the single guard: ExperimentRepository.add
        raises ExperimentNameConflictError on a duplicate, so a pre-check query is
        redundant (and would not close the concurrent-create race anyway). The
        model is only required to exist (ModelNotFoundError -> 404 otherwise).
        """
        model = ModelRepository(session).require_by_name(model_name)
        experiment = Experiment(
            name=name,
            model_id=model.id,
            generation_config=generation_config,
            description=description,
        )
        saved = ExperimentRepository(session).add(experiment)
        session.commit()
        return ExperimentView(experiment=saved, model_name=model.name)

    def get(self, session: Session, experiment_id: UUID) -> ExperimentView:
        """Return the experiment and its model name, or raise (404)."""
        experiment = self._require_experiment(session, experiment_id)
        model = ModelRepository(session).require_by_id(experiment.model_id)
        return ExperimentView(experiment=experiment, model_name=model.name)

    def list_recent(self, session: Session, limit: int) -> list[ExperimentView]:
        """Return recent experiments paired with their model names (bounded).

        Model names are resolved with a single catalog read and an in-memory map,
        not one query per experiment, so the list stays free of an N+1.
        """
        experiments = ExperimentRepository(session).list_recent(limit)
        names = {model.id: model.name for model in ModelRepository(session).list_all()}
        return [
            ExperimentView(experiment=experiment, model_name=names.get(experiment.model_id, "unknown"))
            for experiment in experiments
        ]

    def run(
        self,
        session: Session,
        experiment_id: UUID,
        input_text: str,
        *,
        metrics: list[str] | None = None,
    ) -> ExperimentRunResult:
        """Run the experiment's config once, then score it when metrics are named.

        The pipeline is infer -> persist inference -> (when metrics are named)
        evaluate via arc-eval -> persist scores -> record the experiment-inference
        association last. Raises ExperimentNotFoundError for an unknown experiment
        and ModelNotFoundError if its model was removed.
        """
        experiment = self._require_experiment(session, experiment_id)
        model = ModelRepository(session).require_by_id(experiment.model_id)
        inference = self._inference.run_for_experiment(
            session,
            model=model,
            input_text=input_text,
            config=experiment.generation_config,
        )
        outcome = self._evaluation.evaluate_inference(session, inference, metrics) if metrics else None
        ExperimentRunRepository(session).add(ExperimentRun(experiment_id=experiment.id, inference_id=inference.id))
        session.commit()
        logger.info(
            "experiment run complete",
            extra={
                "experiment_id": str(experiment.id),
                "model_name": model.name,
                "inference_id": str(inference.id),
                "latency_ms": inference.latency_ms,
                "metric_count": len(metrics) if metrics else 0,
                "evaluation_status": outcome.status.value if outcome is not None else "not_requested",
            },
        )
        return ExperimentRunResult(inference=inference, evaluation=outcome)

    def results(self, session: Session, experiment_id: UUID) -> ExperimentResults:
        self._require_experiment(session, experiment_id)
        aggregates = ExperimentRepository(session).aggregate_scores(experiment_id)
        return ExperimentResults(experiment_id=experiment_id, metrics=aggregates)

    def compare(
        self,
        session: Session,
        experiment_id_a: UUID,
        experiment_id_b: UUID,
    ) -> list[ExperimentResults]:
        """Aggregate scores for both experiments, in the order given."""
        return [
            self.results(session, experiment_id_a),
            self.results(session, experiment_id_b),
        ]

    def _require_experiment(self, session: Session, experiment_id: UUID) -> Experiment:
        experiment = ExperimentRepository(session).get(experiment_id)
        if experiment is None:
            raise ExperimentNotFoundError(f"Experiment not found: {experiment_id}")
        return experiment
