"""Experiment CLI: create, run, compare.

python -m arc_model_lab.cli.experiments create --name exp-a --model-name qwen2.5-1.5b-instruct
python -m arc_model_lab.cli.experiments run --experiment-id <uuid> --input-text "..."
python -m arc_model_lab.cli.experiments compare --experiment-id <uuid> --other-id <uuid>

``run`` loads the model and generates (like ``model smoke``); ``create`` and
``compare`` only touch the database.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import TypeVar
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from arc_model_lab.clients.arc_eval_client import EvalSettings, build_arc_eval_client
from arc_model_lab.config import get_settings
from arc_model_lab.db.base import create_engine_from_url, create_session_factory
from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import (
    DomainError,
    Experiment,
    ExperimentMetricAggregate,
    GenerationConfig,
)
from arc_model_lab.domain.generation import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
)
from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.experiment_service import ExperimentService
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.inference_workflow import InferenceWorkflow
from arc_model_lab.services.model_service import ModelService


def _session_factory() -> sessionmaker[Session]:
    return create_session_factory(create_engine_from_url(get_settings().database_url))


_T = TypeVar("_T")


def _in_session(operation: Callable[[Session], _T]) -> _T:
    """Run a session-scoped operation, surfacing domain errors as clean CLI exits.

    Keeps the subcommands from each re-deriving how a bad id or a conflict should
    look to a CLI user: any ``DomainError`` becomes a one-line ``SystemExit`` rather
    than a traceback.
    """
    try:
        with _session_factory()() as session:
            return operation(session)
    except DomainError as exc:
        raise SystemExit(str(exc)) from exc


def _experiment_service() -> ExperimentService:
    settings = get_settings()
    inference_service = InferenceService(ModelService(settings), settings.model_name)
    evaluation_service = EvaluationService(build_arc_eval_client(EvalSettings()))
    return ExperimentService(InferenceWorkflow(inference_service, evaluation_service))


def _print_aggregates(experiment_id: UUID, aggregates: list[ExperimentMetricAggregate]) -> None:
    if not aggregates:
        print(f"{experiment_id}\t(no scores)")
        return
    for aggregate in aggregates:
        print(f"{experiment_id}\t{aggregate.metric_name}\t{aggregate.average_score:.3f}\t{aggregate.evaluated_count}")


def _create(name: str, model_name: str, config: GenerationConfig) -> None:
    service = _experiment_service()

    def operation(session: Session) -> Experiment:
        model = ModelRepository(session).get_by_name(model_name)
        if model is None:
            raise SystemExit(f"Model not found: {model_name}")
        return service.create(session, Experiment(name=name, model_id=model.id, generation_config=config))

    experiment = _in_session(operation)
    print(f"{experiment.id}\t{experiment.name}\t{model_name}")


def _run(experiment_id: UUID, input_text: str, metrics: list[str] | None) -> None:
    service = _experiment_service()
    result = _in_session(lambda session: service.run(session, experiment_id, input_text, metrics=metrics))
    scores = "-"
    if result.evaluation is not None:
        scores = ", ".join(f"{r.metric_name}={r.score:.3f}" for r in result.evaluation.results) or "-"
    print(f"{result.inference.id}\t{result.inference.output_text}\t{scores}")


def _compare(experiment_id: UUID, other_id: UUID) -> None:
    service = _experiment_service()
    comparison = _in_session(lambda session: service.compare(session, experiment_id, other_id))
    for result in comparison:
        _print_aggregates(result.experiment_id, result.metrics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment operations.")
    sub = parser.add_subparsers(dest="command", required=True)

    create_parser = sub.add_parser("create", help="Create an experiment")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--model-name", required=True)
    create_parser.add_argument("--num-beams", type=int, default=DEFAULT_NUM_BEAMS)
    create_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    create_parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)

    run_parser = sub.add_parser("run", help="Run an experiment by id")
    run_parser.add_argument("--experiment-id", required=True, type=UUID)
    run_parser.add_argument("--input-text", required=True)
    run_parser.add_argument("--metrics", nargs="*", default=None)

    compare_parser = sub.add_parser("compare", help="Compare two experiments by id")
    compare_parser.add_argument("--experiment-id", required=True, type=UUID)
    compare_parser.add_argument("--other-id", required=True, type=UUID)

    args = parser.parse_args(argv)

    if args.command == "create":
        config = GenerationConfig(
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        _create(args.name, args.model_name, config)
    elif args.command == "run":
        _run(args.experiment_id, args.input_text, args.metrics)
    elif args.command == "compare":  # pragma: no branch - argparse restricts the command set
        _compare(args.experiment_id, args.other_id)


if __name__ == "__main__":
    main()
