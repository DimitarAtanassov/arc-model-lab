"""Evaluation CLI: run, replay, backfill.

python -m arc_model_lab.cli.evaluations run --inference-id <uuid>
python -m arc_model_lab.cli.evaluations replay --limit 100
python -m arc_model_lab.cli.evaluations backfill --since 2026-01-01 --until 2026-02-01 --limit 500

``replay`` and ``backfill`` are fail-closed per item: a failed evaluation is
counted and the run continues, and the command exits non-zero if any item
failed. Evaluations are idempotent (upsert on the unique metric key), so a
command is safe to re-run.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from arc_model_lab.clients.arc_eval_client import EvalSettings, build_arc_eval_client
from arc_model_lab.config import get_settings
from arc_model_lab.db.base import create_engine_from_url, create_session_factory
from arc_model_lab.db.repositories import InferenceRepository
from arc_model_lab.domain import EvaluationOutcome, EvaluationStatus
from arc_model_lab.services.evaluation_service import EvaluationService

_DEFAULT_LIMIT = 100
# The summarization metric set; callers override it with --metrics.
_DEFAULT_METRICS = ("faithfulness", "answer_relevance")


def _session_factory() -> sessionmaker[Session]:
    return create_session_factory(create_engine_from_url(get_settings().database_url))


def _evaluation_service() -> EvaluationService:
    client = build_arc_eval_client(EvalSettings())
    if client is None:
        raise SystemExit("ARC_EVAL_SERVICE_URL is not set; cannot run evaluations.")
    return EvaluationService(client)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _print_outcome(inference_id: UUID, outcome: EvaluationOutcome) -> None:
    scores = ", ".join(f"{result.metric_name}={result.score:.3f}" for result in outcome.results) or "-"
    print(f"{inference_id}\t{outcome.status}\t{scores}")


def _run(inference_id: UUID, metrics: list[str]) -> None:
    service = _evaluation_service()
    with _session_factory()() as session:
        inference = InferenceRepository(session).get(inference_id)
        if inference is None:
            raise SystemExit(f"Inference not found: {inference_id}")
        outcome = service.evaluate_inference(session, inference, metrics)
    _print_outcome(inference_id, outcome)
    if outcome.status is EvaluationStatus.FAILED:
        raise SystemExit(1)


def _process_batch(
    *,
    created_after: datetime | None,
    created_before: datetime | None,
    limit: int,
    metrics: list[str],
) -> None:
    service = _evaluation_service()
    completed = 0
    failed = 0
    with _session_factory()() as session:
        pending = InferenceRepository(session).list_unevaluated(
            limit=limit, created_after=created_after, created_before=created_before
        )
        for inference in pending:
            outcome = service.evaluate_inference(session, inference, metrics)
            _print_outcome(inference.id, outcome)
            if outcome.status is EvaluationStatus.COMPLETED:
                completed += 1
            else:
                failed += 1
    print(f"processed={len(pending)} completed={completed} failed={failed}")
    if failed:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluation operations.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Evaluate a single inference by id")
    run_parser.add_argument("--inference-id", required=True, type=UUID)
    run_parser.add_argument("--metrics", nargs="+", default=list(_DEFAULT_METRICS))

    replay_parser = sub.add_parser("replay", help="Evaluate unevaluated inferences")
    replay_parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    replay_parser.add_argument("--metrics", nargs="+", default=list(_DEFAULT_METRICS))

    backfill_parser = sub.add_parser("backfill", help="Evaluate unevaluated inferences in a time range")
    backfill_parser.add_argument("--since", type=_parse_timestamp, default=None)
    backfill_parser.add_argument("--until", type=_parse_timestamp, default=None)
    backfill_parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    backfill_parser.add_argument("--metrics", nargs="+", default=list(_DEFAULT_METRICS))

    args = parser.parse_args(argv)

    if args.command == "run":
        _run(args.inference_id, args.metrics)
    elif args.command == "replay":
        _process_batch(created_after=None, created_before=None, limit=args.limit, metrics=args.metrics)
    elif args.command == "backfill":  # pragma: no branch - argparse restricts the command set
        _process_batch(created_after=args.since, created_before=args.until, limit=args.limit, metrics=args.metrics)


if __name__ == "__main__":
    main()
