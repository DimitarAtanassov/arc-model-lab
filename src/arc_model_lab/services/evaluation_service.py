"""Evaluation workflow: score one inference via arc-eval and persist the results.

Evaluation is deliberately separate from inference. It runs *after* the inference
row is already committed, in its own unit of work, so a slow or broken evaluator
can never corrupt inference storage. Online requests fail open (a transport or
schema failure yields a ``FAILED`` outcome, not a 5xx); an unknown metric is the
exception, a caller error that propagates as ``UnknownMetricError`` (404). When no
client is wired for the environment the outcome is ``SKIPPED``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.clients.arc_eval_client import (
    ArcEvalClient,
    EvalMetadata,
    EvalMetricResult,
    EvalRequest,
)
from arc_model_lab.db.repositories import EvaluationResultRepository, InferenceRepository
from arc_model_lab.domain import (
    EvaluationError,
    EvaluationOutcome,
    EvaluationResult,
    EvaluationStatus,
    Inference,
    InferenceNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoredInference:
    """An inference scored by arc-eval, not yet persisted.

    The seam between the network-bound scoring step and the DB-bound persistence
    step: a batch fans scoring out concurrently, then persists serially on one
    session. ``results`` is empty unless ``status`` is ``COMPLETED``.
    """

    inference: Inference
    results: tuple[EvaluationResult, ...]
    status: EvaluationStatus


class EvaluationService:
    """Evaluates one inference via arc-eval and stores the resulting scores."""

    def __init__(self, client: ArcEvalClient | None) -> None:
        self._client = client

    async def evaluate_inference(
        self,
        session: AsyncSession,
        inference: Inference,
        metrics: list[str],
    ) -> EvaluationOutcome:
        """Score ``inference`` against ``metrics`` and persist the results.

        Returns a ``SKIPPED`` outcome when evaluation is not configured, a
        ``FAILED`` outcome when the eval call fails (fail-open, nothing
        persisted), or a ``COMPLETED`` outcome with the stored results. An unknown
        metric raises :class:`~arc_model_lab.domain.UnknownMetricError` (a caller
        error) instead of failing open, so the endpoint can surface it as 404.
        """
        return await self.persist(session, await self.score(inference, metrics))

    async def score(self, inference: Inference, metrics: list[str]) -> ScoredInference:
        """Score one inference via arc-eval without persisting (network only).

        Isolating the network step from the DB write lets a batch fan scoring out
        concurrently. Returns a ``SKIPPED`` scoring when no client is configured
        and a ``FAILED`` scoring (fail-open) when the eval call errors; an unknown
        metric still raises :class:`~arc_model_lab.domain.UnknownMetricError`.
        """
        if self._client is None:
            return ScoredInference(inference=inference, results=(), status=EvaluationStatus.SKIPPED)

        try:
            response = await self._client.evaluate(_build_request(inference, metrics))
        except EvaluationError:
            logger.warning(
                "evaluation failed; failing open",
                extra={"inference_id": str(inference.id)},
                exc_info=True,
            )
            return ScoredInference(inference=inference, results=(), status=EvaluationStatus.FAILED)

        results = tuple(_to_result(inference, metric) for metric in response.results)
        return ScoredInference(inference=inference, results=results, status=EvaluationStatus.COMPLETED)

    async def persist(self, session: AsyncSession, scored: ScoredInference) -> EvaluationOutcome:
        """Persist a scored inference's results, or pass a non-completed status through.

        A ``SKIPPED`` or ``FAILED`` scoring writes nothing. A ``COMPLETED`` scoring
        upserts its metric rows (idempotent on the unique key) and commits.
        """
        if scored.status is not EvaluationStatus.COMPLETED:
            return EvaluationOutcome(status=scored.status)
        persisted = await EvaluationResultRepository(session).upsert_many(list(scored.results))
        await session.commit()
        return EvaluationOutcome(status=EvaluationStatus.COMPLETED, results=tuple(persisted))

    async def evaluate_inference_by_id(
        self, session: AsyncSession, inference_id: UUID, metrics: list[str]
    ) -> EvaluationOutcome:
        """Load the inference with ``inference_id`` and score it against ``metrics``.

        The standalone counterpart to an experiment run: it scores an inference
        that already exists, with no experiment involved. Raises
        :class:`InferenceNotFoundError` (404) when no inference has that id, then
        delegates to :meth:`evaluate_inference`, so the skip, fail-open, and
        unknown-metric behavior is identical.
        """
        inference = await InferenceRepository(session).get(inference_id)
        if inference is None:
            raise InferenceNotFoundError(f"Inference not found: {inference_id}")
        return await self.evaluate_inference(session, inference, metrics)


def _build_request(inference: Inference, metrics: list[str]) -> EvalRequest:
    return EvalRequest(
        input_text=inference.input_text,
        output_text=inference.output_text,
        prompt=inference.prompt,
        metrics=metrics,
        metadata=EvalMetadata(
            inference_id=str(inference.id),
            model_id=str(inference.model_id),
        ),
    )


def _to_result(inference: Inference, metric: EvalMetricResult) -> EvaluationResult:
    return EvaluationResult(
        inference_id=inference.id,
        metric_name=metric.metric_name,
        score=metric.score,
        reasoning=metric.reasoning,
        evaluator_name=metric.evaluator_name,
        evaluator_version=metric.evaluator_version,
    )
