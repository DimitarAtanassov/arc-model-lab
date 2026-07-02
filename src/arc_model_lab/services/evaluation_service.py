"""Evaluation workflow: score one inference via arc-eval and persist the results.

Evaluation is deliberately separate from inference. It runs *after* the inference
row is already committed, in its own unit of work, so a slow or broken evaluator
can never corrupt inference storage. Online requests fail open (a transport or
schema failure yields a ``FAILED`` outcome, not a 5xx); when no client is wired
for the environment the outcome is ``SKIPPED``.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import EvaluationResultRepository
from arc_model_lab.domain import (
    EvaluationError,
    EvaluationOutcome,
    EvaluationResult,
    EvaluationStatus,
    Inference,
)
from arc_model_lab.services.arc_eval_client import (
    ArcEvalClient,
    EvalMetadata,
    EvalMetricResult,
    EvalRequest,
)

logger = logging.getLogger(__name__)

# arc-model-lab only performs summarization today, so the task type is a
# constant rather than configuration.
_TASK_TYPE = "summarization"


class EvaluationService:
    """Evaluates one inference via arc-eval and stores the resulting scores."""

    def __init__(self, client: ArcEvalClient | None) -> None:
        self._client = client

    def evaluate_inference(self, session: Session, inference: Inference) -> EvaluationOutcome:
        """Score ``inference`` and persist the results in a fresh transaction.

        Returns a ``SKIPPED`` outcome when evaluation is not configured, a
        ``FAILED`` outcome when the eval call fails (fail-open, nothing
        persisted), or a ``COMPLETED`` outcome with the stored results.
        """
        if self._client is None:
            return EvaluationOutcome(status=EvaluationStatus.SKIPPED)

        try:
            response = self._client.evaluate(_build_request(inference))
        except EvaluationError:
            logger.warning(
                "evaluation failed; failing open",
                extra={"inference_id": str(inference.id)},
                exc_info=True,
            )
            return EvaluationOutcome(status=EvaluationStatus.FAILED)

        results = [_to_result(inference, metric) for metric in response.results]
        persisted = EvaluationResultRepository(session).upsert_many(results)
        session.commit()
        return EvaluationOutcome(status=EvaluationStatus.COMPLETED, results=tuple(persisted))


def _build_request(inference: Inference) -> EvalRequest:
    return EvalRequest(
        task_type=_TASK_TYPE,
        input_text=inference.input_text,
        output_text=inference.output_text,
        prompt=inference.prompt,
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
