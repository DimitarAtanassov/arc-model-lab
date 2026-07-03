"""Application service: run an inference and, when asked, evaluate it.

One entry point for the "generate, then optionally score" use case. The HTTP
route is a thin adapter over it, and the coming experiments route and the CLI can
reuse the same sequence instead of re-implementing it in the transport layer.

The single call to evaluation in :meth:`InferenceWorkflow.run` is the seam: it is
synchronous today; moving evaluation to an asynchronous path (enqueue for a
worker) replaces only that call, leaving the route and the CLI untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from arc_model_lab.domain import EvaluationOutcome, Inference
from arc_model_lab.services.evaluation_service import (
    DEFAULT_TASK_TYPE,
    EvaluationService,
)
from arc_model_lab.services.inference_service import InferenceService


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """An inference and, when metrics were requested, its evaluation outcome."""

    inference: Inference
    evaluation: EvaluationOutcome | None = None


class InferenceWorkflow:
    """Runs inference, then evaluation when the caller requests metrics."""

    def __init__(
        self,
        inference_service: InferenceService,
        evaluation_service: EvaluationService,
    ) -> None:
        self._inference = inference_service
        self._evaluation = evaluation_service

    def run(
        self,
        session: Session,
        *,
        input_text: str,
        metrics: list[str] | None = None,
        task_type: str = DEFAULT_TASK_TYPE,
    ) -> InferenceResult:
        """Generate an inference and evaluate it only when metrics are requested."""
        inference = self._inference.summarize(session, input_text)
        if not metrics:
            return InferenceResult(inference=inference)
        outcome = self._evaluation.evaluate_inference(session, inference, metrics, task_type=task_type)
        return InferenceResult(inference=inference, evaluation=outcome)
