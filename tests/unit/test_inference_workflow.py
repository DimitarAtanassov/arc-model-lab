"""Unit tests for InferenceWorkflow: the generate-then-optionally-evaluate seam."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy.orm import Session

from arc_model_lab.domain import EvaluationOutcome, EvaluationStatus, Inference
from arc_model_lab.services.inference_workflow import InferenceWorkflow


def _inference() -> Inference:
    return Inference(
        model_id=uuid4(),
        input_text="source",
        prompt="rendered",
        output_text="summary",
        latency_ms=1,
    )


def test_run_without_metrics_skips_evaluation() -> None:
    inference = _inference()
    inference_service = MagicMock()
    inference_service.summarize.return_value = inference
    evaluation_service = MagicMock()
    session = MagicMock(spec=Session)

    result = InferenceWorkflow(inference_service, evaluation_service).run(
        session, input_text="source", model_name=None, metrics=None
    )

    assert result.inference is inference
    assert result.evaluation is None
    inference_service.summarize.assert_called_once_with(session, "source", None)
    evaluation_service.evaluate_inference.assert_not_called()


def test_run_with_metrics_evaluates_and_threads_task_type() -> None:
    inference = _inference()
    outcome = EvaluationOutcome(status=EvaluationStatus.COMPLETED)
    inference_service = MagicMock()
    inference_service.summarize.return_value = inference
    evaluation_service = MagicMock()
    evaluation_service.evaluate_inference.return_value = outcome
    session = MagicMock(spec=Session)

    result = InferenceWorkflow(inference_service, evaluation_service).run(
        session,
        input_text="source",
        metrics=["faithfulness"],
        task_type="question_answering",
    )

    assert result.inference is inference
    assert result.evaluation is outcome
    evaluation_service.evaluate_inference.assert_called_once_with(
        session, inference, ["faithfulness"], task_type="question_answering"
    )
