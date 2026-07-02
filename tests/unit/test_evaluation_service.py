"""Unit tests for EvaluationService branching (skip, fail-open) and request build."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy.orm import Session

from arc_model_lab.domain import EvaluationError, EvaluationStatus, Inference
from arc_model_lab.services import evaluation_service as module
from arc_model_lab.services.evaluation_service import EvaluationService


def _inference() -> Inference:
    return Inference(
        model_id=uuid4(),
        input_text="source text",
        prompt="rendered prompt",
        output_text="the summary",
        latency_ms=10,
    )


def test_evaluate_without_client_is_skipped() -> None:
    outcome = EvaluationService(None).evaluate_inference(MagicMock(spec=Session), _inference())

    assert outcome.status is EvaluationStatus.SKIPPED
    assert outcome.results == ()


def test_evaluate_fails_open_when_client_raises() -> None:
    client = MagicMock()
    client.evaluate.side_effect = EvaluationError("down")
    session = MagicMock(spec=Session)

    outcome = EvaluationService(client).evaluate_inference(session, _inference())

    assert outcome.status is EvaluationStatus.FAILED
    assert outcome.results == ()
    session.commit.assert_not_called()


def test_build_request_maps_inference_fields() -> None:
    inference = _inference()

    request = module._build_request(inference)

    assert request.task_type == "summarization"
    assert request.input_text == "source text"
    assert request.output_text == "the summary"
    assert request.prompt == "rendered prompt"
    assert request.metadata.inference_id == str(inference.id)
    assert request.metadata.model_id == str(inference.model_id)
