from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.clients.arc_eval_client import ArcEvalClient
from arc_model_lab.domain import (
    EvaluationError,
    EvaluationStatus,
    Inference,
    InferenceNotFoundError,
    UnknownMetricError,
)
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


async def test_evaluate_without_client_is_skipped() -> None:
    outcome = await EvaluationService(None).evaluate_inference(
        MagicMock(spec=AsyncSession), _inference(), ["faithfulness"]
    )

    assert outcome.status is EvaluationStatus.SKIPPED
    assert outcome.results == ()


async def test_evaluate_fails_open_when_client_raises() -> None:
    client = MagicMock(spec=ArcEvalClient)
    client.evaluate = AsyncMock(side_effect=EvaluationError("down"))
    session = MagicMock(spec=AsyncSession)

    outcome = await EvaluationService(client).evaluate_inference(session, _inference(), ["faithfulness"])

    assert outcome.status is EvaluationStatus.FAILED
    assert outcome.results == ()
    session.commit.assert_not_called()


async def test_unknown_metric_propagates_and_does_not_fail_open() -> None:
    client = MagicMock(spec=ArcEvalClient)
    client.evaluate = AsyncMock(side_effect=UnknownMetricError("unknown metric 'nope'"))
    session = MagicMock(spec=AsyncSession)

    # An unknown metric is a caller error, not an infra failure: it must surface,
    # not be swallowed into a FAILED outcome.
    with pytest.raises(UnknownMetricError):
        await EvaluationService(client).evaluate_inference(session, _inference(), ["nope"])

    session.commit.assert_not_called()


def test_build_request_maps_inference_fields() -> None:
    inference = _inference()

    request = module._build_request(inference, ["faithfulness"])

    assert request.input_text == "source text"
    assert request.output_text == "the summary"
    assert request.prompt == "rendered prompt"
    assert request.metrics == ["faithfulness"]
    assert request.metadata.inference_id == str(inference.id)
    assert request.metadata.model_id == str(inference.model_id)


async def test_evaluate_by_id_raises_when_inference_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = MagicMock()
    repository.get = AsyncMock(return_value=None)
    monkeypatch.setattr(module, "InferenceRepository", lambda session: repository)

    with pytest.raises(InferenceNotFoundError):
        await EvaluationService(MagicMock()).evaluate_inference_by_id(
            MagicMock(spec=AsyncSession), uuid4(), ["faithfulness"]
        )


async def test_evaluate_by_id_loads_then_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    inference = _inference()
    repository = MagicMock()
    repository.get = AsyncMock(return_value=inference)
    monkeypatch.setattr(module, "InferenceRepository", lambda session: repository)

    # No client, so evaluate_inference short-circuits to SKIPPED. That proves the
    # id was resolved to the loaded inference and handed to evaluate_inference.
    outcome = await EvaluationService(None).evaluate_inference_by_id(
        MagicMock(spec=AsyncSession), inference.id, ["faithfulness"]
    )

    assert outcome.status is EvaluationStatus.SKIPPED
    repository.get.assert_called_once_with(inference.id)
