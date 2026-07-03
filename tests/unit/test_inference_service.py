"""Unit tests for InferenceService input guards."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from arc_model_lab.domain import (
    GenerationConfig,
    InputTooLargeError,
    Model,
    ModelInactiveError,
    ModelNotFoundError,
    ModelStatus,
    Provider,
)
from arc_model_lab.services import inference_service as inference_service_module
from arc_model_lab.services.inference_service import InferenceService, RunContext
from arc_model_lab.services.model_service import ModelService


def test_summarize_rejects_oversized_input(fake_model_service: ModelService) -> None:
    service = InferenceService(fake_model_service, "test-model")

    with pytest.raises(InputTooLargeError):
        service.summarize(MagicMock(spec=Session), "x" * 60_000)


def _deployed_model(status: ModelStatus) -> Model:
    return Model(
        name="deployed",
        provider=Provider.HUGGINGFACE,
        model_id="x/y",
        tokenizer_id="x/y",
        status=status,
    )


def test_summarize_resolves_the_deployed_model_and_raises_when_missing(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = MagicMock()
    repository.get_by_name.return_value = None
    monkeypatch.setattr(inference_service_module, "ModelRepository", lambda session: repository)

    service = InferenceService(fake_model_service, "deployed")

    with pytest.raises(ModelNotFoundError):
        service.summarize(MagicMock(spec=Session), "hello")
    repository.get_by_name.assert_called_once_with("deployed")


def test_summarize_raises_when_deployed_model_inactive(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = MagicMock()
    repository.get_by_name.return_value = _deployed_model(ModelStatus.INACTIVE)
    monkeypatch.setattr(inference_service_module, "ModelRepository", lambda session: repository)

    service = InferenceService(fake_model_service, "deployed")

    with pytest.raises(ModelInactiveError):
        service.summarize(MagicMock(spec=Session), "hello")


def test_summarize_with_explicit_model_skips_deployed_resolution(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = MagicMock()
    monkeypatch.setattr(inference_service_module, "ModelRepository", lambda session: repository)
    model = _deployed_model(ModelStatus.ACTIVE)
    context = RunContext(
        model=model,
        config=GenerationConfig(max_input_tokens=1024, max_new_tokens=256, num_beams=1),
        experiment_id=uuid4(),
    )

    service = InferenceService(fake_model_service, "deployed")
    inference = service.summarize(MagicMock(spec=Session), "hello", context)

    assert inference.model_id == model.id
    assert inference.experiment_id == context.experiment_id
    repository.get_by_name.assert_not_called()
