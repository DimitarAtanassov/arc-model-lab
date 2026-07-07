"""Unit tests for InferenceService: model resolution, guards, and persistence."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from arc_model_lab.config import Settings
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
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ChatMessage, GenerationResult, ModelService


def _config() -> GenerationConfig:
    return GenerationConfig(temperature=0.0, max_output_tokens=64)


def _model(name: str = "m") -> Model:
    return Model(name=name, provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y")


def _patch_model_repo(monkeypatch: pytest.MonkeyPatch, model: Model | None) -> MagicMock:
    repository = MagicMock()
    # The service resolves through require_by_name, which returns the model or
    # raises ModelNotFoundError; mirror that contract on the mocked seam.
    if model is None:
        repository.require_by_name.side_effect = ModelNotFoundError("Model not found")
    else:
        repository.require_by_name.return_value = model
    monkeypatch.setattr(inference_service_module, "ModelRepository", lambda session: repository)
    return repository


def _patch_inference_repo(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    added: list[object] = []
    repository = MagicMock()
    repository.add.side_effect = lambda inference: (added.append(inference), inference)[1]
    monkeypatch.setattr(inference_service_module, "InferenceRepository", lambda session: repository)
    return added


class _CapturingModelService(ModelService):
    """Records the resolved GenerationConfig handed to generate (loads no weights)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.configs: list[GenerationConfig | None] = []

    def generate(
        self, model: Model, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> GenerationResult:
        self.configs.append(config)
        return GenerationResult(prompt="p", output_text="o", prompt_tokens=1, completion_tokens=1, latency_ms=1)


def test_summarize_raises_when_model_missing(fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _patch_model_repo(monkeypatch, None)
    service = InferenceService(fake_model_service)

    with pytest.raises(ModelNotFoundError):
        service.summarize(MagicMock(spec=Session), model_name="missing", input_text="hello")
    repository.require_by_name.assert_called_once_with("missing")


def test_summarize_raises_when_model_inactive(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    inactive = Model(
        name="m", provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y", status=ModelStatus.INACTIVE
    )
    _patch_model_repo(monkeypatch, inactive)

    with pytest.raises(ModelInactiveError):
        InferenceService(fake_model_service).summarize(MagicMock(spec=Session), model_name="m", input_text="hello")


def test_summarize_rejects_oversized_input(fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())
    service = InferenceService(fake_model_service)

    with pytest.raises(InputTooLargeError):
        service.summarize(MagicMock(spec=Session), model_name="m", input_text="x" * 60_000)


def test_summarize_persists_inference(fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    _patch_model_repo(monkeypatch, model)
    added = _patch_inference_repo(monkeypatch)

    inference = InferenceService(fake_model_service).summarize(
        MagicMock(spec=Session), model_name="m", input_text="hello"
    )

    assert inference.model_id == model.id
    assert added
    assert added[0] is inference


def test_run_for_experiment_uses_the_given_model(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    added = _patch_inference_repo(monkeypatch)
    model = _model()

    inference = InferenceService(fake_model_service).run_for_experiment(
        MagicMock(spec=Session),
        model=model,
        input_text="hello",
        config=_config(),
    )

    assert inference.model_id == model.id
    # The experiment path resolves nothing by name; the model is passed in and the
    # inference carries no experiment reference.
    assert added
    assert added[0] is inference


def test_summarize_uses_server_default_config_when_temperature_omitted(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    server = settings.model_copy(update={"temperature": 0.5, "max_output_tokens": 512})
    model_service = _CapturingModelService(server)

    InferenceService(model_service).summarize(MagicMock(spec=Session), model_name="m", input_text="hello")

    # No caller temperature: the whole config comes from the server settings.
    assert model_service.configs == [GenerationConfig(temperature=0.5, max_output_tokens=512)]


def test_summarize_applies_caller_temperature_over_server_default(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    server = settings.model_copy(update={"temperature": 0.5, "max_output_tokens": 512})
    model_service = _CapturingModelService(server)

    InferenceService(model_service).summarize(
        MagicMock(spec=Session), model_name="m", input_text="hello", temperature=0.9
    )

    # Caller temperature wins; output length stays the server default.
    assert model_service.configs == [GenerationConfig(temperature=0.9, max_output_tokens=512)]
