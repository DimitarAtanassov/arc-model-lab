from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.config import Settings
from arc_model_lab.domain import (
    GenerationConfig,
    GenerationPreset,
    InputTooLargeError,
    Model,
    ModelInactiveError,
    ModelNotFoundError,
    ModelStatus,
    PresetNotFoundError,
    PresetStatus,
    Provider,
)
from arc_model_lab.services import inference_service as inference_service_module
from arc_model_lab.services import preset_service as preset_service_module
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ChatMessage, GenerationResult, ModelService
from arc_model_lab.services.preset_service import PresetService

_CAP = Settings().max_output_tokens_cap


def _model(name: str = "m") -> Model:
    return Model(name=name, provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y")


def _service(model_service: ModelService) -> InferenceService:
    return InferenceService(model_service, PresetService(_CAP), _CAP)


def _patch_preset_repo(monkeypatch: pytest.MonkeyPatch, preset: GenerationPreset | None) -> MagicMock:
    """Mock the preset repo seam PresetService.get resolves through."""
    repository = MagicMock()
    repository.get = AsyncMock(return_value=preset)
    monkeypatch.setattr(preset_service_module, "PresetRepository", lambda session: repository)
    return repository


def _patch_model_repo(monkeypatch: pytest.MonkeyPatch, model: Model | None) -> MagicMock:
    repository = MagicMock()
    # The service resolves through require_by_name, which returns the model or
    # raises ModelNotFoundError; mirror that contract on the mocked (async) seam.
    if model is None:
        repository.require_by_name = AsyncMock(side_effect=ModelNotFoundError("Model not found"))
    else:
        repository.require_by_name = AsyncMock(return_value=model)
    monkeypatch.setattr(inference_service_module, "ModelRepository", lambda session: repository)
    return repository


def _patch_inference_repo(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    added: list[object] = []
    repository = MagicMock()
    repository.add = AsyncMock(side_effect=lambda inference: (added.append(inference), inference)[1])
    monkeypatch.setattr(inference_service_module, "InferenceRepository", lambda session: repository)
    return added


class _CapturingModelService(ModelService):
    """Records the messages and GenerationConfig handed to generate (loads no weights)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.configs: list[GenerationConfig | None] = []
        self.messages: list[list[ChatMessage]] = []

    def generate(
        self, model: Model, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> GenerationResult:
        self.configs.append(config)
        self.messages.append(messages)
        return GenerationResult(prompt="p", output_text="o", prompt_tokens=1, completion_tokens=1, latency_ms=1)


async def test_infer_raises_when_model_missing(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _patch_model_repo(monkeypatch, None)

    with pytest.raises(ModelNotFoundError):
        await _service(fake_model_service).infer(MagicMock(spec=AsyncSession), model_name="missing", input_text="hello")
    repository.require_by_name.assert_called_once_with("missing")


async def test_infer_raises_when_model_inactive(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    inactive = Model(
        name="m", provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y", status=ModelStatus.INACTIVE
    )
    _patch_model_repo(monkeypatch, inactive)

    with pytest.raises(ModelInactiveError):
        await _service(fake_model_service).infer(MagicMock(spec=AsyncSession), model_name="m", input_text="hello")


async def test_infer_rejects_oversized_input(fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())

    with pytest.raises(InputTooLargeError):
        await _service(fake_model_service).infer(MagicMock(spec=AsyncSession), model_name="m", input_text="x" * 60_000)


async def test_infer_persists_inference(fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    _patch_model_repo(monkeypatch, model)
    added = _patch_inference_repo(monkeypatch)

    inference = await _service(fake_model_service).infer(
        MagicMock(spec=AsyncSession), model_name="m", input_text="hello"
    )

    assert inference.model_id == model.id
    assert added
    assert added[0] is inference


async def test_infer_sends_raw_input_as_a_single_user_turn(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    model_service = _CapturingModelService(settings)

    await _service(model_service).infer(MagicMock(spec=AsyncSession), model_name="m", input_text="hello")

    # The raw input is the single user turn, with no system framing.
    assert model_service.messages == [[{"role": "user", "content": "hello"}]]


async def test_infer_uses_server_default_config_when_no_overrides(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    server = settings.model_copy(update={"temperature": 0.5, "max_output_tokens": 512})
    model_service = _CapturingModelService(server)

    await _service(model_service).infer(MagicMock(spec=AsyncSession), model_name="m", input_text="hello")

    # No preset and no model_params: the whole config comes from the server settings.
    assert model_service.configs == [GenerationConfig(temperature=0.5, max_output_tokens=512)]


async def test_model_params_override_server_default(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    server = settings.model_copy(update={"temperature": 0.5, "max_output_tokens": 512})
    model_service = _CapturingModelService(server)

    await _service(model_service).infer(
        MagicMock(spec=AsyncSession), model_name="m", input_text="hello", model_params={"temperature": 0.9}
    )

    # The override wins on temperature; the untouched knob stays the server default.
    assert model_service.configs == [GenerationConfig(temperature=0.9, max_output_tokens=512)]


async def test_preset_seeds_config_and_is_persisted_on_the_row(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model_repo(monkeypatch, _model())
    added = _patch_inference_repo(monkeypatch)
    preset = GenerationPreset(name="balanced", config=GenerationConfig(do_sample=True, temperature=0.8))
    _patch_preset_repo(monkeypatch, preset)
    model_service = _CapturingModelService(settings)

    inference = await _service(model_service).infer(
        MagicMock(spec=AsyncSession), model_name="m", input_text="hello", preset_id=preset.id
    )

    # The preset seeds decoding, and its id is recorded on the row for lineage.
    assert model_service.configs[0].temperature == 0.8
    assert model_service.configs[0].do_sample is True
    assert inference.preset_id == preset.id
    assert added[0].preset_id == preset.id


async def test_model_params_win_over_preset(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    preset = GenerationPreset(name="balanced", config=GenerationConfig(do_sample=True, temperature=0.8, top_p=0.9))
    _patch_preset_repo(monkeypatch, preset)
    model_service = _CapturingModelService(settings)

    await _service(model_service).infer(
        MagicMock(spec=AsyncSession),
        model_name="m",
        input_text="hello",
        preset_id=preset.id,
        model_params={"temperature": 1.2},
    )

    # Override beats the preset on temperature; the preset's other knob is inherited.
    assert model_service.configs[0].temperature == 1.2
    assert model_service.configs[0].top_p == 0.9


async def test_infer_raises_when_preset_unknown(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    _patch_preset_repo(monkeypatch, None)
    model_service = _CapturingModelService(settings)

    with pytest.raises(PresetNotFoundError):
        await _service(model_service).infer(
            MagicMock(spec=AsyncSession), model_name="m", input_text="hello", preset_id=_model().id
        )


async def test_infer_raises_when_preset_archived(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    archived = GenerationPreset(name="old", config=GenerationConfig(), status=PresetStatus.ARCHIVED)
    _patch_preset_repo(monkeypatch, archived)
    model_service = _CapturingModelService(settings)

    with pytest.raises(PresetNotFoundError):
        await _service(model_service).infer(
            MagicMock(spec=AsyncSession), model_name="m", input_text="hello", preset_id=archived.id
        )
