from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.config import Settings
from arc_model_lab.domain import (
    GenerationConfig,
    InputTooLargeError,
    Model,
    ModelInactiveError,
    ModelNotFoundError,
    ModelStatus,
    PromptInput,
    PromptTemplate,
    Provider,
)
from arc_model_lab.prompts.loader import PromptLibrary
from arc_model_lab.services import inference_service as inference_service_module
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ChatMessage, GenerationResult, ModelService


def _config() -> GenerationConfig:
    return GenerationConfig(temperature=0.0, max_output_tokens=64)


def _model(name: str = "m") -> Model:
    return Model(name=name, provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y")


def _prompt(
    text: str = "hello", *, template: str | None = None, variables: dict[str, str] | None = None
) -> PromptInput:
    return PromptInput(input_text=text, template=template, variables=variables or {})


def _service(model_service: ModelService, *, templates: dict[str, PromptTemplate] | None = None) -> InferenceService:
    return InferenceService(model_service, PromptLibrary(templates or {}))


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
        await _service(fake_model_service).infer(MagicMock(spec=AsyncSession), model_name="missing", prompt=_prompt())
    repository.require_by_name.assert_called_once_with("missing")


async def test_infer_raises_when_model_inactive(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    inactive = Model(
        name="m", provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y", status=ModelStatus.INACTIVE
    )
    _patch_model_repo(monkeypatch, inactive)

    with pytest.raises(ModelInactiveError):
        await _service(fake_model_service).infer(MagicMock(spec=AsyncSession), model_name="m", prompt=_prompt())


async def test_infer_rejects_oversized_input(fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())

    with pytest.raises(InputTooLargeError):
        await _service(fake_model_service).infer(
            MagicMock(spec=AsyncSession), model_name="m", prompt=_prompt("x" * 60_000)
        )


async def test_infer_persists_inference(fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    _patch_model_repo(monkeypatch, model)
    added = _patch_inference_repo(monkeypatch)

    inference = await _service(fake_model_service).infer(MagicMock(spec=AsyncSession), model_name="m", prompt=_prompt())

    assert inference.model_id == model.id
    assert added
    assert added[0] is inference


async def test_infer_sends_raw_input_as_a_single_user_turn(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    model_service = _CapturingModelService(settings)

    await _service(model_service).infer(MagicMock(spec=AsyncSession), model_name="m", prompt=_prompt("hello"))

    # No template: the raw input is the single user turn, with no system framing.
    assert model_service.messages == [[{"role": "user", "content": "hello"}]]


async def test_infer_renders_a_named_template_with_variables(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    model_service = _CapturingModelService(settings)
    templates = {
        "translate": PromptTemplate(
            name="translate",
            user_template="To {language}:\n\n{input_text}",
            system_template="Be {tone}.",
        )
    }

    await _service(model_service, templates=templates).infer(
        MagicMock(spec=AsyncSession),
        model_name="m",
        prompt=_prompt("hola", template="translate", variables={"language": "English", "tone": "formal"}),
    )

    assert model_service.messages == [
        [
            {"role": "system", "content": "Be formal."},
            {"role": "user", "content": "To English:\n\nhola"},
        ]
    ]


async def test_infer_renders_a_template_without_a_system_message(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    model_service = _CapturingModelService(settings)
    templates = {"bare": PromptTemplate(name="bare", user_template="Q: {input_text}")}

    await _service(model_service, templates=templates).infer(
        MagicMock(spec=AsyncSession), model_name="m", prompt=_prompt("why", template="bare")
    )

    assert model_service.messages == [[{"role": "user", "content": "Q: why"}]]


async def test_run_named_allows_an_inactive_model(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    inactive = Model(
        name="m", provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y", status=ModelStatus.INACTIVE
    )
    _patch_model_repo(monkeypatch, inactive)
    added = _patch_inference_repo(monkeypatch)

    inference = await _service(fake_model_service).run_named(
        MagicMock(spec=AsyncSession),
        model_name="m",
        prompt=_prompt(),
        config=_config(),
        allow_inactive=True,
    )

    assert inference.model_id == inactive.id
    assert added
    assert added[0] is inference


async def test_run_named_rejects_an_inactive_model_when_not_allowed(
    fake_model_service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    inactive = Model(
        name="m", provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y", status=ModelStatus.INACTIVE
    )
    _patch_model_repo(monkeypatch, inactive)

    with pytest.raises(ModelInactiveError):
        await _service(fake_model_service).run_named(
            MagicMock(spec=AsyncSession),
            model_name="m",
            prompt=_prompt(),
            config=_config(),
            allow_inactive=False,
        )


async def test_infer_uses_server_default_config_when_temperature_omitted(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    server = settings.model_copy(update={"temperature": 0.5, "max_output_tokens": 512})
    model_service = _CapturingModelService(server)

    await _service(model_service).infer(MagicMock(spec=AsyncSession), model_name="m", prompt=_prompt())

    # No caller temperature: the whole config comes from the server settings.
    assert model_service.configs == [GenerationConfig(temperature=0.5, max_output_tokens=512)]


async def test_infer_applies_caller_temperature_over_server_default(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_model_repo(monkeypatch, _model())
    _patch_inference_repo(monkeypatch)
    server = settings.model_copy(update={"temperature": 0.5, "max_output_tokens": 512})
    model_service = _CapturingModelService(server)

    await _service(model_service).infer(MagicMock(spec=AsyncSession), model_name="m", prompt=_prompt(), temperature=0.9)

    # Caller temperature wins; output length stays the server default.
    assert model_service.configs == [GenerationConfig(temperature=0.9, max_output_tokens=512)]
