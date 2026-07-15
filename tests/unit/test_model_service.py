from __future__ import annotations

import pytest
import torch

from arc_model_lab.config import Settings
from arc_model_lab.domain import GenerationConfig, GenerationError, Model, ModelLoadError, Provider
from arc_model_lab.services import model_service as model_service_module
from arc_model_lab.services.model_service import (
    ChatMessage,
    ModelService,
    RuntimeModel,
    _cache_key,
    _select_device,
)

_MESSAGES: list[ChatMessage] = [{"role": "user", "content": "hi"}]


def _model(*, revision: str | None = None, adapter_path: str | None = None) -> Model:
    return Model(
        name="unit-model",
        provider=Provider.HUGGINGFACE,
        model_id="unit/model",
        tokenizer_id="unit/model",
        revision=revision,
        adapter_path=adapter_path,
    )


def _service() -> ModelService:
    return ModelService(Settings())


class _FakeInputs(dict):
    def to(self, device: str) -> _FakeInputs:
        return self


class _FakeTokenizer:
    def __init__(self, *, prompt: str = "PROMPT", decoded: str = " summary ") -> None:
        self._prompt = prompt
        self._decoded = decoded

    def apply_chat_template(self, messages: object, *, tokenize: bool, add_generation_prompt: bool) -> str:
        return self._prompt

    def __call__(self, prompt: object, *, return_tensors: str, truncation: bool, max_length: int) -> _FakeInputs:
        return _FakeInputs(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids: object, *, skip_special_tokens: bool) -> str:
        return self._decoded


class _FakeModel:
    def to(self, device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def generate(self, *, input_ids: torch.Tensor, max_new_tokens: int, **kwargs: object) -> torch.Tensor:
        return torch.cat([input_ids, torch.tensor([[4, 5]])], dim=1)


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(name: str, *, revision: str | None, cache_dir: str) -> _FakeTokenizer:
        return _FakeTokenizer()


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(name: str, *, revision: str | None, torch_dtype: str, cache_dir: str) -> _FakeModel:
        return _FakeModel()


def test_cache_key_includes_revision_and_adapter() -> None:
    assert _cache_key(_model(revision="v2", adapter_path="/a")) == "unit-model:v2:/a"


def test_cache_key_defaults_when_revision_and_adapter_missing() -> None:
    assert _cache_key(_model()) == "unit-model:default:no-adapter"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_select_device_auto(monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    assert _select_device("auto") == expected


def test_select_device_explicit_cpu_is_honored_without_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert _select_device("cpu") == "cpu"


@pytest.mark.parametrize("preference", ["cuda", "mps"])
def test_select_device_explicit_accelerator_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch, preference: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ModelLoadError):
        _select_device(preference)  # type: ignore[arg-type]


def test_load_caches_runtime_and_places_on_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_service_module, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(model_service_module, "AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(model_service_module, "_select_device", lambda _preference="auto": "cpu")
    service = _service()
    model = _model()

    runtime = service.load(model)

    assert runtime.device == "cpu"
    assert isinstance(runtime.tokenizer, _FakeTokenizer)
    assert isinstance(runtime.model, _FakeModel)
    assert service.load(model) is runtime  # second call is served from cache


def test_load_wraps_failure_in_model_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        @staticmethod
        def from_pretrained(name: str, *, revision: str | None, cache_dir: str) -> _FakeTokenizer:
            raise RuntimeError("weights unreachable")

    monkeypatch.setattr(model_service_module, "AutoTokenizer", _Boom)

    with pytest.raises(ModelLoadError, match="Failed to load model"):
        _service().load(_model())


def test_generate_returns_only_newly_generated_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    runtime = RuntimeModel(
        tokenizer=_FakeTokenizer(prompt="P", decoded=" hello world "),
        model=_FakeModel(),
        device="cpu",
    )
    monkeypatch.setattr(service, "load", lambda _model: runtime)

    result = service.generate(_model(), _MESSAGES)

    assert result.prompt == "P"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2
    assert result.output_text == "hello world"
    assert result.latency_ms >= 0


def test_generate_wraps_failure_in_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BoomTokenizer(_FakeTokenizer):
        def apply_chat_template(self, messages: object, *, tokenize: bool, add_generation_prompt: bool) -> str:
            raise RuntimeError("template failed")

    service = _service()
    runtime = RuntimeModel(tokenizer=_BoomTokenizer(), model=_FakeModel(), device="cpu")
    monkeypatch.setattr(service, "load", lambda _model: runtime)

    with pytest.raises(GenerationError, match="Text generation failed"):
        service.generate(_model(), _MESSAGES)


@pytest.mark.parametrize(
    ("temperature", "expect_sampling"),
    [(0.0, False), (0.7, True)],
)
def test_generate_threads_decoding_config(
    monkeypatch: pytest.MonkeyPatch, temperature: float, expect_sampling: bool
) -> None:
    # temperature 0 stays greedy (do_sample False, no temperature kwarg); above 0
    # enables sampling and forwards the temperature to the runtime.
    captured: dict[str, object] = {}

    class _CapturingModel(_FakeModel):
        def generate(self, *, input_ids: torch.Tensor, max_new_tokens: int, **kwargs: object) -> torch.Tensor:
            captured["max_new_tokens"] = max_new_tokens
            captured.update(kwargs)
            return torch.cat([input_ids, torch.tensor([[4, 5]])], dim=1)

    service = _service()
    runtime = RuntimeModel(tokenizer=_FakeTokenizer(), model=_CapturingModel(), device="cpu")
    monkeypatch.setattr(service, "load", lambda _model: runtime)

    service.generate(_model(), _MESSAGES, GenerationConfig(temperature=temperature, max_output_tokens=16))

    assert captured["max_new_tokens"] == 16
    assert captured["do_sample"] is expect_sampling
    assert ("temperature" in captured) is expect_sampling
    if expect_sampling:
        assert captured["temperature"] == temperature
