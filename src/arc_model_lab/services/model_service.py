"""Model loading and text generation for a HuggingFace causal instruct model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from arc_model_lab.config import Settings
from arc_model_lab.domain import GenerationError, Model, ModelLoadError


class ChatMessage(TypedDict):
    """A single chat turn for the tokenizer's chat template."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(slots=True, frozen=True)
class RuntimeModel:
    """A loaded runtime: tokenizer, weights, and the device they live on."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: str


@dataclass(slots=True, frozen=True)
class GenerationResult:
    prompt: str
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cache_key(model: Model) -> str:
    revision = model.revision or "default"
    adapter = model.adapter_path or "no-adapter"
    return f"{model.name}:{revision}:{adapter}"


class ModelService:
    """Loads catalog models on demand and caches each runtime in process.

    Runtimes are keyed by ``name:revision:adapter`` so several models and revisions
    coexist. Weights download lazily on first use.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, RuntimeModel] = {}

    def load(self, model: Model) -> RuntimeModel:
        key = _cache_key(model)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model.tokenizer_id,
                revision=model.revision,
                cache_dir=self._settings.model_cache_dir,
            )
            runtime_model = cast(
                PreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    model.model_id,
                    revision=model.revision,
                    torch_dtype="auto",
                    cache_dir=self._settings.model_cache_dir,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - surface any load failure as a domain error
            raise ModelLoadError(f"Failed to load model '{model.model_id}'") from exc

        device = _select_device()
        # transformers stubs mistype nn.Module.to/.eval on PreTrainedModel.
        runtime_model.to(device)  # type: ignore[arg-type]
        runtime_model.eval()  # type: ignore[no-untyped-call]
        runtime = RuntimeModel(tokenizer=tokenizer, model=runtime_model, device=device)
        self._cache[key] = runtime
        return runtime

    def generate(self, model: Model, messages: list[ChatMessage]) -> GenerationResult:
        runtime = self.load(model)
        try:
            prompt = cast(
                str,
                runtime.tokenizer.apply_chat_template(
                    cast("list[dict[str, str]]", messages),
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )
            inputs = runtime.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._settings.max_input_tokens,
            ).to(runtime.device)
            prompt_tokens = int(inputs["input_ids"].shape[1])

            start = time.perf_counter()
            with torch.no_grad():
                output_ids = runtime.model.generate(  # type: ignore[operator]
                    **inputs,
                    max_new_tokens=self._settings.max_new_tokens,
                    num_beams=self._settings.num_beams,
                )
            latency_ms = int((time.perf_counter() - start) * 1000)

            # Causal LMs return prompt + completion; keep only the newly generated tail.
            completion_ids = output_ids[0][prompt_tokens:]
            completion_tokens = int(completion_ids.shape[0])
            output_text = cast(str, runtime.tokenizer.decode(completion_ids, skip_special_tokens=True))

            return GenerationResult(
                prompt=prompt,
                output_text=output_text.strip(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001 - surface any runtime failure as a domain error
            raise GenerationError("Text generation failed") from exc
