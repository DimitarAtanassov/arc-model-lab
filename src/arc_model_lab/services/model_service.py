"""Model loading and text generation for a HuggingFace causal instruct model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from arc_model_lab.config import Settings
from arc_model_lab.domain.models import Model


class ChatMessage(TypedDict):
    """A single chat turn for the tokenizer's chat template."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(slots=True, frozen=True)
class GenerationResult:
    prompt: str
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class ModelService:
    """Loads a causal LM once at startup and reuses it for every request.

    ``descriptor`` is the domain metadata used to register the model in the database.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None
        self._device: str = "cpu"
        self.descriptor = Model(
            name=settings.model_name,
            provider=settings.model_provider,
            model_id=settings.model_id,
            tokenizer_id=settings.tokenizer_id,
            adapter_path=settings.adapter_path,
        )

    def load(self) -> None:
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self._settings.tokenizer_id)
        model = AutoModelForCausalLM.from_pretrained(self._settings.model_id, torch_dtype="auto")
        model.to(self._device)
        model.eval()
        self._model = model

    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() before generate().")

        tokenizer = self._tokenizer
        prompt: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._settings.max_input_tokens,
        ).to(self._device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._settings.max_new_tokens,
                num_beams=self._settings.num_beams,
            )
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Causal LMs return prompt + completion; keep only the newly generated tail.
        completion_ids = output_ids[0][prompt_tokens:]
        completion_tokens = int(completion_ids.shape[0])
        output_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        return GenerationResult(
            prompt=prompt,
            output_text=output_text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
