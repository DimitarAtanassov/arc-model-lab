"""Inference workflow: build chat messages, run the model, persist the result."""

from __future__ import annotations

from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import InferenceRepository, ModelRepository
from arc_model_lab.domain import (
    Inference,
    InputTooLargeError,
    ModelInactiveError,
    ModelNotFoundError,
    ModelStatus,
)
from arc_model_lab.services.model_service import ChatMessage, ModelService

# Reject oversized payloads before they reach the tokenizer.
_MAX_INPUT_CHARS = 50_000

_SUMMARY_SYSTEM_PROMPT = (
    "You are a precise assistant that writes clear, concise summaries. Capture the key points and leave out filler."
)
_SUMMARY_INSTRUCTION = "Summarize the following text:\n\n{text}"


def build_summary_messages(input_text: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": _SUMMARY_INSTRUCTION.format(text=input_text)},
    ]


class InferenceService:
    """Coordinates a single summarization request end to end.

    The model is resolved per request from the catalog by name. Unknown names
    raise ``ModelNotFoundError``; non-active models raise ``ModelInactiveError``.
    The commit happens here so a row is persisted before any success response.
    """

    def __init__(self, model_service: ModelService, default_model_name: str) -> None:
        self._model_service = model_service
        self._default_model_name = default_model_name

    def summarize(self, session: Session, input_text: str, model_name: str | None = None) -> Inference:
        if len(input_text) > _MAX_INPUT_CHARS:
            raise InputTooLargeError(f"Input exceeds {_MAX_INPUT_CHARS} characters")

        name = model_name or self._default_model_name
        model = ModelRepository(session).get_by_name(name)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {name}")
        if model.status != ModelStatus.ACTIVE:
            raise ModelInactiveError(f"Model is not active: {name}")

        messages = build_summary_messages(input_text)
        result = self._model_service.generate(model, messages)

        inference = Inference(
            model_id=model.id,
            input_text=input_text,
            prompt=result.prompt,
            output_text=result.output_text,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        persisted = InferenceRepository(session).add(inference)
        session.commit()
        return persisted
