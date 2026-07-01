"""Inference workflow: build chat messages, run the model, persist the result."""

from __future__ import annotations

from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import InferenceRepository
from arc_model_lab.domain import Inference, InputTooLargeError, Model
from arc_model_lab.services.model_service import ChatMessage, ModelService

# Reject oversized payloads before they reach the tokenizer.
_MAX_INPUT_CHARS = 50_000

_SUMMARY_SYSTEM_PROMPT = (
    "You are a precise assistant that writes clear, concise summaries. "
    "Capture the key points and leave out filler."
)
_SUMMARY_INSTRUCTION = "Summarize the following text:\n\n{text}"


def build_summary_messages(input_text: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": _SUMMARY_INSTRUCTION.format(text=input_text)},
    ]


class InferenceService:
    """Coordinates a single summarization request end to end.

    ``model`` is the registered domain model (with its persisted id), resolved
    once at startup and reused as the foreign key for every inference row.
    """

    def __init__(self, model_service: ModelService, model: Model) -> None:
        self._model_service = model_service
        self._model = model

    def summarize(self, session: Session, input_text: str) -> Inference:
        if len(input_text) > _MAX_INPUT_CHARS:
            raise InputTooLargeError(f"Input exceeds {_MAX_INPUT_CHARS} characters")

        messages = build_summary_messages(input_text)
        result = self._model_service.generate(messages)

        inference = Inference(
            model_id=self._model.id,
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
