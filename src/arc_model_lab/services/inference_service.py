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

    The model is not chosen by the caller: every request runs on the deployed
    model named in configuration. A deployed model that is absent from the
    catalog raises ``ModelNotFoundError`` and one that is not active raises
    ``ModelInactiveError``; both are server-side misconfigurations, not client
    input. The commit happens here so a row is persisted before any success
    response.
    """

    def __init__(self, model_service: ModelService, deployed_model_name: str) -> None:
        self._model_service = model_service
        self._deployed_model_name = deployed_model_name

    def summarize(self, session: Session, input_text: str) -> Inference:
        if len(input_text) > _MAX_INPUT_CHARS:
            raise InputTooLargeError(f"Input exceeds {_MAX_INPUT_CHARS} characters")

        model = ModelRepository(session).get_by_name(self._deployed_model_name)
        if model is None:
            raise ModelNotFoundError(f"Deployed model not found: {self._deployed_model_name}")
        if model.status != ModelStatus.ACTIVE:
            raise ModelInactiveError(f"Deployed model is not active: {self._deployed_model_name}")

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
