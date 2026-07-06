"""Inference execution: build chat messages, run the model, persist the result."""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID

from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import InferenceRepository, ModelRepository
from arc_model_lab.domain import (
    GenerationConfig,
    Inference,
    InputTooLargeError,
    Model,
    ModelInactiveError,
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
    """Runs one summarization request end to end and persists the result.

    The caller names the model: ``/inference`` passes a ``model_name`` that this
    service resolves against the catalog (a missing name is ``ModelNotFoundError``
    -> 404; a non-active model is ``ModelInactiveError`` -> 409, so deactivating a
    model takes it out of online serving). Experiments instead pass an
    already-resolved model (bypassing the active gate on purpose, to evaluate
    candidates), their generation config, and the experiment id to tag the row.
    The commit happens here so a row is persisted before any success response.
    """

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    def summarize(
        self, session: Session, *, model_name: str, input_text: str, temperature: float | None = None
    ) -> Inference:
        """Resolve the named model and run one summarization for ``/inference``.

        Decoding defaults to the server config (``ARC_TEMPERATURE`` and
        ``ARC_MAX_OUTPUT_TOKENS``); a caller-supplied ``temperature`` overrides
        only that knob. Output length is not caller-controlled here.

        Raises :class:`ModelNotFoundError` (404) when no catalog model has that
        name, :class:`ModelInactiveError` (409) when the model is not active, and
        :class:`InputTooLargeError` (413) for an oversized payload.
        """
        model = self._resolve_model(session, model_name)
        config = self._model_service.default_generation_config()
        if temperature is not None:
            config = replace(config, temperature=temperature)
        return self._generate_and_store(session, model=model, input_text=input_text, config=config, experiment_id=None)

    def run_for_experiment(
        self,
        session: Session,
        *,
        model: Model,
        input_text: str,
        config: GenerationConfig,
        experiment_id: UUID,
    ) -> Inference:
        """Run one summarization for an experiment, tagging the row with its id."""
        return self._generate_and_store(
            session, model=model, input_text=input_text, config=config, experiment_id=experiment_id
        )

    def _generate_and_store(
        self,
        session: Session,
        *,
        model: Model,
        input_text: str,
        config: GenerationConfig,
        experiment_id: UUID | None,
    ) -> Inference:
        if len(input_text) > _MAX_INPUT_CHARS:
            raise InputTooLargeError(f"Input exceeds {_MAX_INPUT_CHARS} characters")

        messages = build_summary_messages(input_text)
        result = self._model_service.generate(model, messages, config)

        inference = Inference(
            model_id=model.id,
            input_text=input_text,
            prompt=result.prompt,
            output_text=result.output_text,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            experiment_id=experiment_id,
        )
        persisted = InferenceRepository(session).add(inference)
        session.commit()
        return persisted

    def _resolve_model(self, session: Session, model_name: str) -> Model:
        model = ModelRepository(session).require_by_name(model_name)
        if model.status != ModelStatus.ACTIVE:
            raise ModelInactiveError(f"Model is not active: {model_name}")
        return model
