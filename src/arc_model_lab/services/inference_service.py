"""Inference execution: build chat messages, run the model, persist the result."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import (
    EvaluationResultRepository,
    InferenceRepository,
    ModelRepository,
)
from arc_model_lab.domain import (
    EvaluationResult,
    GenerationConfig,
    Inference,
    InferenceNotFoundError,
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


@dataclass(frozen=True, slots=True)
class InferenceDetailView:
    """One inference paired with its persisted evaluation scores, for the read API."""

    inference: Inference
    evaluations: list[EvaluationResult]


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
    candidates) and their generation config; the experiment-inference link is
    recorded separately by ``ExperimentService``, so the inference row stays free
    of any experiment reference. The commit happens here so a row is persisted
    before any success response.
    """

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    async def summarize(
        self, session: AsyncSession, *, model_name: str, input_text: str, temperature: float | None = None
    ) -> Inference:
        """Resolve the named model and run one summarization for ``/inference``.

        Decoding defaults to the server config (``ARC_TEMPERATURE`` and
        ``ARC_MAX_OUTPUT_TOKENS``); a caller-supplied ``temperature`` overrides
        only that knob. Output length is not caller-controlled here.

        Raises :class:`ModelNotFoundError` (404) when no catalog model has that
        name, :class:`ModelInactiveError` (409) when the model is not active, and
        :class:`InputTooLargeError` (413) for an oversized payload.
        """
        model = await self._resolve_model(session, model_name)
        config = self._model_service.default_generation_config()
        if temperature is not None:
            config = replace(config, temperature=temperature)
        return await self._generate_and_store(session, model=model, input_text=input_text, config=config)

    async def run_for_experiment(
        self,
        session: AsyncSession,
        *,
        model: Model,
        input_text: str,
        config: GenerationConfig,
    ) -> Inference:
        """Run one summarization under an experiment's model and config.

        The inference is stored with no experiment reference; ``ExperimentService``
        records the experiment-inference association after this returns.
        """
        return await self._generate_and_store(session, model=model, input_text=input_text, config=config)

    async def _generate_and_store(
        self,
        session: AsyncSession,
        *,
        model: Model,
        input_text: str,
        config: GenerationConfig,
    ) -> Inference:
        if len(input_text) > _MAX_INPUT_CHARS:
            raise InputTooLargeError(f"Input exceeds {_MAX_INPUT_CHARS} characters")

        messages = build_summary_messages(input_text)
        # Generation is CPU/GPU-bound and blocking; run it off the event loop so a
        # request never stalls the loop for other requests.
        result = await asyncio.to_thread(self._model_service.generate, model, messages, config)

        inference = Inference(
            model_id=model.id,
            input_text=input_text,
            prompt=result.prompt,
            output_text=result.output_text,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        persisted = await InferenceRepository(session).add(inference)
        await session.commit()
        return persisted

    async def _resolve_model(self, session: AsyncSession, model_name: str) -> Model:
        model = await ModelRepository(session).require_by_name(model_name)
        if model.status != ModelStatus.ACTIVE:
            raise ModelInactiveError(f"Model is not active: {model_name}")
        return model

    async def list_recent(self, session: AsyncSession, limit: int) -> list[Inference]:
        """Return the most recent inferences for the history surface (bounded)."""
        return await InferenceRepository(session).list_recent(limit)

    async def get_detail(self, session: AsyncSession, inference_id: UUID) -> InferenceDetailView:
        """Return one inference with its evaluation scores, or raise (404).

        Raises :class:`InferenceNotFoundError` when no inference has that id.
        """
        inference = await InferenceRepository(session).get(inference_id)
        if inference is None:
            raise InferenceNotFoundError(f"Inference not found: {inference_id}")
        evaluations = await EvaluationResultRepository(session).list_for_inference(inference_id)
        return InferenceDetailView(inference=inference, evaluations=evaluations)
