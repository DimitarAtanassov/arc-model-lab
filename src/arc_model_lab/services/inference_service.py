from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import InferenceRepository, ModelRepository
from arc_model_lab.domain import (
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


def build_summary_messages(input_text: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": _SUMMARY_INSTRUCTION.format(text=input_text)},
    ]


class InferenceService:
    """Runs one summarization request end to end and persists the result.

    /inference names an active model; deactivating a model takes it out of online
    serving (409). /v1/inference:run is the service-to-service path the eval
    service calls to run a candidate model with an explicit generation config, and
    may run an inactive model (allow_inactive). The commit happens here so a row is
    persisted before any success response.
    """

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    async def summarize(
        self, session: AsyncSession, *, model_name: str, input_text: str, temperature: float | None = None
    ) -> Inference:
        """Resolve the named active model and run one summarization for /inference.

        Decoding defaults to the server config (ARC_TEMPERATURE and
        ARC_MAX_OUTPUT_TOKENS); a caller-supplied temperature overrides only that
        knob. Raises ModelNotFoundError (404), ModelInactiveError (409), and
        InputTooLargeError (413).
        """
        model = await self._resolve(session, model_name, allow_inactive=False)
        config = self._model_service.default_generation_config()
        if temperature is not None:
            config = replace(config, temperature=temperature)
        return await self._generate_and_store(session, model=model, input_text=input_text, config=config)

    async def run_named(
        self,
        session: AsyncSession,
        *,
        model_name: str,
        input_text: str,
        config: GenerationConfig,
        allow_inactive: bool,
    ) -> Inference:
        """Run one summarization for a named model with an explicit config.

        Backs the service-to-service POST /v1/inference:run: the eval service runs a
        candidate model (which may be inactive) for an experiment. Raises
        ModelNotFoundError (404) for an unknown model and ModelInactiveError (409)
        when the model is not active and allow_inactive is false.
        """
        model = await self._resolve(session, model_name, allow_inactive=allow_inactive)
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

    async def _resolve(self, session: AsyncSession, model_name: str, *, allow_inactive: bool) -> Model:
        model = await ModelRepository(session).require_by_name(model_name)
        if not allow_inactive and model.status != ModelStatus.ACTIVE:
            raise ModelInactiveError(f"Model is not active: {model_name}")
        return model

    async def list_recent(self, session: AsyncSession, limit: int) -> list[Inference]:
        """Return the most recent inferences for the history surface (bounded)."""
        return await InferenceRepository(session).list_recent(limit)

    async def get(self, session: AsyncSession, inference_id: UUID) -> Inference:
        """Return one inference by id, or raise InferenceNotFoundError (404)."""
        inference = await InferenceRepository(session).get(inference_id)
        if inference is None:
            raise InferenceNotFoundError(f"Inference not found: {inference_id}")
        return inference
