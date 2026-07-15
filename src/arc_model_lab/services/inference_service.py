from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import InferenceRepository, ModelRepository
from arc_model_lab.domain import (
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


class InferenceService:
    """Runs one inference request end to end and persists the result.

    /inference names an active model and runs its input_text through the model
    unframed; deactivating a model takes it out of online serving (409). The
    commit happens here so a row is persisted before any success response.
    """

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    async def infer(
        self, session: AsyncSession, *, model_name: str, input_text: str, temperature: float | None = None
    ) -> Inference:
        """Resolve the named active model and run one inference for /inference.

        Decoding defaults to the server config (ARC_TEMPERATURE and
        ARC_MAX_OUTPUT_TOKENS); a caller-supplied temperature overrides only that
        knob. Raises ModelNotFoundError (404), ModelInactiveError (409), and
        InputTooLargeError (413).
        """
        model = await self._resolve(session, model_name)
        config = self._model_service.default_generation_config()
        if temperature is not None:
            config = replace(config, temperature=temperature)

        if len(input_text) > _MAX_INPUT_CHARS:
            raise InputTooLargeError(f"Input exceeds {_MAX_INPUT_CHARS} characters")

        messages: list[ChatMessage] = [{"role": "user", "content": input_text}]
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
            generation_config=config,
        )
        persisted = await InferenceRepository(session).add(inference)
        await session.commit()
        return persisted

    async def _resolve(self, session: AsyncSession, model_name: str) -> Model:
        model = await ModelRepository(session).require_by_name(model_name)
        if model.status != ModelStatus.ACTIVE:
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
