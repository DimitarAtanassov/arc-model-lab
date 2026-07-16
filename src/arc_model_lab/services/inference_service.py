from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
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
from arc_model_lab.domain.generation import resolve_generation_config
from arc_model_lab.services.model_service import ChatMessage, ModelService
from arc_model_lab.services.preset_service import PresetService

# Reject oversized payloads before they reach the tokenizer.
_MAX_INPUT_CHARS = 50_000


class InferenceService:
    """Runs one inference request end to end and persists the result.

    /inference names an active model and runs its input_text through the model
    unframed; deactivating a model takes it out of online serving (409). The
    commit happens here so a row is persisted before any success response.
    """

    def __init__(self, model_service: ModelService, preset_service: PresetService, max_output_tokens_cap: int) -> None:
        self._model_service = model_service
        self._preset_service = preset_service
        self._max_output_tokens_cap = max_output_tokens_cap

    async def infer(
        self,
        session: AsyncSession,
        *,
        model_name: str,
        input_text: str,
        preset_id: UUID | None = None,
        model_params: Mapping[str, Any] | None = None,
    ) -> Inference:
        """Resolve the named active model and run one inference for /inference.

        Decoding is resolved by precedence: call ``model_params`` > ``preset_id`` >
        server defaults (ARC_TEMPERATURE, ARC_MAX_OUTPUT_TOKENS). The resolved config
        is re-validated by the one GenerationConfig constructor, so an illegal merge
        (for example a beam preset plus a top_p override) is a 422, and it is what the
        row persists alongside the preset reference. Raises ModelNotFoundError (404),
        ModelInactiveError (409), PresetNotFoundError (404), InputTooLargeError (413),
        and InvalidGenerationConfigError (422).
        """
        model = await self._resolve(session, model_name)
        preset = await self._preset_service.get(session, preset_id) if preset_id is not None else None
        config = resolve_generation_config(
            self._model_service.default_generation_config(),
            preset.config if preset is not None else None,
            model_params or {},
            max_output_tokens_cap=self._max_output_tokens_cap,
        )

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
            preset_id=preset_id,
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
