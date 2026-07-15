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
    PromptInput,
)
from arc_model_lab.prompts import PromptLibrary
from arc_model_lab.services.model_service import ChatMessage, ModelService

# Reject oversized payloads before they reach the tokenizer.
_MAX_INPUT_CHARS = 50_000


class InferenceService:
    """Runs one inference request end to end and persists the result.

    /inference names an active model; deactivating a model takes it out of online
    serving (409). /v1/inference:run is the service-to-service path the eval
    service calls to run a candidate model with an explicit generation config, and
    may run an inactive model (allow_inactive). A request runs raw (its input_text
    sent to the model unframed) or through a named prompt template (which frames
    input_text and is filled by the request's variables). The commit happens here so
    a row is persisted before any success response.
    """

    def __init__(self, model_service: ModelService, prompts: PromptLibrary) -> None:
        self._model_service = model_service
        self._prompts = prompts

    async def infer(
        self, session: AsyncSession, *, model_name: str, prompt: PromptInput, temperature: float | None = None
    ) -> Inference:
        """Resolve the named active model and run one inference for /inference.

        Decoding defaults to the server config (ARC_TEMPERATURE and
        ARC_MAX_OUTPUT_TOKENS); a caller-supplied temperature overrides only that
        knob. Raises ModelNotFoundError (404), ModelInactiveError (409),
        PromptTemplateNotFoundError (404), PromptRenderError (422), and
        InputTooLargeError (413).
        """
        model = await self._resolve(session, model_name, allow_inactive=False)
        config = self._model_service.default_generation_config()
        if temperature is not None:
            config = replace(config, temperature=temperature)
        return await self._generate_and_store(session, model=model, prompt=prompt, config=config)

    async def run_named(
        self,
        session: AsyncSession,
        *,
        model_name: str,
        prompt: PromptInput,
        config: GenerationConfig,
        allow_inactive: bool,
    ) -> Inference:
        """Run one inference for a named model with an explicit config.

        Backs the service-to-service POST /v1/inference:run: the eval service runs a
        candidate model (which may be inactive) for an experiment. Raises
        ModelNotFoundError (404), ModelInactiveError (409),
        PromptTemplateNotFoundError (404), and PromptRenderError (422).
        """
        model = await self._resolve(session, model_name, allow_inactive=allow_inactive)
        return await self._generate_and_store(session, model=model, prompt=prompt, config=config)

    async def _generate_and_store(
        self,
        session: AsyncSession,
        *,
        model: Model,
        prompt: PromptInput,
        config: GenerationConfig,
    ) -> Inference:
        if len(prompt.input_text) > _MAX_INPUT_CHARS:
            raise InputTooLargeError(f"Input exceeds {_MAX_INPUT_CHARS} characters")

        messages = self._build_messages(prompt)
        # Generation is CPU/GPU-bound and blocking; run it off the event loop so a
        # request never stalls the loop for other requests.
        result = await asyncio.to_thread(self._model_service.generate, model, messages, config)

        inference = Inference(
            model_id=model.id,
            input_text=prompt.input_text,
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

    def _build_messages(self, prompt: PromptInput) -> list[ChatMessage]:
        """Build chat messages: raw input as one user turn, or a rendered template.

        No template sends input_text unframed. A named template frames it (as
        {input_text}) and its variables fill the rest; an unknown template is a 404
        and a bad variable set a 422, both raised before generation.
        """
        if prompt.template is None:
            return [{"role": "user", "content": prompt.input_text}]
        template = self._prompts.require(prompt.template)
        rendered = template.render(input_text=prompt.input_text, variables=prompt.variables)
        messages: list[ChatMessage] = []
        if rendered.system is not None:
            messages.append({"role": "system", "content": rendered.system})
        messages.append({"role": "user", "content": rendered.user})
        return messages

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
