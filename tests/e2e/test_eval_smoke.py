from __future__ import annotations

import os

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arc_model_lab.clients.arc_eval_client import EvalSettings, build_arc_eval_client
from arc_model_lab.db.repositories import (
    EvaluationResultRepository,
    InferenceRepository,
    ModelRepository,
)
from arc_model_lab.domain import EvaluationStatus, Inference, Model, Provider
from arc_model_lab.services.evaluation_service import EvaluationService

pytestmark = [
    pytest.mark.eval_smoke,
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("ARC_EVAL_SERVICE_URL"),
        reason="ARC_EVAL_SERVICE_URL not set; live arc-eval smoke skipped",
    ),
]


async def test_live_eval_persists_results(session_factory: async_sessionmaker[AsyncSession]) -> None:
    client = build_arc_eval_client(EvalSettings())
    assert client is not None, "ARC_EVAL_SERVICE_URL must be set for the smoke test"
    service = EvaluationService(client)

    async with session_factory() as session:
        model = await ModelRepository(session).upsert(
            Model(name="smoke-model", provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y")
        )
        inference = await InferenceRepository(session).add(
            Inference(
                model_id=model.id,
                input_text="The ARC platform evaluates model outputs for quality signals.",
                prompt="Summarize the following text: The ARC platform evaluates model outputs.",
                output_text="ARC evaluates model output quality.",
                latency_ms=5,
            )
        )
        await session.commit()

        outcome = await service.evaluate_inference(session, inference, ["faithfulness"])

        assert outcome.status is EvaluationStatus.COMPLETED
        assert await EvaluationResultRepository(session).list_for_inference(inference.id)
