"""End-to-end evaluation smoke test against a live arc-eval.

Skipped unless ``ARC_EVAL_SERVICE_URL`` points at a reachable arc-eval. It
fabricates an inference (no model weights are loaded) and evaluates it, so it
verifies the real HTTP contract and persistence without a GPU. Run with
``make eval.smoke``.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy.orm import Session, sessionmaker

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


def test_live_eval_persists_results(session_factory: sessionmaker[Session]) -> None:
    client = build_arc_eval_client(EvalSettings())
    assert client is not None, "ARC_EVAL_SERVICE_URL must be set for the smoke test"
    service = EvaluationService(client)

    with session_factory() as session:
        model = ModelRepository(session).upsert(
            Model(name="smoke-model", provider=Provider.HUGGINGFACE, model_id="x/y", tokenizer_id="x/y")
        )
        inference = InferenceRepository(session).add(
            Inference(
                model_id=model.id,
                input_text="The ARC platform evaluates model outputs for quality signals.",
                prompt="Summarize the following text: The ARC platform evaluates model outputs.",
                output_text="ARC evaluates model output quality.",
                latency_ms=5,
            )
        )
        session.commit()

        outcome = service.evaluate_inference(session, inference)

        assert outcome.status is EvaluationStatus.COMPLETED
        assert EvaluationResultRepository(session).list_for_inference(inference.id)
