from __future__ import annotations

import asyncio
from typing import cast
from uuid import uuid4

from arc_model_lab.domain import EvaluationOutcome, EvaluationStatus, Inference
from arc_model_lab.services.evaluation_batch import evaluate_batch
from arc_model_lab.services.evaluation_service import EvaluationService, ScoredInference


def _inference() -> Inference:
    return Inference(model_id=uuid4(), input_text="in", prompt="p", output_text="out", latency_ms=1)


class _RecordingService:
    """Fake service recording peak concurrency and persistence order.

    Every inference scores COMPLETED unless its id is in fail_ids. score
    tracks how many calls overlap so a test can assert the semaphore bound.
    """

    def __init__(self, fail_ids: set[object] | None = None) -> None:
        self._fail_ids = fail_ids or set()
        self.active = 0
        self.max_active = 0
        self.persisted: list[object] = []

    async def score(self, inference: Inference, metrics: list[str]) -> ScoredInference:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)  # yield so overlapping scores can interleave
        self.active -= 1
        status = EvaluationStatus.FAILED if inference.id in self._fail_ids else EvaluationStatus.COMPLETED
        return ScoredInference(inference=inference, results=(), status=status)

    async def persist(self, session: object, scored: ScoredInference) -> EvaluationOutcome:
        self.persisted.append(scored.inference.id)
        return EvaluationOutcome(status=scored.status)


async def test_evaluate_batch_returns_empty_for_no_inferences() -> None:
    service = _RecordingService()

    results = await evaluate_batch(object(), cast(EvaluationService, service), [], metrics=["m"], concurrency=4)

    assert results == []
    assert service.persisted == []


async def test_evaluate_batch_scores_and_persists_each_in_input_order() -> None:
    service = _RecordingService()
    inferences = [_inference() for _ in range(3)]

    results = await evaluate_batch(object(), cast(EvaluationService, service), inferences, metrics=["m"], concurrency=2)

    assert [result.inference_id for result in results] == [inference.id for inference in inferences]
    assert all(result.outcome.status is EvaluationStatus.COMPLETED for result in results)
    # Persistence is serial and follows input order.
    assert service.persisted == [inference.id for inference in inferences]


async def test_evaluate_batch_bounds_concurrency_to_the_semaphore() -> None:
    service = _RecordingService()
    inferences = [_inference() for _ in range(6)]

    await evaluate_batch(object(), cast(EvaluationService, service), inferences, metrics=["m"], concurrency=2)

    assert service.max_active <= 2


async def test_evaluate_batch_passes_each_items_status_through() -> None:
    inferences = [_inference() for _ in range(2)]
    service = _RecordingService(fail_ids={inferences[0].id})

    results = await evaluate_batch(object(), cast(EvaluationService, service), inferences, metrics=["m"], concurrency=4)

    assert results[0].outcome.status is EvaluationStatus.FAILED
    assert results[1].outcome.status is EvaluationStatus.COMPLETED
