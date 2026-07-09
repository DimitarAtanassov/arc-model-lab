from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.domain import EvaluationOutcome, Inference
from arc_model_lab.services.evaluation_service import EvaluationService, ScoredInference


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    """One inference's evaluation outcome within a batch."""

    inference_id: UUID
    outcome: EvaluationOutcome


async def evaluate_batch(
    session: AsyncSession,
    service: EvaluationService,
    inferences: list[Inference],
    *,
    metrics: list[str],
    concurrency: int,
) -> list[BatchItemResult]:
    """Score inferences concurrently (bounded), then persist each serially.

    At most concurrency arc-eval calls are in flight at once. Persistence runs
    in input order after scoring completes; each item's outcome mirrors the
    single-inference path (skipped, fail-open failed, or completed). An unknown
    metric propagates as UnknownMetricError and aborts the batch, since it is a
    caller error that would affect every item.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _score(inference: Inference) -> ScoredInference:
        async with semaphore:
            return await service.score(inference, metrics)

    scored = await asyncio.gather(*(_score(inference) for inference in inferences))
    results: list[BatchItemResult] = []
    for item in scored:
        outcome = await service.persist(session, item)
        results.append(BatchItemResult(inference_id=item.inference.id, outcome=outcome))
    return results
