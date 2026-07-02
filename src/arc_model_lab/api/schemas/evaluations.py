"""Evaluation contract nested in the summarize response.

The inline envelope is intentionally lean: the score and its provenance, not the
full reasoning text (which is persisted and queryable via ``evaluation_results``).
"""

from __future__ import annotations

from pydantic import BaseModel

from arc_model_lab.domain import EvaluationOutcome, EvaluationStatus


class EvaluationResultOut(BaseModel):
    metric_name: str
    score: float
    evaluator_name: str
    evaluator_version: str | None = None


class EvaluationEnvelope(BaseModel):
    status: EvaluationStatus
    results: list[EvaluationResultOut] = []

    @classmethod
    def from_outcome(cls, outcome: EvaluationOutcome) -> EvaluationEnvelope:
        return cls(
            status=outcome.status,
            results=[
                EvaluationResultOut(
                    metric_name=result.metric_name,
                    score=result.score,
                    evaluator_name=result.evaluator_name,
                    evaluator_version=result.evaluator_version,
                )
                for result in outcome.results
            ],
        )
