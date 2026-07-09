from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.domain import EvaluationOutcome, EvaluationStatus


class EvaluationRequest(BaseModel):
    """Body for evaluating an existing inference: the metrics to score against.

    metrics is required and non-empty: calling this endpoint is the intent to
    score, so there is no "evaluate nothing" case (unlike an experiment run, where
    omitting metrics means skip scoring).
    """

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(min_length=1, description="Metrics to score the inference against.")


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
