from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.api.schemas.evaluations import EvaluationEnvelope
from arc_model_lab.domain import (
    EvaluationOutcome,
    Experiment,
    ExperimentMetricAggregate,
    ExperimentResults,
    GenerationConfig,
    Inference,
)
from arc_model_lab.domain.generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_TEMPERATURE,
)


class GenerationConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=MAX_TEMPERATURE)
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=1)

    def to_domain(self) -> GenerationConfig:
        return GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    @classmethod
    def from_domain(cls, config: GenerationConfig) -> GenerationConfigSchema:
        return cls(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )


class ExperimentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1, description="Unique experiment name.")
    description: str | None = None
    model_name: str = Field(min_length=1, description="Catalog model to run under this experiment.")
    generation_config: GenerationConfigSchema = Field(default_factory=GenerationConfigSchema)


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: UUID  # noqa: A003 - mirrors the domain primary key
    name: str
    description: str | None
    model_id: UUID
    model_name: str
    generation_config: GenerationConfigSchema
    created_at: datetime

    @classmethod
    def from_domain(cls, experiment: Experiment, model_name: str) -> ExperimentResponse:
        return cls(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            model_id=experiment.model_id,
            model_name=model_name,
            generation_config=GenerationConfigSchema.from_domain(experiment.generation_config),
            created_at=experiment.created_at,
        )


class ExperimentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_text: str = Field(min_length=1, description="Text to summarize under the experiment.")
    metrics: list[str] | None = Field(default=None, description="Metrics to score the output against.")


class ExperimentRunResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: UUID  # noqa: A003 - mirrors the domain primary key
    model_id: UUID
    input_text: str
    prompt: str
    output_text: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    experiment_id: UUID
    created_at: datetime
    evaluation: EvaluationEnvelope | None = None

    @classmethod
    def from_run(
        cls,
        experiment_id: UUID,
        inference: Inference,
        evaluation: EvaluationOutcome | None = None,
    ) -> ExperimentRunResponse:
        """Shape an experiment run: its inference and, when scored, its evaluation.

        The experiment_id comes from the run context, not the inference:
        inference is decoupled from experiments and never carries one.
        /inference returns neither this id nor an evaluation, which keeps the
        two endpoints orthogonal.
        """
        return cls(
            id=inference.id,
            model_id=inference.model_id,
            input_text=inference.input_text,
            prompt=inference.prompt,
            output_text=inference.output_text,
            latency_ms=inference.latency_ms,
            prompt_tokens=inference.prompt_tokens,
            completion_tokens=inference.completion_tokens,
            experiment_id=experiment_id,
            created_at=inference.created_at,
            evaluation=EvaluationEnvelope.from_outcome(evaluation) if evaluation is not None else None,
        )


class MetricAggregateOut(BaseModel):
    metric_name: str
    average_score: float
    evaluated_count: int

    @classmethod
    def from_domain(cls, aggregate: ExperimentMetricAggregate) -> MetricAggregateOut:
        return cls(
            metric_name=aggregate.metric_name,
            average_score=aggregate.average_score,
            evaluated_count=aggregate.evaluated_count,
        )


class ExperimentResultsResponse(BaseModel):
    experiment_id: UUID
    metrics: list[MetricAggregateOut]

    @classmethod
    def from_domain(cls, result: ExperimentResults) -> ExperimentResultsResponse:
        return cls(
            experiment_id=result.experiment_id,
            metrics=[MetricAggregateOut.from_domain(aggregate) for aggregate in result.metrics],
        )


class ExperimentComparisonResponse(BaseModel):
    experiments: list[ExperimentResultsResponse]

    @classmethod
    def from_domain(cls, results: list[ExperimentResults]) -> ExperimentComparisonResponse:
        return cls(experiments=[ExperimentResultsResponse.from_domain(result) for result in results])
