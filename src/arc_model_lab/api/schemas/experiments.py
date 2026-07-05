"""Request/response contracts for the experiments endpoints.

The run endpoint reuses :class:`InferenceResponse` because an experiment run
produces exactly one inference (and optional scores), the same shape as
``/inference``. ``GenerationConfigSchema`` forbids unknown keys, so a knob the
runtime ignores (for example ``temperature``) is rejected with 422 at the
boundary rather than silently dropped.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.domain import (
    Experiment,
    ExperimentMetricAggregate,
    ExperimentResults,
    GenerationConfig,
)
from arc_model_lab.domain.generation import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
)


class GenerationConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_input_tokens: int = Field(default=DEFAULT_MAX_INPUT_TOKENS, ge=1)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1)
    num_beams: int = Field(default=DEFAULT_NUM_BEAMS, ge=1)

    def to_domain(self) -> GenerationConfig:
        return GenerationConfig(
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )

    @classmethod
    def from_domain(cls, config: GenerationConfig) -> GenerationConfigSchema:
        return cls(
            max_input_tokens=config.max_input_tokens,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
        )


class ExperimentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1, description="Unique experiment name.")
    description: str | None = None
    model_id: UUID = Field(description="Catalog model to run under this experiment.")
    generation_config: GenerationConfigSchema = Field(default_factory=GenerationConfigSchema)
    created_by: str | None = Field(
        default=None,
        description="Optional caller-supplied label; not authenticated, so not a trustworthy attribution.",
    )

    def to_domain(self) -> Experiment:
        return Experiment(
            name=self.name,
            model_id=self.model_id,
            generation_config=self.generation_config.to_domain(),
            description=self.description,
            created_by=self.created_by,
        )


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: UUID  # noqa: A003 - mirrors the domain primary key
    name: str
    description: str | None
    model_id: UUID
    prompt_version_id: UUID | None
    generation_config: GenerationConfigSchema
    created_by: str | None
    created_at: datetime

    @classmethod
    def from_domain(cls, experiment: Experiment) -> ExperimentResponse:
        return cls(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            model_id=experiment.model_id,
            prompt_version_id=experiment.prompt_version_id,
            generation_config=GenerationConfigSchema.from_domain(experiment.generation_config),
            created_by=experiment.created_by,
            created_at=experiment.created_at,
        )


class ExperimentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_text: str = Field(min_length=1, description="Text to summarize under the experiment.")
    metrics: list[str] | None = Field(default=None, description="Metrics to score the output against.")


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
