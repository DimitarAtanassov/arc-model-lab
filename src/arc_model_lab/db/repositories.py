from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import exists, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.models import (
    EvaluationResultRecord,
    ExperimentRecord,
    ExperimentRunRecord,
    InferenceRecord,
    ModelRecord,
)
from arc_model_lab.domain import (
    CorruptStoredDataError,
    EvaluationResult,
    Experiment,
    ExperimentMetricAggregate,
    ExperimentNameConflictError,
    ExperimentRun,
    GenerationConfig,
    Inference,
    InvalidGenerationConfigError,
    Model,
    ModelNotFoundError,
    ModelStatus,
    Provider,
)


class ModelRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_name(self, name: str) -> Model | None:
        record = await self._session.scalar(select(ModelRecord).where(ModelRecord.name == name))
        return _to_model(record) if record is not None else None

    async def get_by_id(self, model_id: UUID) -> Model | None:
        record = await self._session.get(ModelRecord, model_id)
        return _to_model(record) if record is not None else None

    async def require_by_name(self, name: str) -> Model:
        """Return the model with this name, or raise ModelNotFoundError (404)."""
        model = await self.get_by_name(name)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {name}")
        return model

    async def require_by_id(self, model_id: UUID) -> Model:
        """Return the model with this id, or raise ModelNotFoundError (404)."""
        model = await self.get_by_id(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {model_id}")
        return model

    async def add(self, model: Model) -> Model:
        self._session.add(_to_model_record(model))
        await self._session.flush()
        return model

    async def list_all(self) -> list[Model]:
        records = (await self._session.scalars(select(ModelRecord).order_by(ModelRecord.name))).all()
        return [_to_model(record) for record in records]

    async def upsert(self, model: Model) -> Model:
        record = await self._session.scalar(select(ModelRecord).where(ModelRecord.name == model.name))
        if record is None:
            return await self.add(model)
        record.provider = model.provider
        record.model_id = model.model_id
        record.tokenizer_id = model.tokenizer_id
        record.revision = model.revision
        record.adapter_path = model.adapter_path
        record.status = model.status
        await self._session.flush()
        await self._session.refresh(record)
        return _to_model(record)

    async def set_status(self, name: str, status: ModelStatus) -> Model | None:
        record = await self._session.scalar(select(ModelRecord).where(ModelRecord.name == name))
        if record is None:
            return None
        record.status = status
        await self._session.flush()
        await self._session.refresh(record)
        return _to_model(record)


class InferenceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, inference: Inference) -> Inference:
        self._session.add(_to_inference_record(inference))
        await self._session.flush()
        return inference

    async def get(self, inference_id: UUID) -> Inference | None:
        record = await self._session.get(InferenceRecord, inference_id)
        return _to_inference(record) if record is not None else None

    async def list_unevaluated(
        self,
        *,
        limit: int,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[Inference]:
        """Return inferences that have no evaluation results yet, oldest first.

        Used by replay/backfill. The optional half-open [created_after,
        created_before) window scopes a backfill to a time range.
        """
        unevaluated = ~exists().where(EvaluationResultRecord.inference_id == InferenceRecord.id)
        stmt = select(InferenceRecord).where(unevaluated)
        if created_after is not None:
            stmt = stmt.where(InferenceRecord.created_at >= created_after)
        if created_before is not None:
            stmt = stmt.where(InferenceRecord.created_at < created_before)
        stmt = stmt.order_by(InferenceRecord.created_at).limit(limit)
        records = (await self._session.scalars(stmt)).all()
        return [_to_inference(record) for record in records]

    async def list_recent(self, limit: int) -> list[Inference]:
        """Return the most recent inferences, newest first (bounded page size).

        Backs the history table. Ordering is by created_at descending; the
        caller bounds limit so the collection is never unbounded.
        """
        records = (
            await self._session.scalars(
                select(InferenceRecord).order_by(InferenceRecord.created_at.desc()).limit(limit)
            )
        ).all()
        return [_to_inference(record) for record in records]


class EvaluationResultRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_many(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        """Insert results, refreshing score/reasoning on the unique key.

        The unique key (inference_id, metric_name, evaluator_name) makes this
        idempotent: replaying an already-scored inference updates the score in
        place instead of raising or duplicating. created_at is preserved as
        the first-evaluated time.
        """
        if not results:
            return []
        insert_stmt = pg_insert(EvaluationResultRecord).values(
            [_evaluation_result_values(result) for result in results]
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            constraint="uq_evaluation_results_inference_metric_evaluator",
            set_={
                "score": insert_stmt.excluded.score,
                "reasoning": insert_stmt.excluded.reasoning,
                "evaluator_version": insert_stmt.excluded.evaluator_version,
            },
        )
        await self._session.execute(upsert_stmt)
        await self._session.flush()
        return results

    async def list_for_inference(self, inference_id: UUID) -> list[EvaluationResult]:
        records = (
            await self._session.scalars(
                select(EvaluationResultRecord)
                .where(EvaluationResultRecord.inference_id == inference_id)
                .order_by(EvaluationResultRecord.metric_name)
            )
        ).all()
        return [_to_evaluation_result(record) for record in records]


class ExperimentRepository:
    # Names the DB-generated unique constraint (see the metadata naming
    # convention in db.base) so a duplicate name is told apart from other
    # integrity violations such as the model_id foreign key.
    _NAME_CONSTRAINT = "uq_experiments_name"

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, experiment: Experiment) -> Experiment:
        """Insert an experiment, mapping a duplicate name to a domain conflict.

        The unique constraint is the authoritative guard (including against a
        concurrent create); only that violation is a caller conflict. Any other
        IntegrityError (for example the model_id foreign key) propagates unchanged
        so it is never misreported as a name conflict.
        """
        self._session.add(_to_experiment_record(experiment))
        try:
            await self._session.flush()
        except IntegrityError as exc:
            if _violates_constraint(exc, self._NAME_CONSTRAINT):
                raise ExperimentNameConflictError(f"Experiment name already exists: {experiment.name}") from exc
            raise
        return experiment

    async def get(self, experiment_id: UUID) -> Experiment | None:
        record = await self._session.get(ExperimentRecord, experiment_id)
        return _to_experiment(record) if record is not None else None

    async def get_by_name(self, name: str) -> Experiment | None:
        record = await self._session.scalar(select(ExperimentRecord).where(ExperimentRecord.name == name))
        return _to_experiment(record) if record is not None else None

    async def list_recent(self, limit: int) -> list[Experiment]:
        """Return the most recent experiments, newest first (bounded page size)."""
        records = (
            await self._session.scalars(
                select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc()).limit(limit)
            )
        ).all()
        return [_to_experiment(record) for record in records]

    async def aggregate_scores(self, experiment_id: UUID) -> list[ExperimentMetricAggregate]:
        """Average score and count per metric across the experiment's evaluations.

        Aggregation joins evaluation results to the experiment through the
        experiment_runs association (not a column on inference), so comparison
        stays indexable and needs no application-side joins.
        """
        stmt = (
            select(
                EvaluationResultRecord.metric_name,
                func.avg(EvaluationResultRecord.score).label("average_score"),
                func.count().label("evaluated_count"),
            )
            .join(
                ExperimentRunRecord,
                ExperimentRunRecord.inference_id == EvaluationResultRecord.inference_id,
            )
            .where(ExperimentRunRecord.experiment_id == experiment_id)
            .group_by(EvaluationResultRecord.metric_name)
            .order_by(EvaluationResultRecord.metric_name)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            ExperimentMetricAggregate(
                metric_name=row.metric_name,
                average_score=float(row.average_score),
                evaluated_count=int(row.evaluated_count),
            )
            for row in rows
        ]


class ExperimentRunRepository:
    """Persists the association between an inference and its experiment.

    Write-only: the read path for experiment runs is the aggregate join in
    ExperimentRepository.aggregate_scores, so a per-row lookup is not
    needed here.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, run: ExperimentRun) -> ExperimentRun:
        self._session.add(_to_experiment_run_record(run))
        await self._session.flush()
        return run


def _to_model(record: ModelRecord) -> Model:
    return Model(
        id=record.id,
        name=record.name,
        provider=Provider(record.provider),
        model_id=record.model_id,
        tokenizer_id=record.tokenizer_id,
        revision=record.revision,
        adapter_path=record.adapter_path,
        status=ModelStatus(record.status),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_model_record(model: Model) -> ModelRecord:
    return ModelRecord(
        id=model.id,
        name=model.name,
        provider=model.provider,
        model_id=model.model_id,
        tokenizer_id=model.tokenizer_id,
        revision=model.revision,
        adapter_path=model.adapter_path,
        status=model.status,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _to_inference(record: InferenceRecord) -> Inference:
    return Inference(
        id=record.id,
        model_id=record.model_id,
        input_text=record.input_text,
        prompt=record.prompt,
        output_text=record.output_text,
        latency_ms=record.latency_ms,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        created_at=record.created_at,
    )


def _to_inference_record(inference: Inference) -> InferenceRecord:
    return InferenceRecord(
        id=inference.id,
        model_id=inference.model_id,
        input_text=inference.input_text,
        prompt=inference.prompt,
        output_text=inference.output_text,
        latency_ms=inference.latency_ms,
        prompt_tokens=inference.prompt_tokens,
        completion_tokens=inference.completion_tokens,
        created_at=inference.created_at,
    )


def _to_evaluation_result(record: EvaluationResultRecord) -> EvaluationResult:
    return EvaluationResult(
        id=record.id,
        inference_id=record.inference_id,
        metric_name=record.metric_name,
        score=record.score,
        reasoning=record.reasoning,
        evaluator_name=record.evaluator_name,
        evaluator_version=record.evaluator_version,
        created_at=record.created_at,
    )


def _evaluation_result_values(result: EvaluationResult) -> dict[str, object]:
    return {
        "id": result.id,
        "inference_id": result.inference_id,
        "metric_name": result.metric_name,
        "score": result.score,
        "reasoning": result.reasoning,
        "evaluator_name": result.evaluator_name,
        "evaluator_version": result.evaluator_version,
        "created_at": result.created_at,
    }


def _to_experiment(record: ExperimentRecord) -> Experiment:
    return Experiment(
        id=record.id,
        name=record.name,
        description=record.description,
        model_id=record.model_id,
        generation_config=_load_generation_config(record),
        created_at=record.created_at,
    )


def _load_generation_config(record: ExperimentRecord) -> GenerationConfig:
    """Rebuild the stored generation config, treating corruption as a server fault.

    Stored config passed validation on write, so a failure reading it back is data
    corruption, not a client error: re-raise as CorruptStoredDataError (500) rather
    than let InvalidGenerationConfigError surface a read fault as a 422.
    """
    try:
        return GenerationConfig.from_mapping(record.generation_config)
    except InvalidGenerationConfigError as exc:
        raise CorruptStoredDataError(f"Corrupt generation config for experiment {record.id}: {exc}") from exc


def _to_experiment_record(experiment: Experiment) -> ExperimentRecord:
    return ExperimentRecord(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        model_id=experiment.model_id,
        generation_config=experiment.generation_config.to_dict(),
        created_at=experiment.created_at,
    )


def _to_experiment_run_record(run: ExperimentRun) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        id=run.id,
        experiment_id=run.experiment_id,
        inference_id=run.inference_id,
        created_at=run.created_at,
    )


def _violates_constraint(error: IntegrityError, constraint: str) -> bool:
    """True when the IntegrityError is a violation of the named DB constraint.

    Reads the constraint name from the psycopg diagnostics so a unique-name
    violation can be told apart from other integrity failures (for example a
    foreign key). Falls back to False when the driver exposes no constraint name,
    so an unclassifiable error is re-raised rather than mislabeled.
    """
    diagnostic = getattr(error.orig, "diag", None)
    return bool(getattr(diagnostic, "constraint_name", None) == constraint)
