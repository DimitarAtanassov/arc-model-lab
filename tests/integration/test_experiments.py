"""Experiment repository and service round-trips against a real Postgres.

Covers the mapping round-trip, the SQL aggregation, model validation on create,
and that a run links its inference to the experiment through experiment_runs. The
model runtime is faked (no weight downloads) and evaluation is disabled, so these
exercise the experiment engine, not arc-eval.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.config import Settings
from arc_model_lab.db.models import ExperimentRecord, ExperimentRunRecord
from arc_model_lab.db.repositories import (
    EvaluationResultRepository,
    ExperimentRepository,
    ExperimentRunRepository,
    InferenceRepository,
    ModelRepository,
)
from arc_model_lab.domain import (
    CorruptStoredDataError,
    EvaluationResult,
    Experiment,
    ExperimentNameConflictError,
    ExperimentNotFoundError,
    ExperimentRun,
    GenerationConfig,
    Inference,
    Model,
    ModelNotFoundError,
    ModelStatus,
    Provider,
)
from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.experiment_service import ExperimentService
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ChatMessage, GenerationResult, ModelService

pytestmark = pytest.mark.integration


class _CapturingModelService(ModelService):
    """Model-runtime double that records the generation config it was handed."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.configs: list[GenerationConfig | None] = []

    def generate(
        self, model: Model, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> GenerationResult:
        self.configs.append(config)
        return GenerationResult(
            prompt="fake-prompt",
            output_text="fake summary",
            prompt_tokens=3,
            completion_tokens=2,
            latency_ms=1,
        )


def _config() -> GenerationConfig:
    return GenerationConfig(temperature=0.0, max_output_tokens=256)


async def _persist_model(
    session: AsyncSession, name: str = "base", *, status: ModelStatus = ModelStatus.ACTIVE
) -> Model:
    model = await ModelRepository(session).upsert(
        Model(
            name=name,
            provider=Provider.HUGGINGFACE,
            model_id="org/model",
            tokenizer_id="org/model",
            status=status,
        )
    )
    await session.commit()
    return model


def _experiment(model_id: object, name: str = "exp") -> Experiment:
    return Experiment(name=name, model_id=model_id, generation_config=_config())  # type: ignore[arg-type]


def _service(model_service: ModelService) -> ExperimentService:
    return ExperimentService(InferenceService(model_service), EvaluationService(None))


async def _create(
    service: ExperimentService, session: AsyncSession, model_name: str, *, name: str = "exp"
) -> Experiment:
    """Create an experiment via the service and return the stored domain entity."""
    view = await service.create(session, name=name, model_name=model_name, generation_config=_config())
    return view.experiment


async def test_add_and_get_round_trips(db_session: AsyncSession) -> None:
    model = await _persist_model(db_session)
    repo = ExperimentRepository(db_session)
    saved = await repo.add(_experiment(model.id))
    await db_session.commit()

    fetched = await repo.get(saved.id)
    assert fetched is not None
    assert fetched.name == "exp"
    assert fetched.model_id == model.id
    assert fetched.generation_config == _config()


async def test_get_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await ExperimentRepository(db_session).get(uuid4()) is None


async def test_run_allows_an_inactive_model(db_session: AsyncSession, fake_model_service: ModelService) -> None:
    # Experiments deliberately bypass the /inference active gate so a candidate
    # model can be evaluated before it is activated.
    model = await _persist_model(db_session, name="candidate", status=ModelStatus.INACTIVE)
    service = _service(fake_model_service)
    experiment = await _create(service, db_session, model.name, name="candidate-exp")

    result = await service.run(db_session, experiment.id, "summarize this")

    link = await db_session.scalar(
        select(ExperimentRunRecord).where(ExperimentRunRecord.inference_id == result.inference.id)
    )
    assert link is not None
    assert link.experiment_id == experiment.id


async def test_aggregate_scores_averages_by_metric(db_session: AsyncSession) -> None:
    model = await _persist_model(db_session)
    repo = ExperimentRepository(db_session)
    experiment = await repo.add(_experiment(model.id))
    await db_session.commit()

    inference_repo = InferenceRepository(db_session)
    run_repo = ExperimentRunRepository(db_session)
    eval_repo = EvaluationResultRepository(db_session)
    for score in (0.6, 0.8):
        inference = await inference_repo.add(
            Inference(
                model_id=model.id,
                input_text="in",
                prompt="p",
                output_text="out",
                latency_ms=1,
            )
        )
        await db_session.commit()
        await run_repo.add(ExperimentRun(experiment_id=experiment.id, inference_id=inference.id))
        await eval_repo.upsert_many(
            [
                EvaluationResult(
                    inference_id=inference.id,
                    metric_name="faithfulness",
                    score=score,
                    evaluator_name="summary-faithfulness",
                )
            ]
        )
        await db_session.commit()

    aggregates = await repo.aggregate_scores(experiment.id)

    assert len(aggregates) == 1
    assert aggregates[0].metric_name == "faithfulness"
    assert aggregates[0].evaluated_count == 2
    assert aggregates[0].average_score == pytest.approx(0.7)


async def test_create_rejects_unknown_model(db_session: AsyncSession, fake_model_service: ModelService) -> None:
    service = _service(fake_model_service)
    with pytest.raises(ModelNotFoundError):
        await service.create(db_session, name="exp", model_name="does-not-exist", generation_config=_config())


async def test_run_links_inference_to_experiment(db_session: AsyncSession, fake_model_service: ModelService) -> None:
    model = await _persist_model(db_session)
    service = _service(fake_model_service)
    experiment = await _create(service, db_session, model.name)

    result = await service.run(db_session, experiment.id, "summarize this")

    assert result.evaluation is None
    # The inference is persisted; the experiment link lives in experiment_runs,
    # not on the inference row.
    stored = await InferenceRepository(db_session).get(result.inference.id)
    assert stored is not None
    link = await db_session.scalar(
        select(ExperimentRunRecord).where(ExperimentRunRecord.inference_id == result.inference.id)
    )
    assert link is not None
    assert link.experiment_id == experiment.id


async def test_get_by_name_round_trips(db_session: AsyncSession) -> None:
    model = await _persist_model(db_session)
    repo = ExperimentRepository(db_session)
    await repo.add(_experiment(model.id, name="named"))
    await db_session.commit()

    fetched = await repo.get_by_name("named")

    assert fetched is not None
    assert fetched.name == "named"
    assert await repo.get_by_name("absent") is None


async def test_create_rejects_duplicate_name(db_session: AsyncSession, fake_model_service: ModelService) -> None:
    model = await _persist_model(db_session)
    service = _service(fake_model_service)
    await _create(service, db_session, model.name, name="dup")

    with pytest.raises(ExperimentNameConflictError):
        await _create(service, db_session, model.name, name="dup")


async def test_results_rejects_unknown_experiment(db_session: AsyncSession, fake_model_service: ModelService) -> None:
    service = _service(fake_model_service)

    with pytest.raises(ExperimentNotFoundError):
        await service.results(db_session, uuid4())


async def test_compare_rejects_unknown_experiment(db_session: AsyncSession, fake_model_service: ModelService) -> None:
    model = await _persist_model(db_session)
    service = _service(fake_model_service)
    known = await _create(service, db_session, model.name, name="known")

    with pytest.raises(ExperimentNotFoundError):
        await service.compare(db_session, known.id, uuid4())


async def test_run_uses_the_experiment_generation_config(db_session: AsyncSession, settings: Settings) -> None:
    model = await _persist_model(db_session)
    capturing = _CapturingModelService(settings)
    service = _service(capturing)
    config = GenerationConfig(temperature=0.7, max_output_tokens=32)
    view = await service.create(db_session, name="cfg", model_name=model.name, generation_config=config)
    experiment = view.experiment

    await service.run(db_session, experiment.id, "summarize this")

    assert capturing.configs == [config]


async def test_add_reraises_foreign_key_violation(db_session: AsyncSession) -> None:
    # A non-existent model_id violates the RESTRICT foreign key, not the name
    # unique constraint: it must surface as IntegrityError, not a name conflict.
    repo = ExperimentRepository(db_session)

    with pytest.raises(IntegrityError):
        await repo.add(_experiment(uuid4(), name="orphan"))


async def test_get_raises_on_corrupt_stored_generation_config(db_session: AsyncSession) -> None:
    model = await _persist_model(db_session)
    db_session.add(
        ExperimentRecord(
            id=uuid4(),
            name="corrupt",
            model_id=model.id,
            generation_config={"temperature": 1},
        )
    )
    await db_session.commit()

    with pytest.raises(CorruptStoredDataError):
        await ExperimentRepository(db_session).get_by_name("corrupt")
