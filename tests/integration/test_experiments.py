"""Experiment repository and service round-trips against a real Postgres.

Covers the mapping round-trip, the SQL aggregation, model validation on create,
and that a run tags its inference with the experiment id. The model runtime is
faked (no weight downloads) and evaluation is disabled, so these exercise the
experiment engine, not arc-eval.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from arc_model_lab.config import Settings
from arc_model_lab.db.models import ExperimentRecord
from arc_model_lab.db.repositories import (
    EvaluationResultRepository,
    ExperimentRepository,
    InferenceRepository,
    ModelRepository,
)
from arc_model_lab.domain import (
    CorruptStoredDataError,
    EvaluationResult,
    Experiment,
    ExperimentNameConflictError,
    ExperimentNotFoundError,
    GenerationConfig,
    Inference,
    Model,
    ModelNotFoundError,
    Provider,
)
from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.experiment_service import ExperimentService
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.inference_workflow import InferenceWorkflow
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
    return GenerationConfig(max_input_tokens=1024, max_new_tokens=256, num_beams=1)


def _persist_model(session: Session, name: str = "base") -> Model:
    model = ModelRepository(session).upsert(
        Model(name=name, provider=Provider.HUGGINGFACE, model_id="org/model", tokenizer_id="org/model")
    )
    session.commit()
    return model


def _experiment(model_id: object, name: str = "exp") -> Experiment:
    return Experiment(name=name, model_id=model_id, generation_config=_config())  # type: ignore[arg-type]


def _service(model_service: ModelService) -> ExperimentService:
    workflow = InferenceWorkflow(
        InferenceService(model_service, deployed_model_name="unused"),
        EvaluationService(None),
    )
    return ExperimentService(workflow)


def test_add_and_get_round_trips(db_session: Session) -> None:
    model = _persist_model(db_session)
    repo = ExperimentRepository(db_session)
    saved = repo.add(_experiment(model.id))
    db_session.commit()

    fetched = repo.get(saved.id)
    assert fetched is not None
    assert fetched.name == "exp"
    assert fetched.model_id == model.id
    assert fetched.generation_config == _config()


def test_get_returns_none_when_absent(db_session: Session) -> None:
    assert ExperimentRepository(db_session).get(uuid4()) is None


def test_aggregate_scores_averages_by_metric(db_session: Session) -> None:
    model = _persist_model(db_session)
    repo = ExperimentRepository(db_session)
    experiment = repo.add(_experiment(model.id))
    db_session.commit()

    inference_repo = InferenceRepository(db_session)
    eval_repo = EvaluationResultRepository(db_session)
    for score in (0.6, 0.8):
        inference = inference_repo.add(
            Inference(
                model_id=model.id,
                input_text="in",
                prompt="p",
                output_text="out",
                latency_ms=1,
                experiment_id=experiment.id,
            )
        )
        db_session.commit()
        eval_repo.upsert_many(
            [
                EvaluationResult(
                    inference_id=inference.id,
                    metric_name="faithfulness",
                    score=score,
                    evaluator_name="summary-faithfulness",
                )
            ]
        )
        db_session.commit()

    aggregates = repo.aggregate_scores(experiment.id)

    assert len(aggregates) == 1
    assert aggregates[0].metric_name == "faithfulness"
    assert aggregates[0].evaluated_count == 2
    assert aggregates[0].average_score == pytest.approx(0.7)


def test_create_rejects_unknown_model(db_session: Session, fake_model_service: ModelService) -> None:
    service = _service(fake_model_service)
    with pytest.raises(ModelNotFoundError):
        service.create(db_session, _experiment(uuid4()))


def test_run_tags_inference_with_experiment_id(db_session: Session, fake_model_service: ModelService) -> None:
    model = _persist_model(db_session)
    service = _service(fake_model_service)
    experiment = service.create(db_session, _experiment(model.id))

    result = service.run(db_session, experiment.id, "summarize this")

    assert result.evaluation is None
    assert result.inference.experiment_id == experiment.id
    stored = InferenceRepository(db_session).get(result.inference.id)
    assert stored is not None
    assert stored.experiment_id == experiment.id


def test_get_by_name_round_trips(db_session: Session) -> None:
    model = _persist_model(db_session)
    repo = ExperimentRepository(db_session)
    repo.add(_experiment(model.id, name="named"))
    db_session.commit()

    fetched = repo.get_by_name("named")

    assert fetched is not None
    assert fetched.name == "named"
    assert repo.get_by_name("absent") is None


def test_create_rejects_duplicate_name(db_session: Session, fake_model_service: ModelService) -> None:
    model = _persist_model(db_session)
    service = _service(fake_model_service)
    service.create(db_session, _experiment(model.id, name="dup"))

    with pytest.raises(ExperimentNameConflictError):
        service.create(db_session, _experiment(model.id, name="dup"))


def test_results_rejects_unknown_experiment(db_session: Session, fake_model_service: ModelService) -> None:
    service = _service(fake_model_service)

    with pytest.raises(ExperimentNotFoundError):
        service.results(db_session, uuid4())


def test_compare_rejects_unknown_experiment(db_session: Session, fake_model_service: ModelService) -> None:
    model = _persist_model(db_session)
    service = _service(fake_model_service)
    known = service.create(db_session, _experiment(model.id, name="known"))

    with pytest.raises(ExperimentNotFoundError):
        service.compare(db_session, known.id, uuid4())


def test_run_uses_the_experiment_generation_config(db_session: Session, settings: Settings) -> None:
    model = _persist_model(db_session)
    capturing = _CapturingModelService(settings)
    service = _service(capturing)
    config = GenerationConfig(max_input_tokens=128, max_new_tokens=32, num_beams=3)
    experiment = service.create(db_session, Experiment(name="cfg", model_id=model.id, generation_config=config))

    service.run(db_session, experiment.id, "summarize this")

    assert capturing.configs == [config]


def test_compare_with_the_same_id_returns_two_entries(db_session: Session, fake_model_service: ModelService) -> None:
    model = _persist_model(db_session)
    service = _service(fake_model_service)
    experiment = service.create(db_session, _experiment(model.id, name="solo"))

    comparison = service.compare(db_session, experiment.id, experiment.id)

    assert [result.experiment_id for result in comparison] == [experiment.id, experiment.id]


def test_add_reraises_foreign_key_violation(db_session: Session) -> None:
    # A non-existent model_id violates the RESTRICT foreign key, not the name
    # unique constraint: it must surface as IntegrityError, not a name conflict.
    repo = ExperimentRepository(db_session)

    with pytest.raises(IntegrityError):
        repo.add(_experiment(uuid4(), name="orphan"))


def test_get_raises_on_corrupt_stored_generation_config(db_session: Session) -> None:
    model = _persist_model(db_session)
    db_session.add(
        ExperimentRecord(
            id=uuid4(),
            name="corrupt",
            model_id=model.id,
            generation_config={"temperature": 1},
        )
    )
    db_session.commit()

    with pytest.raises(CorruptStoredDataError):
        ExperimentRepository(db_session).get_by_name("corrupt")
