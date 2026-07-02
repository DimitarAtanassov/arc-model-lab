"""Repository round-trips against a real Postgres.

Covers the mapping in both directions and the found/absent branches that the
API-level tests do not exercise.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import (
    EvaluationResultRepository,
    InferenceRepository,
    ModelRepository,
)
from arc_model_lab.domain import (
    EvaluationResult,
    Inference,
    Model,
    ModelStatus,
    Provider,
)

pytestmark = pytest.mark.integration


def _model(name: str, *, model_id: str = "org/model", status: ModelStatus = ModelStatus.ACTIVE) -> Model:
    return Model(name=name, provider=Provider.HUGGINGFACE, model_id=model_id, tokenizer_id=model_id, status=status)


def _persist_inference(session: Session, *, model_name: str = "m") -> Inference:
    model = ModelRepository(session).upsert(_model(model_name))
    inference = InferenceRepository(session).add(
        Inference(model_id=model.id, input_text="in", prompt="p", output_text="out", latency_ms=5)
    )
    session.commit()
    return inference


def _result(
    inference_id: UUID, *, metric: str = "faithfulness", score: float = 0.5, version: str = "v1"
) -> EvaluationResult:
    return EvaluationResult(
        inference_id=inference_id,
        metric_name=metric,
        score=score,
        evaluator_name="summary-faithfulness",
        evaluator_version=version,
    )


def test_get_by_name_returns_none_when_absent(db_session: Session) -> None:
    assert ModelRepository(db_session).get_by_name("ghost") is None


def test_list_all_returns_models_ordered_by_name(db_session: Session) -> None:
    repo = ModelRepository(db_session)
    repo.add(_model("beta"))
    repo.add(_model("alpha"))

    assert [m.name for m in repo.list_all()] == ["alpha", "beta"]


def test_upsert_inserts_then_updates_without_duplicating(db_session: Session) -> None:
    repo = ModelRepository(db_session)
    repo.upsert(_model("m", model_id="first/id"))

    updated = repo.upsert(_model("m", model_id="second/id", status=ModelStatus.INACTIVE))

    assert updated.model_id == "second/id"
    assert updated.status is ModelStatus.INACTIVE
    assert len(repo.list_all()) == 1


def test_set_status_updates_existing_model(db_session: Session) -> None:
    repo = ModelRepository(db_session)
    repo.add(_model("m", status=ModelStatus.ACTIVE))

    updated = repo.set_status("m", ModelStatus.DEPRECATED)

    assert updated is not None
    assert updated.status is ModelStatus.DEPRECATED


def test_set_status_returns_none_when_absent(db_session: Session) -> None:
    assert ModelRepository(db_session).set_status("ghost", ModelStatus.INACTIVE) is None


def test_inference_add_and_get_round_trip(db_session: Session) -> None:
    ModelRepository(db_session).add(_model("m"))
    model = ModelRepository(db_session).get_by_name("m")
    assert model is not None

    repo = InferenceRepository(db_session)
    inference = Inference(
        model_id=model.id,
        input_text="in",
        prompt="p",
        output_text="out",
        latency_ms=5,
        prompt_tokens=2,
        completion_tokens=3,
    )
    repo.add(inference)

    fetched = repo.get(inference.id)
    assert fetched is not None
    assert fetched.model_id == model.id
    assert fetched.output_text == "out"


def test_upsert_many_is_idempotent_on_the_metric_key(db_session: Session) -> None:
    inference = _persist_inference(db_session)
    repo = EvaluationResultRepository(db_session)

    repo.upsert_many([_result(inference.id, score=0.5, version="v1")])
    db_session.commit()
    repo.upsert_many([_result(inference.id, score=0.9, version="v2")])
    db_session.commit()

    stored = repo.list_for_inference(inference.id)
    assert len(stored) == 1
    assert stored[0].score == 0.9
    assert stored[0].evaluator_version == "v2"


def test_upsert_many_with_no_results_is_a_noop(db_session: Session) -> None:
    inference = _persist_inference(db_session)
    repo = EvaluationResultRepository(db_session)

    assert repo.upsert_many([]) == []
    assert repo.list_for_inference(inference.id) == []


def test_list_unevaluated_returns_only_rows_without_results(db_session: Session) -> None:
    evaluated = _persist_inference(db_session, model_name="m1")
    unevaluated = _persist_inference(db_session, model_name="m2")
    EvaluationResultRepository(db_session).upsert_many([_result(evaluated.id)])
    db_session.commit()

    pending = InferenceRepository(db_session).list_unevaluated(limit=10)

    assert [inference.id for inference in pending] == [unevaluated.id]


def test_list_unevaluated_honors_the_created_before_window(db_session: Session) -> None:
    inference = _persist_inference(db_session)

    before = InferenceRepository(db_session).list_unevaluated(limit=10, created_before=inference.created_at)
    after = InferenceRepository(db_session).list_unevaluated(limit=10, created_after=inference.created_at)

    assert before == []
    assert [row.id for row in after] == [inference.id]


def test_inference_get_returns_none_when_absent(db_session: Session) -> None:
    assert InferenceRepository(db_session).get(uuid4()) is None
