from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import (
    EvaluationResultRepository,
    InferenceRepository,
    ModelRepository,
)
from arc_model_lab.domain import (
    EvaluationResult,
    Inference,
    Model,
    ModelNotFoundError,
    ModelStatus,
    Provider,
)

pytestmark = pytest.mark.integration


def _model(name: str, *, model_id: str = "org/model", status: ModelStatus = ModelStatus.ACTIVE) -> Model:
    return Model(name=name, provider=Provider.HUGGINGFACE, model_id=model_id, tokenizer_id=model_id, status=status)


async def _persist_inference(session: AsyncSession, *, model_name: str = "m") -> Inference:
    model = await ModelRepository(session).upsert(_model(model_name))
    inference = await InferenceRepository(session).add(
        Inference(model_id=model.id, input_text="in", prompt="p", output_text="out", latency_ms=5)
    )
    await session.commit()
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


async def test_get_by_name_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await ModelRepository(db_session).get_by_name("ghost") is None


async def test_require_by_id_returns_model_or_raises(db_session: AsyncSession) -> None:
    repo = ModelRepository(db_session)
    model = await repo.upsert(_model("present"))
    await db_session.commit()

    assert (await repo.require_by_id(model.id)).id == model.id
    with pytest.raises(ModelNotFoundError):
        await repo.require_by_id(uuid4())


async def test_list_all_returns_models_ordered_by_name(db_session: AsyncSession) -> None:
    repo = ModelRepository(db_session)
    await repo.add(_model("beta"))
    await repo.add(_model("alpha"))

    assert [m.name for m in await repo.list_all()] == ["alpha", "beta"]


async def test_upsert_inserts_then_updates_without_duplicating(db_session: AsyncSession) -> None:
    repo = ModelRepository(db_session)
    await repo.upsert(_model("m", model_id="first/id"))

    updated = await repo.upsert(_model("m", model_id="second/id", status=ModelStatus.INACTIVE))

    assert updated.model_id == "second/id"
    assert updated.status is ModelStatus.INACTIVE
    assert len(await repo.list_all()) == 1


async def test_set_status_updates_existing_model(db_session: AsyncSession) -> None:
    repo = ModelRepository(db_session)
    await repo.add(_model("m", status=ModelStatus.ACTIVE))

    updated = await repo.set_status("m", ModelStatus.DEPRECATED)

    assert updated is not None
    assert updated.status is ModelStatus.DEPRECATED


async def test_set_status_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await ModelRepository(db_session).set_status("ghost", ModelStatus.INACTIVE) is None


async def test_inference_add_and_get_round_trip(db_session: AsyncSession) -> None:
    await ModelRepository(db_session).add(_model("m"))
    model = await ModelRepository(db_session).get_by_name("m")
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
    await repo.add(inference)

    fetched = await repo.get(inference.id)
    assert fetched is not None
    assert fetched.model_id == model.id
    assert fetched.output_text == "out"


async def test_upsert_many_is_idempotent_on_the_metric_key(db_session: AsyncSession) -> None:
    inference = await _persist_inference(db_session)
    repo = EvaluationResultRepository(db_session)

    await repo.upsert_many([_result(inference.id, score=0.5, version="v1")])
    await db_session.commit()
    await repo.upsert_many([_result(inference.id, score=0.9, version="v2")])
    await db_session.commit()

    stored = await repo.list_for_inference(inference.id)
    assert len(stored) == 1
    assert stored[0].score == 0.9
    assert stored[0].evaluator_version == "v2"


async def test_upsert_many_with_no_results_is_a_noop(db_session: AsyncSession) -> None:
    inference = await _persist_inference(db_session)
    repo = EvaluationResultRepository(db_session)

    assert await repo.upsert_many([]) == []
    assert await repo.list_for_inference(inference.id) == []


async def test_list_unevaluated_returns_only_rows_without_results(db_session: AsyncSession) -> None:
    evaluated = await _persist_inference(db_session, model_name="m1")
    unevaluated = await _persist_inference(db_session, model_name="m2")
    await EvaluationResultRepository(db_session).upsert_many([_result(evaluated.id)])
    await db_session.commit()

    pending = await InferenceRepository(db_session).list_unevaluated(limit=10)

    assert [inference.id for inference in pending] == [unevaluated.id]


async def test_list_unevaluated_honors_the_created_before_window(db_session: AsyncSession) -> None:
    inference = await _persist_inference(db_session)

    before = await InferenceRepository(db_session).list_unevaluated(limit=10, created_before=inference.created_at)
    after = await InferenceRepository(db_session).list_unevaluated(limit=10, created_after=inference.created_at)

    assert before == []
    assert [row.id for row in after] == [inference.id]


async def test_inference_get_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await InferenceRepository(db_session).get(uuid4()) is None
