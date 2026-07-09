from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import InferenceRepository, ModelRepository
from arc_model_lab.domain import Inference, Model, ModelNotFoundError, ModelStatus, Provider

pytestmark = pytest.mark.integration


def _model(name: str, *, model_id: str = "org/model", status: ModelStatus = ModelStatus.ACTIVE) -> Model:
    return Model(name=name, provider=Provider.HUGGINGFACE, model_id=model_id, tokenizer_id=model_id, status=status)


async def test_get_by_name_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await ModelRepository(db_session).get_by_name("ghost") is None


async def test_require_by_id_returns_model_or_raises(db_session: AsyncSession) -> None:
    repo = ModelRepository(db_session)
    model = await repo.upsert(_model("present"))
    await db_session.commit()

    assert (await repo.require_by_id(model.id)).id == model.id
    with pytest.raises(ModelNotFoundError):
        await repo.require_by_id(uuid4())


async def test_require_by_name_raises_when_absent(db_session: AsyncSession) -> None:
    with pytest.raises(ModelNotFoundError):
        await ModelRepository(db_session).require_by_name("ghost")


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


async def test_get_by_id_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await ModelRepository(db_session).get_by_id(uuid4()) is None


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


async def test_inference_list_recent_orders_newest_first(db_session: AsyncSession) -> None:
    model = await ModelRepository(db_session).upsert(_model("m"))
    repo = InferenceRepository(db_session)
    first = Inference(model_id=model.id, input_text="a", prompt="p", output_text="o", latency_ms=1)
    second = Inference(model_id=model.id, input_text="b", prompt="p", output_text="o", latency_ms=1)
    await repo.add(first)
    await repo.add(second)
    await db_session.commit()

    recent = await repo.list_recent(10)

    assert {inference.id for inference in recent} == {first.id, second.id}
    assert len(recent) == 2


async def test_inference_get_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await InferenceRepository(db_session).get(uuid4()) is None
