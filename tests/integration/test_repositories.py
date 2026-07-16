from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import InferenceRepository, ModelRepository
from arc_model_lab.domain import (
    GenerationConfig,
    Inference,
    Model,
    ModelNotFoundError,
    ModelStatus,
    Provider,
)

pytestmark = pytest.mark.integration


def _model(name: str, *, model_id: str = "org/model", status: ModelStatus = ModelStatus.ACTIVE) -> Model:
    return Model(name=name, provider=Provider.HUGGINGFACE, model_id=model_id, tokenizer_id=model_id, status=status)


async def test_get_by_name_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await ModelRepository(db_session).get_by_name("ghost") is None


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
    # Distinct created_at so the ordering is deterministic and actually asserted;
    # rows written in one transaction would otherwise share now() and tie.
    older = Inference(
        model_id=model.id,
        input_text="a",
        prompt="p",
        output_text="o",
        latency_ms=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = Inference(
        model_id=model.id,
        input_text="b",
        prompt="p",
        output_text="o",
        latency_ms=1,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    await repo.add(older)
    await repo.add(newer)
    await db_session.commit()

    recent = await repo.list_recent(10)

    assert [inference.id for inference in recent] == [newer.id, older.id]


async def test_inference_get_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await InferenceRepository(db_session).get(uuid4()) is None


async def test_inference_persists_generation_config(db_session: AsyncSession) -> None:
    # The resolved decoding config is durable: it survives the JSONB round trip so
    # a stored inference reproduces the call that made it.
    model = await ModelRepository(db_session).upsert(_model("m"))
    repo = InferenceRepository(db_session)
    inference = Inference(
        model_id=model.id,
        input_text="in",
        prompt="p",
        output_text="out",
        latency_ms=5,
        generation_config=GenerationConfig(temperature=0.7, max_output_tokens=128),
    )
    await repo.add(inference)

    fetched = await repo.get(inference.id)
    assert fetched is not None
    assert fetched.generation_config == GenerationConfig(temperature=0.7, max_output_tokens=128)
