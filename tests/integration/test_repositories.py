"""Repository round-trips against a real Postgres.

Covers the mapping in both directions and the found/absent branches that the
API-level tests do not exercise.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import InferenceRepository, ModelRepository
from arc_model_lab.domain import Inference, Model, ModelStatus, Provider

pytestmark = pytest.mark.integration


def _model(name: str, *, model_id: str = "org/model", status: ModelStatus = ModelStatus.ACTIVE) -> Model:
    return Model(name=name, provider=Provider.HUGGINGFACE, model_id=model_id, tokenizer_id=model_id, status=status)


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


def test_inference_get_returns_none_when_absent(db_session: Session) -> None:
    assert InferenceRepository(db_session).get(uuid4()) is None
