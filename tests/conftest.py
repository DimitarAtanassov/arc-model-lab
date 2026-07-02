"""Shared test fixtures and model-runtime fakes.

The real model runtime is never loaded in tests: ``FakeModelService`` overrides
``load``/``generate`` so CI never downloads weights. A real Postgres runs via
testcontainers for anything that touches the database.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from arc_model_lab.api.dependencies import get_evaluation_service, get_inference_service
from arc_model_lab.config import Settings
from arc_model_lab.db.base import Base, create_engine_from_url, create_session_factory
from arc_model_lab.db.models import EvaluationResultRecord, InferenceRecord, ModelRecord
from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import GenerationError, Model, ModelLoadError, Provider
from arc_model_lab.main import create_app
from arc_model_lab.services.arc_eval_client import ArcEvalClient
from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ChatMessage, GenerationResult, ModelService

_TEST_MODEL_NAME = "test-model"


class FakeModelService(ModelService):
    """Model-runtime double: never loads weights, returns deterministic output."""

    def generate(self, model: Model, messages: list[ChatMessage]) -> GenerationResult:
        return GenerationResult(
            prompt="fake-prompt",
            output_text="fake summary",
            prompt_tokens=3,
            completion_tokens=2,
            latency_ms=1,
        )


class FailingModelService(FakeModelService):
    """Model-runtime double whose generation always fails."""

    def generate(self, model: Model, messages: list[ChatMessage]) -> GenerationResult:
        raise GenerationError("boom")


class ModelLoadFailingModelService(FakeModelService):
    """Model-runtime double whose weights never load (maps to HTTP 503)."""

    def generate(self, model: Model, messages: list[ChatMessage]) -> GenerationResult:
        raise ModelLoadError("model temporarily unavailable")


def _test_model() -> Model:
    return Model(
        name=_TEST_MODEL_NAME,
        provider=Provider.HUGGINGFACE,
        model_id="test/model",
        tokenizer_id="test/model",
    )


def build_app(
    model_service: ModelService,
    session_factory: sessionmaker[Session],
    *,
    eval_client: ArcEvalClient | None = None,
) -> FastAPI:
    """Build an app wired to a test session factory and a fake model runtime."""
    app = create_app()
    app.state.session_factory = session_factory
    with session_factory() as session:
        ModelRepository(session).upsert(_test_model())
        session.commit()
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(model_service, _TEST_MODEL_NAME)
    app.dependency_overrides[get_evaluation_service] = lambda: EvaluationService(eval_client)
    return app


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def fake_model_service(settings: Settings) -> FakeModelService:
    return FakeModelService(settings)


@pytest.fixture(scope="session")
def _postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16", driver="psycopg") as container:
        yield container.get_connection_url()


@pytest.fixture(scope="session")
def engine(_postgres_url: str) -> Iterator[Engine]:
    eng = create_engine_from_url(_postgres_url)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return create_session_factory(engine)


@pytest.fixture(autouse=True)
def _isolate_db(request: pytest.FixtureRequest) -> Iterator[None]:
    """Truncate tables before any test that touches the database.

    Cleaning at setup (rather than teardown) avoids a fixture-finalization
    ordering bug: this autouse fixture is set up before ``session_factory`` and
    therefore torn down after it, so ``session_factory`` is no longer resolvable
    from a post-yield finalizer. Because this fixture is autouse it still runs
    before the ``client`` fixture seeds the test model, so each DB test starts
    from a clean slate.
    """
    if "session_factory" in request.fixturenames:
        factory: sessionmaker[Session] = request.getfixturevalue("session_factory")
        with factory() as session:
            session.execute(delete(EvaluationResultRecord))
            session.execute(delete(InferenceRecord))
            session.execute(delete(ModelRecord))
            session.commit()
    return


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        yield session


@pytest.fixture
def client(fake_model_service: FakeModelService, session_factory: sessionmaker[Session]) -> TestClient:
    return TestClient(build_app(fake_model_service, session_factory))


@pytest.fixture
def make_client(
    fake_model_service: FakeModelService, session_factory: sessionmaker[Session]
) -> Callable[[ArcEvalClient | None], TestClient]:
    """Return a factory that builds a TestClient wired to a given eval client."""

    def _make(eval_client: ArcEvalClient | None = None) -> TestClient:
        return TestClient(build_app(fake_model_service, session_factory, eval_client=eval_client))

    return _make


@pytest.fixture
def failing_client(settings: Settings, session_factory: sessionmaker[Session]) -> TestClient:
    return TestClient(build_app(FailingModelService(settings), session_factory))


@pytest.fixture
def model_load_failing_client(settings: Settings, session_factory: sessionmaker[Session]) -> TestClient:
    return TestClient(build_app(ModelLoadFailingModelService(settings), session_factory))
