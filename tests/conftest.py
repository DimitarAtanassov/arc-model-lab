"""Shared test fixtures and model-runtime fakes.

The real model runtime is never loaded in tests: ``FakeModelService`` overrides
``load``/``generate`` so CI never downloads weights. A real Postgres runs via
testcontainers for anything that touches the database.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from arc_model_lab.api.dependencies import get_inference_service
from arc_model_lab.config import Settings
from arc_model_lab.db.base import Base, create_engine_from_url, create_session_factory
from arc_model_lab.db.models import InferenceRecord, ModelRecord
from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import GenerationError
from arc_model_lab.main import create_app
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ChatMessage, GenerationResult, ModelService


class FakeModelService(ModelService):
    """Model-runtime double: loads nothing, returns deterministic output."""

    def load(self) -> None:
        self._device = "cpu"

    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        return GenerationResult(
            prompt="fake-prompt",
            output_text="fake summary",
            prompt_tokens=3,
            completion_tokens=2,
            latency_ms=1,
        )


class FailingModelService(FakeModelService):
    """Model-runtime double whose generation always fails."""

    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        raise GenerationError("boom")


def build_app(model_service: ModelService, session_factory: sessionmaker[Session]) -> FastAPI:
    """Build an app wired to a test session factory and a fake model runtime."""
    app = create_app()
    app.state.session_factory = session_factory
    with session_factory() as session:
        ModelRepository(session).get_or_create(model_service.descriptor)
        session.commit()
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(model_service)
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
    """Truncate tables after any test that touched the database."""
    yield
    if "session_factory" not in request.fixturenames:
        return
    factory: sessionmaker[Session] = request.getfixturevalue("session_factory")
    with factory() as session:
        session.execute(delete(InferenceRecord))
        session.execute(delete(ModelRecord))
        session.commit()


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        yield session


@pytest.fixture
def client(fake_model_service: FakeModelService, session_factory: sessionmaker[Session]) -> TestClient:
    return TestClient(build_app(fake_model_service, session_factory))


@pytest.fixture
def failing_client(settings: Settings, session_factory: sessionmaker[Session]) -> TestClient:
    return TestClient(build_app(FailingModelService(settings), session_factory))
