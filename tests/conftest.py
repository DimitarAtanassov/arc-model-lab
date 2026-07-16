from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from arc_model_lab.api.dependencies import get_inference_service
from arc_model_lab.config import Settings
from arc_model_lab.db import models as _models  # noqa: F401 - register ORM tables on Base.metadata
from arc_model_lab.db.base import Base, create_async_engine_from_url, create_async_session_factory
from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import GenerationConfig, GenerationError, Model, ModelLoadError, Provider
from arc_model_lab.main import create_app
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_catalog_service import ModelCatalogService
from arc_model_lab.services.model_service import ChatMessage, GenerationResult, ModelService
from arc_model_lab.services.preset_service import PresetService

_TEST_MODEL_NAME = "test-model"
# Truncated together under CASCADE, so foreign-key order does not matter.
_TABLES = ("inference", "models", "generation_preset")


class FakeModelService(ModelService):
    """Model-runtime double: never loads weights, returns deterministic output."""

    def generate(
        self, model: Model, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> GenerationResult:
        return GenerationResult(
            prompt="fake-prompt",
            output_text="fake summary",
            prompt_tokens=3,
            completion_tokens=2,
            latency_ms=1,
        )


class FailingModelService(FakeModelService):
    """Model-runtime double whose generation always fails."""

    def generate(
        self, model: Model, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> GenerationResult:
        raise GenerationError("boom")


class ModelLoadFailingModelService(FakeModelService):
    """Model-runtime double whose weights never load (maps to HTTP 503)."""

    def generate(
        self, model: Model, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> GenerationResult:
        raise ModelLoadError("model temporarily unavailable")


def _test_model() -> Model:
    return Model(
        name=_TEST_MODEL_NAME,
        provider=Provider.HUGGINGFACE,
        model_id="test/model",
        tokenizer_id="test/model",
    )


def _free_port() -> int:
    """Reserve an ephemeral TCP port for the local Postgres cluster."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _start_testcontainer() -> tuple[str, Callable[[], None]] | None:
    """Start Postgres in a container, or return None when unavailable.

    The canonical CI path. Returns None when the library is missing, the
    Docker daemon is down, or the image registry is blocked. Set
    ARC_SKIP_TESTCONTAINER=1 to skip the attempt outright in known-blocked
    environments and fall straight through to the local cluster.
    """
    if os.environ.get("ARC_SKIP_TESTCONTAINER"):
        return None
    try:
        from testcontainers.postgres import PostgresContainer  # noqa: PLC0415 - optional, lazy for graceful fallback
    except ImportError:
        return None
    try:
        container = PostgresContainer("postgres:16", driver="psycopg")
        container.start()
    except Exception:  # noqa: BLE001 - any container failure (daemon down, registry blocked)
        return None
    return container.get_connection_url(), container.stop


def _start_local_postgres() -> tuple[str, Callable[[], None]] | None:
    """Start an ephemeral cluster with on-PATH initdb/pg_ctl, or None.

    A dev fallback for machines where the container registry is blocked. Real
    Postgres in a temp dir on a random port, discarded at session end; None
    when the binaries are not installed.
    """
    initdb = shutil.which("initdb")
    pg_ctl = shutil.which("pg_ctl")
    if initdb is None or pg_ctl is None:
        return None

    root = Path(tempfile.mkdtemp(prefix="arc-model-lab-pg-"))
    data_dir = root / "data"
    port = _free_port()
    try:
        subprocess.run(  # noqa: S603 - fixed argv, executable resolved from PATH
            [initdb, "-D", str(data_dir), "-U", "postgres", "--auth=trust", "-N"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(  # noqa: S603 - fixed argv, executable resolved from PATH
            [
                pg_ctl,
                "-D",
                str(data_dir),
                "-w",
                "-l",
                str(root / "log"),
                "-o",
                f"-p {port} -k {root} -h 127.0.0.1",
                "start",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(root, ignore_errors=True)
        return None

    def _stop() -> None:
        subprocess.run(  # noqa: S603 - fixed argv, executable resolved from PATH
            [pg_ctl, "-D", str(data_dir), "-m", "immediate", "-w", "stop"],
            check=False,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(root, ignore_errors=True)

    return f"postgresql+psycopg://postgres@127.0.0.1:{port}/postgres", _stop


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A Postgres URL with the schema created, for the whole session.

    Prefers a testcontainer (CI); falls back to a local cluster when the registry
    is blocked. Schema creation runs on a throwaway sync engine.
    """
    provider = _start_testcontainer() or _start_local_postgres()
    if provider is None:
        pytest.skip("no Postgres available (container registry blocked, no local initdb)")
    url, stop = provider
    sync_engine = create_engine(url)
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()
    try:
        yield url
    finally:
        stop()


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    """A function-scoped async engine on the shared database (per-test event loop)."""
    eng = create_async_engine_from_url(database_url)
    yield eng
    await eng.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_async_session_factory(engine)


@pytest.fixture(autouse=True)
def _isolate_db(request: pytest.FixtureRequest) -> None:
    """Truncate all tables before any DB-touching test.

    Runs on a throwaway sync engine (not the async session_factory) so it has
    no event-loop or fixture-finalization coupling and simply resets state before
    each test that pulls in the database.
    """
    if "session_factory" not in request.fixturenames:
        return
    url = request.getfixturevalue("database_url")
    sync_engine = create_engine(url)
    with sync_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(_TABLES)} CASCADE"))
    sync_engine.dispose()


@pytest.fixture
async def db_session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


async def build_app(
    model_service: ModelService,
    session_factory: async_sessionmaker[AsyncSession],
) -> FastAPI:
    """Build an app wired to a test session factory and a fake model runtime.

    The catalog is seeded with one active model (_TEST_MODEL_NAME); an
    /inference request names it. A request that names an absent model gets a
    404 from the model lookup.
    """
    app = create_app()
    app.state.session_factory = session_factory
    async with session_factory() as session:
        await ModelRepository(session).upsert(_test_model())
        await session.commit()
    preset_service = PresetService(app.state.settings.max_output_tokens_cap)
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(
        model_service, preset_service, app.state.settings.max_output_tokens_cap
    )
    # The catalog read service has no I/O seam to fake, so wire the real one into
    # app state the way lifespan does; the read endpoints resolve it from there.
    app.state.model_catalog_service = ModelCatalogService()
    app.state.preset_service = preset_service
    return app


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def fake_model_service(settings: Settings) -> FakeModelService:
    return FakeModelService(settings)


@pytest.fixture
async def client(
    fake_model_service: FakeModelService, session_factory: async_sessionmaker[AsyncSession]
) -> AsyncIterator[AsyncClient]:
    app = await build_app(fake_model_service, session_factory)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
        yield async_client


@pytest.fixture
async def failing_client(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession]
) -> AsyncIterator[AsyncClient]:
    app = await build_app(FailingModelService(settings), session_factory)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
        yield async_client


@pytest.fixture
async def model_load_failing_client(
    settings: Settings, session_factory: async_sessionmaker[AsyncSession]
) -> AsyncIterator[AsyncClient]:
    app = await build_app(ModelLoadFailingModelService(settings), session_factory)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
        yield async_client
