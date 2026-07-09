from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from arc_model_lab import main as main_module
from arc_model_lab.config import Settings, get_settings
from arc_model_lab.main import create_app, run
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ModelService


def test_lifespan_populates_application_state() -> None:
    app = create_app()

    with TestClient(app):
        assert app.state.engine is not None
        assert isinstance(app.state.session_factory, async_sessionmaker)
        assert isinstance(app.state.model_service, ModelService)
        assert isinstance(app.state.inference_service, InferenceService)


def test_create_app_honors_injected_settings() -> None:
    settings = Settings()

    assert create_app(settings).state.settings is settings


def test_run_invokes_uvicorn_with_configured_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(target: str, *, host: str, port: int) -> None:
        captured.update(target=target, host=host, port=port)

    monkeypatch.setattr(main_module.uvicorn, "run", _fake_run)

    run()

    settings = get_settings()
    assert captured == {"target": "arc_model_lab.main:app", "host": settings.api_host, "port": settings.api_port}


def test_lifespan_closes_eval_client_and_disposes_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = False
    disposed = False

    class _DummyEvalClient:
        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    class _DummyEngine:
        async def dispose(self) -> None:
            nonlocal disposed
            disposed = True

    monkeypatch.setattr(main_module, "create_async_engine_from_url", lambda *args, **kwargs: _DummyEngine())
    monkeypatch.setattr(main_module, "create_async_session_factory", lambda engine: object())
    monkeypatch.setattr(main_module, "ModelService", lambda settings: object())
    monkeypatch.setattr(main_module, "build_arc_eval_client", lambda settings: _DummyEvalClient())
    monkeypatch.setattr(main_module, "InferenceService", lambda model_service: object())
    monkeypatch.setattr(main_module, "EvaluationService", lambda eval_client: object())

    app = create_app(Settings(database_url="postgresql://example/test"))

    async def _exercise() -> None:
        async with main_module.lifespan(app):
            pass

    asyncio.run(_exercise())

    assert closed is True
    assert disposed is True
