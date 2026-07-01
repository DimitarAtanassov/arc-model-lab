"""Composition root: lifespan wiring and the console entrypoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from arc_model_lab import main as main_module
from arc_model_lab.config import Settings, get_settings
from arc_model_lab.main import create_app, run
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ModelService


def test_lifespan_populates_application_state() -> None:
    app = create_app()

    with TestClient(app):
        assert app.state.engine is not None
        assert isinstance(app.state.session_factory, sessionmaker)
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
