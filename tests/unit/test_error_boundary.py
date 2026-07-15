from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_inference_service, get_session
from arc_model_lab.domain import InvalidGenerationConfigError
from arc_model_lab.main import create_app


class _BoomService:
    """An inference-service double whose infer always fails non-domain."""

    def infer(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("unexpected")


def _fake_session() -> Iterator[Session]:
    yield MagicMock(spec=Session)


def test_unhandled_error_returns_500_with_correlation_id() -> None:
    app = create_app()
    app.dependency_overrides[get_session] = _fake_session
    app.dependency_overrides[get_inference_service] = _BoomService

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/inference", json={"model_name": "m", "input_text": "hi"})

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert body["correlation_id"]


def test_invalid_generation_config_without_message_uses_default_detail() -> None:
    app = create_app()

    @app.get("/_invalid-generation")
    def _invalid_generation() -> None:
        raise InvalidGenerationConfigError()

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/_invalid-generation")

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid generation config"}
