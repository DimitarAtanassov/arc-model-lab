"""Unit test for the catch-all error boundary: safe 500 body plus correlation id."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from arc_model_lab.api import errors as api_errors
from arc_model_lab.api.dependencies import get_inference_service, get_session
from arc_model_lab.domain import CorruptStoredDataError, InvalidGenerationConfigError
from arc_model_lab.main import create_app


class _BoomService:
    """An inference-service double whose summarize always fails non-domain."""

    def summarize(self, *args: object, **kwargs: object) -> object:
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


def test_corrupt_stored_data_returns_safe_500_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    logged: dict[str, object] = {}

    def _fake_error(message: str, *args: object, **kwargs: object) -> None:
        logged["message"] = message
        logged["kwargs"] = kwargs

    monkeypatch.setattr(api_errors.logger, "error", _fake_error)

    @app.get("/_corrupt")
    def _corrupt() -> None:
        raise CorruptStoredDataError("bad stored json")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/_corrupt")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert logged["message"] == "Corrupt stored data"
    kwargs = logged["kwargs"]
    assert isinstance(kwargs, dict)
    assert isinstance(kwargs["exc_info"], CorruptStoredDataError)
