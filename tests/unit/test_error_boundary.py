"""Unit test for the catch-all error boundary: safe 500 body plus correlation id."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_inference_workflow, get_session
from arc_model_lab.main import create_app


class _BoomWorkflow:
    """A workflow double whose run always fails with a non-domain error."""

    def run(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("unexpected")


def _fake_session() -> Iterator[Session]:
    yield MagicMock(spec=Session)


def test_unhandled_error_returns_500_with_correlation_id() -> None:
    app = create_app()
    app.dependency_overrides[get_session] = _fake_session
    app.dependency_overrides[get_inference_workflow] = _BoomWorkflow

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/inference", json={"input_text": "hi"})

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert body["correlation_id"]
