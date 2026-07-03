"""API behavior for POST /inference."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_valid_request_returns_201(client: TestClient) -> None:
    response = client.post("/inference", json={"input_text": "summarize me"})
    assert response.status_code == 201


def test_empty_input_returns_422(client: TestClient) -> None:
    response = client.post("/inference", json={"input_text": ""})
    assert response.status_code == 422


def test_model_name_field_is_rejected(client: TestClient) -> None:
    # The caller cannot pick a model; a stale model_name is forbidden, not ignored.
    response = client.post("/inference", json={"input_text": "hi", "model_name": "anything"})
    assert response.status_code == 422


def test_oversized_input_returns_413(client: TestClient) -> None:
    response = client.post("/inference", json={"input_text": "x" * 60_000})
    assert response.status_code == 413


def test_generation_failure_returns_500(failing_client: TestClient) -> None:
    response = failing_client.post("/inference", json={"input_text": "hi"})
    assert response.status_code == 500


def test_model_load_failure_returns_503(model_load_failing_client: TestClient) -> None:
    response = model_load_failing_client.post("/inference", json={"input_text": "hi"})
    assert response.status_code == 503
