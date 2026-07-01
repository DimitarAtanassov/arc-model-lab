"""API behavior for POST /summarize."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_valid_request_returns_201(client: TestClient) -> None:
    response = client.post("/summarize", json={"input_text": "summarize me"})
    assert response.status_code == 201


def test_empty_input_returns_422(client: TestClient) -> None:
    response = client.post("/summarize", json={"input_text": ""})
    assert response.status_code == 422


def test_unknown_model_returns_404(client: TestClient) -> None:
    response = client.post("/summarize", json={"input_text": "hi", "model_name": "does-not-exist"})
    assert response.status_code == 404


def test_oversized_input_returns_413(client: TestClient) -> None:
    response = client.post("/summarize", json={"input_text": "x" * 60_000})
    assert response.status_code == 413


def test_generation_failure_returns_500(failing_client: TestClient) -> None:
    response = failing_client.post("/summarize", json={"input_text": "hi"})
    assert response.status_code == 500
