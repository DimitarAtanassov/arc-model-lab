"""API behavior for the model-catalog read endpoints.

The shared ``client`` fixture seeds one active model (``test-model``); these
tests read it back through the HTTP surface.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

_MODEL = "test-model"


def test_list_models_returns_the_seeded_model(client: TestClient) -> None:
    response = client.get("/models")

    assert response.status_code == 200
    names = [entry["name"] for entry in response.json()]
    assert names == [_MODEL]


def test_list_models_shapes_each_entry(client: TestClient) -> None:
    entry = client.get("/models").json()[0]

    assert entry["name"] == _MODEL
    assert entry["provider"] == "huggingface"
    assert entry["model_id"] == "test/model"
    assert entry["status"] == "active"


def test_get_model_by_name_returns_it(client: TestClient) -> None:
    response = client.get(f"/models/{_MODEL}")

    assert response.status_code == 200
    assert response.json()["name"] == _MODEL


def test_get_unknown_model_returns_404(client: TestClient) -> None:
    assert client.get("/models/does-not-exist").status_code == 404
