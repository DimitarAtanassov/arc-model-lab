from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration

_MODEL = "test-model"


async def test_list_models_returns_the_seeded_model(client: AsyncClient) -> None:
    response = await client.get("/models")

    assert response.status_code == 200
    names = [entry["name"] for entry in response.json()]
    assert names == [_MODEL]


async def test_list_models_shapes_each_entry(client: AsyncClient) -> None:
    entry = (await client.get("/models")).json()[0]

    assert entry["name"] == _MODEL
    assert entry["provider"] == "huggingface"
    assert entry["model_id"] == "test/model"
    assert entry["status"] == "active"


async def test_get_model_by_name_returns_it(client: AsyncClient) -> None:
    response = await client.get(f"/models/{_MODEL}")

    assert response.status_code == 200
    assert response.json()["name"] == _MODEL


async def test_get_unknown_model_returns_404(client: AsyncClient) -> None:
    assert (await client.get("/models/does-not-exist")).status_code == 404
