from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration

_UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"


def _body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "name": "balanced",
        "description": "mild sampling",
        "config": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_output_tokens": 512},
    }
    body.update(overrides)
    return body


async def _create(client: AsyncClient, **overrides: object) -> dict[str, object]:
    response = await client.post("/presets", json=_body(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


async def test_create_returns_201_with_resolved_config(client: AsyncClient) -> None:
    created = await _create(client)
    assert created["name"] == "balanced"
    assert created["status"] == "active"
    assert created["config"]["temperature"] == 0.7
    assert created["config"]["top_p"] == 0.9


async def test_create_unknown_knob_returns_422(client: AsyncClient) -> None:
    # config is the registry allow-list; an unknown knob is rejected at the boundary.
    response = await client.post("/presets", json=_body(config={"nonsense": 1}))
    assert response.status_code == 422


async def test_create_out_of_range_value_returns_422(client: AsyncClient) -> None:
    response = await client.post("/presets", json=_body(config={"do_sample": True, "temperature": 5.0}))
    assert response.status_code == 422


async def test_create_over_cap_output_tokens_returns_422(client: AsyncClient) -> None:
    # The server output cap (default 2048) is enforced before persistence.
    response = await client.post("/presets", json=_body(config={"max_output_tokens": 100_000}))
    assert response.status_code == 422


async def test_create_illegal_combination_returns_422(client: AsyncClient) -> None:
    # Beam search cannot combine with sampling parameters (cross-field rule).
    response = await client.post("/presets", json=_body(config={"num_beams": 4, "top_p": 0.9}))
    assert response.status_code == 422


async def test_create_duplicate_active_name_returns_409(client: AsyncClient) -> None:
    await _create(client, name="dup")
    response = await client.post("/presets", json=_body(name="dup"))
    assert response.status_code == 409


async def test_list_returns_active_presets(client: AsyncClient) -> None:
    await _create(client, name="a")
    await _create(client, name="b")

    response = await client.get("/presets")

    assert response.status_code == 200
    names = {item["name"] for item in response.json()}
    assert {"a", "b"} <= names


async def test_get_returns_one_preset(client: AsyncClient) -> None:
    created = await _create(client, name="fetch-me")
    response = await client.get(f"/presets/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


async def test_get_unknown_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/presets/{_UNKNOWN_ID}")
    assert response.status_code == 404


async def test_patch_updates_description_and_config(client: AsyncClient) -> None:
    created = await _create(client, name="patch-me")
    response = await client.patch(
        f"/presets/{created['id']}",
        json={"description": "updated", "config": {"do_sample": True, "temperature": 1.2}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["description"] == "updated"
    assert body["config"]["temperature"] == 1.2


async def test_patch_invalid_config_returns_422(client: AsyncClient) -> None:
    created = await _create(client, name="patch-bad")
    response = await client.patch(f"/presets/{created['id']}", json={"config": {"num_beams": 4, "top_p": 0.9}})
    assert response.status_code == 422


async def test_patch_unknown_returns_404(client: AsyncClient) -> None:
    response = await client.patch(f"/presets/{_UNKNOWN_ID}", json={"description": "x"})
    assert response.status_code == 404


async def test_archive_returns_204_and_hides_preset(client: AsyncClient) -> None:
    created = await _create(client, name="archive-me")

    archived = await client.delete(f"/presets/{created['id']}")
    assert archived.status_code == 204

    # An archived preset 404s from get and drops out of the active listing.
    assert (await client.get(f"/presets/{created['id']}")).status_code == 404
    names = {item["name"] for item in (await client.get("/presets")).json()}
    assert "archive-me" not in names


async def test_archive_unknown_returns_404(client: AsyncClient) -> None:
    response = await client.delete(f"/presets/{_UNKNOWN_ID}")
    assert response.status_code == 404


async def test_name_is_reusable_after_archive_via_api(client: AsyncClient) -> None:
    first = await _create(client, name="recyclable")
    assert (await client.delete(f"/presets/{first['id']}")).status_code == 204

    # The name is free again once the original is archived.
    revived = await _create(client, name="recyclable")
    assert revived["id"] != first["id"]
