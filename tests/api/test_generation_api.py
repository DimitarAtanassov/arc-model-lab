from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from arc_model_lab.config import Settings
from arc_model_lab.domain.generation_params import REGISTRY
from arc_model_lab.main import create_app

# GET /generation/params reads only settings, so it needs no database. The app is
# built directly with a settings override to assert the effective cap is reported.

_CAP = 4096


@pytest.fixture
async def params_client() -> AsyncIterator[AsyncClient]:
    app = create_app(Settings(max_output_tokens_cap=_CAP))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_returns_200(params_client: AsyncClient) -> None:
    assert (await params_client.get("/generation/params")).status_code == 200


async def test_reports_the_effective_cap_from_settings(params_client: AsyncClient) -> None:
    # The cap is the configured runtime value, not a static registry constant.
    body = (await params_client.get("/generation/params")).json()
    assert body["max_output_tokens_cap"] == _CAP


async def test_serves_the_full_registry(params_client: AsyncClient) -> None:
    body = (await params_client.get("/generation/params")).json()
    names = {param["name"] for param in body["params"]}
    assert names == {spec.name for spec in REGISTRY}


async def test_each_param_carries_its_descriptor(params_client: AsyncClient) -> None:
    body = (await params_client.get("/generation/params")).json()
    param = next(p for p in body["params"] if p["name"] == "temperature")
    assert param == {
        "name": "temperature",
        "kind": "float",
        "minimum": 0.0,
        "maximum": 2.0,
        "default": 0.0,
        "tier": "core",
        "group": "sampling",
    }


async def test_max_output_tokens_reports_no_static_ceiling(params_client: AsyncClient) -> None:
    # Its ceiling is the cap reported alongside, so the static maximum is null.
    body = (await params_client.get("/generation/params")).json()
    param = next(p for p in body["params"] if p["name"] == "max_output_tokens")
    assert param["minimum"] == 1
    assert param["maximum"] is None
    assert param["default"] == 256


async def test_stop_default_serializes_as_empty_list(params_client: AsyncClient) -> None:
    body = (await params_client.get("/generation/params")).json()
    param = next(p for p in body["params"] if p["name"] == "stop")
    assert param["kind"] == "str_list"
    assert param["default"] == []
