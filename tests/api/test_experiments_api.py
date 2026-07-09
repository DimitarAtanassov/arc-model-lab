"""API behavior for the experiments endpoints.

These reuse the shared ``client`` fixture: ``get_experiment_service`` composes the
same overridden inference and evaluation services, so runs use the fake model
runtime and evaluation is disabled (no arc-eval needed).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration

_UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"
_MODEL = "test-model"
_CONFIG = {"temperature": 0.0, "max_output_tokens": 32}


async def _create(client: AsyncClient, *, name: str = "exp-a") -> dict[str, object]:
    response = await client.post(
        "/experiments",
        json={"name": name, "model_name": _MODEL, "generation_config": _CONFIG},
    )
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


async def test_create_returns_201_and_echoes_config(client: AsyncClient) -> None:
    body = await _create(client)

    assert body["name"] == "exp-a"
    assert body["model_name"] == _MODEL
    assert body["generation_config"] == _CONFIG
    # model_id is present on every response, coherent with the inference-shaped
    # responses; model_name is the friendly config identity alongside it.
    assert body["model_id"]
    assert "created_by" not in body
    assert "prompt_version_id" not in body


async def test_create_rejects_unknown_generation_knob(client: AsyncClient) -> None:
    response = await client.post(
        "/experiments",
        json={"name": "bad", "model_name": _MODEL, "generation_config": {"num_beams": 2}},
    )

    assert response.status_code == 422


async def test_create_with_unknown_model_returns_404(client: AsyncClient) -> None:
    response = await client.post("/experiments", json={"name": "ghost", "model_name": "does-not-exist"})
    assert response.status_code == 404


async def test_get_unknown_experiment_returns_404(client: AsyncClient) -> None:
    assert (await client.get(f"/experiments/{_UNKNOWN_ID}")).status_code == 404


async def test_get_returns_the_experiment(client: AsyncClient) -> None:
    created = await _create(client, name="fetch-me")

    body = (await client.get(f"/experiments/{created['id']}")).json()

    assert body["name"] == "fetch-me"
    assert body["model_name"] == _MODEL


async def test_run_returns_the_inference(client: AsyncClient) -> None:
    experiment = await _create(client, name="run-exp")

    response = await client.post(f"/experiments/{experiment['id']}/run", json={"input_text": "summarize me"})

    assert response.status_code == 201, response.text
    assert response.json()["output_text"] == "fake summary"


async def test_run_unknown_experiment_returns_404(client: AsyncClient) -> None:
    response = await client.post(f"/experiments/{_UNKNOWN_ID}/run", json={"input_text": "x"})
    assert response.status_code == 404


async def test_results_are_empty_before_evaluation(client: AsyncClient) -> None:
    experiment = await _create(client, name="results-exp")
    await client.post(f"/experiments/{experiment['id']}/run", json={"input_text": "summarize me"})

    response = await client.get(f"/experiments/{experiment['id']}/results")

    assert response.status_code == 200
    assert response.json() == {"experiment_id": experiment["id"], "metrics": []}


async def test_results_unknown_experiment_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/experiments/{_UNKNOWN_ID}/results")
    assert response.status_code == 404


async def test_compare_unknown_experiment_returns_404(client: AsyncClient) -> None:
    known = await _create(client, name="known")

    response = await client.get(f"/experiments/{known['id']}/compare/{_UNKNOWN_ID}")

    assert response.status_code == 404


async def test_compare_returns_both_experiments(client: AsyncClient) -> None:
    left = await _create(client, name="left")
    right = await _create(client, name="right")

    response = await client.get(f"/experiments/{left['id']}/compare/{right['id']}")

    assert response.status_code == 200
    assert response.json() == {
        "experiments": [
            {"experiment_id": left["id"], "metrics": []},
            {"experiment_id": right["id"], "metrics": []},
        ]
    }


async def test_create_duplicate_name_returns_409(client: AsyncClient) -> None:
    await _create(client, name="dupe")

    response = await client.post(
        "/experiments",
        json={"name": "dupe", "model_name": _MODEL, "generation_config": _CONFIG},
    )

    assert response.status_code == 409


async def test_run_response_includes_experiment_id(client: AsyncClient) -> None:
    experiment = await _create(client, name="tagged")

    response = await client.post(f"/experiments/{experiment['id']}/run", json={"input_text": "summarize me"})

    assert response.status_code == 201, response.text
    assert response.json()["experiment_id"] == experiment["id"]


async def test_list_returns_created_experiments_with_model_names(client: AsyncClient) -> None:
    await _create(client, name="exp-1")
    await _create(client, name="exp-2")

    response = await client.get("/experiments")

    assert response.status_code == 200
    entries = response.json()
    assert {entry["name"] for entry in entries} == {"exp-1", "exp-2"}
    assert all(entry["model_name"] == _MODEL for entry in entries)
