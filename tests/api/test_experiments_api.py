"""API behavior for the experiments endpoints.

These reuse the shared ``client`` fixture: ``get_experiment_service`` composes the
same overridden inference and evaluation services, so runs use the fake model
runtime and evaluation is disabled (no arc-eval needed).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

_UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"
_MODEL = "test-model"
_CONFIG = {"temperature": 0.0, "max_output_tokens": 32}


def _create(client: TestClient, *, name: str = "exp-a") -> dict[str, object]:
    response = client.post(
        "/experiments",
        json={"name": name, "model_name": _MODEL, "generation_config": _CONFIG},
    )
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def test_create_returns_201_and_echoes_config(client: TestClient) -> None:
    body = _create(client)

    assert body["name"] == "exp-a"
    assert body["model_name"] == _MODEL
    assert body["generation_config"] == _CONFIG
    # The internal model id is not leaked; the API speaks model names.
    assert "model_id" not in body
    assert "created_by" not in body


def test_create_rejects_unknown_generation_knob(client: TestClient) -> None:
    response = client.post(
        "/experiments",
        json={"name": "bad", "model_name": _MODEL, "generation_config": {"num_beams": 2}},
    )

    assert response.status_code == 422


def test_create_with_unknown_model_returns_404(client: TestClient) -> None:
    response = client.post("/experiments", json={"name": "ghost", "model_name": "does-not-exist"})
    assert response.status_code == 404


def test_get_unknown_experiment_returns_404(client: TestClient) -> None:
    assert client.get(f"/experiments/{_UNKNOWN_ID}").status_code == 404


def test_get_returns_the_experiment(client: TestClient) -> None:
    created = _create(client, name="fetch-me")

    body = client.get(f"/experiments/{created['id']}").json()

    assert body["name"] == "fetch-me"
    assert body["model_name"] == _MODEL


def test_run_returns_the_inference(client: TestClient) -> None:
    experiment = _create(client, name="run-exp")

    response = client.post(f"/experiments/{experiment['id']}/run", json={"input_text": "summarize me"})

    assert response.status_code == 201, response.text
    assert response.json()["output_text"] == "fake summary"


def test_run_unknown_experiment_returns_404(client: TestClient) -> None:
    response = client.post(f"/experiments/{_UNKNOWN_ID}/run", json={"input_text": "x"})
    assert response.status_code == 404


def test_results_are_empty_before_evaluation(client: TestClient) -> None:
    experiment = _create(client, name="results-exp")
    client.post(f"/experiments/{experiment['id']}/run", json={"input_text": "summarize me"})

    response = client.get(f"/experiments/{experiment['id']}/results")

    assert response.status_code == 200
    assert response.json() == {"experiment_id": experiment["id"], "metrics": []}


def test_results_unknown_experiment_returns_404(client: TestClient) -> None:
    response = client.get(f"/experiments/{_UNKNOWN_ID}/results")
    assert response.status_code == 404


def test_compare_unknown_experiment_returns_404(client: TestClient) -> None:
    known = _create(client, name="known")

    response = client.get(f"/experiments/{known['id']}/compare/{_UNKNOWN_ID}")

    assert response.status_code == 404


def test_compare_returns_both_experiments(client: TestClient) -> None:
    left = _create(client, name="left")
    right = _create(client, name="right")

    response = client.get(f"/experiments/{left['id']}/compare/{right['id']}")

    assert response.status_code == 200
    assert response.json() == {
        "experiments": [
            {"experiment_id": left["id"], "metrics": []},
            {"experiment_id": right["id"], "metrics": []},
        ]
    }


def test_create_duplicate_name_returns_409(client: TestClient) -> None:
    _create(client, name="dupe")

    response = client.post(
        "/experiments",
        json={"name": "dupe", "model_name": _MODEL, "generation_config": _CONFIG},
    )

    assert response.status_code == 409


def test_run_response_includes_experiment_id(client: TestClient) -> None:
    experiment = _create(client, name="tagged")

    response = client.post(f"/experiments/{experiment['id']}/run", json={"input_text": "summarize me"})

    assert response.status_code == 201, response.text
    assert response.json()["experiment_id"] == experiment["id"]


def test_compare_same_experiment_returns_two_entries(client: TestClient) -> None:
    experiment = _create(client, name="solo-api")

    response = client.get(f"/experiments/{experiment['id']}/compare/{experiment['id']}")

    assert response.status_code == 200
    experiments = response.json()["experiments"]
    assert [entry["experiment_id"] for entry in experiments] == [experiment["id"], experiment["id"]]
