"""API behavior for POST /inference."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

_MODEL = "test-model"


def _body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {"model_name": _MODEL, "input_text": "summarize me"}
    body.update(overrides)
    return body


def test_valid_request_returns_201(client: TestClient) -> None:
    response = client.post("/inference", json=_body())
    assert response.status_code == 201


def test_response_omits_experiment_id_and_evaluation(client: TestClient) -> None:
    # /inference is pure inference: no experiment id, no scores in the response.
    body = client.post("/inference", json=_body()).json()
    assert "experiment_id" not in body
    assert "evaluation" not in body


def test_empty_input_returns_422(client: TestClient) -> None:
    response = client.post("/inference", json=_body(input_text=""))
    assert response.status_code == 422


def test_missing_model_name_returns_422(client: TestClient) -> None:
    response = client.post("/inference", json={"input_text": "hi"})
    assert response.status_code == 422


def test_unknown_model_returns_404(client: TestClient) -> None:
    response = client.post("/inference", json=_body(model_name="does-not-exist"))
    assert response.status_code == 404


def test_temperature_out_of_range_returns_422(client: TestClient) -> None:
    response = client.post("/inference", json=_body(temperature=5.0))
    assert response.status_code == 422


def test_metrics_field_is_rejected(client: TestClient) -> None:
    # Evaluation moved out of /inference; a stale metrics field is forbidden.
    response = client.post("/inference", json=_body(metrics=["faithfulness"]))
    assert response.status_code == 422


def test_max_output_tokens_field_is_rejected(client: TestClient) -> None:
    # Output length is a server default now, not a caller knob on /inference.
    response = client.post("/inference", json=_body(max_output_tokens=128))
    assert response.status_code == 422


def test_oversized_input_returns_413(client: TestClient) -> None:
    response = client.post("/inference", json=_body(input_text="x" * 60_000))
    assert response.status_code == 413


def test_generation_failure_returns_500(failing_client: TestClient) -> None:
    response = failing_client.post("/inference", json=_body())
    assert response.status_code == 500


def test_model_load_failure_returns_503(model_load_failing_client: TestClient) -> None:
    response = model_load_failing_client.post("/inference", json=_body())
    assert response.status_code == 503
