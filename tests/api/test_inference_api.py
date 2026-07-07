"""API behavior for POST /inference."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from arc_model_lab.db.repositories import (
    EvaluationResultRepository,
    InferenceRepository,
    ModelRepository,
)
from arc_model_lab.domain import EvaluationResult, Inference, Model, ModelStatus, Provider

pytestmark = pytest.mark.integration

_MODEL = "test-model"
_UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"


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


def test_explicit_temperature_is_accepted(client: TestClient) -> None:
    # Temperature is an optional caller override; a valid value is honored.
    response = client.post("/inference", json=_body(temperature=0.7))
    assert response.status_code == 201


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


def test_inactive_model_returns_409(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    # Deactivating a model takes it out of /inference serving (409), the safety lever.
    with session_factory() as session:
        ModelRepository(session).upsert(
            Model(
                name="disabled",
                provider=Provider.HUGGINGFACE,
                model_id="x/y",
                tokenizer_id="x/y",
                status=ModelStatus.INACTIVE,
            )
        )
        session.commit()

    response = client.post("/inference", json=_body(model_name="disabled"))
    assert response.status_code == 409


def _persist_inference(
    session_factory: sessionmaker[Session],
    *,
    input_text: str = "summarize me",
    output_text: str = "fake summary",
) -> Inference:
    """Insert an inference against the seeded model, bypassing the HTTP surface."""
    with session_factory() as session:
        model = ModelRepository(session).require_by_name(_MODEL)
        inference = InferenceRepository(session).add(
            Inference(
                model_id=model.id,
                input_text=input_text,
                prompt="p",
                output_text=output_text,
                latency_ms=5,
            )
        )
        session.commit()
        return inference


def test_list_inferences_returns_the_created_inference(client: TestClient) -> None:
    client.post("/inference", json=_body())

    response = client.get("/inference")

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["input_preview"] == "summarize me"
    assert items[0]["output_preview"] == "fake summary"


def test_list_inference_preview_truncates_long_text(client: TestClient, session_factory: sessionmaker[Session]) -> None:
    _persist_inference(session_factory, input_text="word " * 100)

    preview: str = client.get("/inference").json()[0]["input_preview"]

    assert preview.endswith("\u2026")
    assert len(preview) <= 160


def test_get_inference_detail_includes_evaluation_scores(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    inference = _persist_inference(session_factory)
    with session_factory() as session:
        EvaluationResultRepository(session).upsert_many(
            [
                EvaluationResult(
                    inference_id=inference.id,
                    metric_name="faithfulness",
                    score=0.75,
                    evaluator_name="summary-faithfulness",
                    evaluator_version="v1",
                )
            ]
        )
        session.commit()

    body = client.get(f"/inference/{inference.id}").json()

    assert body["output_text"] == "fake summary"
    assert body["evaluations"] == [
        {
            "metric_name": "faithfulness",
            "score": 0.75,
            "reasoning": None,
            "evaluator_name": "summary-faithfulness",
            "evaluator_version": "v1",
            "created_at": body["evaluations"][0]["created_at"],
        }
    ]


def test_get_unknown_inference_returns_404(client: TestClient) -> None:
    assert client.get(f"/inference/{_UNKNOWN_ID}").status_code == 404
