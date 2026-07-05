"""Integration: an experiment run evaluates against a mocked arc-eval.

Exercises the full experiment path (infer -> store -> eval call -> persistence ->
response) with a real Postgres and an ``httpx.MockTransport`` standing in for
arc-eval. ``/inference`` itself never evaluates; evaluation lives in the run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from arc_model_lab.clients.arc_eval_client import ArcEvalClient
from arc_model_lab.db.models import EvaluationResultRecord, InferenceRecord

pytestmark = pytest.mark.integration

ClientFactory = Callable[[ArcEvalClient | None], TestClient]

_MODEL = "test-model"
_METRICS = ["faithfulness", "answer_relevance"]


def _create_experiment(client: TestClient, *, name: str = "eval-exp") -> str:
    response = client.post("/experiments", json={"name": name, "model_name": _MODEL})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _run(client: TestClient, experiment_id: str, *, metrics: list[str] | None) -> httpx.Response:
    body: dict[str, object] = {"input_text": "A long article."}
    if metrics is not None:
        body["metrics"] = metrics
    return client.post(f"/experiments/{experiment_id}/run", json=body)


def _scores_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    # The contract: no task_type, explicit metrics, prompt, and correlation ids.
    assert "task_type" not in body
    assert body["metrics"] == _METRICS
    assert body["output_text"] == "fake summary"
    assert body["prompt"]
    assert body["metadata"]["inference_id"]
    return httpx.Response(
        200,
        json={
            "results": [
                {
                    "metric_name": "faithfulness",
                    "score": 0.91,
                    "reasoning": "ok",
                    "evaluator_name": "summary-faithfulness",
                    "evaluator_version": "v1",
                },
                {
                    "metric_name": "answer_relevance",
                    "score": 0.8,
                    "reasoning": None,
                    "evaluator_name": "summary-answer-relevance",
                    "evaluator_version": "v1",
                },
            ]
        },
    )


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> ArcEvalClient:
    return ArcEvalClient(httpx.Client(transport=httpx.MockTransport(handler), base_url="http://arc-eval.test"))


def _unexpected_call_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError("arc-eval must not be called when no metrics are requested")


def test_run_with_metrics_persists_evaluation_results(make_client: ClientFactory, db_session: Session) -> None:
    client = make_client(_mock_client(_scores_handler))
    experiment_id = _create_experiment(client)

    response = _run(client, experiment_id, metrics=_METRICS)

    assert response.status_code == 201
    body = response.json()
    assert body["experiment_id"] == experiment_id
    assert body["evaluation"]["status"] == "completed"
    metrics = {result["metric_name"]: result["score"] for result in body["evaluation"]["results"]}
    assert metrics == {"faithfulness": 0.91, "answer_relevance": 0.8}

    rows = db_session.execute(select(EvaluationResultRecord)).scalars().all()
    assert {row.metric_name for row in rows} == {"faithfulness", "answer_relevance"}
    assert all(row.inference_id == UUID(body["id"]) for row in rows)


def test_run_without_metrics_does_not_evaluate(make_client: ClientFactory, db_session: Session) -> None:
    # Even with an eval client wired, omitting metrics skips evaluation entirely.
    client = make_client(_mock_client(_unexpected_call_handler))
    experiment_id = _create_experiment(client)

    response = _run(client, experiment_id, metrics=None)

    assert response.status_code == 201
    assert response.json()["evaluation"] is None
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []


def test_run_evaluation_fails_open_on_unavailable(make_client: ClientFactory, db_session: Session) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "down"})

    client = make_client(_mock_client(unavailable))
    experiment_id = _create_experiment(client)

    response = _run(client, experiment_id, metrics=["faithfulness"])

    assert response.status_code == 201
    assert response.json()["evaluation"]["status"] == "failed"
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []


def test_run_skips_evaluation_without_eval_client(client: TestClient, db_session: Session) -> None:
    # Metrics requested, but no eval client is configured for this environment.
    experiment_id = _create_experiment(client)

    response = _run(client, experiment_id, metrics=["faithfulness"])

    assert response.status_code == 201
    assert response.json()["evaluation"]["status"] == "skipped"
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []


def test_unknown_metric_returns_404_and_keeps_the_inference(make_client: ClientFactory, db_session: Session) -> None:
    def unknown_metric(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "unknown metric 'nope'"})

    client = make_client(_mock_client(unknown_metric))
    experiment_id = _create_experiment(client)

    response = _run(client, experiment_id, metrics=["nope"])

    # arc-eval's 404 is surfaced to the caller, not failed open.
    assert response.status_code == 404
    assert response.json()["detail"] == "unknown metric 'nope'"
    # The inference itself succeeded and is persisted; only evaluation was rejected.
    assert len(db_session.execute(select(InferenceRecord)).scalars().all()) == 1
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []
