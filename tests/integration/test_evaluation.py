"""Integration: POST /inference evaluates against a mocked arc-eval when metrics are given.

Exercises the full online path (inference -> eval call -> persistence -> response)
with a real Postgres and an ``httpx.MockTransport`` standing in for arc-eval.
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

_METRICS = ["faithfulness", "answer_relevance"]


def _scores_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["task_type"] == "summarization"
    assert body["metrics"] == _METRICS
    assert body["output_text"] == "fake summary"
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


def test_inference_with_metrics_persists_evaluation_results(make_client: ClientFactory, db_session: Session) -> None:
    client = make_client(_mock_client(_scores_handler))

    response = client.post("/inference", json={"input_text": "A long article.", "metrics": _METRICS})

    assert response.status_code == 201
    body = response.json()
    assert body["evaluation"]["status"] == "completed"
    metrics = {result["metric_name"]: result["score"] for result in body["evaluation"]["results"]}
    assert metrics == {"faithfulness": 0.91, "answer_relevance": 0.8}

    rows = db_session.execute(select(EvaluationResultRecord)).scalars().all()
    assert {row.metric_name for row in rows} == {"faithfulness", "answer_relevance"}
    assert all(row.inference_id == UUID(body["id"]) for row in rows)


def test_inference_without_metrics_does_not_evaluate(make_client: ClientFactory, db_session: Session) -> None:
    # Even with an eval client wired, omitting metrics skips evaluation entirely.
    client = make_client(_mock_client(_unexpected_call_handler))

    response = client.post("/inference", json={"input_text": "A long article."})

    assert response.status_code == 201
    assert response.json()["evaluation"] is None
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []


def test_inference_evaluation_fails_open_on_unavailable(make_client: ClientFactory, db_session: Session) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "down"})

    client = make_client(_mock_client(unavailable))

    response = client.post("/inference", json={"input_text": "hi", "metrics": ["faithfulness"]})

    assert response.status_code == 201
    assert response.json()["evaluation"]["status"] == "failed"
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []


def test_inference_skips_evaluation_without_eval_client(client: TestClient, db_session: Session) -> None:
    # Metrics requested, but no eval client is configured for this environment.
    response = client.post("/inference", json={"input_text": "hi", "metrics": ["faithfulness"]})

    assert response.status_code == 201
    assert response.json()["evaluation"]["status"] == "skipped"
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []


def test_unknown_metric_returns_404_and_keeps_the_inference(make_client: ClientFactory, db_session: Session) -> None:
    def unknown_metric(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "unknown metric 'nope'"})

    client = make_client(_mock_client(unknown_metric))

    response = client.post("/inference", json={"input_text": "hi", "metrics": ["nope"]})

    # arc-eval's 404 is surfaced to the caller, not failed open.
    assert response.status_code == 404
    assert response.json()["detail"] == "unknown metric 'nope'"
    # The inference itself succeeded and is persisted; only evaluation was rejected.
    assert len(db_session.execute(select(InferenceRecord)).scalars().all()) == 1
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []
