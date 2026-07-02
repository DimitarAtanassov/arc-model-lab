"""Integration: POST /summarize?evaluate=true against a mocked arc-eval.

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

from arc_model_lab.db.models import EvaluationResultRecord
from arc_model_lab.services.arc_eval_client import ArcEvalClient

pytestmark = pytest.mark.integration

ClientFactory = Callable[[ArcEvalClient | None], TestClient]


def _scores_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["task_type"] == "summarization"
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


def test_summarize_with_evaluate_persists_results(make_client: ClientFactory, db_session: Session) -> None:
    client = make_client(_mock_client(_scores_handler))

    response = client.post("/summarize", json={"input_text": "A long article.", "evaluate": True})

    assert response.status_code == 201
    body = response.json()
    assert body["evaluation"]["status"] == "completed"
    metrics = {result["metric_name"]: result["score"] for result in body["evaluation"]["results"]}
    assert metrics == {"faithfulness": 0.91, "answer_relevance": 0.8}

    rows = db_session.execute(select(EvaluationResultRecord)).scalars().all()
    assert {row.metric_name for row in rows} == {"faithfulness", "answer_relevance"}
    assert all(row.inference_id == UUID(body["id"]) for row in rows)


def test_summarize_with_evaluate_fails_open_on_unavailable(make_client: ClientFactory, db_session: Session) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "down"})

    client = make_client(_mock_client(unavailable))

    response = client.post("/summarize", json={"input_text": "hi", "evaluate": True})

    assert response.status_code == 201
    assert response.json()["evaluation"]["status"] == "failed"
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []


def test_summarize_with_evaluate_but_no_client_is_skipped(client: TestClient) -> None:
    response = client.post("/summarize", json={"input_text": "hi", "evaluate": True})

    assert response.status_code == 201
    assert response.json()["evaluation"]["status"] == "skipped"


def test_summarize_without_evaluate_omits_evaluation(client: TestClient, db_session: Session) -> None:
    response = client.post("/summarize", json={"input_text": "hi"})

    assert response.status_code == 201
    assert response.json()["evaluation"] is None
    assert db_session.execute(select(EvaluationResultRecord)).scalars().all() == []
