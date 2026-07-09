"""Integration: an experiment run evaluates against a mocked arc-eval.

Exercises the full experiment path (infer -> store -> eval call -> persistence ->
response) with a real Postgres and an ``httpx.MockTransport`` standing in for
arc-eval. ``/inference`` itself never evaluates; evaluation lives in the run.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from uuid import UUID

import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.clients.arc_eval_client import ArcEvalClient
from arc_model_lab.db.models import EvaluationResultRecord, InferenceRecord

pytestmark = pytest.mark.integration

ClientFactory = Callable[[ArcEvalClient | None], Awaitable[AsyncClient]]

_MODEL = "test-model"
_METRICS = ["faithfulness", "answer_relevance"]
_UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"


async def _create_inference(client: AsyncClient) -> str:
    """Create one inference via /inference and return its id."""
    response = await client.post("/inference", json={"model_name": _MODEL, "input_text": "A long article."})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _create_experiment(client: AsyncClient, *, name: str = "eval-exp") -> str:
    response = await client.post("/experiments", json={"name": name, "model_name": _MODEL})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _run(client: AsyncClient, experiment_id: str, *, metrics: list[str] | None) -> httpx.Response:
    body: dict[str, object] = {"input_text": "A long article."}
    if metrics is not None:
        body["metrics"] = metrics
    return await client.post(f"/experiments/{experiment_id}/run", json=body)


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
    return ArcEvalClient(httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://arc-eval.test"))


def _unexpected_call_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError("arc-eval must not be called when no metrics are requested")


async def test_run_with_metrics_persists_evaluation_results(
    make_client: ClientFactory, db_session: AsyncSession
) -> None:
    client = await make_client(_mock_client(_scores_handler))
    experiment_id = await _create_experiment(client)

    response = await _run(client, experiment_id, metrics=_METRICS)

    assert response.status_code == 201
    body = response.json()
    assert body["experiment_id"] == experiment_id
    assert body["evaluation"]["status"] == "completed"
    metrics = {result["metric_name"]: result["score"] for result in body["evaluation"]["results"]}
    assert metrics == {"faithfulness": 0.91, "answer_relevance": 0.8}

    rows = (await db_session.execute(select(EvaluationResultRecord))).scalars().all()
    assert {row.metric_name for row in rows} == {"faithfulness", "answer_relevance"}
    assert all(row.inference_id == UUID(body["id"]) for row in rows)


async def test_run_without_metrics_does_not_evaluate(make_client: ClientFactory, db_session: AsyncSession) -> None:
    # Even with an eval client wired, omitting metrics skips evaluation entirely.
    client = await make_client(_mock_client(_unexpected_call_handler))
    experiment_id = await _create_experiment(client)

    response = await _run(client, experiment_id, metrics=None)

    assert response.status_code == 201
    assert response.json()["evaluation"] is None
    assert (await db_session.execute(select(EvaluationResultRecord))).scalars().all() == []


async def test_run_evaluation_fails_open_on_unavailable(make_client: ClientFactory, db_session: AsyncSession) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "down"})

    client = await make_client(_mock_client(unavailable))
    experiment_id = await _create_experiment(client)

    response = await _run(client, experiment_id, metrics=["faithfulness"])

    assert response.status_code == 201
    assert response.json()["evaluation"]["status"] == "failed"
    assert (await db_session.execute(select(EvaluationResultRecord))).scalars().all() == []


async def test_run_skips_evaluation_without_eval_client(client: AsyncClient, db_session: AsyncSession) -> None:
    # Metrics requested, but no eval client is configured for this environment.
    experiment_id = await _create_experiment(client)

    response = await _run(client, experiment_id, metrics=["faithfulness"])

    assert response.status_code == 201
    assert response.json()["evaluation"]["status"] == "skipped"
    assert (await db_session.execute(select(EvaluationResultRecord))).scalars().all() == []


async def test_unknown_metric_returns_404_and_keeps_the_inference(
    make_client: ClientFactory, db_session: AsyncSession
) -> None:
    def unknown_metric(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "unknown metric 'nope'"})

    client = await make_client(_mock_client(unknown_metric))
    experiment_id = await _create_experiment(client)

    response = await _run(client, experiment_id, metrics=["nope"])

    # arc-eval's 404 is surfaced to the caller, not failed open.
    assert response.status_code == 404
    assert response.json()["detail"] == "unknown metric 'nope'"
    # The inference itself succeeded and is persisted; only evaluation was rejected.
    assert len((await db_session.execute(select(InferenceRecord))).scalars().all()) == 1
    assert (await db_session.execute(select(EvaluationResultRecord))).scalars().all() == []


# --- Standalone evaluation endpoint: POST /inference/{id}/evaluate ------------
# Evaluation without an experiment: score an inference that already exists.


async def test_evaluate_endpoint_scores_existing_inference(
    make_client: ClientFactory, db_session: AsyncSession
) -> None:
    client = await make_client(_mock_client(_scores_handler))
    inference_id = await _create_inference(client)

    response = await client.post(f"/inference/{inference_id}/evaluate", json={"metrics": _METRICS})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    scores = {result["metric_name"]: result["score"] for result in body["results"]}
    assert scores == {"faithfulness": 0.91, "answer_relevance": 0.8}

    rows = (await db_session.execute(select(EvaluationResultRecord))).scalars().all()
    assert {row.metric_name for row in rows} == {"faithfulness", "answer_relevance"}
    assert all(row.inference_id == UUID(inference_id) for row in rows)


async def test_evaluate_endpoint_unknown_inference_returns_404(make_client: ClientFactory) -> None:
    # The eval client must not be called: a missing inference is a 404 before it.
    client = await make_client(_mock_client(_unexpected_call_handler))

    response = await client.post(f"/inference/{_UNKNOWN_ID}/evaluate", json={"metrics": ["faithfulness"]})

    assert response.status_code == 404


async def test_evaluate_endpoint_requires_at_least_one_metric(make_client: ClientFactory) -> None:
    client = await make_client(_mock_client(_unexpected_call_handler))
    inference_id = await _create_inference(client)

    response = await client.post(f"/inference/{inference_id}/evaluate", json={"metrics": []})

    assert response.status_code == 422


async def test_evaluate_endpoint_skips_without_eval_client(client: AsyncClient, db_session: AsyncSession) -> None:
    inference_id = await _create_inference(client)

    response = await client.post(f"/inference/{inference_id}/evaluate", json={"metrics": ["faithfulness"]})

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert (await db_session.execute(select(EvaluationResultRecord))).scalars().all() == []


async def test_evaluate_endpoint_unknown_metric_returns_404(
    make_client: ClientFactory, db_session: AsyncSession
) -> None:
    def unknown_metric(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "unknown metric 'nope'"})

    client = await make_client(_mock_client(unknown_metric))
    inference_id = await _create_inference(client)

    response = await client.post(f"/inference/{inference_id}/evaluate", json={"metrics": ["nope"]})

    assert response.status_code == 404
    assert response.json()["detail"] == "unknown metric 'nope'"
    assert (await db_session.execute(select(EvaluationResultRecord))).scalars().all() == []
