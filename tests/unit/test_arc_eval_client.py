"""Unit tests for the arc-eval HTTP client and its failure modes."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from arc_model_lab.domain import EvaluationError
from arc_model_lab.services.arc_eval_client import ArcEvalClient, EvalMetadata, EvalRequest

_VALID_BODY = {
    "results": [
        {
            "metric_name": "faithfulness",
            "score": 0.91,
            "reasoning": "grounded",
            "evaluator_name": "summary-faithfulness",
            "evaluator_version": "v1",
        }
    ]
}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> ArcEvalClient:
    return ArcEvalClient(httpx.Client(transport=httpx.MockTransport(handler), base_url="http://eval.test"))


def _request() -> EvalRequest:
    return EvalRequest(
        task_type="summarization",
        input_text="source",
        output_text="summary",
        prompt="rendered",
        metadata=EvalMetadata(inference_id="i-1", model_id="m-1"),
    )


def test_evaluate_posts_to_v1_evaluate_and_parses_results() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_VALID_BODY)

    response = _client(handler).evaluate(_request())

    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/v1/evaluate")
    assert captured["json"] == {  # type: ignore[comparison-overlap]
        "task_type": "summarization",
        "input_text": "source",
        "output_text": "summary",
        "prompt": "rendered",
        "metadata": {"inference_id": "i-1", "model_id": "m-1"},
    }
    assert len(response.results) == 1
    assert response.results[0].metric_name == "faithfulness"
    assert response.results[0].score == 0.91


def test_evaluate_raises_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unavailable"})

    with pytest.raises(EvaluationError):
        _client(handler).evaluate(_request())


def test_evaluate_raises_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(EvaluationError):
        _client(handler).evaluate(_request())


def test_evaluate_raises_on_non_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(EvaluationError):
        _client(handler).evaluate(_request())


def test_evaluate_raises_on_unexpected_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"score": 0.5}]})

    with pytest.raises(EvaluationError):
        _client(handler).evaluate(_request())
