from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from arc_model_lab.clients.arc_eval_client import (
    ArcEvalClient,
    EvalMetadata,
    EvalRequest,
    EvalSettings,
    _error_detail,
    build_arc_eval_client,
)
from arc_model_lab.domain import EvaluationError, UnknownMetricError

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
    return ArcEvalClient(httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://eval.test"))


def _request() -> EvalRequest:
    return EvalRequest(
        input_text="source",
        output_text="summary",
        prompt="rendered",
        metrics=["faithfulness"],
        metadata=EvalMetadata(inference_id="i-1", model_id="m-1"),
    )


async def test_evaluate_posts_to_v1_evaluate_and_parses_results() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_VALID_BODY)

    response = await _client(handler).evaluate(_request())

    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/v1/evaluate")
    assert captured["json"] == {  # type: ignore[comparison-overlap]
        "input_text": "source",
        "output_text": "summary",
        "prompt": "rendered",
        "metrics": ["faithfulness"],
        "metadata": {"inference_id": "i-1", "model_id": "m-1"},
    }
    assert len(response.results) == 1
    assert response.results[0].metric_name == "faithfulness"
    assert response.results[0].score == 0.91


async def test_evaluate_raises_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unavailable"})

    with pytest.raises(EvaluationError):
        await _client(handler).evaluate(_request())


async def test_evaluate_raises_unknown_metric_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "unknown metric 'nope'"})

    # A 404 is a caller error (the metric is not defined), distinct from the
    # fail-open EvaluationError, and it carries arc-eval's detail through.
    with pytest.raises(UnknownMetricError, match="unknown metric 'nope'"):
        await _client(handler).evaluate(_request())


async def test_evaluate_raises_unknown_metric_with_default_message_when_404_detail_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "no detail"})

    with pytest.raises(UnknownMetricError, match="requested metric is not defined"):
        await _client(handler).evaluate(_request())


async def test_evaluate_raises_unknown_metric_with_default_message_when_404_body_is_not_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not json")

    with pytest.raises(UnknownMetricError, match="requested metric is not defined"):
        await _client(handler).evaluate(_request())


def test_error_detail_returns_none_for_non_object_json_payload() -> None:
    response = httpx.Response(404, json=["not", "a", "dict"])

    assert _error_detail(response) is None


async def test_evaluate_raises_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(EvaluationError):
        await _client(handler).evaluate(_request())


async def test_evaluate_raises_on_non_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(EvaluationError):
        await _client(handler).evaluate(_request())


async def test_evaluate_raises_on_unexpected_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"score": 0.5}]})

    with pytest.raises(EvaluationError):
        await _client(handler).evaluate(_request())


def test_build_arc_eval_client_returns_none_without_service_url() -> None:
    assert build_arc_eval_client(EvalSettings(service_url="")) is None


async def test_build_arc_eval_client_creates_client_when_configured() -> None:
    client = build_arc_eval_client(EvalSettings(service_url="http://eval.test", timeout_seconds=12.5))

    assert client is not None
    assert client._http.base_url == httpx.URL("http://eval.test")
    assert client._http.timeout.connect == 12.5
    assert client._http.timeout.read == 12.5
    assert client._http.timeout.write == 12.5
    assert client._http.timeout.pool == 12.5

    await client.aclose()


async def test_aclose_closes_underlying_http_client() -> None:
    client = ArcEvalClient(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_VALID_BODY)))
    )

    await client.aclose()
    assert client._http.is_closed
