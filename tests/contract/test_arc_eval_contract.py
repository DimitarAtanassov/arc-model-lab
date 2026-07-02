"""Consumer contract tests for the arc-eval ``/v1/evaluate`` boundary.

arc-model-lab cannot import the provider package, so the provider's schema is
encoded here as a fixture. If arc-eval changes its request or response contract,
these tests fail and surface the break before production does.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arc_model_lab.clients.arc_eval_client import (
    CONTRACT_VERSION,
    EvalMetadata,
    EvalRequest,
    EvalResponse,
)

pytestmark = pytest.mark.contract


def test_request_serializes_to_arc_eval_schema() -> None:
    request = EvalRequest(
        task_type="summarization",
        input_text="source",
        output_text="summary",
        prompt="rendered",
        metadata=EvalMetadata(inference_id="i-1", model_id="m-1"),
    )

    payload = request.model_dump(mode="json")

    assert set(payload) == {"task_type", "input_text", "output_text", "prompt", "metrics", "metadata"}
    assert isinstance(payload["task_type"], str)
    assert isinstance(payload["input_text"], str)
    assert isinstance(payload["output_text"], str)
    assert payload["prompt"] is None or isinstance(payload["prompt"], str)
    assert payload["metrics"] is None or isinstance(payload["metrics"], list)
    assert payload["metadata"] == {"inference_id": "i-1", "model_id": "m-1"}


def test_request_carries_optional_metrics() -> None:
    payload = EvalRequest(
        task_type="summarization",
        input_text="source",
        output_text="summary",
        metrics=["faithfulness"],
    ).model_dump(mode="json")

    assert payload["metrics"] == ["faithfulness"]


def test_request_requires_output_text() -> None:
    with pytest.raises(ValidationError):
        EvalRequest.model_validate({"task_type": "summarization", "input_text": "source"})


def test_response_parses_provider_payload() -> None:
    payload = {
        "results": [
            {
                "metric_name": "faithfulness",
                "score": 0.91,
                "reasoning": "grounded in the source",
                "evaluator_name": "summary-faithfulness",
                "evaluator_version": "v1",
            },
            {
                "metric_name": "answer_relevance",
                "score": 0.8,
                "reasoning": None,
                "evaluator_name": "summary-answer-relevance",
                "evaluator_version": None,
            },
        ]
    }

    response = EvalResponse.model_validate(payload)

    assert [result.metric_name for result in response.results] == ["faithfulness", "answer_relevance"]
    assert response.results[0].score == 0.91


def test_response_allows_empty_results() -> None:
    response = EvalResponse.model_validate({"results": []})

    assert response.results == []


def test_response_exposes_provider_contract_version() -> None:
    response = EvalResponse.model_validate({"contract_version": "1.0.0", "results": []})

    assert response.contract_version == "1.0.0"


def test_response_without_contract_version_still_parses() -> None:
    # An older provider that omits the field must not break the consumer.
    assert EvalResponse.model_validate({"results": []}).contract_version is None


def test_client_version_matches_the_provider_contract() -> None:
    # arc-eval emits this version today. Bump CONTRACT_VERSION in lockstep when
    # the wire contract changes so drift fails a test instead of a request.
    provider_version = "1.0.0"
    assert CONTRACT_VERSION == provider_version


@pytest.mark.parametrize("missing", ["metric_name", "score", "evaluator_name"])
def test_response_requires_core_metric_fields(missing: str) -> None:
    metric = {
        "metric_name": "faithfulness",
        "score": 0.9,
        "evaluator_name": "summary-faithfulness",
    }
    del metric[missing]

    with pytest.raises(ValidationError):
        EvalResponse.model_validate({"results": [metric]})
