from __future__ import annotations

import pytest
from pydantic import ValidationError

from arc_model_lab.api.schemas.inference import InferenceResponse, InferenceRunRequest

pytestmark = pytest.mark.contract

# The provider half of the /v1/inference:run contract. arc-eval-service defines
# the consumer half in its own tests/contract/test_lab_inference_contract.py;
# keep the two in lockstep so a field rename fails a test here instead of a
# production 422 mid-experiment.

# The exact body arc-eval-service's LabInferenceClient sends.
_CONSUMER_REQUEST_BODY = {
    "model_name": "candidate",
    "input_text": "source",
    "generation_config": {"temperature": 0.0, "max_output_tokens": 64},
    "allow_inactive": True,
}

# The exact fields arc-eval-service's InferenceResult reads back.
_CONSUMED_RESPONSE_FIELDS = {
    "id",
    "model_id",
    "input_text",
    "prompt",
    "output_text",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "created_at",
}


def test_run_request_accepts_the_consumer_body() -> None:
    request = InferenceRunRequest.model_validate(_CONSUMER_REQUEST_BODY)

    assert request.model_name == "candidate"
    assert request.allow_inactive is True
    config = request.generation_config.to_domain()
    assert (config.temperature, config.max_output_tokens) == (0.0, 64)


def test_run_request_rejects_an_unknown_field() -> None:
    # extra="forbid" turns contract drift into a boundary 422 rather than a
    # silently ignored field.
    with pytest.raises(ValidationError):
        InferenceRunRequest.model_validate({**_CONSUMER_REQUEST_BODY, "task_type": "summarize"})


def test_response_exposes_exactly_the_fields_the_consumer_reads() -> None:
    assert set(InferenceResponse.model_fields) == _CONSUMED_RESPONSE_FIELDS
