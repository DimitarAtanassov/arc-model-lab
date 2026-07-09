from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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


async def test_valid_request_returns_201(client: AsyncClient) -> None:
    response = await client.post("/inference", json=_body())
    assert response.status_code == 201


async def test_response_omits_experiment_id_and_evaluation(client: AsyncClient) -> None:
    # /inference is pure inference: no experiment id, no scores in the response.
    body = (await client.post("/inference", json=_body())).json()
    assert "experiment_id" not in body
    assert "evaluation" not in body


async def test_empty_input_returns_422(client: AsyncClient) -> None:
    response = await client.post("/inference", json=_body(input_text=""))
    assert response.status_code == 422


async def test_missing_model_name_returns_422(client: AsyncClient) -> None:
    response = await client.post("/inference", json={"input_text": "hi"})
    assert response.status_code == 422


async def test_unknown_model_returns_404(client: AsyncClient) -> None:
    response = await client.post("/inference", json=_body(model_name="does-not-exist"))
    assert response.status_code == 404


async def test_temperature_out_of_range_returns_422(client: AsyncClient) -> None:
    response = await client.post("/inference", json=_body(temperature=5.0))
    assert response.status_code == 422


async def test_explicit_temperature_is_accepted(client: AsyncClient) -> None:
    # Temperature is an optional caller override; a valid value is honored.
    response = await client.post("/inference", json=_body(temperature=0.7))
    assert response.status_code == 201


async def test_metrics_field_is_rejected(client: AsyncClient) -> None:
    # Evaluation moved out of /inference; a stale metrics field is forbidden.
    response = await client.post("/inference", json=_body(metrics=["faithfulness"]))
    assert response.status_code == 422


async def test_max_output_tokens_field_is_rejected(client: AsyncClient) -> None:
    # Output length is a server default now, not a caller knob on /inference.
    response = await client.post("/inference", json=_body(max_output_tokens=128))
    assert response.status_code == 422


async def test_oversized_input_returns_413(client: AsyncClient) -> None:
    response = await client.post("/inference", json=_body(input_text="x" * 60_000))
    assert response.status_code == 413


async def test_generation_failure_returns_500(failing_client: AsyncClient) -> None:
    response = await failing_client.post("/inference", json=_body())
    assert response.status_code == 500


async def test_model_load_failure_returns_503(model_load_failing_client: AsyncClient) -> None:
    response = await model_load_failing_client.post("/inference", json=_body())
    assert response.status_code == 503


async def test_inactive_model_returns_409(
    client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    # Deactivating a model takes it out of /inference serving (409), the safety lever.
    async with session_factory() as session:
        await ModelRepository(session).upsert(
            Model(
                name="disabled",
                provider=Provider.HUGGINGFACE,
                model_id="x/y",
                tokenizer_id="x/y",
                status=ModelStatus.INACTIVE,
            )
        )
        await session.commit()

    response = await client.post("/inference", json=_body(model_name="disabled"))
    assert response.status_code == 409


async def _persist_inference(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    input_text: str = "summarize me",
    output_text: str = "fake summary",
) -> Inference:
    """Insert an inference against the seeded model, bypassing the HTTP surface."""
    async with session_factory() as session:
        model = await ModelRepository(session).require_by_name(_MODEL)
        inference = await InferenceRepository(session).add(
            Inference(
                model_id=model.id,
                input_text=input_text,
                prompt="p",
                output_text=output_text,
                latency_ms=5,
            )
        )
        await session.commit()
        return inference


async def test_list_inferences_returns_the_created_inference(client: AsyncClient) -> None:
    await client.post("/inference", json=_body())

    response = await client.get("/inference")

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["input_preview"] == "summarize me"
    assert items[0]["output_preview"] == "fake summary"


async def test_list_inference_preview_truncates_long_text(
    client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await _persist_inference(session_factory, input_text="word " * 100)

    preview: str = (await client.get("/inference")).json()[0]["input_preview"]

    assert preview.endswith("\u2026")
    assert len(preview) <= 160


async def test_get_inference_detail_includes_evaluation_scores(
    client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    inference = await _persist_inference(session_factory)
    async with session_factory() as session:
        await EvaluationResultRepository(session).upsert_many(
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
        await session.commit()

    body = (await client.get(f"/inference/{inference.id}")).json()

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


async def test_get_unknown_inference_returns_404(client: AsyncClient) -> None:
    assert (await client.get(f"/inference/{_UNKNOWN_ID}")).status_code == 404
