from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arc_model_lab.db.repositories import InferenceRepository, ModelRepository
from arc_model_lab.domain import Inference, Model, ModelStatus, Provider

pytestmark = pytest.mark.integration

_MODEL = "test-model"
_UNKNOWN_ID = "00000000-0000-0000-0000-000000000000"


def _body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {"model_name": _MODEL, "input_text": "summarize me"}
    body.update(overrides)
    return body


async def _inactive_model(session_factory: async_sessionmaker[AsyncSession], name: str) -> None:
    async with session_factory() as session:
        await ModelRepository(session).upsert(
            Model(
                name=name,
                provider=Provider.HUGGINGFACE,
                model_id="x/y",
                tokenizer_id="x/y",
                status=ModelStatus.INACTIVE,
            )
        )
        await session.commit()


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


async def test_top_level_temperature_is_rejected(client: AsyncClient) -> None:
    # Regression: temperature is no longer a top-level field; it lives only inside
    # model_params. A legacy top-level temperature is an unknown field (422).
    response = await client.post("/inference", json=_body(temperature=0.7))
    assert response.status_code == 422


async def test_model_params_temperature_is_accepted(client: AsyncClient) -> None:
    response = await client.post("/inference", json=_body(model_params={"temperature": 0.7}))
    assert response.status_code == 201


async def test_model_params_temperature_out_of_range_returns_422(client: AsyncClient) -> None:
    response = await client.post("/inference", json=_body(model_params={"temperature": 5.0}))
    assert response.status_code == 422


async def test_model_params_unknown_knob_returns_422(client: AsyncClient) -> None:
    # model_params is the registry allow-list; an unknown knob is a 422.
    response = await client.post("/inference", json=_body(model_params={"nonsense": 1}))
    assert response.status_code == 422


async def test_model_params_illegal_combination_returns_422(client: AsyncClient) -> None:
    # Beam search cannot combine with sampling params; the merged config is a 422.
    response = await client.post("/inference", json=_body(model_params={"num_beams": 2, "top_p": 0.9}))
    assert response.status_code == 422


async def test_unknown_preset_returns_404(client: AsyncClient) -> None:
    response = await client.post("/inference", json=_body(preset_id=_UNKNOWN_ID))
    assert response.status_code == 404


async def test_preset_seeds_config_and_is_recorded_on_the_response(client: AsyncClient) -> None:
    created = await client.post(
        "/presets",
        json={"name": "balanced", "config": {"do_sample": True, "temperature": 0.8}},
    )
    assert created.status_code == 201
    preset_id = created.json()["id"]

    response = await client.post("/inference", json=_body(preset_id=preset_id))

    assert response.status_code == 201
    body = response.json()
    # The response carries the resolved config and the informing preset id.
    assert body["preset_id"] == preset_id
    assert body["generation_config"]["temperature"] == 0.8
    assert body["generation_config"]["do_sample"] is True


async def test_model_params_win_over_preset_in_the_response(client: AsyncClient) -> None:
    created = await client.post(
        "/presets",
        json={"name": "seed", "config": {"do_sample": True, "temperature": 0.8, "top_p": 0.9}},
    )
    preset_id = created.json()["id"]

    body = (await client.post("/inference", json=_body(preset_id=preset_id, model_params={"temperature": 1.2}))).json()

    # Override beats the preset on temperature; the preset's other knob is inherited.
    assert body["generation_config"]["temperature"] == 1.2
    assert body["generation_config"]["top_p"] == 0.9


async def test_metrics_field_is_rejected(client: AsyncClient) -> None:
    # Evaluation moved out of the lab; a stale metrics field is forbidden.
    response = await client.post("/inference", json=_body(metrics=["faithfulness"]))
    assert response.status_code == 422


async def test_max_output_tokens_field_is_rejected(client: AsyncClient) -> None:
    # Output length is a server default on /inference, not a caller knob.
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
    await _inactive_model(session_factory, "disabled")
    response = await client.post("/inference", json=_body(model_name="disabled"))
    assert response.status_code == 409


async def test_prompt_template_field_is_rejected(client: AsyncClient) -> None:
    # Prompt templating moved out of the lab; a stale prompt_template is forbidden.
    response = await client.post("/inference", json=_body(prompt_template="summarize"))
    assert response.status_code == 422


async def test_variables_field_is_rejected(client: AsyncClient) -> None:
    # variables only made sense with a template; the field no longer exists.
    response = await client.post("/inference", json=_body(variables={"target_language": "French"}))
    assert response.status_code == 422


async def test_service_to_service_run_endpoint_is_gone(client: AsyncClient) -> None:
    # The /v1/inference:run seam had a single consumer (arc-eval-service) that no
    # longer calls the lab, so the endpoint was removed.
    response = await client.post("/v1/inference:run", json={"model_name": _MODEL, "input_text": "hi"})
    assert response.status_code == 404


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


async def test_get_inference_returns_the_inference(
    client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    inference = await _persist_inference(session_factory)

    body = (await client.get(f"/inference/{inference.id}")).json()

    assert body["id"] == str(inference.id)
    assert body["output_text"] == "fake summary"
    assert "evaluations" not in body


async def test_get_unknown_inference_returns_404(client: AsyncClient) -> None:
    assert (await client.get(f"/inference/{_UNKNOWN_ID}")).status_code == 404


async def test_resolved_config_and_preset_id_persist_on_the_row(
    client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    # Persist a preset, run an inference that layers an override on it, then read the
    # row straight from the DB to prove the resolved config and lineage link are stored.
    created = await client.post(
        "/presets",
        json={"name": "balanced", "config": {"do_sample": True, "temperature": 0.8, "top_p": 0.9}},
    )
    preset_id = created.json()["id"]

    body = (await client.post("/inference", json=_body(preset_id=preset_id, model_params={"temperature": 1.2}))).json()

    async with session_factory() as session:
        row = await InferenceRepository(session).get(UUID(body["id"]))

    assert row is not None
    assert row.preset_id == UUID(preset_id)
    # The row stores the resolved config (override > preset), not the raw inputs.
    assert row.generation_config.temperature == 1.2
    assert row.generation_config.top_p == 0.9
    assert row.generation_config.do_sample is True
