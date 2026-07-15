from __future__ import annotations

from uuid import UUID, uuid4

from arc_model_lab.domain import Inference, Model, Provider


def test_model_generates_id_and_aware_timestamp() -> None:
    model = Model(
        name="m",
        provider=Provider.HUGGINGFACE,
        model_id="org/model",
        tokenizer_id="org/model",
    )

    assert isinstance(model.id, UUID)
    assert model.created_at.tzinfo is not None


def test_each_inference_gets_a_unique_id() -> None:
    model_id = uuid4()
    first = Inference(model_id=model_id, input_text="in", prompt="p", output_text="out", latency_ms=1)
    second = Inference(model_id=model_id, input_text="in", prompt="p", output_text="out", latency_ms=1)

    assert first.id != second.id
