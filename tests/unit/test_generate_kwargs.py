from __future__ import annotations

from arc_model_lab.domain import GenerationConfig
from arc_model_lab.domain.generation import to_generate_kwargs

# to_generate_kwargs is a pure mapper: no model, no tokenizer. Each test asserts
# that a config emits exactly the generate kwargs valid for its derived mode, and
# nothing from the other modes.

_SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p"}
_BEAM_KEYS = {"num_beams", "length_penalty", "early_stopping"}


def test_greedy_emits_only_length_and_do_sample_false() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(max_output_tokens=64))

    assert kwargs == {"max_new_tokens": 64, "do_sample": False}


def test_greedy_carries_no_sampling_or_beam_keys() -> None:
    kwargs = to_generate_kwargs(GenerationConfig())

    assert not (_SAMPLING_KEYS | _BEAM_KEYS) & kwargs.keys()


def test_sampling_via_temperature_emits_temperature() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(temperature=0.7, max_output_tokens=32))

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_new_tokens"] == 32


def test_sampling_emits_only_set_sampling_params() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(do_sample=True, top_p=0.9, top_k=40))

    assert kwargs["do_sample"] is True
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 40
    # Unset sampling params are omitted, and temperature 0 is not forwarded.
    assert "temperature" not in kwargs
    assert "min_p" not in kwargs


def test_sampling_emits_min_p_when_set() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(do_sample=True, min_p=0.05))

    assert kwargs["min_p"] == 0.05


def test_sampling_omits_zero_temperature() -> None:
    # do_sample=true with temperature 0 samples, but the 0 is the transformers
    # default and is not worth forwarding.
    kwargs = to_generate_kwargs(GenerationConfig(do_sample=True, top_p=0.9))

    assert kwargs["do_sample"] is True
    assert "temperature" not in kwargs


def test_beam_emits_beam_params_and_no_sampling() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(num_beams=4, length_penalty=1.5, early_stopping=True))

    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 4
    assert kwargs["length_penalty"] == 1.5
    assert kwargs["early_stopping"] is True
    assert not _SAMPLING_KEYS & kwargs.keys()


def test_beam_omits_unset_beam_params() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(num_beams=2))

    assert kwargs["num_beams"] == 2
    assert "length_penalty" not in kwargs
    assert "early_stopping" not in kwargs


def test_repetition_controls_emit_in_any_mode() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(repetition_penalty=1.2, no_repeat_ngram_size=3))

    assert kwargs["repetition_penalty"] == 1.2
    assert kwargs["no_repeat_ngram_size"] == 3


def test_min_new_tokens_is_emitted_when_set() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(min_new_tokens=8, max_output_tokens=64))

    assert kwargs["min_new_tokens"] == 8


def test_stop_maps_to_stop_strings() -> None:
    kwargs = to_generate_kwargs(GenerationConfig(stop=["END", "STOP"]))

    assert kwargs["stop_strings"] == ["END", "STOP"]


def test_empty_stop_emits_no_stop_strings() -> None:
    assert "stop_strings" not in to_generate_kwargs(GenerationConfig())
