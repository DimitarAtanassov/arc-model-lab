from __future__ import annotations

import pytest

from arc_model_lab.domain import GenerationConfig, InvalidGenerationConfigError

# The constructor's __post_init__ is the single validation entry point: every
# construction path (the HTTP schema's to_domain, server defaults, dataclasses
# replace) funnels through it, so testing it directly covers them all.


def test_defaults_are_greedy() -> None:
    # The one place the "default decoding" values live; the schema derives from it.
    assert GenerationConfig() == GenerationConfig(temperature=0.0, max_output_tokens=256)


def test_temperature_is_coerced_to_float() -> None:
    # temperature=1 is stored as 1.0 so the annotation and stored value agree.
    assert GenerationConfig(temperature=1, max_output_tokens=256).temperature == 1.0


@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_max_output_tokens(value: int) -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=0.0, max_output_tokens=value)


def test_rejects_non_integer_max_output_tokens() -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=0.0, max_output_tokens="big")  # type: ignore[arg-type]


def test_rejects_bool_max_output_tokens() -> None:
    # bool is an int subclass; reject it so max_output_tokens=True cannot slip through.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=0.0, max_output_tokens=True)


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_rejects_out_of_range_temperature(value: float) -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=value, max_output_tokens=256)


def test_rejects_bool_temperature() -> None:
    # bool is an int subclass; reject it so temperature=True cannot slip through.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=True, max_output_tokens=256)


def test_rejects_non_numeric_temperature() -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature="hot", max_output_tokens=256)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_direct_construction_rejects_non_positive_max_output_tokens(value: int) -> None:
    # Guards the CLI path, which builds the config directly rather than via from_mapping.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=0.0, max_output_tokens=value)


def test_direct_construction_rejects_out_of_range_temperature() -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=2.5, max_output_tokens=256)


def test_int_temperature_is_normalized_to_float() -> None:
    # The float annotation is honored: an int temperature is coerced to a float,
    # so stored and serialized values stay consistent (1 -> 1.0).
    config = GenerationConfig(temperature=1, max_output_tokens=256)

    assert isinstance(config.temperature, float)
    assert config.temperature == 1.0
