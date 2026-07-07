"""Unit tests for GenerationConfig validation (the experiment config boundary)."""

from __future__ import annotations

import pytest

from arc_model_lab.domain import GenerationConfig, InvalidGenerationConfigError

_VALID = {"temperature": 0.0, "max_output_tokens": 256}


def test_from_mapping_builds_config() -> None:
    assert GenerationConfig.from_mapping(_VALID) == GenerationConfig(temperature=0.0, max_output_tokens=256)


def test_from_mapping_rejects_unknown_key() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="Unknown"):
        GenerationConfig.from_mapping({**_VALID, "num_beams": 2})


def test_from_mapping_rejects_missing_key() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="Missing"):
        GenerationConfig.from_mapping({"temperature": 0.0})


@pytest.mark.parametrize("value", [0, -1])
def test_from_mapping_rejects_non_positive_max_output_tokens(value: int) -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_mapping({**_VALID, "max_output_tokens": value})


def test_from_mapping_rejects_non_integer_max_output_tokens() -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_mapping({**_VALID, "max_output_tokens": "big"})


def test_from_mapping_rejects_bool_max_output_tokens() -> None:
    # bool is an int subclass; reject it so max_output_tokens=True cannot slip through.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_mapping({**_VALID, "max_output_tokens": True})


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_from_mapping_rejects_out_of_range_temperature(value: float) -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_mapping({**_VALID, "temperature": value})


def test_from_mapping_rejects_bool_temperature() -> None:
    # bool is an int subclass; reject it so temperature=True cannot slip through.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_mapping({**_VALID, "temperature": True})


def test_to_dict_round_trips() -> None:
    config = GenerationConfig(temperature=0.7, max_output_tokens=128)
    assert GenerationConfig.from_mapping(config.to_dict()) == config


def test_defaults_are_greedy() -> None:
    # The one place the "default decoding" values live; schema and CLI derive from it.
    assert GenerationConfig() == GenerationConfig(temperature=0.0, max_output_tokens=256)


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
