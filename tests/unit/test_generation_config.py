"""Unit tests for GenerationConfig validation (the experiment config boundary)."""

from __future__ import annotations

import pytest

from arc_model_lab.domain import GenerationConfig, InvalidGenerationConfigError

_VALID = {"max_input_tokens": 1024, "max_new_tokens": 256, "num_beams": 1}


def test_from_mapping_builds_config() -> None:
    assert GenerationConfig.from_mapping(_VALID) == GenerationConfig(
        max_input_tokens=1024, max_new_tokens=256, num_beams=1
    )


def test_from_mapping_rejects_unknown_key() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="Unknown"):
        GenerationConfig.from_mapping({**_VALID, "temperature": 0.2})


def test_from_mapping_rejects_missing_key() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="Missing"):
        GenerationConfig.from_mapping({"max_input_tokens": 1024, "max_new_tokens": 256})


@pytest.mark.parametrize("value", [0, -1])
def test_from_mapping_rejects_non_positive(value: int) -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_mapping({**_VALID, "num_beams": value})


def test_from_mapping_rejects_non_integer() -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_mapping({**_VALID, "max_input_tokens": "big"})


def test_from_mapping_rejects_bool() -> None:
    # bool is an int subclass; reject it so num_beams=True cannot slip through.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_mapping({**_VALID, "num_beams": True})


def test_to_dict_round_trips() -> None:
    config = GenerationConfig(max_input_tokens=512, max_new_tokens=128, num_beams=2)
    assert GenerationConfig.from_mapping(config.to_dict()) == config
