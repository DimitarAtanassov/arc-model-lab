from __future__ import annotations

import pytest

from arc_model_lab.domain import GenerationConfig, InvalidGenerationConfigError
from arc_model_lab.domain.generation import enforce_output_cap

_CAP = 2048


def test_within_cap_is_accepted() -> None:
    enforce_output_cap(GenerationConfig(max_output_tokens=_CAP), _CAP)


def test_over_cap_is_rejected() -> None:
    with pytest.raises(InvalidGenerationConfigError) as exc:
        enforce_output_cap(GenerationConfig(max_output_tokens=_CAP + 1), _CAP)
    # The 422 message names the field and the cap so the caller can fix it.
    assert "max_output_tokens" in str(exc.value)
    assert str(_CAP) in str(exc.value)


def test_cap_moves_with_the_setting() -> None:
    # The ceiling is server-authoritative, not a static registry value.
    low_cap = 128
    with pytest.raises(InvalidGenerationConfigError):
        enforce_output_cap(GenerationConfig(max_output_tokens=256), low_cap)
    enforce_output_cap(GenerationConfig(max_output_tokens=256), 4096)
