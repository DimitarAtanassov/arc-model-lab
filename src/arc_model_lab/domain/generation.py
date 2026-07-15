from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from arc_model_lab.domain.exceptions import InvalidGenerationConfigError

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 256
# The largest temperature a caller may request. Above this, sampling degenerates
# into noise for the instruct models this service runs.
MAX_TEMPERATURE = 2.0


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """The decoding parameters for one generation.

    The default is greedy decoding (temperature 0): a reproducible baseline.
    Validation runs in __post_init__ so no construction path (HTTP schema,
    CLI, or storage) can build an out-of-range config.
    """

    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        # Frozen dataclass: normalize through object.__setattr__ so the stored
        # values honor the annotations (temperature=1 becomes 1.0) and the
        # validators' coerced return values are used rather than discarded.
        object.__setattr__(self, "temperature", _temperature(self.temperature))
        object.__setattr__(self, "max_output_tokens", _positive_int("max_output_tokens", self.max_output_tokens))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping for persistence (the inference row)."""
        return {"temperature": self.temperature, "max_output_tokens": self.max_output_tokens}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GenerationConfig:
        """Rebuild from a stored mapping, defaulting any missing knob.

        A pre-capture row persisted ``{}`` and rehydrates to the greedy default;
        every value is re-validated through the constructor.
        """
        return cls(
            temperature=data.get("temperature", DEFAULT_TEMPERATURE),
            max_output_tokens=data.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
        )


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidGenerationConfigError(f"{name} must be an integer")
    if value < 1:
        raise InvalidGenerationConfigError(f"{name} must be >= 1")
    return value


def _temperature(value: object) -> float:
    # bool is an int subclass; reject it so temperature=True cannot slip through.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidGenerationConfigError("temperature must be a number")
    if value < 0:
        raise InvalidGenerationConfigError("temperature must be >= 0")
    if value > MAX_TEMPERATURE:
        raise InvalidGenerationConfigError(f"temperature must be <= {MAX_TEMPERATURE}")
    return float(value)
