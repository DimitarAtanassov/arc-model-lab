"""Generation configuration: the decoding knobs a run may vary.

A run varies two parameters: ``temperature`` (0 is greedy and deterministic,
above 0 enables sampling) and ``max_output_tokens``. Holding them in a typed
value object means a caller cannot request a knob the runtime ignores: unknown
keys are rejected by :meth:`GenerationConfig.from_mapping`, and out-of-range
values are rejected at construction by ``__post_init__``.

These default constants are the reproducible baseline for experiment configs.
The server's runtime settings (``Settings.temperature`` and
``Settings.max_output_tokens``) default from them so "default decoding" has one
definition; ``/inference`` resolves its default from those runtime settings, not
from these constants directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from arc_model_lab.domain.exceptions import InvalidGenerationConfigError

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 256
# The largest temperature a caller may request. Above this, sampling degenerates
# into noise for the instruct models this service runs.
MAX_TEMPERATURE = 2.0

_FIELDS = ("temperature", "max_output_tokens")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """The decoding parameters for one generation.

    The default is greedy decoding (``temperature`` 0): a reproducible baseline.
    Validation runs in ``__post_init__`` so no construction path (HTTP schema,
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

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> GenerationConfig:
        """Build a config from a raw mapping (stored JSON or other untyped input).

        Validates only the keys (unknown or missing); the constructor's
        ``__post_init__`` owns the value bounds, so no field is validated twice.
        This is the entry point for configs that did not pass the typed HTTP schema.
        """
        unknown = sorted(set(data) - set(_FIELDS))
        if unknown:
            raise InvalidGenerationConfigError(f"Unknown generation parameters: {unknown}")
        missing = sorted(set(_FIELDS) - set(data))
        if missing:
            raise InvalidGenerationConfigError(f"Missing generation parameters: {missing}")
        values: dict[str, Any] = {name: data[name] for name in _FIELDS}
        return cls(**values)


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
