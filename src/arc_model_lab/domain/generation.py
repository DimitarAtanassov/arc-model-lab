"""Generation configuration: the deterministic decoding knobs a run may vary.

The runtime is deterministic (greedy or beam search, no sampling), so these are
the only parameters an experiment can change. Holding them in a typed value
object means an experiment cannot request a knob the runtime ignores: unknown
keys are rejected at the boundary by :meth:`GenerationConfig.from_mapping`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from arc_model_lab.domain.exceptions import InvalidGenerationConfigError

_FIELDS = ("max_input_tokens", "max_new_tokens", "num_beams")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """The deterministic decoding parameters for one generation."""

    max_input_tokens: int
    max_new_tokens: int
    num_beams: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> GenerationConfig:
        """Build a config from a raw mapping, rejecting unknown or invalid knobs.

        Raises :class:`InvalidGenerationConfigError` when a key is unknown, a
        required key is missing, or a value is not a positive integer. This is the
        single validation point for experiment ``generation_config``.
        """
        unknown = sorted(set(data) - set(_FIELDS))
        if unknown:
            raise InvalidGenerationConfigError(f"Unknown generation parameters: {unknown}")
        try:
            values = {name: _positive_int(name, data[name]) for name in _FIELDS}
        except KeyError as exc:
            raise InvalidGenerationConfigError(f"Missing generation parameter: {exc.args[0]}") from exc
        return cls(**values)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidGenerationConfigError(f"{name} must be an integer")
    if value < 1:
        raise InvalidGenerationConfigError(f"{name} must be >= 1")
    return value
