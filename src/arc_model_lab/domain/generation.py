"""Generation configuration: the deterministic decoding knobs a run may vary.

The runtime is deterministic (greedy or beam search, no sampling), so these are
the only parameters an experiment can change. Holding them in a typed value
object means an experiment cannot request a knob the runtime ignores: unknown
keys are rejected by :meth:`GenerationConfig.from_mapping`, and out-of-range
values are rejected at construction by ``__post_init__``.

The default values live here once so the HTTP schema and the CLI cannot drift
apart on what "default decoding" means.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from arc_model_lab.domain.exceptions import InvalidGenerationConfigError

DEFAULT_MAX_INPUT_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 256
# Greedy decoding: the deterministic, reproducible baseline for experiments. The
# deployed model may decode differently (see ``Settings.num_beams``); an experiment
# opts into that by setting ``num_beams`` explicitly.
DEFAULT_NUM_BEAMS = 1

_FIELDS = ("max_input_tokens", "max_new_tokens", "num_beams")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """The deterministic decoding parameters for one generation.

    Defaults are greedy decoding: a reproducible experiment baseline, independent
    of the deployed model's own decoding. Validation runs in ``__post_init__`` so
    no construction path (HTTP schema, CLI, or storage) can build an out-of-range
    config.
    """

    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    num_beams: int = DEFAULT_NUM_BEAMS

    def __post_init__(self) -> None:
        for name in _FIELDS:
            _positive_int(name, getattr(self, name))

    def to_dict(self) -> dict[str, int]:
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
