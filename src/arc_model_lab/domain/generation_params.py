"""The parameter registry: the single source of truth for decoding bounds.

One `ParamSpec` per tuneable decoding knob. The domain validator, the API
boundary, and the `GET /generation/params` metadata endpoint all read this
registry, so a bound is defined exactly once. Adding a knob is one entry here
plus its mapping in `to_generate_kwargs`.

Every `minimum`/`maximum` is a *static* value. A knob whose real ceiling depends
on another field (`min_new_tokens`, bounded above by `max_output_tokens`) carries
only its static floor here; the cross-field ceiling is enforced in
`GenerationConfig.__post_init__`. The one runtime-sourced ceiling is
`max_output_tokens`, whose maximum is `None` here and resolved from the
`ARC_MAX_OUTPUT_TOKENS_CAP` setting at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# stop is a list knob whose bound is item count and per-item length rather than a
# scalar range, so it cannot ride the ParamSpec min/max fields. The constants live
# here so the registry module stays the single home for every decoding bound.
MAX_STOP_ITEMS = 4
MAX_STOP_ITEM_CHARS = 32

# The floor and default for max_output_tokens are static; its ceiling is the
# server-authoritative ARC_MAX_OUTPUT_TOKENS_CAP setting, resolved at the boundary.
MIN_OUTPUT_TOKENS = 1
DEFAULT_OUTPUT_TOKENS = 256

DEFAULT_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

# seed is a 32-bit unsigned value, matching transformers.set_seed's range.
MAX_SEED = 2**32 - 1


class ParamKind(StrEnum):
    """The value shape of a decoding knob, for boundary and UI rendering."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR_LIST = "str_list"


class ParamTier(StrEnum):
    """core is the universal-safe set shown by default; advanced is behind a disclosure."""

    CORE = "core"
    ADVANCED = "advanced"


class ParamGroup(StrEnum):
    """The UI grouping a knob belongs to."""

    LENGTH = "length"
    SAMPLING = "sampling"
    REPETITION = "repetition"
    BEAM = "beam"
    DETERMINISM = "determinism"
    STOPPING = "stopping"


# A spec default is metadata for the UI and the boundary, not a live value, so it
# spans every knob's value shape.
ParamDefault = int | float | bool | tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One decoding knob's static description: shape, bound, default, and grouping."""

    name: str
    kind: ParamKind
    minimum: float | None
    maximum: float | None
    default: ParamDefault
    tier: ParamTier
    group: ParamGroup


# One entry per row of spec 0001 §1.2, in the order the UI renders them. This is
# the authoritative bound table; nothing else defines these numbers.
REGISTRY: tuple[ParamSpec, ...] = (
    ParamSpec(
        "max_output_tokens",
        ParamKind.INT,
        MIN_OUTPUT_TOKENS,
        None,
        DEFAULT_OUTPUT_TOKENS,
        ParamTier.CORE,
        ParamGroup.LENGTH,
    ),
    ParamSpec("min_new_tokens", ParamKind.INT, 0, None, None, ParamTier.ADVANCED, ParamGroup.LENGTH),
    ParamSpec("do_sample", ParamKind.BOOL, None, None, False, ParamTier.CORE, ParamGroup.SAMPLING),
    ParamSpec(
        "temperature", ParamKind.FLOAT, 0.0, MAX_TEMPERATURE, DEFAULT_TEMPERATURE, ParamTier.CORE, ParamGroup.SAMPLING
    ),
    ParamSpec("top_p", ParamKind.FLOAT, 0.0, 1.0, None, ParamTier.CORE, ParamGroup.SAMPLING),
    ParamSpec("top_k", ParamKind.INT, 0, 1000, None, ParamTier.ADVANCED, ParamGroup.SAMPLING),
    ParamSpec("min_p", ParamKind.FLOAT, 0.0, 1.0, None, ParamTier.ADVANCED, ParamGroup.SAMPLING),
    ParamSpec("repetition_penalty", ParamKind.FLOAT, 1.0, 2.0, None, ParamTier.ADVANCED, ParamGroup.REPETITION),
    ParamSpec("no_repeat_ngram_size", ParamKind.INT, 0, 10, None, ParamTier.ADVANCED, ParamGroup.REPETITION),
    ParamSpec("num_beams", ParamKind.INT, 1, 8, 1, ParamTier.ADVANCED, ParamGroup.BEAM),
    ParamSpec("length_penalty", ParamKind.FLOAT, -2.0, 2.0, None, ParamTier.ADVANCED, ParamGroup.BEAM),
    ParamSpec("early_stopping", ParamKind.BOOL, None, None, False, ParamTier.ADVANCED, ParamGroup.BEAM),
    ParamSpec("seed", ParamKind.INT, 0, MAX_SEED, None, ParamTier.CORE, ParamGroup.DETERMINISM),
    ParamSpec("stop", ParamKind.STR_LIST, None, None, (), ParamTier.ADVANCED, ParamGroup.STOPPING),
)

_SPECS_BY_NAME: dict[str, ParamSpec] = {spec.name: spec for spec in REGISTRY}


def spec(name: str) -> ParamSpec:
    """Return the ParamSpec for a knob name, or raise KeyError for an unknown knob."""
    return _SPECS_BY_NAME[name]
