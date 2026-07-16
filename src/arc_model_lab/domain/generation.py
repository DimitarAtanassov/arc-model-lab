from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from arc_model_lab.domain.exceptions import InvalidGenerationConfigError
from arc_model_lab.domain.generation_params import (
    DEFAULT_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_STOP_ITEM_CHARS,
    MAX_STOP_ITEMS,
    MAX_TEMPERATURE,
    ParamKind,
    ParamSpec,
    spec,
)

# Re-exported for the config module and the inference schema, which imported these
# names from here before the taxonomy grew.
DEFAULT_MAX_OUTPUT_TOKENS = DEFAULT_OUTPUT_TOKENS

__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_TEMPERATURE",
    "MAX_TEMPERATURE",
    "DecodingMode",
    "GenerationConfig",
    "enforce_output_cap",
    "resolve_generation_config",
    "to_generate_kwargs",
]


class DecodingMode(StrEnum):
    """The one decoding mode a config resolves to, derived from its fields.

    A config never mixes modes: sampling and beam parameters are valid only in
    their own mode, and cross-mode combinations are rejected at construction.
    """

    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM = "beam"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """The validated decoding parameters for one generation.

    The default is greedy decoding: a reproducible baseline. Validation runs in
    __post_init__ against the parameter registry (the single source of bounds),
    so no construction path (HTTP schema, merge, or storage) can build an
    out-of-range or contradictory config. The config also derives one decoding
    mode and rejects cross-mode conflicts a flat dataclass cannot forbid by type.

    ``do_sample`` is tri-state: ``None`` means "derive from the sampling params",
    while ``True``/``False`` are explicit and are checked against the derived mode.
    """

    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    do_sample: bool | None = None
    min_new_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    num_beams: int = 1
    length_penalty: float | None = None
    early_stopping: bool = False
    seed: int | None = None
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Frozen dataclass: normalize through object.__setattr__ so coerced values
        # (int temperature -> float, stop list -> tuple) are the ones stored.
        s = object.__setattr__
        s(self, "max_output_tokens", _int("max_output_tokens", self.max_output_tokens))
        s(self, "temperature", _float("temperature", self.temperature))
        s(self, "do_sample", _optional_bool("do_sample", self.do_sample))
        s(self, "min_new_tokens", _int("min_new_tokens", self.min_new_tokens, allow_none=True))
        s(self, "top_p", _float("top_p", self.top_p, allow_none=True))
        s(self, "top_k", _int("top_k", self.top_k, allow_none=True))
        s(self, "min_p", _float("min_p", self.min_p, allow_none=True))
        s(self, "repetition_penalty", _float("repetition_penalty", self.repetition_penalty, allow_none=True))
        s(self, "no_repeat_ngram_size", _int("no_repeat_ngram_size", self.no_repeat_ngram_size, allow_none=True))
        s(self, "num_beams", _int("num_beams", self.num_beams))
        s(self, "length_penalty", _float("length_penalty", self.length_penalty, allow_none=True))
        s(self, "early_stopping", _bool("early_stopping", self.early_stopping))
        s(self, "seed", _int("seed", self.seed, allow_none=True))
        s(self, "stop", _stop(self.stop))
        self._validate_combination()

    @property
    def mode(self) -> DecodingMode:
        """The decoding mode this config resolves to (valid only after construction)."""
        if self.num_beams > 1:
            return DecodingMode.BEAM
        if self.do_sample is True or (self.do_sample is None and self._sampling_params_present):
            return DecodingMode.SAMPLING
        return DecodingMode.GREEDY

    @property
    def _sampling_params_present(self) -> bool:
        return self.top_p is not None or self.top_k is not None or self.min_p is not None or self.temperature > 0

    @property
    def _beam_params_present(self) -> bool:
        return self.length_penalty is not None or self.early_stopping

    def _validate_combination(self) -> None:
        # Cross-field rules a single dataclass cannot express by type. Each rejects
        # rather than silently coerces, so an impossible request fails loudly (422).
        if self.min_new_tokens is not None and self.min_new_tokens > self.max_output_tokens:
            raise InvalidGenerationConfigError(
                f"min_new_tokens ({self.min_new_tokens}) must be <= max_output_tokens ({self.max_output_tokens})"
            )

        if self.num_beams > 1:
            if self.do_sample is True:
                raise InvalidGenerationConfigError("beam search (num_beams > 1) cannot combine with do_sample=true")
            if self._sampling_params_present:
                raise InvalidGenerationConfigError(
                    "beam search (num_beams > 1) cannot combine with sampling parameters "
                    "(temperature > 0, top_p, top_k, min_p)"
                )
            return  # beam mode: beam params are valid, sampling params already excluded.

        # num_beams == 1: beam-only params are meaningless here.
        if self._beam_params_present:
            raise InvalidGenerationConfigError("length_penalty and early_stopping require beam search (num_beams > 1)")
        # Explicit greedy must not carry sampling params.
        if self.do_sample is False and self._sampling_params_present:
            raise InvalidGenerationConfigError(
                "sampling parameters (temperature > 0, top_p, top_k, min_p) require do_sample=true"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping for persistence (the inference row).

        Only knobs that carry a meaningful value are emitted, so a greedy default
        stays compact and from_dict rehydrates the rest. max_output_tokens and
        temperature are always present, preserving the pre-taxonomy row shape.
        """
        data: dict[str, Any] = {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if self.do_sample is not None:
            data["do_sample"] = self.do_sample
        for name in (
            "min_new_tokens",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "length_penalty",
            "seed",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.num_beams != 1:
            data["num_beams"] = self.num_beams
        if self.early_stopping:
            data["early_stopping"] = self.early_stopping
        if self.stop:
            data["stop"] = list(self.stop)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GenerationConfig:
        """Rebuild from a stored mapping, defaulting any missing knob.

        A pre-capture row persisted ``{}`` and rehydrates to the greedy default;
        every value is re-validated through the constructor.
        """
        stop = data.get("stop", ())
        return cls(
            max_output_tokens=data.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
            temperature=data.get("temperature", DEFAULT_TEMPERATURE),
            do_sample=data.get("do_sample"),
            min_new_tokens=data.get("min_new_tokens"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            min_p=data.get("min_p"),
            repetition_penalty=data.get("repetition_penalty"),
            no_repeat_ngram_size=data.get("no_repeat_ngram_size"),
            num_beams=data.get("num_beams", 1),
            length_penalty=data.get("length_penalty"),
            early_stopping=data.get("early_stopping", False),
            seed=data.get("seed"),
            stop=tuple(stop) if isinstance(stop, (list, tuple)) else stop,
        )


def enforce_output_cap(config: GenerationConfig, max_output_tokens_cap: int) -> None:
    """Reject a config whose output budget exceeds the server-authoritative cap.

    ``max_output_tokens`` carries no static ceiling in the registry: the ceiling is
    the runtime ``ARC_MAX_OUTPUT_TOKENS_CAP`` setting, so it is enforced here rather
    than in ``__post_init__``. A violation is a 422 that names the field and the cap.
    """
    if config.max_output_tokens > max_output_tokens_cap:
        raise InvalidGenerationConfigError(
            f"max_output_tokens ({config.max_output_tokens}) exceeds the server cap ({max_output_tokens_cap})"
        )


def resolve_generation_config(
    default: GenerationConfig,
    preset: GenerationConfig | None,
    overrides: Mapping[str, Any],
    *,
    max_output_tokens_cap: int,
) -> GenerationConfig:
    """Merge decoding config by precedence: call overrides > preset > server default.

    Pure and unit-testable with no database and no model. It layers the sparse
    ``to_dict`` payloads (each emits only meaningfully-set knobs), so a preset or an
    override touches exactly the knobs it names and inherits the rest. The merged
    result is re-validated by the one ``GenerationConfig`` constructor, so an illegal
    cross-mode combination the merge produced (for example a beam-mode preset plus a
    ``top_p`` override) is rejected as a 422, and the output cap is re-checked.
    """
    merged: dict[str, Any] = dict(default.to_dict())
    if preset is not None:
        merged.update(preset.to_dict())
    merged.update(overrides)
    config = GenerationConfig.from_dict(merged)
    enforce_output_cap(config, max_output_tokens_cap)
    return config


def to_generate_kwargs(config: GenerationConfig) -> dict[str, Any]:
    """Map a validated config to transformers.generate keyword arguments.

    Pure and unit-testable with no model loaded. It emits only the kwargs valid
    for the config's derived mode, so a greedy call carries no sampling knobs and
    a sampling call carries no beam knobs. ``stop`` maps to ``stop_strings``; the
    caller passes the tokenizer alongside it (transformers requires it).
    """
    mode = config.mode
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_output_tokens,
        "do_sample": mode is DecodingMode.SAMPLING,
    }
    if mode is DecodingMode.SAMPLING:
        kwargs.update(_sampling_kwargs(config))
    elif mode is DecodingMode.BEAM:
        kwargs.update(_beam_kwargs(config))
    kwargs.update(_universal_kwargs(config))
    if config.stop:
        kwargs["stop_strings"] = list(config.stop)
    return kwargs


def _sampling_kwargs(config: GenerationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.temperature > 0:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        kwargs["top_k"] = config.top_k
    if config.min_p is not None:
        kwargs["min_p"] = config.min_p
    return kwargs


def _beam_kwargs(config: GenerationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"num_beams": config.num_beams}
    if config.length_penalty is not None:
        kwargs["length_penalty"] = config.length_penalty
    if config.early_stopping:
        kwargs["early_stopping"] = config.early_stopping
    return kwargs


def _universal_kwargs(config: GenerationConfig) -> dict[str, Any]:
    # Valid in any mode.
    kwargs: dict[str, Any] = {}
    if config.repetition_penalty is not None:
        kwargs["repetition_penalty"] = config.repetition_penalty
    if config.no_repeat_ngram_size is not None:
        kwargs["no_repeat_ngram_size"] = config.no_repeat_ngram_size
    if config.min_new_tokens is not None:
        kwargs["min_new_tokens"] = config.min_new_tokens
    return kwargs


def _bounds(name: str) -> ParamSpec:
    return spec(name)


def _int(name: str, value: object, *, allow_none: bool = False) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise InvalidGenerationConfigError(f"{name} must be an integer")
    # bool is an int subclass; reject it so a flag cannot pose as a count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidGenerationConfigError(f"{name} must be an integer")
    _check_range(name, value)
    return value


def _float(name: str, value: object, *, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise InvalidGenerationConfigError(f"{name} must be a number")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidGenerationConfigError(f"{name} must be a number")
    _check_range(name, value)
    return float(value)


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise InvalidGenerationConfigError(f"{name} must be a boolean")
    return value


def _optional_bool(name: str, value: object) -> bool | None:
    if value is None:
        return None
    return _bool(name, value)


def _check_range(name: str, value: int | float) -> None:
    bounds = _bounds(name)
    if bounds.minimum is not None and value < bounds.minimum:
        raise InvalidGenerationConfigError(f"{name} must be >= {_fmt(bounds.minimum, bounds.kind)}")
    if bounds.maximum is not None and value > bounds.maximum:
        raise InvalidGenerationConfigError(f"{name} must be <= {_fmt(bounds.maximum, bounds.kind)}")


def _fmt(bound: float, kind: ParamKind) -> str:
    return str(int(bound)) if kind is ParamKind.INT else str(bound)


def _stop(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise InvalidGenerationConfigError("stop must be a list of strings")
    items = tuple(value)
    if len(items) > MAX_STOP_ITEMS:
        raise InvalidGenerationConfigError(f"stop accepts at most {MAX_STOP_ITEMS} items")
    for item in items:
        if not isinstance(item, str):
            raise InvalidGenerationConfigError("stop must be a list of strings")
        if not 1 <= len(item) <= MAX_STOP_ITEM_CHARS:
            raise InvalidGenerationConfigError(f"each stop string must be 1 to {MAX_STOP_ITEM_CHARS} characters")
    return items
