from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from arc_model_lab.domain.generation_params import REGISTRY, ParamSpec

# The JSON-safe shape of a registry default across every knob's value type.
ParamDefaultJson = int | float | bool | list[str] | None


class GenerationParams(BaseModel):
    """The decoding-knob allow-list at the API boundary (preset config, model_params).

    Its fields are exactly the registry knobs, so an unknown key is a 422 under
    ``extra="forbid"``. It deliberately carries no numeric bounds: the bounds live
    once in the registry and are enforced by ``GenerationConfig`` when the service
    builds the config, so this model never duplicates a bound. ``to_config_dict``
    emits only the knobs the caller actually set, which is what the precedence merge
    and preset storage overlay.
    """

    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int | None = None
    min_new_tokens: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    num_beams: int | None = None
    length_penalty: float | None = None
    early_stopping: bool | None = None
    seed: int | None = None
    stop: list[str] | None = None

    def to_config_dict(self) -> dict[str, Any]:
        """The knobs the caller explicitly set, for the precedence merge or storage."""
        return self.model_dump(exclude_unset=True)


class ParamSpecResponse(BaseModel):
    """One decoding knob's static description, as the UI renders it.

    ``maximum`` is null for a knob whose ceiling is cross-field or runtime-sourced
    (``min_new_tokens`` is bounded by ``max_output_tokens``; ``max_output_tokens``
    by the server cap reported alongside this list).
    """

    name: str
    kind: str
    minimum: float | None
    maximum: float | None
    default: ParamDefaultJson
    tier: str
    group: str

    @classmethod
    def from_spec(cls, spec: ParamSpec) -> ParamSpecResponse:
        default: ParamDefaultJson = list(spec.default) if isinstance(spec.default, tuple) else spec.default
        return cls(
            name=spec.name,
            kind=spec.kind.value,
            minimum=spec.minimum,
            maximum=spec.maximum,
            default=default,
            tier=spec.tier.value,
            group=spec.group.value,
        )


class GenerationParamsResponse(BaseModel):
    """The parameter registry plus the effective runtime cap for the UI to render.

    ``max_output_tokens_cap`` is the configured ceiling, not a static registry
    value, so the UI renders the operator's real bound.
    """

    model_config = ConfigDict(protected_namespaces=())

    max_output_tokens_cap: int
    params: list[ParamSpecResponse]

    @classmethod
    def build(cls, max_output_tokens_cap: int) -> GenerationParamsResponse:
        return cls(
            max_output_tokens_cap=max_output_tokens_cap,
            params=[ParamSpecResponse.from_spec(spec) for spec in REGISTRY],
        )
