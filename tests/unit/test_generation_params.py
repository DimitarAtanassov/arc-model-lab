from __future__ import annotations

import pytest

from arc_model_lab.domain.generation_params import (
    DEFAULT_OUTPUT_TOKENS,
    MAX_SEED,
    MIN_OUTPUT_TOKENS,
    REGISTRY,
    ParamGroup,
    ParamKind,
    ParamTier,
    spec,
)

# The registry is the single source of decoding bounds; these tests pin its shape
# so a drift (a renamed knob, a dropped bound) fails here before it reaches the
# domain validator, the boundary, or the metadata endpoint.

_EXPECTED_NAMES = {
    "max_output_tokens",
    "min_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "num_beams",
    "length_penalty",
    "early_stopping",
    "seed",
    "stop",
}


def test_registry_covers_the_full_taxonomy() -> None:
    assert {s.name for s in REGISTRY} == _EXPECTED_NAMES


def test_registry_names_are_unique() -> None:
    names = [s.name for s in REGISTRY]
    assert len(names) == len(set(names))


def test_spec_returns_matching_entry() -> None:
    assert spec("temperature").name == "temperature"


def test_spec_raises_for_unknown_knob() -> None:
    with pytest.raises(KeyError):
        spec("does_not_exist")


def test_every_bounded_spec_has_min_le_max() -> None:
    for s in REGISTRY:
        if s.minimum is not None and s.maximum is not None:
            assert s.minimum <= s.maximum, s.name


def test_max_output_tokens_carries_floor_and_default_but_no_static_ceiling() -> None:
    # Its ceiling is the runtime ARC_MAX_OUTPUT_TOKENS_CAP, not a registry constant.
    s = spec("max_output_tokens")
    assert s.minimum == MIN_OUTPUT_TOKENS
    assert s.maximum is None
    assert s.default == DEFAULT_OUTPUT_TOKENS


def test_min_new_tokens_carries_only_a_static_floor() -> None:
    # Its ceiling (max_output_tokens) is cross-field, enforced in the domain, not here.
    s = spec("min_new_tokens")
    assert s.minimum == 0
    assert s.maximum is None


def test_seed_upper_bound_is_32_bit() -> None:
    assert spec("seed").maximum == MAX_SEED


@pytest.mark.parametrize(
    ("name", "default"),
    [("do_sample", False), ("num_beams", 1), ("temperature", 0.0), ("early_stopping", False), ("stop", ())],
)
def test_scalar_defaults_match_the_domain(name: str, default: object) -> None:
    assert spec(name).default == default


def test_kinds_tiers_and_groups_are_registry_enums() -> None:
    for s in REGISTRY:
        assert isinstance(s.kind, ParamKind)
        assert isinstance(s.tier, ParamTier)
        assert isinstance(s.group, ParamGroup)
