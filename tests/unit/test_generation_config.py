from __future__ import annotations

import pytest

from arc_model_lab.domain import GenerationConfig, InvalidGenerationConfigError
from arc_model_lab.domain.generation import DecodingMode, resolve_generation_config

# The constructor's __post_init__ is the single validation entry point: every
# construction path (the HTTP schema's to_domain, server defaults, dataclasses
# replace) funnels through it, so testing it directly covers them all.


def test_defaults_are_greedy() -> None:
    # The one place the "default decoding" values live; the schema derives from it.
    assert GenerationConfig() == GenerationConfig(temperature=0.0, max_output_tokens=256)


def test_temperature_is_coerced_to_float() -> None:
    # temperature=1 is stored as 1.0 so the annotation and stored value agree.
    assert GenerationConfig(temperature=1, max_output_tokens=256).temperature == 1.0


@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_max_output_tokens(value: int) -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=0.0, max_output_tokens=value)


def test_rejects_non_integer_max_output_tokens() -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=0.0, max_output_tokens="big")  # type: ignore[arg-type]


def test_rejects_bool_max_output_tokens() -> None:
    # bool is an int subclass; reject it so max_output_tokens=True cannot slip through.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=0.0, max_output_tokens=True)


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_rejects_out_of_range_temperature(value: float) -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=value, max_output_tokens=256)


def test_rejects_bool_temperature() -> None:
    # bool is an int subclass; reject it so temperature=True cannot slip through.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=True, max_output_tokens=256)


def test_rejects_non_numeric_temperature() -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature="hot", max_output_tokens=256)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1])
def test_direct_construction_rejects_non_positive_max_output_tokens(value: int) -> None:
    # Guards the CLI path, which builds the config directly rather than via from_mapping.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=0.0, max_output_tokens=value)


def test_direct_construction_rejects_out_of_range_temperature() -> None:
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(temperature=2.5, max_output_tokens=256)


def test_int_temperature_is_normalized_to_float() -> None:
    # The float annotation is honored: an int temperature is coerced to a float,
    # so stored and serialized values stay consistent (1 -> 1.0).
    config = GenerationConfig(temperature=1, max_output_tokens=256)

    assert isinstance(config.temperature, float)
    assert config.temperature == 1.0


def test_to_dict_round_trips_through_from_dict() -> None:
    # Persistence contract: what to_dict writes, from_dict reads back unchanged.
    config = GenerationConfig(temperature=0.7, max_output_tokens=128)

    assert GenerationConfig.from_dict(config.to_dict()) == config


def test_from_dict_defaults_missing_knobs() -> None:
    # A pre-capture row persisted {} rehydrates to the greedy default.
    assert GenerationConfig.from_dict({}) == GenerationConfig()


def test_from_dict_revalidates_stored_values() -> None:
    # A corrupt stored config is rejected at read, not silently trusted.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig.from_dict({"temperature": 9.0, "max_output_tokens": 256})


# --- Extended taxonomy: per-field bounds (registry-driven) --------------------


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("min_new_tokens", -1),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("top_k", -1),
        ("top_k", 1001),
        ("min_p", -0.1),
        ("min_p", 1.1),
        ("repetition_penalty", 0.9),
        ("repetition_penalty", 2.1),
        ("no_repeat_ngram_size", -1),
        ("no_repeat_ngram_size", 11),
        ("num_beams", 0),
        ("num_beams", 9),
        ("length_penalty", -2.1),
        ("length_penalty", 2.1),
        ("seed", -1),
        ("seed", 2**32),
    ],
)
def test_rejects_out_of_range_field(field: str, bad_value: float) -> None:
    # Every registry bound is enforced at construction. length_penalty rides on
    # num_beams so its out-of-range value is paired with a valid beam count.
    kwargs: dict[str, object] = {field: bad_value}
    if field == "length_penalty":
        kwargs["num_beams"] = 2
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "good_value"),
    [
        ("min_new_tokens", 0),
        ("top_p", 0.9),
        ("top_k", 40),
        ("min_p", 0.05),
        ("repetition_penalty", 1.1),
        ("no_repeat_ngram_size", 3),
        ("num_beams", 8),
        ("seed", 0),
        ("seed", 2**32 - 1),
    ],
)
def test_accepts_in_range_field(field: str, good_value: int | float) -> None:
    config = GenerationConfig(**{field: good_value})  # type: ignore[arg-type]
    assert getattr(config, field) == good_value


@pytest.mark.parametrize("field", ["top_k", "num_beams", "no_repeat_ngram_size", "seed", "min_new_tokens"])
def test_rejects_bool_for_int_fields(field: str) -> None:
    # bool is an int subclass; a flag must not pose as a count.
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(**{field: True})  # type: ignore[arg-type]


def test_rejects_none_for_required_int_field() -> None:
    # A required int (max_output_tokens) cannot be null; only opt-in knobs may be.
    with pytest.raises(InvalidGenerationConfigError, match="must be an integer"):
        GenerationConfig(max_output_tokens=None)  # type: ignore[arg-type]


def test_rejects_none_for_required_float_field() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="must be a number"):
        GenerationConfig(temperature=None)  # type: ignore[arg-type]


def test_rejects_non_bool_early_stopping() -> None:
    # early_stopping is a strict flag; a truthy non-bool must not slip through.
    with pytest.raises(InvalidGenerationConfigError, match="must be a boolean"):
        GenerationConfig(num_beams=2, early_stopping=1)  # type: ignore[arg-type]


# --- Decoding-mode derivation -------------------------------------------------


def test_default_is_greedy_mode() -> None:
    assert GenerationConfig().mode is DecodingMode.GREEDY


def test_temperature_derives_sampling_mode() -> None:
    # The legacy /inference path sets only temperature and leaves do_sample unset.
    assert GenerationConfig(temperature=0.7).mode is DecodingMode.SAMPLING


@pytest.mark.parametrize("field", ["top_p", "min_p"])
def test_sampling_shape_param_derives_sampling_mode(field: str) -> None:
    assert GenerationConfig(**{field: 0.5}).mode is DecodingMode.SAMPLING  # type: ignore[arg-type]


def test_top_k_derives_sampling_mode() -> None:
    assert GenerationConfig(top_k=40).mode is DecodingMode.SAMPLING


def test_explicit_do_sample_true_is_sampling_mode() -> None:
    assert GenerationConfig(do_sample=True).mode is DecodingMode.SAMPLING


def test_num_beams_gt_one_is_beam_mode() -> None:
    assert GenerationConfig(num_beams=2).mode is DecodingMode.BEAM


def test_explicit_do_sample_false_stays_greedy() -> None:
    assert GenerationConfig(do_sample=False).mode is DecodingMode.GREEDY


# --- Illegal cross-field combinations (each a 422 at the boundary) ------------


def test_sampling_param_with_explicit_greedy_is_rejected() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="require do_sample=true"):
        GenerationConfig(do_sample=False, top_p=0.9)


def test_temperature_with_explicit_greedy_is_rejected() -> None:
    # temperature > 0 is a sampling-shape parameter; do_sample=false contradicts it.
    with pytest.raises(InvalidGenerationConfigError, match="require do_sample=true"):
        GenerationConfig(do_sample=False, temperature=0.7)


def test_beam_with_do_sample_true_is_rejected() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="cannot combine with do_sample=true"):
        GenerationConfig(num_beams=2, do_sample=True)


@pytest.mark.parametrize("field", ["top_p", "top_k", "min_p"])
def test_beam_with_sampling_param_is_rejected(field: str) -> None:
    with pytest.raises(InvalidGenerationConfigError, match="cannot combine with sampling"):
        GenerationConfig(num_beams=2, **{field: 0.5 if field != "top_k" else 40})  # type: ignore[arg-type]


def test_beam_with_temperature_is_rejected() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="cannot combine with sampling"):
        GenerationConfig(num_beams=2, temperature=0.7)


@pytest.mark.parametrize("field", ["length_penalty", "early_stopping"])
def test_beam_only_param_without_beam_is_rejected(field: str) -> None:
    value: float | bool = -1.0 if field == "length_penalty" else True
    with pytest.raises(InvalidGenerationConfigError, match="require beam search"):
        GenerationConfig(**{field: value})  # type: ignore[arg-type]


def test_min_new_tokens_above_max_output_tokens_is_rejected() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="must be <= max_output_tokens"):
        GenerationConfig(min_new_tokens=100, max_output_tokens=50)


def test_min_new_tokens_equal_to_max_output_tokens_is_allowed() -> None:
    config = GenerationConfig(min_new_tokens=50, max_output_tokens=50)
    assert config.min_new_tokens == 50


# --- Valid mode-specific compositions -----------------------------------------


def test_beam_accepts_beam_only_params() -> None:
    config = GenerationConfig(num_beams=4, length_penalty=1.5, early_stopping=True)
    assert config.mode is DecodingMode.BEAM
    assert config.length_penalty == 1.5
    assert config.early_stopping is True


def test_sampling_accepts_all_sampling_params() -> None:
    config = GenerationConfig(do_sample=True, temperature=0.8, top_p=0.9, top_k=40, min_p=0.05)
    assert config.mode is DecodingMode.SAMPLING


def test_repetition_penalty_valid_in_greedy_mode() -> None:
    # Repetition controls are mode-independent.
    config = GenerationConfig(repetition_penalty=1.2, no_repeat_ngram_size=3)
    assert config.mode is DecodingMode.GREEDY


# --- stop validation ----------------------------------------------------------


def test_stop_accepts_up_to_four_items() -> None:
    config = GenerationConfig(stop=["a", "bb", "ccc", "dddd"])
    assert config.stop == ("a", "bb", "ccc", "dddd")


def test_stop_normalizes_list_to_tuple() -> None:
    # The frozen dataclass stores an immutable tuple, not the caller's list.
    assert isinstance(GenerationConfig(stop=["x"]).stop, tuple)


def test_stop_rejects_more_than_four_items() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="at most 4 items"):
        GenerationConfig(stop=["a", "b", "c", "d", "e"])


def test_stop_rejects_empty_string() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="1 to 32 characters"):
        GenerationConfig(stop=[""])


def test_stop_rejects_overlong_string() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="1 to 32 characters"):
        GenerationConfig(stop=["x" * 33])


def test_stop_rejects_bare_string() -> None:
    # A bare string would iterate into characters; only a list of strings is valid.
    with pytest.raises(InvalidGenerationConfigError, match="list of strings"):
        GenerationConfig(stop="stop")  # type: ignore[arg-type]


def test_stop_rejects_non_string_item() -> None:
    with pytest.raises(InvalidGenerationConfigError, match="list of strings"):
        GenerationConfig(stop=[1])  # type: ignore[list-item]


# --- Persistence roundtrip across the full taxonomy ---------------------------


def test_rich_config_round_trips_through_to_dict() -> None:
    config = GenerationConfig(
        max_output_tokens=128,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        min_new_tokens=8,
        seed=123,
        stop=["END"],
    )

    assert GenerationConfig.from_dict(config.to_dict()) == config


def test_beam_config_round_trips_through_to_dict() -> None:
    config = GenerationConfig(num_beams=4, length_penalty=1.5, early_stopping=True, max_output_tokens=64)

    assert GenerationConfig.from_dict(config.to_dict()) == config


def test_to_dict_omits_unset_knobs() -> None:
    # A greedy default emits only the two always-present keys, keeping the JSONB lean.
    assert GenerationConfig().to_dict() == {"max_output_tokens": 256, "temperature": 0.0}


def test_from_dict_defaults_new_knobs_to_none() -> None:
    config = GenerationConfig.from_dict({"max_output_tokens": 256, "temperature": 0.0})
    assert config.do_sample is None
    assert config.top_p is None
    assert config.num_beams == 1
    assert config.stop == ()


# resolve_generation_config is the pure precedence merge: call model_params >
# preset > server default. No database, no model; the merged result is re-validated
# by the one GenerationConfig constructor, and the output cap is re-checked.

_CAP = 2048


def test_merge_returns_default_when_no_preset_or_overrides() -> None:
    default = GenerationConfig(temperature=0.5, max_output_tokens=512)

    resolved = resolve_generation_config(default, None, {}, max_output_tokens_cap=_CAP)

    assert resolved == default


def test_merge_preset_overrides_default() -> None:
    default = GenerationConfig(temperature=0.5, max_output_tokens=512)
    preset = GenerationConfig(do_sample=True, temperature=0.8, max_output_tokens=128)

    resolved = resolve_generation_config(default, preset, {}, max_output_tokens_cap=_CAP)

    # The preset's set knobs win, including its own output budget (to_dict always
    # carries max_output_tokens and temperature).
    assert resolved.temperature == 0.8
    assert resolved.do_sample is True
    assert resolved.max_output_tokens == 128


def test_merge_inherits_knobs_neither_layer_sets() -> None:
    default = GenerationConfig(temperature=0.5, max_output_tokens=512)
    preset = GenerationConfig(do_sample=True, temperature=0.8)

    resolved = resolve_generation_config(default, preset, {}, max_output_tokens_cap=_CAP)

    # top_p is set by neither layer, so it stays unset (to_dict omits it when unset).
    assert resolved.top_p is None


def test_merge_overrides_beat_preset_and_default() -> None:
    default = GenerationConfig(temperature=0.5, max_output_tokens=512)
    preset = GenerationConfig(do_sample=True, temperature=0.8, top_p=0.9, max_output_tokens=128)

    resolved = resolve_generation_config(default, preset, {"temperature": 1.2}, max_output_tokens_cap=_CAP)

    # Override wins on temperature; the preset's untouched knobs are inherited.
    assert resolved.temperature == 1.2
    assert resolved.top_p == 0.9
    assert resolved.max_output_tokens == 128


def test_merge_revalidates_illegal_combination() -> None:
    # A beam-mode preset plus a sampling override is an illegal merged config (422).
    default = GenerationConfig()
    preset = GenerationConfig(num_beams=4, length_penalty=1.5)

    with pytest.raises(InvalidGenerationConfigError):
        resolve_generation_config(default, preset, {"top_p": 0.9}, max_output_tokens_cap=_CAP)


def test_merge_enforces_output_cap_on_the_resolved_config() -> None:
    default = GenerationConfig()

    with pytest.raises(InvalidGenerationConfigError):
        resolve_generation_config(default, None, {"max_output_tokens": _CAP + 1}, max_output_tokens_cap=_CAP)
