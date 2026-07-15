from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from arc_model_lab.domain import PromptTemplateNotFoundError
from arc_model_lab.prompts import load_prompt_library


def test_bundled_library_loads_summarize_and_translate() -> None:
    names = load_prompt_library().names()

    assert "summarize" in names
    assert "translate" in names


def test_summarize_needs_no_extra_variables() -> None:
    template = load_prompt_library().require("summarize")

    assert template.required_variables == frozenset()


def test_translate_requires_target_language() -> None:
    template = load_prompt_library().require("translate")

    assert template.required_variables == frozenset({"target_language"})


def test_require_unknown_template_raises() -> None:
    with pytest.raises(PromptTemplateNotFoundError):
        load_prompt_library().require("does-not-exist")


def test_get_returns_none_for_unknown_template() -> None:
    assert load_prompt_library().get("does-not-exist") is None


def test_loads_from_a_custom_directory(tmp_path: Path) -> None:
    (tmp_path / "greet.yaml").write_text("user: 'Hello {input_text}'\n", encoding="utf-8")

    library = load_prompt_library(tmp_path)

    assert library.names() == ["greet"]
    assert library.require("greet").required_variables == frozenset()


def test_non_mapping_template_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a mapping"):
        load_prompt_library(tmp_path)


def test_malformed_placeholder_fails_to_load(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("user: 'Hi {input_text[0]}'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="only simple names"):
        load_prompt_library(tmp_path)


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("user: 'Hi {input_text}'\nnope: 1\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_prompt_library(tmp_path)
