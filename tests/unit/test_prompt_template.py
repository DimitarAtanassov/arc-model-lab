from __future__ import annotations

import pytest

from arc_model_lab.domain import PromptInput, PromptRenderError, PromptTemplate


def _template(user: str, system: str | None = None) -> PromptTemplate:
    return PromptTemplate(name="t", user_template=user, system_template=system)


def test_prompt_input_defaults_to_no_template() -> None:
    prompt = PromptInput(input_text="hello")

    assert prompt.template is None
    assert dict(prompt.variables) == {}


def test_required_variables_excludes_reserved_input_text() -> None:
    template = _template("Translate to {language}:\n\n{input_text}")

    assert template.required_variables == frozenset({"language"})


def test_required_variables_spans_system_and_user() -> None:
    template = _template(user="{greeting}, {input_text}", system="Use a {tone} tone.")

    assert template.required_variables == frozenset({"greeting", "tone"})


def test_render_fills_placeholders_from_variables_and_input_text() -> None:
    template = _template(user="To {language}:\n\n{input_text}", system="Be {tone}.")

    rendered = template.render(input_text="hi", variables={"language": "French", "tone": "formal"})

    assert rendered.system == "Be formal."
    assert rendered.user == "To French:\n\nhi"


def test_render_without_system_returns_none_system() -> None:
    rendered = _template("Q: {input_text}").render(input_text="hi", variables={})

    assert rendered.system is None
    assert rendered.user == "Q: hi"


def test_render_rejects_missing_variable() -> None:
    with pytest.raises(PromptRenderError, match="Missing"):
        _template("To {language}: {input_text}").render(input_text="hi", variables={})


def test_render_rejects_unknown_variable() -> None:
    with pytest.raises(PromptRenderError, match="Unknown"):
        _template("{input_text}").render(input_text="hi", variables={"nope": "x"})


def test_render_rejects_reserved_variable() -> None:
    with pytest.raises(PromptRenderError, match="input_text"):
        _template("{input_text}").render(input_text="hi", variables={"input_text": "override"})


def test_render_does_not_reparse_variable_values() -> None:
    # A value that looks like a placeholder is inert: substituted, never re-expanded.
    rendered = _template("{input_text}").render(input_text="{language}", variables={})

    assert rendered.user == "{language}"


def test_template_with_attribute_placeholder_is_rejected() -> None:
    # Only bare names are allowed; attribute access must not build a template.
    with pytest.raises(ValueError, match="only simple names"):
        _ = _template("{input_text.__class__}").required_variables
