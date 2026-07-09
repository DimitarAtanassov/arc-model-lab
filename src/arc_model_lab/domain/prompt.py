from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass, field

from arc_model_lab.domain.exceptions import PromptRenderError

# The single variable every template may reference without the caller listing it
# in `variables`: it carries the request's input_text. Reserving one name keeps an
# unambiguous source for the primary content and lets a template frame it.
RESERVED_VARIABLE = "input_text"

_FORMATTER = string.Formatter()


def _placeholders(template: str) -> frozenset[str]:
    """Return the bare ``{name}`` fields in a template.

    Only simple names are allowed: attribute or index access (``{a.b}``,
    ``{a[0]}``) is rejected so a template can never reach into a value. This runs
    when the library loads, so a malformed template fails boot, not a request.
    """
    names: set[str] = set()
    for _text, field_name, _spec, _conversion in _FORMATTER.parse(template):
        if field_name is None:
            continue
        if not field_name.isidentifier():
            raise ValueError(f"Unsupported placeholder '{{{field_name}}}': only simple names are allowed")
        names.add(field_name)
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A template rendered into an optional system message and a user message."""

    user: str
    system: str | None = None


@dataclass(frozen=True, slots=True)
class PromptInput:
    """What to prompt the model with.

    Without a ``template`` the ``input_text`` is sent to the model unframed. With
    one, the template frames ``input_text`` (available as ``{input_text}``) and
    ``variables`` fill its remaining placeholders.
    """

    input_text: str
    template: str | None = None
    variables: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A named, reusable prompt loaded from one YAML file.

    Placeholders use ``{name}``. Every template may reference ``{input_text}``; its
    other placeholders are its required variables, supplied by the caller.
    """

    name: str
    user_template: str
    system_template: str | None = None
    description: str | None = None

    @property
    def required_variables(self) -> frozenset[str]:
        """The placeholders the caller must supply (every one except input_text)."""
        placeholders = _placeholders(self.user_template)
        if self.system_template is not None:
            placeholders |= _placeholders(self.system_template)
        return placeholders - {RESERVED_VARIABLE}

    def render(self, *, input_text: str, variables: Mapping[str, str]) -> RenderedPrompt:
        """Render the template, or raise ``PromptRenderError`` (422) on a mismatch.

        The caller must supply exactly the required variables: a missing one, an
        unknown one (a likely typo), or the reserved ``input_text`` is rejected.
        Values are substituted, never re-parsed, so a value containing braces is
        inert.
        """
        provided = set(variables)
        if RESERVED_VARIABLE in provided:
            raise PromptRenderError(f"'{RESERVED_VARIABLE}' is set from the request body, not variables")
        required = self.required_variables
        missing = required - provided
        if missing:
            raise PromptRenderError(f"Missing variables for template '{self.name}': {sorted(missing)}")
        unknown = provided - required
        if unknown:
            raise PromptRenderError(f"Unknown variables for template '{self.name}': {sorted(unknown)}")
        values: dict[str, str] = {RESERVED_VARIABLE: input_text, **variables}
        user = self.user_template.format_map(values)
        system = self.system_template.format_map(values) if self.system_template is not None else None
        return RenderedPrompt(user=user, system=system)
