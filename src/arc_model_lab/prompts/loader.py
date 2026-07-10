"""Loading and lookup for prompt templates stored as one YAML file per template.

Templates are data, not code: each ``*.yaml`` in this package is validated once at
boot into an immutable :class:`PromptLibrary`, so a malformed template fails
startup rather than a live request. PyYAML is available transitively through
transformers (a core dependency); this module is its only consumer.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from arc_model_lab.domain import PromptTemplate, PromptTemplateNotFoundError

_BUNDLED_DIR = Path(__file__).parent


class PromptDefinition(BaseModel):
    """The YAML shape of one prompt-template file, validated at load."""

    model_config = ConfigDict(extra="forbid")

    user: str = Field(min_length=1)
    system: str | None = None
    description: str | None = None


class PromptLibrary:
    """The loaded prompt templates, keyed by name (the file's stem)."""

    def __init__(self, templates: dict[str, PromptTemplate]) -> None:
        self._templates = templates

    def get(self, name: str) -> PromptTemplate | None:
        return self._templates.get(name)

    def require(self, name: str) -> PromptTemplate:
        """Return the named template, or raise PromptTemplateNotFoundError (404)."""
        template = self._templates.get(name)
        if template is None:
            raise PromptTemplateNotFoundError(f"Prompt template not found: {name}")
        return template

    def names(self) -> list[str]:
        """Return the registered template names, sorted."""
        return sorted(self._templates)


def load_prompt_library(directory: Path | None = None) -> PromptLibrary:
    """Load and validate every ``*.yaml`` prompt in a directory (bundled by default).

    Placeholders are validated here (via ``required_variables``), so a malformed
    template fails boot rather than a live request.
    """
    root = directory if directory is not None else _BUNDLED_DIR
    templates: dict[str, PromptTemplate] = {}
    for path in sorted(root.glob("*.yaml")):
        template = _load_template(path)
        templates[template.name] = template
    return PromptLibrary(templates)


def _load_template(path: Path) -> PromptTemplate:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt template is not a mapping: {path}")
    definition = PromptDefinition.model_validate(raw)
    template = PromptTemplate(
        name=path.stem,
        user_template=definition.user,
        system_template=definition.system,
        description=definition.description,
    )
    # Access required_variables now so a malformed placeholder fails boot, not a request.
    _ = template.required_variables
    return template
