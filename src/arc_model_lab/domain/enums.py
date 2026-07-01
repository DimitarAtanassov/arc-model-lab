"""Domain enumerations."""

from __future__ import annotations

from enum import StrEnum


class Provider(StrEnum):
    HUGGINGFACE = "huggingface"


class ModelStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
