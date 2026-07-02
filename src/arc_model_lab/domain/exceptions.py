"""Domain errors. The API layer maps these to HTTP status codes."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class ModelNotFoundError(DomainError):
    """The requested model is not registered in the catalog."""


class ModelInactiveError(DomainError):
    """The requested model exists but is not available for inference."""


class ModelLoadError(DomainError):
    """A model's weights or tokenizer could not be loaded."""


class GenerationError(DomainError):
    """Text generation failed."""


class InputTooLargeError(DomainError):
    """Input exceeds the maximum accepted size."""


class EvaluationError(DomainError):
    """Calling the evaluation service failed or returned an unusable response."""


class UnknownMetricError(DomainError):
    """A requested evaluation metric is not defined by the evaluation service.

    Distinct from :class:`EvaluationError`: an unknown metric is a caller mistake
    (surfaced as 404), not an infrastructure failure that evaluation fails open on.
    """
