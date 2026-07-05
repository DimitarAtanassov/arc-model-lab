"""Domain errors. The API layer maps these to HTTP status codes."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class ModelNotFoundError(DomainError):
    """The requested model is not registered in the catalog."""


class DeployedModelUnavailableError(DomainError):
    """The configured deployed model is missing or inactive.

    A server-side misconfiguration, not a client mistake: callers of ``/inference``
    do not choose the model, so this surfaces as 503, not 404/409.
    """


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


class ExperimentNotFoundError(DomainError):
    """The requested experiment does not exist."""


class ExperimentNameConflictError(DomainError):
    """An experiment with the requested name already exists."""


class InvalidGenerationConfigError(DomainError):
    """A generation config names an unknown knob or an invalid value.

    A client-boundary validation error (422). Its read-path counterpart is
    :class:`CorruptStoredDataError`: the same failure on data loaded from storage
    is a server fault (500), not a client mistake.
    """


class CorruptStoredDataError(DomainError):
    """Persisted data failed validation when read back into the domain.

    A server-side data-integrity fault: the repository raises it when stored JSON
    (for example an experiment's generation config) cannot be rebuilt, so it
    surfaces as 500, unlike the 422 a client boundary returns for invalid input.
    """
