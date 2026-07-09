from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class ModelNotFoundError(DomainError):
    """The requested model is not registered in the catalog."""


class ModelInactiveError(DomainError):
    """The requested model exists but is not active for online inference.

    /inference serves only active models, so deactivating a model takes it out
    of production serving (409). Experiments deliberately bypass this gate so a
    candidate model can be evaluated before it is activated.
    """


class ModelLoadError(DomainError):
    """A model's weights or tokenizer could not be loaded."""


class GenerationError(DomainError):
    """Text generation failed."""


class InputTooLargeError(DomainError):
    """Input exceeds the maximum accepted size."""


class InferenceNotFoundError(DomainError):
    """The requested inference does not exist."""


class InvalidGenerationConfigError(DomainError):
    """A generation config names an unknown knob or an invalid value (422)."""
