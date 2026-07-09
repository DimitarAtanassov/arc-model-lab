from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from arc_model_lab.domain import (
    GenerationError,
    InferenceNotFoundError,
    InputTooLargeError,
    InvalidGenerationConfigError,
    ModelInactiveError,
    ModelLoadError,
    ModelNotFoundError,
    PromptRenderError,
    PromptTemplateNotFoundError,
)

logger = logging.getLogger(__name__)


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


async def _model_not_found(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_404_NOT_FOUND, str(exc) or "Model not found")


async def _model_inactive(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_409_CONFLICT, str(exc) or "Model is not active")


async def _inference_not_found(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_404_NOT_FOUND, str(exc) or "Inference not found")


async def _invalid_generation_config(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc) or "Invalid generation config")


async def _input_too_large(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_413_CONTENT_TOO_LARGE, str(exc) or "Input too large")


async def _model_load_error(request: Request, exc: Exception) -> Response:
    # Client gets a safe message; the real cause (download failure, OOM, bad
    # cache dir, ...) is only recoverable from the server log.
    logger.error("Model load failed", exc_info=exc)
    return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is temporarily unavailable")


async def _generation_error(request: Request, exc: Exception) -> Response:
    logger.error("Text generation failed", exc_info=exc)
    return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Text generation failed")


async def _prompt_template_not_found(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_404_NOT_FOUND, str(exc) or "Prompt template not found")


async def _prompt_render_error(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc) or "Invalid prompt variables")


async def _unhandled(request: Request, exc: Exception) -> Response:
    """Last-resort boundary: log with a correlation id, return a safe 500 body.

    The real cause stays in the server log (keyed by correlation_id); the
    client gets a generic message and the same id to quote in a support request.
    """
    correlation_id = str(uuid4())
    logger.error(
        "Unhandled error",
        exc_info=exc,
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "correlation_id": correlation_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ModelNotFoundError, _model_not_found)
    app.add_exception_handler(ModelInactiveError, _model_inactive)
    app.add_exception_handler(InferenceNotFoundError, _inference_not_found)
    app.add_exception_handler(InvalidGenerationConfigError, _invalid_generation_config)
    app.add_exception_handler(InputTooLargeError, _input_too_large)
    app.add_exception_handler(ModelLoadError, _model_load_error)
    app.add_exception_handler(GenerationError, _generation_error)
    app.add_exception_handler(PromptTemplateNotFoundError, _prompt_template_not_found)
    app.add_exception_handler(PromptRenderError, _prompt_render_error)
    app.add_exception_handler(Exception, _unhandled)
