"""Maps domain errors to HTTP responses with safe, client-facing messages."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from arc_model_lab.domain import (
    GenerationError,
    InputTooLargeError,
    ModelInactiveError,
    ModelLoadError,
    ModelNotFoundError,
    UnknownMetricError,
)

logger = logging.getLogger(__name__)


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


async def _model_not_found(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_404_NOT_FOUND, str(exc) or "Model not found")


async def _unknown_metric(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_404_NOT_FOUND, str(exc) or "Requested metric does not exist")


async def _model_inactive(request: Request, exc: Exception) -> Response:
    return _error(status.HTTP_409_CONFLICT, str(exc) or "Model is not active")


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


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ModelNotFoundError, _model_not_found)
    app.add_exception_handler(ModelInactiveError, _model_inactive)
    app.add_exception_handler(InputTooLargeError, _input_too_large)
    app.add_exception_handler(ModelLoadError, _model_load_error)
    app.add_exception_handler(GenerationError, _generation_error)
    app.add_exception_handler(UnknownMetricError, _unknown_metric)
