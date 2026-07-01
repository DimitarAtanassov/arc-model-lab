"""FastAPI dependency wiring. Shared singletons are read from application state."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session, sessionmaker

from arc_model_lab.services.inference_service import InferenceService


def get_session(request: Request) -> Iterator[Session]:
    """Yield a request-scoped session, committing on success.

    If the handler raises, the surrounding ``with`` block closes the session,
    which discards (rolls back) the uncommitted transaction before the exception
    propagates.
    """
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as session:
        yield session
        session.commit()


def get_inference_service(request: Request) -> InferenceService:
    service: InferenceService = request.app.state.inference_service
    return service
