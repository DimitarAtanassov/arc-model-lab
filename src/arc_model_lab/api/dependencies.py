from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_catalog_service import ModelCatalogService


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped async session; the context rolls back on error.

    The service layer owns the commit, so a row is persisted before any success
    response is returned.
    """
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with session_factory() as session:
        yield session


def get_inference_service(request: Request) -> InferenceService:
    service: InferenceService = request.app.state.inference_service
    return service


def get_model_catalog_service(request: Request) -> ModelCatalogService:
    service: ModelCatalogService = request.app.state.model_catalog_service
    return service
