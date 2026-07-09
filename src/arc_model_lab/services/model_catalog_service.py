from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import Model


class ModelCatalogService:
    """Lists catalog models and fetches one by name for the read endpoints."""

    async def list_models(self, session: AsyncSession) -> list[Model]:
        """Return every catalog model, ordered by name."""
        return await ModelRepository(session).list_all()

    async def get(self, session: AsyncSession, name: str) -> Model:
        """Return the catalog model with this name, or raise ModelNotFoundError (404)."""
        return await ModelRepository(session).require_by_name(name)
