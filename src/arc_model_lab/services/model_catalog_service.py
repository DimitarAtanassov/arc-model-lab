"""Read access to the model catalog (the ``models`` table).

A thin read service over :class:`ModelRepository`. The catalog is seeded and
maintained out of band (the seed script and the models CLI), so the HTTP read
surface only needs to list it and fetch one entry by name. This is deliberately
separate from :class:`~arc_model_lab.services.model_service.ModelService`, which
owns the HuggingFace runtime (weights and generation), not persistence.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.domain import Model


class ModelCatalogService:
    """Lists catalog models and fetches one by name for the read endpoints."""

    def list_models(self, session: Session) -> list[Model]:
        """Return every catalog model, ordered by name."""
        return ModelRepository(session).list_all()

    def get(self, session: Session, name: str) -> Model:
        """Return the catalog model with this name, or raise ``ModelNotFoundError`` (404)."""
        return ModelRepository(session).require_by_name(name)
