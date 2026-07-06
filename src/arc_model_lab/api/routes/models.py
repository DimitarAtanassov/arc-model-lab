"""The model-catalog read endpoints: list the catalog and fetch one entry by name.

Reads only. The catalog is populated out of band (the seed script and the models
CLI); this surface exposes it to the console. Handlers are synchronous to match
the rest of the service.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from arc_model_lab.api.dependencies import get_model_catalog_service, get_session
from arc_model_lab.api.schemas.models import ModelResponse
from arc_model_lab.services.model_catalog_service import ModelCatalogService

SessionDep = Annotated[Session, Depends(get_session)]
ServiceDep = Annotated[ModelCatalogService, Depends(get_model_catalog_service)]

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelResponse])
def list_models(session: SessionDep, service: ServiceDep) -> list[ModelResponse]:
    return [ModelResponse.from_domain(model) for model in service.list_models(session)]


@router.get("/{name}", response_model=ModelResponse)
def get_model(name: str, session: SessionDep, service: ServiceDep) -> ModelResponse:
    """Return one catalog model by its unique name, or 404 if absent."""
    return ModelResponse.from_domain(service.get(session, name))
