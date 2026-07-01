"""Application composition root: wiring, lifespan, and the ASGI ``app`` object."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from arc_model_lab.api.routes import router
from arc_model_lab.config import Settings, get_settings
from arc_model_lab.db.base import create_engine_from_url, create_session_factory
from arc_model_lab.db.repositories import ModelRepository
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    engine = create_engine_from_url(settings.database_url, echo=settings.db_echo)
    session_factory = create_session_factory(engine)

    model_service = ModelService(settings)
    model_service.load()

    with session_factory() as session:
        registered_model = ModelRepository(session).get_or_create(model_service.descriptor)
        session.commit()

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.model_service = model_service
    app.state.inference_service = InferenceService(model_service, registered_model)

    try:
        yield
    finally:
        engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="arc-model-lab", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or get_settings()
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    """Console-script entrypoint that starts the ASGI server."""
    settings = get_settings()
    uvicorn.run("arc_model_lab.main:app", host=settings.api_host, port=settings.api_port)
