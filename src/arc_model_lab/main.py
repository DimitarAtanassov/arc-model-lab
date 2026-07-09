from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from arc_model_lab.api.errors import register_exception_handlers
from arc_model_lab.api.routes import router
from arc_model_lab.clients.arc_eval_client import EvalSettings, build_arc_eval_client
from arc_model_lab.config import Settings, get_settings
from arc_model_lab.db.base import create_async_engine_from_url, create_async_session_factory
from arc_model_lab.services.evaluation_service import EvaluationService
from arc_model_lab.services.inference_service import InferenceService
from arc_model_lab.services.model_catalog_service import ModelCatalogService
from arc_model_lab.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    engine = create_async_engine_from_url(settings.database_url, echo=settings.db_echo)
    session_factory = create_async_session_factory(engine)
    model_service = ModelService(settings)
    eval_settings = EvalSettings()
    eval_client = build_arc_eval_client(eval_settings)

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.eval_settings = eval_settings
    app.state.model_service = model_service
    app.state.inference_service = InferenceService(model_service)
    app.state.evaluation_service = EvaluationService(eval_client)
    app.state.model_catalog_service = ModelCatalogService()

    try:
        yield
    finally:
        if eval_client is not None:
            await eval_client.aclose()
        await engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="arc-model-lab", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or get_settings()
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    """Console-script entrypoint that starts the ASGI server."""
    settings = get_settings()
    uvicorn.run("arc_model_lab.main:app", host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    run()
