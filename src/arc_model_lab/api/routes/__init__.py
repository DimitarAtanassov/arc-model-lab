from fastapi import APIRouter

from arc_model_lab.api.routes import health, summarize

router = APIRouter()
router.include_router(health.router)
router.include_router(summarize.router)

__all__ = ["router"]
