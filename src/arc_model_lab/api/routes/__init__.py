from fastapi import APIRouter

from arc_model_lab.api.routes import health, inference

router = APIRouter()
router.include_router(health.router)
router.include_router(inference.router)

__all__ = ["router"]
