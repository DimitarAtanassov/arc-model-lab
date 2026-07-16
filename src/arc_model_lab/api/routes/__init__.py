from fastapi import APIRouter

from arc_model_lab.api.routes import generation, health, inference, models, presets

router = APIRouter()
router.include_router(health.router)
router.include_router(models.router)
router.include_router(inference.router)
router.include_router(generation.router)
router.include_router(presets.router)

__all__ = ["router"]
