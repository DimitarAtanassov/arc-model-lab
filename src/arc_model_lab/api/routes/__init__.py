from fastapi import APIRouter

from arc_model_lab.api.routes import experiments, health, inference

router = APIRouter()
router.include_router(health.router)
router.include_router(inference.router)
router.include_router(experiments.router)

__all__ = ["router"]
