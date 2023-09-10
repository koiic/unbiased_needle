from fastapi import APIRouter

from .endpoints import ml

router = APIRouter()
router.include_router(ml.ml_router, prefix="/ml")
