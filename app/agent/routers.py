from fastapi import APIRouter

from app.agent.miniapps.betareport.router import router as betareport_router

router = APIRouter(prefix="/agents", tags=["agents"])
router.include_router(betareport_router)
