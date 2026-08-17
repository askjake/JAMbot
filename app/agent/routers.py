from fastapi import APIRouter

from app.agent.miniapps.betareport.router import router as betareport_router
from app.agent.t2i_runtime_status import t2i_runtime_persistence_payload

router = APIRouter(prefix="/agents", tags=["agents"])
router.include_router(betareport_router)


@router.get("/t2i-runtime-persistence")
async def t2i_runtime_persistence() -> dict[str, object]:
    """Read-only, non-secret persistence evidence for guarded T2I runtime."""
    return t2i_runtime_persistence_payload()
