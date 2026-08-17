"""
Provider-neutral model routing metrics API router.

Primary endpoints:
- GET  /rest/api/v1/internal/model-routing-stats
- POST /rest/api/v1/internal/model-routing-stats/reset

The legacy route names are kept as deprecated aliases so old dashboards do not
break, but the returned metric names are provider-neutral.
"""

from fastapi import APIRouter
from app.agent.complexity_detector import routing_metrics

router = APIRouter(tags=["internal"])


@router.get("/internal/model-routing-stats")
async def get_model_routing_stats():
    """Return current provider-neutral model-role routing statistics."""
    return routing_metrics.get_stats()


@router.post("/internal/model-routing-stats/reset")
async def reset_model_routing_stats():
    """Reset routing metrics for testing and dashboard maintenance."""
    routing_metrics.reset()
    return {"status": "reset", "message": "Model routing metrics cleared"}


@router.get("/internal/legacy-opus-routing-stats", deprecated=True)
async def get_legacy_opus_routing_stats():
    """Deprecated compatibility alias for old dashboards."""
    return await get_model_routing_stats()


@router.post("/internal/legacy-opus-routing-stats/reset", deprecated=True)
async def reset_legacy_opus_routing_stats():
    """Deprecated compatibility alias for old dashboards."""
    return await reset_model_routing_stats()
