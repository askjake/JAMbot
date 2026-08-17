from fastapi import APIRouter

router = APIRouter(prefix="/analytics", tags=["analytics"])

# NOTE:
# At the moment, analytics is primarily driven by background jobs
# (see `app.analytics.service` and `app.analytics.review`).
# This router is intentionally minimal so the FastAPI app can import it
# without exposing additional public endpoints yet. You can extend this
# later with read-only endpoints for ChatSummary / BackendInsight if needed.
