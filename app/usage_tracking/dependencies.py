from typing import Annotated
from fastapi import Depends
from app.usage_tracking.service import get_usage_tracking_service, UsageTrackingService

UsageTrackingServiceDep = Annotated[
    UsageTrackingService, Depends(get_usage_tracking_service)
]
