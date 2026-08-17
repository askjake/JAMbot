from typing import Optional
from functools import cached_property
import datetime
import logging

from pydantic import BaseModel, computed_field
from .constants import MODEL_PRICING

logger = logging.getLogger(__name__)


class UsageTrackingCreate(BaseModel):
    owner_id: str
    chat_id: str
    model: str
    task: str
    input_tokens: Optional[int] = 0
    input_cache_read: Optional[int] = 0
    input_cache_create: Optional[int] = 0
    output_tokens: Optional[int] = 0

    @computed_field
    @cached_property
    def input_cost(self) -> float:
        pricing = MODEL_PRICING.get(self.model)
        if pricing is None:
            logger.warning(f"Model '{self.model}' not found in MODEL_PRICING. Defaulting cost to 0.")
            return 0.0
        return (
            pricing["cache_read"] * self.input_cache_read
            + pricing["cache_create"] * self.input_cache_create
            + pricing["input"] * self.input_tokens
        )

    @computed_field
    @cached_property
    def output_cost(self) -> float:
        pricing = MODEL_PRICING.get(self.model)
        if pricing is None:
            logger.warning(f"Model '{self.model}' not found in MODEL_PRICING. Defaulting cost to 0.")
            return 0.0
        return pricing["output"] * self.output_tokens


class TokenUsageResp(BaseModel):
    input_token: int = 0
    output_token: int = 0
    cost: float = 0.0
