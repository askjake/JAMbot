import logging
from datetime import datetime, date
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncGenerator, Optional

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.tracers.context import register_configure_hook
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.chat.service import ChatService, get_chat_service
from app.db import get_db_session_ctxmgr

from .schemas import UsageTrackingCreate, TokenUsageResp
from .repository import UsageTrackingRepository

logger = logging.getLogger(__name__)
# Dedicated logger for token drift analysis.
# Set log level to INFO on this logger to activate drift output in production.
# Example: logging.getLogger("app.usage_tracking.token_drift").setLevel(logging.INFO)
token_drift_logger = logging.getLogger("app.usage_tracking.token_drift")
settings = get_settings()


def get_usage_tracking_service():
    return UsageTrackingService()


def _feed_calibration(profile: dict, usage_metadata: dict) -> None:
    """
    Feed actual token counts back to the adaptive calibration system.
    
    Retrieves the pre-invocation estimate from the token_counter stash and
    compares it against Bedrock API actual count. Updates the calibration factor
    so future offline estimates converge toward Bedrock API true counts.
    """
    try:
        from app.message.token_counter import (
            get_last_estimate,
            record_calibration_point,
            get_calibration_stats,
        )
    except ImportError:
        return  # token_counter module not available

    estimated = get_last_estimate()
    if estimated <= 0:
        return
    
    actual = usage_metadata.get("input_tokens", 0)
    # Include cache tokens in the actual count (they were part of the input)
    cache_read = usage_metadata.get("input_token_details", {}).get("cache_read", 0)
    cache_create = usage_metadata.get("input_token_details", {}).get("cache_creation", 0)
    total_actual = actual + cache_read + cache_create
    
    if total_actual <= 0:
        return
    
    try:
        record_calibration_point(estimated, total_actual)
        stats = get_calibration_stats()
        token_drift_logger.info(
            "CALIBRATION_FEEDBACK estimated=%d actual=%d factor=%.4f samples=%d converged=%s",
            estimated, total_actual, stats["factor"], stats["samples"], stats["converged"]
        )
    except Exception as e:
        logger.warning(f"Calibration feedback error (non-fatal): {e}")

def _log_token_drift(model_name: str, usage_metadata: dict) -> None:
    """
    Compare tiktoken-estimated token counts against exact Bedrock API counts and
    emit structured drift metrics to the token_drift_logger.





    The compression pipeline uses tiktoken (cl100k_base) to estimate token counts
    before invoking the model. This function captures the ground-truth counts from
    the Bedrock Converse API response and computes:

      - Cache hit rate: % of input tokens served from Anthropic prompt cache
      - Cache efficiency: ratio of cache_read / (cache_create + cache_read)
      - A structured summary line parseable by log aggregators (e.g. CloudWatch Logs Insights)

    This data answers the key question: how much work is the Bedrock prompt cache
    actually doing, and how accurate is our pre-invocation estimation?

    Note: tiktoken estimation drift vs Bedrock actual is logged when the calling
    layer provides an "estimated_input_tokens" key in the profile dict (Phase 2).
    This function handles Phase 1: emit actuals + cache breakdown unconditionally.
    """
    input_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)
    total_tokens = usage_metadata.get("total_tokens", 0)
    token_details = usage_metadata.get("input_token_details", {})
    cache_read = token_details.get("cache_read", 0)
    cache_creation = token_details.get("cache_creation", 0)

    # Total billed input = standard input + cache_creation write + cache_read
    total_input = input_tokens + cache_creation + cache_read

    # Cache hit rate: what fraction of input tokens came from cache (not re-processed)
    cache_hit_rate = (cache_read / total_input * 100) if total_input > 0 else 0.0

    # Cache efficiency: of all cached content, what fraction was a read-hit vs new write
    cache_total = cache_read + cache_creation
    cache_efficiency = (cache_read / cache_total * 100) if cache_total > 0 else 0.0

    token_drift_logger.info(
        "BEDROCK_TOKEN_ACTUAL"
        " model=%s"
        " input_tokens=%d"
        " output_tokens=%d"
        " total_tokens=%d"
        " cache_read=%d"
        " cache_creation=%d"
        " total_input_billed=%d"
        " cache_hit_rate_pct=%.1f"
        " cache_efficiency_pct=%.1f",
        model_name,
        input_tokens,
        output_tokens,
        total_tokens,
        cache_read,
        cache_creation,
        total_input,
        cache_hit_rate,
        cache_efficiency,
    )


class UsageTrackingService:
    def __init__(
        self,
        usage_tracking_repo: Optional[UsageTrackingRepository] = None,
        chat_service: Optional[ChatService] = None,
    ):
        self.usage_tracking_repo = usage_tracking_repo or UsageTrackingRepository()
        self.chat_service = chat_service or get_chat_service()

    async def get_usage_by_owner(
        self,
        db: AsyncSession,
        user_email: str,
        *,
        before: datetime | date | None = None,
        after: datetime | date | None = None,
    ) -> TokenUsageResp:
        """
        Retrieve usage tracking records for a specific owner, optionally filtered by a timestamp.

        params:
            db (AsyncSession): The database session.
            owner_id (str): The owner's email.
            after (Optional[datetime]): If provided, only records after this timestamp are returned.
        returns:
            List of UsageTracking records.
        """
        records = await self.usage_tracking_repo.get_by_owner(
            db, user_email, before=before, after=after
        )

        resp = TokenUsageResp()
        if records:
            resp.input_token = sum(
                r.input_tokens + r.input_cache_create + r.input_cache_read
                for r in records
            )
            resp.output_token = sum(r.output_tokens for r in records)
            resp.cost = sum(r.input_cost + r.output_cost for r in records)

        return resp

    async def get_usage_by_chat(
        self,
        db: AsyncSession,
        user_email: str,
        chat_id: str,
        is_vault: bool = False,
    ) -> TokenUsageResp:
        """
        Retrieve usage tracking records for a specific chat.

        params:
            db (AsyncSession): The database session.
            user_email (str): The user's email.
            chat_id (str): The chat's uuid.
            is_vault (bool): Whether the chat is in vault mode.
        returns:
            List of UsageTracking records.
        """
        # Verify chat exists and belongs to the user
        await self.chat_service.get_chat_if_authorized(
            db, chat_id, user_email, is_vault_mode=is_vault
        )
        records = await self.usage_tracking_repo.get_by_chat(db, chat_id)

        resp = TokenUsageResp()
        if records:
            resp.input_token = sum(
                r.input_tokens + r.input_cache_create + r.input_cache_read
                for r in records
            )
            resp.output_token = sum(r.output_tokens for r in records)
            resp.cost = sum(r.input_cost + r.output_cost for r in records)

        return resp

    @asynccontextmanager
    async def get_async_usage_metadata_callback(
        self,
        name: str = "usage_metadata_callback",
        profile: Optional[dict[str, str]] = None,
        db: Optional[AsyncSession] = None,
    ) -> AsyncGenerator[UsageMetadataCallbackHandler, None]:
        """Get usage metadata callback.

        Get context manager for tracking usage metadata across chat model calls using
        ``AIMessage.usage_metadata``.
        Save the usage metadata to DB after the context manager exits.

        Args:
            name (str): The name of the context variable. Defaults to
                ``'usage_metadata_callback'``.

        Example:
            .. code-block:: python

                from langchain.chat_models import init_chat_model
                from langchain_core.callbacks import get_usage_metadata_callback

                llm_1 = init_chat_model(model="openai:gpt-4o-mini")
                llm_2 = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

                with get_usage_metadata_callback() as cb:
                    llm_1.invoke("Hello")
                    llm_2.invoke("Hello")
                    print(cb.usage_metadata)

            .. code-block:: none

                {'gpt-4o-mini-2024-07-18': {'input_tokens': 8,
                'output_tokens': 10,
                'total_tokens': 18,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}},
                'claude-3-5-haiku-20241022': {'input_tokens': 8,
                'output_tokens': 21,
                'total_tokens': 29,
                'input_token_details': {'cache_read': 0, 'cache_creation': 0}}}

        """

        usage_metadata_callback_var: ContextVar[
            Optional[UsageMetadataCallbackHandler]
        ] = ContextVar(name, default=None)
        register_configure_hook(usage_metadata_callback_var, inheritable=True)
        cb = UsageMetadataCallbackHandler()
        usage_metadata_callback_var.set(cb)
        try:
            yield cb
            if profile:
                # Save usage metadata to DB here if needed
                # {'us.anthropic.claude-sonnet-4-20250514-v1:0': {'input_tokens': 71, 'output_tokens': 789, 'total_tokens': 860, 'input_token_details': {'cache_creation': 0, 'cache_read': 0}}}
                assert db, "Database session must be provided to save usage metadata."
                assert profile.get("owner_email"), "Profile must contain owner_email."
                assert profile.get("chat_id"), "Profile must contain chat_id."
                assert profile.get("task"), "Profile must contain task."

                records = []
                for model_name, usage_metadata in cb.usage_metadata.items():
                    # ── Token drift & cache observability (Phase 1) ───────────────────
                    # Emit structured log of exact Bedrock token counts and cache
                    # breakdown so we can measure prompt-cache effectiveness over time.
                    # grep: BEDROCK_TOKEN_ACTUAL
                    _log_token_drift(model_name, usage_metadata)
                    # ─────────────────────────────────────────────────────────────────
                    # ── Calibration feedback (Phase 2) ────────────────────────────────
                    # If the caller provided an estimated input token count, feed it
                    # back to the adaptive calibration system so future estimates
                    # converge toward Bedrock API actual counts.
                    _feed_calibration(profile, usage_metadata)
                    # ─────────────────────────────────────────────────────────────────
                    record = UsageTrackingCreate(
                        owner_id=profile["owner_email"],
                        chat_id=profile["chat_id"],
                        model=model_name,
                        task=profile["task"],
                        input_tokens=usage_metadata.get("input_tokens", 0),
                        input_cache_read=usage_metadata.get(
                            "input_token_details", {}
                        ).get("cache_read", 0),
                        input_cache_create=usage_metadata.get(
                            "input_token_details", {}
                        ).get("cache_creation", 0),
                        output_tokens=usage_metadata.get("output_tokens", 0),
                    )
                    records.append(record)

                try:
                    await self.usage_tracking_repo.create_many(db, objs_in=records)
                    logger.info(f"Saved {len(records)} usage tracking records")
                except Exception as e:
                    logger.error(
                        f"Unexpected error when saving usage metadata: {e}",
                        exc_info=True,
                    )
        finally:
            usage_metadata_callback_var.set(None)

    async def track_astream_generator(
        self,
        astream: AsyncGenerator[str, None],
        profile: Optional[dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Track usage metadata for an async generator stream.

        params:
            astream (AsyncGenerator): The async generator to track.
            db (AsyncSession): The database session.
            profile (Optional[dict[str, str]]): Optional profile information for tracking.
        returns:
            AsyncGenerator yielding the items from the astream.
        """
        async with get_db_session_ctxmgr() as db:
            async with self.get_async_usage_metadata_callback(profile=profile, db=db):
                async for item in astream:
                    yield item
