"""
Periodic task to check for idle chats and trigger journal generation.

This module runs as a background task that checks ONCE PER DAY for chats
that have been idle for a configurable amount of time and triggers
journal generation for them. Chats are flagged as 'checked' after
evaluation to prevent redundant re-checks.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text, select, and_, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_db_session_ctxmgr
from app.chat.models import Chat
from app.message.models import MessageMD
from app.analytics.models import ChatSummary

logger = logging.getLogger(__name__)
settings = get_settings()

# Default: run once per day (1440 minutes)
_DEFAULT_CHECK_INTERVAL_MINUTES = 1440



async def _analytics_tables_available(db) -> bool:
    """Return False if analytics migrations have not been applied yet."""
    try:
        result = await db.execute(text("SELECT to_regclass('chat_summary')"))
        return bool(result.scalar())
    except Exception:
        await db.rollback()
        return False


async def check_idle_chats(
    idle_threshold_minutes: int = 30,
    min_messages: int = 5,
) -> dict:
    """
    Check for idle chats and trigger journal generation.

    A chat is considered idle if:
    1. It has NOT been previously marked as idle_checked
    2. It has at least min_messages messages
    3. The last message was sent more than idle_threshold_minutes ago
    4. No journal entry has been created for it yet

    After evaluation, every qualifying chat is flagged idle_checked=True
    so it will NOT be scanned again on subsequent runs.

    Args:
        idle_threshold_minutes: Minutes of inactivity before triggering journal
        min_messages: Minimum number of messages required

    Returns:
        dict with counts: checked, triggered, skipped
    """
    from app.analytics.service import summarise_conversation_async
    from app.background_mgr.service import get_task_manager
    import uuid as uuid_module

    stats = {"checked": 0, "triggered": 0, "skipped": 0, "already_summarized": 0}

    try:
        async with get_db_session_ctxmgr() as db:
            if not await _analytics_tables_available(db):
                logger.warning("analytics tables are not available; skipping idle checker until migrations are applied")
                return
            now_naive = datetime.utcnow()
            cutoff_time = now_naive - timedelta(minutes=idle_threshold_minutes)
            recent_cutoff = now_naive - timedelta(days=7)

            # Only fetch chats that have NOT been checked yet
            stmt = (
                select(Chat)
                .where(
                    and_(
                        Chat.idle_checked == False,  # noqa: E712
                        Chat.last_message_at < cutoff_time,
                        Chat.last_message_at > recent_cutoff,
                    )
                )
            )
            result = await db.execute(stmt)
            idle_chats = result.scalars().all()

            logger.info(f"Idle checker: found {len(idle_chats)} unchecked idle chats")

            # Collect IDs of chats we evaluate so we can bulk-flag them
            checked_chat_ids: list = []

            for chat in idle_chats:
                stats["checked"] += 1
                checked_chat_ids.append(chat.chat_id)

                try:
                    # Check if this chat already has a journal entry
                    existing_summary_count = await db.execute(
                        select(func.count(ChatSummary.id)).where(
                            ChatSummary.chat_id == str(chat.chat_id)
                        )
                    )
                    summary_count = existing_summary_count.scalar()

                    if summary_count > 0:
                        logger.debug(
                            f"Chat {chat.chat_id} already has {summary_count} journal "
                            f"{'entry' if summary_count == 1 else 'entries'}, skipping"
                        )
                        stats["already_summarized"] += 1
                        continue

                    # Count messages in this chat
                    message_count_result = await db.execute(
                        select(func.count(MessageMD.message_id)).where(
                            MessageMD.chat_id == chat.chat_id
                        )
                    )
                    message_count = message_count_result.scalar()

                    if message_count < min_messages:
                        logger.debug(
                            f"Chat {chat.chat_id} only has {message_count} messages, skipping"
                        )
                        stats["skipped"] += 1
                        continue

                    # Trigger journal generation
                    task_manager = get_task_manager()
                    task_id = str(uuid_module.uuid4())

                    task_manager.add_task(
                        summarise_conversation_async,
                        task_id,
                        chat_id=str(chat.chat_id),
                        owner_email=chat.owner_id,
                        task_type="chat_summary_idle",
                    )

                    stats["triggered"] += 1
                    logger.info(
                        f"Triggered idle journal for chat {chat.chat_id} "
                        f"(idle {(now_naive - chat.last_message_at).total_seconds() / 60:.0f}m, "
                        f"{message_count} msgs)"
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing idle chat {chat.chat_id}: {e}",
                        exc_info=True,
                    )
                    continue

            # Bulk-flag all evaluated chats as checked so they are never re-scanned
            if checked_chat_ids:
                await db.execute(
                    update(Chat)
                    .where(Chat.chat_id.in_(checked_chat_ids))
                    .values(idle_checked=True)
                )
                await db.commit()
                logger.info(
                    f"Marked {len(checked_chat_ids)} chats as idle_checked=True"
                )

    except Exception as e:
        logger.error(f"Error in check_idle_chats: {e}", exc_info=True)

    logger.info(
        f"Idle checker complete: checked={stats['checked']}, "
        f"triggered={stats['triggered']}, skipped={stats['skipped']}, "
        f"already_summarized={stats['already_summarized']}"
    )
    return stats


async def idle_chat_checker_loop(
    check_interval_minutes: int = _DEFAULT_CHECK_INTERVAL_MINUTES,
    idle_threshold_minutes: int = 30,
    min_messages: int = 5,
) -> None:
    """
    Run the idle chat checker in a loop (once per day by default).

    Args:
        check_interval_minutes: How often to run (default: 1440 = 24h)
        idle_threshold_minutes: Minutes of inactivity before triggering journal
        min_messages: Minimum number of messages required
    """
    logger.info(
        f"Starting idle chat checker loop: "
        f"check_interval={check_interval_minutes}m (~{check_interval_minutes / 60:.1f}h), "
        f"idle_threshold={idle_threshold_minutes}m, "
        f"min_messages={min_messages}"
    )

    while True:
        try:
            await check_idle_chats(
                idle_threshold_minutes=idle_threshold_minutes,
                min_messages=min_messages,
            )
        except Exception as e:
            logger.error(f"Error in idle chat checker loop: {e}")

        # Wait before next check (default: 24 hours)
        await asyncio.sleep(check_interval_minutes * 60)


# Global task reference to allow graceful shutdown
_idle_checker_task: Optional[asyncio.Task] = None


def start_idle_chat_checker(
    check_interval_minutes: int = _DEFAULT_CHECK_INTERVAL_MINUTES,
    idle_threshold_minutes: int = 30,
    min_messages: int = 5,
) -> asyncio.Task:
    """
    Start the idle chat checker as a background task.

    Args:
        check_interval_minutes: How often to run (default: 1440 = 24h)
        idle_threshold_minutes: Minutes of inactivity before triggering journal
        min_messages: Minimum messages required

    Returns:
        The asyncio.Task running the checker
    """
    global _idle_checker_task

    if _idle_checker_task is not None and not _idle_checker_task.done():
        logger.warning("Idle chat checker is already running")
        return _idle_checker_task

    _idle_checker_task = asyncio.create_task(
        idle_chat_checker_loop(
            check_interval_minutes=check_interval_minutes,
            idle_threshold_minutes=idle_threshold_minutes,
            min_messages=min_messages,
        )
    )

    return _idle_checker_task


async def stop_idle_chat_checker(timeout: int = 5) -> None:
    """
    Stop the idle chat checker gracefully.

    Args:
        timeout: Maximum seconds to wait for the task to finish
    """
    global _idle_checker_task

    if _idle_checker_task is None or _idle_checker_task.done():
        logger.info("Idle chat checker is not running")
        return

    logger.info("Stopping idle chat checker...")
    _idle_checker_task.cancel()

    try:
        await asyncio.wait_for(_idle_checker_task, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Idle chat checker did not stop within {timeout}s")
    except asyncio.CancelledError:
        logger.info("Idle chat checker stopped successfully")

    _idle_checker_task = None
