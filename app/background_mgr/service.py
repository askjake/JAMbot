import asyncio
import logging
import inspect
from typing import Any, Callable, Coroutine, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_db_session_ctxmgr

from .schemas import BgTaskSchema, ProgressStatusEnum, BgTaskProgress
from .repository import BgTaskRepository
from .models import BgTask

logger = logging.getLogger(__name__)
settings = get_settings()

class TaskProgressTracker:
    """
    Tracks the progress of a background task and updates the database.
    """
    
    def __init__(
        self,
        task_id: str,
        task_type: Optional[str],
        bg_task_repo: Optional[BgTaskRepository] = None
    ):
        """
        Initialize the progress tracker.
        
        Args:
            task_id: Unique identifier for the task
            db_session: Database session for persistence
        """
        self.task_id = task_id
        self.task_type = task_type or "generic"
        self._lock = asyncio.Lock()
        self._bg_task_repo = bg_task_repo or BgTaskRepository()

    async def _get_task(self, session: AsyncSession) -> BgTask:
        """Retrieve the task progress from db. Create one if it doesn't exist. """

        db_obj = await self._bg_task_repo.get_one_by_id(session, self.task_id)
        if not db_obj:
            new_task = BgTaskSchema(task_id=self.task_id, task_type=self.task_type)
            db_obj = await self._bg_task_repo.create_one(session, obj_in=new_task)
        
        return db_obj

    async def start(self, total: int, message: str = ""):
        """
        Start the task with the given total steps.
        
        Args:
            total: Total number of steps for the task
            message: Optional message to store with the task
        """
        async with self._lock:
            async with get_db_session_ctxmgr() as session:
                task = await self._get_task(session)
                update = {
                    "total": total,
                    "message": message,
                    "status": ProgressStatusEnum.processing
                }
                await self._bg_task_repo.update(session, db_obj=task, obj_in=update)
    
    async def increment(self, progress: int = 1, message: str = ""):
        """
        Increment the task progress.
        
        Args:
            progress: Number of steps to increment
            message: Optional message to update
        """
        async with self._lock:
            async with get_db_session_ctxmgr() as session:
                task = await self._get_task(session)
                
                # Can't update failed task without reset
                if task.status == ProgressStatusEnum.failed:
                    return
                
                if task.progress >= task.total:
                    logger.warning(
                        f"Task progress exceeding task total for: {self.task_type} {self.task_id}"
                    )
                update = {
                    "progress": task.progress + progress
                }
                if message:
                    update["message"] = message
                await self._bg_task_repo.update(session, db_obj=task, obj_in=update)
    
    async def complete(self, message: str = ""):
        """
        Mark the task as complete.
        
        Args:
            message: Optional completion message
        """
        async with self._lock:
            async with get_db_session_ctxmgr() as session:
                task = await self._get_task(session)

                # Can't update failed task without reset
                if task.status == ProgressStatusEnum.failed:
                    return
                
                update = {
                    "progress": task.total,
                    "status": ProgressStatusEnum.ready
                }
                if message:
                    update["message"] = message
                await self._bg_task_repo.update(session, db_obj=task, obj_in=update)
    
    async def fail(self, message: str):
        """
        Mark the task as failed.
        
        Args:
            message: Error message explaining the failure
        """
        async with self._lock:
            async with get_db_session_ctxmgr() as session:
                task = await self._get_task(session)

                # Can't update failed task without reset
                if task.status == ProgressStatusEnum.failed:
                    return

                update = {
                    "status": ProgressStatusEnum.failed,
                    "message": message
                }
                await self._bg_task_repo.update(session, db_obj=task, obj_in=update)
    
    async def reset(self):
        async with self._lock:
            async with get_db_session_ctxmgr() as session:
                task = await self._get_task(session)
                update = {
                    "total": 0,
                    "progress": 0,
                    "message": "",
                    "status": ProgressStatusEnum.queued
                }
                await self._bg_task_repo.update(session, db_obj=task, obj_in=update)

    async def with_tracker(self, coro: Coroutine, message: str = ""):
        """
        Execute a coroutine and track its progress.
        
        Args:
            coro: Coroutine to execute
            message: Optional message to update on success
            
        Returns:
            The result of the coroutine
        """
        try:
            result = await coro
            await self.increment(1, message)
            return result
        except Exception as e:
            logger.exception("Unexpected error caught while executing task.", stack_info=True)
            raise e

class BackgroundTaskManager:
    """
    Manages concurrent background tasks with controlled concurrency limits and tracker.
    
    This class provides a way to run multiple coroutines concurrently while
    limiting the maximum number of tasks that can execute simultaneously.
    It also handles task cleanup and graceful shutdown.
    """
        
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackgroundTaskManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_concurrency: int = 1024):
        """
        Initialize the background task manager.
        
        Args:
            max_concurrency: Maximum number of tasks that can run concurrently.
        """
        if self._initialized:
            return
            
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: dict[asyncio.Task, TaskProgressTracker] = dict()
        self._initialized = True
        self._repo = BgTaskRepository()
    
    async def _run_with_semaphore(
        self,
        coro: Coroutine,
    ) -> Any:
        """
        Run a coroutine with semaphore-controlled concurrency and error handling.
        
        Args:
            coro: The coroutine to execute.
            
        Returns:
            The result of the coroutine execution.
        """
        async with self._semaphore:
            try:
                return await coro
            except Exception as e:
                logger.exception("Unexpected error caught while executing background task.", stack_info=True)
                raise e
    
    def add_task(
        self,
        func: Callable,
        task_id: str,
        /,
        *args,
        task_type: Optional[str] = None,
        **kwargs
    ) -> asyncio.Task:
        """
        Add a coroutine to be executed as a background task.
        
        Args:
            coro: The coroutine to execute.
            callback: Optional callback function to call when task completes.
            
        Returns:
            The created asyncio.Task object.
        """
        # Validate that func supports tracker
        func_sig = inspect.signature(func)
        bg_tracker_param = func_sig.parameters.get("bg_tracker")
        assert(
            bg_tracker_param is not None and
            bg_tracker_param.annotation in [
                TaskProgressTracker,
                Optional[TaskProgressTracker],
                Union[TaskProgressTracker, type(None)]
            ]
        )

        tracker = TaskProgressTracker(
            task_id=task_id,
            task_type=task_type or "generic",
            bg_task_repo=self._repo
        )

        # Handle non-async function
        if inspect.iscoroutinefunction(func):
            coro = func(*args, bg_tracker=tracker, **kwargs)
        else:
            coro = asyncio.to_thread(func, *args, bg_tracker=tracker, **kwargs)
        
        task = asyncio.create_task(self._run_with_semaphore(coro))
        self._tasks[task] = tracker
        task.add_done_callback(self._tasks.pop)

        return task
    
    async def shutdown(self, max_wait_time: int = settings.CLEANUP_TIMEOUT):
        """
        Gracefully shut down all background tasks.
        
        Waits for tasks to complete naturally for a specified time before
        forcibly canceling any remaining tasks.
        
        Args:
            max_wait_time: Maximum seconds to wait for tasks to complete.
        """
        if not self._tasks:
            return
            
        # Wait for a short timeout
        _, pending = await asyncio.wait(self._tasks, timeout=0.1)
        
        # If tasks remain, wait with progress updates
        wait_time = 0
        while pending and wait_time < max_wait_time:
            logger.info(
                f"Waiting on {len(pending)} background tasks to finish. {wait_time}/{max_wait_time}s"
            )
            _, pending = await asyncio.wait(self._tasks, timeout=1)
            wait_time += 1
            
        # Cancel any remaining tasks
        if pending:
            logger.warning(
                f"Shutting down with {len(pending)} unfinished background tasks."
            )
            for task in pending:
                await self._tasks[task].fail("Task killed on server shutdown.")
                task.cancel()
                
            # Wait briefly for cancellations to process
            await asyncio.wait(pending, timeout=1)

    async def get_task_status(
        self,
        session: AsyncSession,
        task_id: str
    ) -> Optional[BgTaskProgress]   :
        progress = await self._repo.get_one_by_id(session, task_id)
        if progress:
            return BgTaskProgress(
                task_id=progress.task_id,
                task_type=progress.task_type,
                progress=progress.calc_progress_percentage(),
                status=progress.status,
                message=progress.message
            )
        else:
            logger.warning(f"Requested non-existent task progress: {task_id}")
            return None

# Singleton instance of the background task manager
_task_manager = BackgroundTaskManager()

# Expose the singleton instance for direct access if needed
def get_task_manager() -> BackgroundTaskManager:
    """Get the singleton instance of the background task manager"""
    return _task_manager