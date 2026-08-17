import contextlib
from typing import Any, AsyncIterator

from app.config import get_settings
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

class Base(DeclarativeBase):
    pass

class DatabaseSessionManager:
    def __init__(self, host: str, engine_kwargs: dict[str, Any] = {}):
        # Production-ready connection pool settings for high concurrency
        default_kwargs = {
            "pool_size": 50,
            "max_overflow": 150,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_timeout": 30,
        }
        default_kwargs.update(engine_kwargs)
        self._engine = create_async_engine(host, **default_kwargs)
        self._sessionmaker = async_sessionmaker(autocommit=False, bind=self._engine)

    async def close(self):
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        await self._engine.dispose()

        self._engine = None
        self._sessionmaker = None

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")

        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise Exception("DatabaseSessionManager is not initialized")

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

settings = get_settings()
sessionmanager = DatabaseSessionManager(settings.POSTGRES_SQLALCHEMY_URL, {"echo": settings.ECHO_SQL})

# For FastAPI Dependency
async def get_db_session():
    async with sessionmanager.session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# For direct use as an async ctx manager
@contextlib.asynccontextmanager
async def get_db_session_ctxmgr():
    async with sessionmanager.session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise