#!/usr/bin/env python3
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.db.base import Base
from app.config import get_settings

# Import all models to register them with Base
from app.chat.models import Chat
from app.message.models import MessageMD
from app.vault.models import VaultCred, VaultSession
from app.user.models import UserState
from app.journal.models import UserJournal
from app.attachment.models import Attachment
from app.chat_group.models import ChatGroup
from app.usage_tracking.models import UsageTracking
from app.analytics.models import ChatSummary, BackendInsight, WebSearch
from app.background_mgr.models import BgTask
from app.releases.models import Release
from app.logs.models import LogIngestionJob
from app.agent_mode.models import AgentModeRun
from app.personality.models import UserPersonalityPreference

async def init_db():
    settings = get_settings()
    print(f"Connecting to: {settings.POSTGRES_SQLALCHEMY_URL}")
    
    engine = create_async_engine(settings.POSTGRES_SQLALCHEMY_URL, echo=True)
    
    async with engine.begin() as conn:
        print("Creating all tables...")
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    print("✅ Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())
