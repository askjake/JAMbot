from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field
from app.core.schemas import Pagination


class JournalAnalysis(BaseModel):
    """Schema matching the LLM output structure for journal analysis"""
    summary: str
    psychoanalysis: Optional[dict] = None
    interaction_patterns: Optional[dict] = None
    technical_profile: Optional[dict] = None  # Added this field
    topics: Optional[dict] = None
    preferences: Optional[dict] = None  # Renamed from user_preferences to match LLM output


class JournalCreate(BaseModel):
    owner_id: str
    chat_id: Optional[UUID] = None
    journal_type: str = "conversation"
    summary: str
    psychoanalysis: Optional[dict] = None
    interaction_patterns: Optional[dict] = None
    user_preferences: Optional[dict] = None  # Kept as user_preferences for DB field
    topics: Optional[dict] = None
    sentiment_analysis: Optional[dict] = None
    conversation_start: Optional[datetime] = None
    conversation_end: Optional[datetime] = None
    message_count: Optional[int] = None


class JournalResponse(BaseModel):
    journal_id: str
    owner_id: str
    chat_id: Optional[str] = None
    journal_type: str
    summary: str
    psychoanalysis: Optional[dict] = None
    interaction_patterns: Optional[dict] = None
    user_preferences: Optional[dict] = None
    topics: Optional[dict] = None
    sentiment_analysis: Optional[dict] = None
    created_at: str
    conversation_start: Optional[str] = None
    conversation_end: Optional[str] = None
    message_count: Optional[int] = None
    status: str


class JournalListResponse(Pagination[JournalResponse]):
    pass


class JournalDetailResponse(JournalResponse):
    pass
