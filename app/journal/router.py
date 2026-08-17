from typing import Annotated, Optional
from fastapi import APIRouter, Path, Query, HTTPException

from app.dependencies import DBSessionDep, UserEmailDep
from app.core.utils import datetime_to_iso_utc, UUID_REGEX
from app.chat.dependencies import ChatServiceDep
from app.chat.exceptions import ChatNotFoundError, NotAuthorizedError
from app.vault.dependencies import VaultKeyDep

from .dependencies import JournalServiceDep
from .schemas import JournalListResponse, JournalDetailResponse, JournalResponse
from .exceptions import JournalNotFoundError

router = APIRouter()


@router.get("/users/me/journals")
async def get_my_journals(
    email: UserEmailDep,
    db_session: DBSessionDep,
    journal_service: JournalServiceDep,
    page: int = Query(ge=1, default=1),
    limit: int = Query(ge=1, le=100, default=50),
    journal_type: Optional[str] = Query(default=None),
) -> JournalListResponse:
    """List user's journal entries with pagination"""
    offset = (page - 1) * limit
    
    journals = await journal_service.get_user_journals(
        db_session, email, limit=limit, offset=offset, journal_type=journal_type
    )
    
    total_docs = await journal_service.count_user_journals(
        db_session, email, journal_type=journal_type
    )
    
    # Convert to response schema
    journal_responses = [
        JournalResponse(
            journal_id=str(j.journal_id),
            owner_id=j.owner_id,
            chat_id=str(j.chat_id) if j.chat_id else None,
            journal_type=j.journal_type,
            summary=j.summary,
            psychoanalysis=j.psychoanalysis,
            interaction_patterns=j.interaction_patterns,
            user_preferences=j.user_preferences,
            topics=j.topics,
            sentiment_analysis=j.sentiment_analysis,
            created_at=datetime_to_iso_utc(j.created_at),
            conversation_start=datetime_to_iso_utc(j.conversation_start) if j.conversation_start else None,
            conversation_end=datetime_to_iso_utc(j.conversation_end) if j.conversation_end else None,
            message_count=j.message_count,
            status=j.status,
        )
        for j in journals
    ]
    
    end_idx = offset + limit
    return JournalListResponse(
        docs=journal_responses,
        totalDocs=total_docs,
        limit=limit,
        page=page,
        totalPages=(total_docs + limit - 1) // limit,
        hasNextPage=end_idx < total_docs,
        nextPage=page + 1 if end_idx < total_docs else None,
        hasPrevPage=page > 1,
        prevPage=page - 1 if page > 1 else None,
    )


@router.get("/users/me/journals/{journal_id}")
async def get_journal_detail(
    journal_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    journal_service: JournalServiceDep,
) -> JournalDetailResponse:
    """Get detailed journal entry"""
    journal = await journal_service.journal_repo.get_one_by_id(db_session, journal_id)
    
    if not journal:
        raise HTTPException(status_code=404, detail="Journal not found")
    
    if journal.owner_id != email:
        raise HTTPException(status_code=403, detail="Not authorized to access this journal")
    
    return JournalDetailResponse(
        journal_id=str(journal.journal_id),
        owner_id=journal.owner_id,
        chat_id=str(journal.chat_id) if journal.chat_id else None,
        journal_type=journal.journal_type,
        summary=journal.summary,
        psychoanalysis=journal.psychoanalysis,
        interaction_patterns=journal.interaction_patterns,
        user_preferences=journal.user_preferences,
        topics=journal.topics,
        sentiment_analysis=journal.sentiment_analysis,
        created_at=datetime_to_iso_utc(journal.created_at),
        conversation_start=datetime_to_iso_utc(journal.conversation_start) if journal.conversation_start else None,
        conversation_end=datetime_to_iso_utc(journal.conversation_end) if journal.conversation_end else None,
        message_count=journal.message_count,
        status=journal.status,
    )


@router.post("/chats/{chat_id}/journals")
async def generate_journal_for_chat(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    journal_service: JournalServiceDep,
    chat_service: ChatServiceDep,
    vault_key: VaultKeyDep,
) -> JournalResponse:
    """Manually trigger journal generation for a chat"""
    try:
        # Verify user has access to this chat
        await chat_service.get_chat_if_authorized(
            db_session, chat_id, email, vault_key != ""
        )
        
        # Generate and store journal
        journal = await journal_service.generate_and_store_journal(
            db_session, chat_id, email, vault_key
        )
        
        if not journal:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate journal entry"
            )
        
        return JournalResponse(
            journal_id=str(journal.journal_id),
            owner_id=journal.owner_id,
            chat_id=str(journal.chat_id) if journal.chat_id else None,
            journal_type=journal.journal_type,
            summary=journal.summary,
            psychoanalysis=journal.psychoanalysis,
            interaction_patterns=journal.interaction_patterns,
            user_preferences=journal.user_preferences,
            topics=journal.topics,
            sentiment_analysis=journal.sentiment_analysis,
            created_at=datetime_to_iso_utc(journal.created_at),
            conversation_start=datetime_to_iso_utc(journal.conversation_start) if journal.conversation_start else None,
            conversation_end=datetime_to_iso_utc(journal.conversation_end) if journal.conversation_end else None,
            message_count=journal.message_count,
            status=journal.status,
        )
        
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")
