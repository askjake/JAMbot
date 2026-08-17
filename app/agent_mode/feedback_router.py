from fastapi import APIRouter, HTTPException
from typing import Optional, List
from app.agent_mode.feedback_collector import FeedbackCollector, FeedbackRating, FeedbackCategory, ResponseFeedback

router = APIRouter(prefix="/feedback", tags=["agent-mode"])
_collector = FeedbackCollector()


@router.post("/submit")
async def submit_feedback(
    message_id: str,
    chat_id: str,
    user_id: str,
    rating: FeedbackRating,
    categories: Optional[List[FeedbackCategory]] = None,
    comment: Optional[str] = None,
    response_text: Optional[str] = None,
):
    """Submit thumbs up/down feedback on an agent response."""
    feedback = _collector.collect_feedback(
        message_id=message_id,
        chat_id=chat_id,
        user_id=user_id,
        rating=rating,
        categories=categories or [],
        comment=comment,
        response_text=response_text,
    )
    return {"feedback_id": str(feedback.feedback_id), "status": "recorded"}


@router.get("/stats")
async def get_feedback_stats(chat_id: Optional[str] = None):
    """Get feedback statistics, optionally filtered by chat."""
    stats = _collector.get_stats(chat_id=chat_id)
    return stats.to_dict()


@router.get("/report")
async def get_improvement_report():
    """Get full feedback improvement report for team review."""
    return {"report": _collector.generate_improvement_report()}