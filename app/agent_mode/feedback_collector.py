# Ported from dish-chat quick-wins
"""
Feedback Collection System for Dish-Chat Agent
===============================================

Allows users to rate responses and provide detailed feedback
to continuously improve the agent.
"""

from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
import json


class FeedbackRating(str, Enum):
    """Simple thumbs up/down rating"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


class FeedbackCategory(str, Enum):
    """Categories for detailed feedback"""
    INCORRECT_INFO = "incorrect_info"
    UNHELPFUL = "unhelpful"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    UNCLEAR = "unclear"
    SECURITY_CONCERN = "security_concern"
    MISSING_VERIFICATION = "missing_verification"
    OTHER = "other"


class ResponseFeedback(BaseModel):
    """Individual feedback on an agent response"""
    feedback_id: UUID = Field(default_factory=uuid4)
    message_id: str
    chat_id: str
    user_id: str
    
    rating: FeedbackRating
    categories: List[FeedbackCategory] = Field(default_factory=list)
    comment: Optional[str] = None
    
    response_text: Optional[str] = None
    prompt_text: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    agent_version: Optional[str] = None
    session_metadata: Optional[Dict] = None
    
    class Config:
        use_enum_values = True


class FeedbackStats(BaseModel):
    """Aggregated feedback statistics"""
    total_feedback: int = 0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    
    @property
    def satisfaction_rate(self) -> float:
        """Percentage of positive feedback"""
        if self.total_feedback == 0:
            return 0.0
        return (self.thumbs_up_count / self.total_feedback) * 100
    
    @property
    def satisfaction_emoji(self) -> str:
        """Emoji representation of satisfaction"""
        rate = self.satisfaction_rate
        if rate >= 90:
            return "🤩"
        elif rate >= 75:
            return "😊"
        elif rate >= 50:
            return "😐"
        elif rate >= 25:
            return "😕"
        else:
            return "😞"
    
    def to_dict(self) -> Dict:
        return {
            "total": self.total_feedback,
            "thumbs_up": self.thumbs_up_count,
            "thumbs_down": self.thumbs_down_count,
            "satisfaction_rate": f"{self.satisfaction_rate:.1f}%",
            "emoji": self.satisfaction_emoji
        }


class FeedbackCollector:
    """Service for collecting and analyzing user feedback"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.feedback_items: List[ResponseFeedback] = []
        
        if storage_path:
            self._load_feedback()
    
    def collect_feedback(self,
                        message_id: str,
                        chat_id: str,
                        user_id: str,
                        rating: FeedbackRating,
                        categories: List[FeedbackCategory] = None,
                        comment: Optional[str] = None,
                        response_text: Optional[str] = None,
                        prompt_text: Optional[str] = None) -> ResponseFeedback:
        """Collect feedback from user"""
        
        feedback = ResponseFeedback(
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
            rating=rating,
            categories=categories or [],
            comment=comment,
            response_text=response_text,
            prompt_text=prompt_text
        )
        
        self.feedback_items.append(feedback)
        
        if self.storage_path:
            self._save_feedback()
        
        return feedback
    
    def get_stats(self, 
                 time_range_days: Optional[int] = None,
                 user_id: Optional[str] = None) -> FeedbackStats:
        """Get feedback statistics"""
        filtered = self.feedback_items
        
        if time_range_days:
            cutoff = datetime.now().timestamp() - (time_range_days * 24 * 60 * 60)
            filtered = [f for f in filtered if f.created_at.timestamp() >= cutoff]
        
        if user_id:
            filtered = [f for f in filtered if f.user_id == user_id]
        
        stats = FeedbackStats(
            total_feedback=len(filtered),
            thumbs_up_count=sum(1 for f in filtered if f.rating == FeedbackRating.THUMBS_UP),
            thumbs_down_count=sum(1 for f in filtered if f.rating == FeedbackRating.THUMBS_DOWN)
        )
        
        return stats
    
    def get_negative_feedback(self, limit: int = 10) -> List[ResponseFeedback]:
        """Get recent negative feedback for review"""
        negative = [f for f in self.feedback_items if f.rating == FeedbackRating.THUMBS_DOWN]
        negative.sort(key=lambda x: x.created_at, reverse=True)
        return negative[:limit]
    
    def get_category_breakdown(self) -> Dict[str, int]:
        """Get count of each feedback category"""
        breakdown = {}
        for feedback in self.feedback_items:
            for category in feedback.categories:
                key = category.value if hasattr(category, 'value') else str(category)
                breakdown[key] = breakdown.get(key, 0) + 1
        return breakdown
    
    def generate_improvement_report(self) -> str:
        """Generate markdown report of improvement opportunities"""
        stats = self.get_stats()
        category_breakdown = self.get_category_breakdown()
        recent_negative = self.get_negative_feedback(5)
        
        lines = []
        lines.append("# Agent Feedback Report\n")
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f"Generated: {time_str}\n")
        lines.append("---\n")
        
        # Overall stats
        lines.append("## Overall Satisfaction\n")
        satisfaction_str = f"{stats.satisfaction_rate:.1f}%"
        lines.append(f"{stats.satisfaction_emoji} **{satisfaction_str}** satisfaction rate\n")
        lines.append(f"- 👍 Thumbs up: {stats.thumbs_up_count}")
        lines.append(f"- 👎 Thumbs down: {stats.thumbs_down_count}")
        lines.append(f"- Total feedback: {stats.total_feedback}\n")
        
        # Category breakdown
        if category_breakdown:
            lines.append("## Issues Reported\n")
            sorted_categories = sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_categories:
                cat_name = category.replace('_', ' ').title()
                lines.append(f"- **{cat_name}**: {count} reports")
            lines.append("")
        
        # Recent negative feedback
        if recent_negative:
            lines.append("## Recent Negative Feedback\n")
            for i, feedback in enumerate(recent_negative, 1):
                time_str = feedback.created_at.strftime('%Y-%m-%d %H:%M')
                lines.append(f"### {i}. {time_str}")
                if feedback.categories:
                    cats = ', '.join(c.value if hasattr(c, 'value') else str(c) for c in feedback.categories)
                    lines.append(f"   - Categories: {cats}")
                if feedback.comment:
                    lines.append(f'   - Comment: "{feedback.comment}"')
                lines.append("")
        
        return "\n".join(lines)
    
    def _save_feedback(self):
        """Save feedback to JSON file"""
        if not self.storage_path:
            return
        
        data = [f.dict() for f in self.feedback_items]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_feedback(self):
        """Load feedback from JSON file"""
        if not self.storage_path:
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.feedback_items = [ResponseFeedback(**item) for item in data]
        except FileNotFoundError:
            pass
