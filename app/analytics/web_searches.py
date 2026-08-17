from fastapi import APIRouter
from sqlalchemy import text
from typing import List, Dict, Any

from app.db.base import sessionmanager
from app.dependencies import UserEmailDep

# Remove the prefix here since it's added in main.py
router = APIRouter(tags=["analytics"])


@router.get("/chat/{chat_id}/web-searches")
async def get_chat_web_searches(
    chat_id: str,
    _current_user: UserEmailDep
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all web searches performed in a specific chat.
    
    Returns:
        List of web searches with query, timestamp, and optionally results
    """
    async with sessionmanager.session() as session:
        result = await session.execute(
            text("""
                SELECT query, results_json, created_at
                FROM web_searches
                WHERE chat_id = :chat_id
                ORDER BY created_at DESC
                LIMIT 50
            """),
            {"chat_id": chat_id},
        )
        rows = result.mappings().all()

    return {
        "items": [
            {
                "query": r["query"],
                "created_at": r["created_at"].isoformat() + "Z",
                "result_count": len(r["results_json"].get("results", [])) if r["results_json"] else 0,
            }
            for r in rows
        ]
    }
