from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/logassist", tags=["logassist"])


@router.get("/journals")
async def list_journals(chat_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """
    Return journal metadata for the frontend.
    
    Normalizes the Log Assist gateway response into a consistent format:
    { "items": [...] }
    
    Args:
        chat_id: Optional filter by chat_id
    
    Returns:
        Dictionary with "items" key containing list of journals
    """
    try:
        # Try to import and use the log_assist_gateway
        from app.tools.log_assist_gateway import logassist_get_journal_files
        
        # Get raw data from the gateway
        raw_response = await logassist_get_journal_files.ainvoke({})
        
        # Parse the response
        import json
        try:
            raw = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        except:
            raw = {}
            raw = {}
        
        # Extract journals from various possible shapes
        journals = raw.get("journals") or raw.get("items") or raw
        
        if isinstance(journals, dict):
            # Convert dict-of-journals -> list-of-journals
            items: List[Dict[str, Any]] = list(journals.values())
        elif isinstance(journals, list):
            items = journals
        else:
            items = []
        
        # Optional: filter by chat_id if provided
        if chat_id and items:
            items = [
                j for j in items
                if j.get("chat_id") in (None, "", chat_id) or j.get("chatId") == chat_id
            ]
        
        logger.info(f"Returning {len(items)} journals for chat_id={chat_id}")
        return {"items": items}
        
    except ImportError:
        # Log Assist gateway not available
        logger.warning("Log Assist gateway not available")
        return {"items": []}
    except Exception as e:
        logger.error(f"Failed to fetch journals: {e}")
        return {"items": []}
