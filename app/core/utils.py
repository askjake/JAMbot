from datetime import datetime, timezone, UTC
from typing import BinaryIO
import logging
from tempfile import SpooledTemporaryFile
import shutil

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.config import get_settings
from .constants import CONTENT_TYPE_MAPPING

logger = logging.getLogger(__name__)
settings = get_settings()

UUID_REGEX = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

def get_timestr_now_utc() -> str:
    """Get time string YYYY-mm-ddTHH:MM:SS.fffZ"""
    return (
        datetime.strftime(datetime.now(tz=timezone.utc), "%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


def get_utc_now_notz() -> datetime:
    "Get a timezone-less datetime object of utc now"
    return datetime.now(UTC).replace(tzinfo=None)


def get_datestr_now() -> str:
    """Get date string YYYY-mm-dd. Using server timezone"""
    return datetime.now().strftime("%Y-%m-%d")

def datetime_to_iso_utc(time_obj: datetime) -> str:
    """Get time string YY-mm-ddTHH:MM:SS.fffZ from provided datetime obj"""
    return time_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def format_message_content(type: str, content: str) -> dict[str, str]:
    """Format a content item"""
    return {"type": type, type: content}


def extract_lc_msg_content(lc_message: BaseMessage) -> dict[int, dict[str, str]]:
    """Extract message content from a Langchain message and form a list of content items:
    If the content is a simple string, return a single element list with text content
    If the content is a list with multiple items:
        If a user message, return the last item,
        If an ai message, return only items with a defined type mapping.

    Return:
        A dict with key as 0-based numerical index and items as content items
    """
    contents = []

    if isinstance(lc_message.content, str):
        contents.append(format_message_content("text", lc_message.content))
    elif isinstance(lc_message.content, list):
        if isinstance(lc_message, HumanMessage):
            # Get last text block, the actual user message. Guaranteed to have one
            i = -1
            content = lc_message.content[i]
            while content.get("type", "") != "text":
                i -= 1
                content = lc_message.content[i]
            contents.append(format_message_content("text", content.get("text", "")))
        else:
            for lc_content in lc_message.content:
                content_type = lc_content.get("type")
                mapped_type = CONTENT_TYPE_MAPPING.get(content_type)

                # Ignore types that're not explicitly defined
                if mapped_type is None:
                    continue

                content_str = lc_content.get(content_type, "")

                # Special handling for langchain reasoning content
                if mapped_type == "reasoning":
                    reasoning = lc_content.get(content_type)
                    content_str = ""
                    if isinstance(reasoning, dict):
                        content_str = reasoning.get("text", "")
                    elif isinstance(reasoning, str):
                        content_str = reasoning

                contents.append(format_message_content(mapped_type, content_str))

    return {i: item for i, item in enumerate(contents)}


def copy_as_spooled_temporary_file(original: BinaryIO) -> SpooledTemporaryFile:
    """
    Creates a copy of a file-like object to a SpooledTemporaryFile.

    Args:
        original: The file-like object to copy.

    Returns:
        A new SpooledTemporaryFile object containing a copy of the original's content.
    """
    original.seek(0)

    # Determine the mode for the new file (e.g., 'w+b' or 'w+t')
    # SpooledTemporaryFile defaults to 'w+b' if mode isn't readily available
    # or correctly inferred, but best to try and match.
    mode = getattr(original, "mode", "w+b")
    # Ensure it's a read/write mode suitable for copying
    if "w" not in mode and "+" not in mode:
        if "b" in mode:
            mode = "w+b"
        else:
            mode = "w+t"

    copied_stf = SpooledTemporaryFile(max_size=settings.SPOOLED_MAX_SIZE, mode=mode)
    shutil.copyfileobj(original, copied_stf)

    original.seek(0)  # Rewind original file
    copied_stf.seek(0)  # Rewind new file for subsequent reads

    return copied_stf
