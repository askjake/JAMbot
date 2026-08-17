from typing import BinaryIO, Any
import base64
import logging

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_aws import ChatBedrockConverse

from app.embedding.schemas import EmbeddedDocument
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

CACHE_POINT_BLOCK = {"cachePoint": {"type": "default"}}


def text_content(text: str) -> dict:
    """Create text input content block"""
    return {
        "type": "text",
        "text": text,
    }


def _detect_image_mime_from_bytes(data: bytes) -> str | None:
    # Model-bound image MIME must agree with the payload bytes.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (
        data.startswith(b"RIFF")
        and len(data) >= 12
        and data[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


def _normalize_declared_image_mime(mime_type: str) -> str:
    declared = str(mime_type or "").strip().lower()
    if declared == "image/jpg":
        return "image/jpeg"
    return declared


def image_content(name: str, mime_type: str, image: BinaryIO) -> list[dict]:
    # Attachment metadata can become stale when preprocessing/transcoding
    # changes image bytes. Bedrock validates format against the byte stream,
    # so byte signature wins at the model boundary. No transcoding occurs here.
    raw = image.read()
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("image_content expected BinaryIO.read() to return bytes")

    raw_bytes = bytes(raw)
    declared = _normalize_declared_image_mime(mime_type)
    detected = _detect_image_mime_from_bytes(raw_bytes)
    effective_mime = detected or declared

    encoded_image = base64.b64encode(raw_bytes).decode(encoding="ascii")

    return [
        {"type": "text", "text": f"User uploaded an image: {name}"},
        {
            "type": "image",
            "source_type": "base64",
            "mime_type": effective_mime,
            "data": encoded_image,
        },
    ]



def doc_content(embedded_doc: EmbeddedDocument, in_ctx: bool) -> dict:
    """Create content block.
    Inject whole document if in_ctx is true, else inject only metadata
    """
    text = (
        "<doc_upload>\n"
        f"User uploaded a document: {embedded_doc.name}\n\n"
        f"Keywords: {', '.join(embedded_doc.keywords)}\n\n"
    )
    if in_ctx:
        fulltext = "".join([p.text for p in embedded_doc.passages])
        text += f"Full text: {fulltext}\n\n"
    else:
        text += (
            f"Summary: {embedded_doc.summary}\n\n"
            "The full text is chunked and indexed into the conversation vector database "
            "and can be retrieved in passages with provided tools.\n\n"
        )

    text += "</doc_upload>"

    return {"type": "text", "text": text}


def stringfy_messages(messages: list[BaseMessage]) -> str:
    """
    Stringfy a list of langchain messages. For AI, only response message is included.
    """
    message_str = ""
    for m in messages:
        m_text = f"{m.type}: "
        if isinstance(m.content, str):
            m_text += m.content
        elif isinstance(m.content, list):
            for item in m.content:
                # Only include items with type text
                item_type = item.get("type")
                if item_type == "text":
                    m_text += item.get(item_type)
        message_str += m_text + "\n\n"

    return message_str


def aggressive_cachept(
    messages: list[BaseMessage], max_cachepts: int, exclude_last_turn: bool = False
) -> list[BaseMessage]:
    """Cachepoint at most the last <max_cachepts> human messages
    
    Args:
        messages: List of messages to add cachepoints to
        max_cachepts: Maximum number of cachepoints to add
        exclude_last_turn: If True, skip the most recent human message
    """

    n_cachept = 0
    skip_first = exclude_last_turn
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            if skip_first:
                skip_first = False
                continue
                
            if isinstance(m.content, str):
                m.content = [text_content(m.content), CACHE_POINT_BLOCK]
                n_cachept += 1
            elif isinstance(m.content, list):
                m.content.append(CACHE_POINT_BLOCK)
                n_cachept += 1

        if n_cachept >= max_cachepts:
            break

    return messages


def cleanup_cachept(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove all cachepoints in a list of messages"""

    for m in messages:
        if isinstance(m, HumanMessage):
            # Cachepoint is added as the last content element
            if m.content and m.content[-1] == CACHE_POINT_BLOCK:
                m.content.pop()

    return messages


def cleanup_tmp(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove all messages marked as temporary"""
    messages[:] = [m for m in messages if not m.additional_kwargs.get("tmp")]
    return messages


def construct_messages(
    messages: list[BaseMessage],
    system_prompt: BaseMessage,
    task_inst: BaseMessage | None = None,
) -> list[BaseMessage]:
    """Construct message list with system prompt first, then messages.
    If task_inst is provided, insert it before the last human message.
    """
    if not task_inst:
        return [system_prompt, *messages]

    # Find the last human message index
    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx is None:
        return [system_prompt, *messages, task_inst]

    return [
        system_prompt,
        *messages[:last_human_idx],
        task_inst,
        *messages[last_human_idx:],
    ]


def set_model_config(model: BaseChatModel, config: dict[str, Any]):
    """
    Set the model config w.r.t the specific model's kwarg format.
    """
    if hasattr(model, "model_kwargs"):
        model.model_kwargs = model.model_kwargs or {}

    temp = config.get("temperature") or settings.DEFAULT_TEMP
    temp = max(min(temp, 1.0), 0)

    if hasattr(model, "temperature"):
        setattr(model, "temperature", temp)
    elif hasattr(model, "model_kwargs"):
        model.model_kwargs["temperature"] = temp
    else:
        logger.warning(
            "Failed setting model temperature due to unsupported model attribute."
        )

    # if (log_p := config.get("log_p")) is not None and isinstance(log_p, float):
    #     log_p = max(min(log_p, 1.0), 0)
    #     if hasattr(model, "log_p"):
    #         setattr(model, "log_p", log_p)
    #     elif hasattr(model,"model_kwargs"):
    #         model.model_kwargs["log_p"] = log_p
    #     else:
    #         logger.warning(
    #             "Failed setting model log_p due to unsupported model attribute."
    #         )

    enable_reasoning = settings.DEFAULT_REASONING
    if config.get("reasoning") is not None:
        enable_reasoning = config.get("reasoning")

    if enable_reasoning:
        if isinstance(model, ChatBedrockConverse):
            model.temperature = 1  # temp needs to be 1 for reasoning mode.
            model.additional_model_request_fields = (
                model.additional_model_request_fields or {}
            )
            model.additional_model_request_fields["thinking"] = {
                "type": "enabled",
                "budget_tokens": settings.REASONING_BUDGET,
            }
        else:
            logger.warning("Failed enable reasoning mode due to unsupported model type")
    else:
        if isinstance(model, ChatBedrockConverse):
            model.temperature = temp  # temp needs to be 1 for reasoning mode.
            model.additional_model_request_fields = (
                model.additional_model_request_fields or {}
            )
            model.additional_model_request_fields.pop("thinking", None)
