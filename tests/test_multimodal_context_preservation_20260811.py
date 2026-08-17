from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.message.compression import (
    MULTIMODAL_IMAGE_TOKEN_ESTIMATE,
    count_message_tokens,
    count_tokens,
    trim_message_history,
)
import app.agent.agents.agentic_rag as agentic_rag


def _image_message(label: str, chars_per_image: int, count: int = 7) -> HumanMessage:
    payload = "A" * chars_per_image
    content = [{"type": "text", "text": label}]
    for _ in range(count):
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": payload,
                },
            }
        )
    return HumanMessage(content=content)


def _text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def test_image_base64_size_does_not_scale_text_token_estimate():
    small = _image_message("atlas benchmark", 1_000)
    huge = _image_message("atlas benchmark", 300_000)

    small_tokens = count_message_tokens([small])
    huge_tokens = count_message_tokens([huge])

    assert small_tokens == huge_tokens
    assert 7 * MULTIMODAL_IMAGE_TOKEN_ESTIMATE <= huge_tokens < 36_000


def test_text_only_counting_semantics_remain_unchanged():
    msg = HumanMessage(content="hello multimodal world")
    assert count_message_tokens([msg]) == count_tokens(msg.content) + 10


def test_trim_history_keeps_current_large_multimodal_turn():
    current = _image_message(
        "This is a blinded image-only benchmark. Use ONLY the seven images.",
        300_000,
    )
    history = [
        HumanMessage(content="Earlier benchmark instructions."),
        AIMessage(content="Acknowledged."),
        current,
    ]

    trimmed = trim_message_history(history, max_tokens=140_000)

    assert trimmed
    assert any(msg is current for msg in trimmed)
    assert any("blinded image-only benchmark" in _text(msg).lower() for msg in trimmed)


def test_bedrock_shape_repair_preserves_original_task_and_followup():
    task = HumanMessage(content="ORIGINAL TASK: analyze seven T2I atlas images.")
    followup = HumanMessage(content="find in your workspace and continue")
    repaired = agentic_rag.ensure_bedrock_converse_message_shape(
        [
            AIMessage(content="leading invalid fragment"),
            task,
            AIMessage(content="intermediate answer"),
            followup,
        ]
    )

    texts = [_text(msg) for msg in repaired]
    assert isinstance(repaired[0], HumanMessage)
    assert "ORIGINAL TASK" in texts[0]
    assert any("find in your workspace" in text for text in texts)


def test_truncate_messages_preserves_current_multimodal_user_turn(monkeypatch):
    monkeypatch.setattr(
        agentic_rag,
        "apply_tiered_compression",
        lambda messages, target_tokens: list(messages),
    )
    monkeypatch.setattr(
        agentic_rag,
        "apply_token_efficiency_layer",
        lambda messages: list(messages),
    )
    monkeypatch.setattr(agentic_rag, "_active_history_budget_tokens", lambda: 140_000)
    monkeypatch.setattr(
        agentic_rag,
        "_active_provider_requires_bedrock_shape",
        lambda: True,
    )

    current = _image_message(
        "This is a blinded image-only benchmark. Use ONLY the seven images.",
        300_000,
    )
    result = agentic_rag.truncate_messages(
        [
            HumanMessage(content="Earlier task."),
            AIMessage(content="Earlier answer."),
            current,
        ],
        max_messages=100,
    )

    texts = [_text(msg) for msg in result]
    assert result
    assert any("blinded image-only benchmark" in text.lower() for text in texts)
    assert not any(
        text.startswith("Continue from the available conversation context")
        for text in texts
    )


def test_followup_retains_prior_image_task_under_bedrock_repair(monkeypatch):
    monkeypatch.setattr(
        agentic_rag,
        "apply_tiered_compression",
        lambda messages, target_tokens: list(messages),
    )
    monkeypatch.setattr(
        agentic_rag,
        "apply_token_efficiency_layer",
        lambda messages: list(messages),
    )
    monkeypatch.setattr(agentic_rag, "_active_history_budget_tokens", lambda: 140_000)
    monkeypatch.setattr(
        agentic_rag,
        "_active_provider_requires_bedrock_shape",
        lambda: True,
    )

    image_task = _image_message("ORIGINAL IMAGE TASK", 300_000)
    followup = HumanMessage(content="do you see the attached images?")

    result = agentic_rag.truncate_messages(
        [
            AIMessage(content="leading fragment"),
            image_task,
            AIMessage(content="answer"),
            followup,
        ],
        max_messages=100,
    )

    texts = [_text(msg) for msg in result]
    assert isinstance(result[0], HumanMessage)
    assert any("ORIGINAL IMAGE TASK" in text for text in texts)
    assert any("do you see the attached images?" in text for text in texts)
