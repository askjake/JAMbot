from pathlib import Path
from app.agent_mode.adaptive_system_prompt import build_system_prompt

FORBIDDEN = ("Claude", "Sonnet", "Opus", "Anthropic", "Bedrock")


def test_chat_prompt_identity_and_contract():
    text = Path("app/agent/agents/prompts/chat_system_prompt.txt").read_text()
    assert "You are Dish-Chat, an internal engineering assistant backed by the configured local LLM provider." in text
    assert "Evidence discipline" in text
    assert "Tool discipline" in text
    assert "verifier pass" in text
    for term in FORBIDDEN:
        assert term not in text


def test_agent_mode_prompt_identity_and_contract():
    text = build_system_prompt("chat-test")
    assert text.splitlines()[0] == "You are Dish-Chat, an internal engineering assistant backed by the configured local LLM provider."
    assert "Evidence discipline" in text
    assert "Tool discipline" in text
    for term in FORBIDDEN:
        assert term not in text
