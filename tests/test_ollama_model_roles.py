from app.config import Settings
from app.core.llm.model_roles import MODEL_ROLES, pure_ollama_chat_mode, resolve_all_model_roles, resolve_model_role, role_for_complexity_score


def test_all_roles_resolve_for_ollama(monkeypatch):
    monkeypatch.setenv("PLLM_PROVIDER", "ollama")
    monkeypatch.setenv("ELLM_PROVIDER", "ollama")
    settings = Settings()
    assert pure_ollama_chat_mode(settings)
    roles = resolve_all_model_roles(settings=settings)
    assert set(roles) == set(MODEL_ROLES)
    for role, cfg in roles.items():
        assert cfg.provider == "ollama"
        assert cfg.model_name
        assert cfg.context_length >= 65536
        assert cfg.max_output_tokens > 0
        assert cfg.keep_alive == "1h"
        assert cfg.cache_key[:2] == ("ollama", role)
        assert cfg.cache_key[-1] == "1h"
    assert roles["tool_worker"].tool_capable is True
    assert roles["verifier"].role == "verifier"


def test_role_for_complexity_score(monkeypatch):
    settings = Settings(PLLM_PROVIDER="ollama", ELLM_PROVIDER="ollama")
    assert role_for_complexity_score(1, settings=settings) == "primary"
    assert role_for_complexity_score(10, settings=settings) == "complex"
    assert resolve_model_role("complex", settings=settings).model_name


def test_role_keep_alive_override(monkeypatch):
    monkeypatch.setenv("PLLM_PROVIDER", "ollama")
    monkeypatch.setenv("ELLM_PROVIDER", "ollama")
    monkeypatch.setenv("MODEL_ROLE_TOOL_WORKER_KEEP_ALIVE", "24h")
    cfg = resolve_model_role("tool_worker", settings=Settings())
    assert cfg.keep_alive == "24h"
    assert cfg.cache_key[-1] == "24h"
