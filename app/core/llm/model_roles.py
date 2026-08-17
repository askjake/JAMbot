"""Provider-neutral model role resolution for Dish-Chat/DishIP.

This module intentionally avoids LangChain imports so config, startup selftests,
and unit tests can run without binding a live model.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

from app.config import Settings, get_settings

MODEL_ROLES: tuple[str, ...] = (
    "primary",
    "efficient",
    "complex",
    "tool_worker",
    "analyst",
    "verifier",
    "title",
    "summary",
)

_PROVIDER_VALUES = {"aws-bedrock", "openai", "anthropic", "ollama"}


@dataclass(frozen=True)
class ModelRoleConfig:
    role: str
    provider: str
    model_name: str
    base_url: str | None
    context_length: int | None
    max_output_tokens: int
    temperature: float | None
    reasoning: bool | None
    tool_capable: bool = False
    keep_alive: str | None = None

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.provider,
            self.role,
            self.model_name,
            self.base_url,
            self.context_length,
            self.max_output_tokens,
            self.keep_alive,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "reasoning": self.reasoning,
            "tool_capable": self.tool_capable,
            "keep_alive": self.keep_alive,
            "cache_key": list(self.cache_key),
        }


def _env_key(role: str, field: str) -> str:
    return f"MODEL_ROLE_{role.upper()}_{field.upper()}"


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    return int(value)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def _as_bool(value: Any, default: bool | None = None) -> bool | None:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _role_dict(settings: Settings, role: str) -> Mapping[str, Any]:
    raw = getattr(settings, "MODEL_ROLES_CONFIG", {}) or {}
    if not isinstance(raw, Mapping):
        return {}
    value = raw.get(role) or raw.get(role.upper()) or {}
    return value if isinstance(value, Mapping) else {}


def _ollama_context(settings: Settings, legacy_ctx: int | None) -> int | None:
    return int(getattr(settings, "OLLAMA_CTX_LEN", None) or legacy_ctx or 32768)


def _default_role_values(settings: Settings, role: str) -> dict[str, Any]:
    primary = {
        "provider": settings.PLLM_PROVIDER,
        "model_name": settings.PLLM_MODEL,
        "base_url": settings.PLLM_API_BASE,
        "context_length": _ollama_context(settings, settings.PLLM_CTX_LEN) if settings.PLLM_PROVIDER == "ollama" else settings.PLLM_CTX_LEN,
        "max_output_tokens": settings.MAX_OUTPUT_COUNT,
        "temperature": settings.DEFAULT_TEMP,
        "reasoning": settings.DEFAULT_REASONING,
        "tool_capable": settings.PLLM_PROVIDER != "ollama",
        "keep_alive": os.environ.get("MODEL_ROLE_PRIMARY_KEEP_ALIVE", settings.OLLAMA_KEEP_ALIVE),
    }
    efficient = {
        "provider": settings.ELLM_PROVIDER or settings.PLLM_PROVIDER,
        "model_name": settings.ELLM_MODEL or settings.PLLM_MODEL,
        "base_url": settings.ELLM_API_BASE or settings.PLLM_API_BASE,
        "context_length": _ollama_context(settings, settings.ELLM_CTX_LEN or settings.PLLM_CTX_LEN) if (settings.ELLM_PROVIDER or settings.PLLM_PROVIDER) == "ollama" else (settings.ELLM_CTX_LEN or settings.PLLM_CTX_LEN),
        "max_output_tokens": 4096,
        "temperature": settings.DEFAULT_TEMP,
        "reasoning": False,
        "tool_capable": (settings.ELLM_PROVIDER or settings.PLLM_PROVIDER) != "ollama",
        "keep_alive": os.environ.get("MODEL_ROLE_EFFICIENT_KEEP_ALIVE", settings.OLLAMA_KEEP_ALIVE),
    }
    complex_role = {
        **primary,
        "model_name": getattr(settings, "COMPLEX_MODEL", None) or getattr(settings, "OPUS_MODEL", None) or settings.PLLM_MODEL,
        "tool_capable": False if settings.PLLM_PROVIDER == "ollama" else primary["tool_capable"],
    }
    tool_worker = {
        **primary,
        "model_name": getattr(settings, "PLLM_TOOL_MODEL", None) or settings.PLLM_MODEL,
        "max_output_tokens": min(settings.MAX_OUTPUT_COUNT, 4096),
        "reasoning": False,
        "tool_capable": True,
    }
    mapping = {
        "primary": primary,
        "efficient": efficient,
        "complex": complex_role,
        "tool_worker": tool_worker,
        "analyst": {**primary, "max_output_tokens": min(settings.MAX_OUTPUT_COUNT, 8192)},
        "verifier": {**primary, "max_output_tokens": 4096, "temperature": 0.0, "reasoning": False, "tool_capable": False},
        "title": {**efficient, "max_output_tokens": 512, "temperature": 0.2, "reasoning": False},
        "summary": {**efficient, "max_output_tokens": 4096, "temperature": 0.3, "reasoning": False},
    }
    return mapping[role]


def _normalize_provider(value: Any) -> str:
    provider = str(value or "").strip()
    if provider not in _PROVIDER_VALUES:
        raise ValueError(f"Unsupported LLM provider {provider!r}; expected one of {sorted(_PROVIDER_VALUES)}")
    return provider


def resolve_model_role(role: str = "primary", settings: Settings | None = None) -> ModelRoleConfig:
    if role not in MODEL_ROLES:
        raise ValueError(f"Unknown model role {role!r}; expected one of {MODEL_ROLES}")
    settings = settings or get_settings()
    defaults = _default_role_values(settings, role)
    cfg = _role_dict(settings, role)
    provider = _normalize_provider(_first(os.getenv(_env_key(role, "PROVIDER")), cfg.get("provider"), defaults["provider"]))
    model_name = _first(os.getenv(_env_key(role, "MODEL")), os.getenv(_env_key(role, "MODEL_NAME")), cfg.get("model"), cfg.get("model_name"), defaults["model_name"])
    base_url = _first(os.getenv(_env_key(role, "BASE_URL")), cfg.get("base_url"), defaults.get("base_url"))
    context_length = _as_int(_first(os.getenv(_env_key(role, "CONTEXT_LENGTH")), os.getenv(_env_key(role, "CTX_LEN")), cfg.get("context_length"), cfg.get("ctx_len"), defaults.get("context_length")))
    max_output_tokens = _as_int(_first(os.getenv(_env_key(role, "MAX_OUTPUT_TOKENS")), os.getenv(_env_key(role, "MAX_OUTPUT")), cfg.get("max_output_tokens"), cfg.get("max_tokens"), defaults.get("max_output_tokens")), settings.MAX_OUTPUT_COUNT)
    temperature = _as_float(_first(os.getenv(_env_key(role, "TEMPERATURE")), cfg.get("temperature"), defaults.get("temperature")))
    reasoning = _as_bool(_first(os.getenv(_env_key(role, "REASONING")), cfg.get("reasoning"), cfg.get("thinking"), defaults.get("reasoning")))
    tool_capable = _as_bool(_first(os.getenv(_env_key(role, "TOOL_CAPABLE")), cfg.get("tool_capable"), defaults.get("tool_capable")), False)
    keep_alive = _first(os.getenv(_env_key(role, 'KEEP_ALIVE')), cfg.get('keep_alive'), defaults.get('keep_alive'))
    if not model_name:
        raise ValueError(f"No model configured for role {role!r} and provider {provider!r}")
    return ModelRoleConfig(role, provider, str(model_name), str(base_url) if base_url else None, context_length, int(max_output_tokens or settings.MAX_OUTPUT_COUNT), temperature, reasoning, bool(tool_capable), str(keep_alive) if keep_alive else None)


def resolve_all_model_roles(settings: Settings | None = None) -> dict[str, ModelRoleConfig]:
    settings = settings or get_settings()
    return {role: resolve_model_role(role, settings=settings) for role in MODEL_ROLES}


def role_for_complexity_score(score: int, threshold: int | None = None, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    threshold = threshold if threshold is not None else getattr(settings, "MODEL_ROUTING_COMPLEXITY_THRESHOLD", getattr(settings, "OPUS_COMPLEXITY_THRESHOLD", 3))
    return "complex" if int(score) >= int(threshold) else "primary"


def is_aws_provider(provider: str | None) -> bool:
    return provider == "aws-bedrock"


def pure_ollama_chat_mode(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return settings.PLLM_PROVIDER == "ollama" and (settings.ELLM_PROVIDER in (None, "ollama"))
