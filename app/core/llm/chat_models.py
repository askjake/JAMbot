from __future__ import annotations

import asyncio
import configparser
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import boto3

try:
    from langchain_core.language_models.chat_models import BaseChatModel
except ModuleNotFoundError:  # Allows config/selftest imports in minimal snapshots.
    BaseChatModel = Any  # type: ignore

from app.config import get_settings
from app.core.llm.model_roles import (
    MODEL_ROLES,
    ModelRoleConfig,
    is_aws_provider,
    pure_ollama_chat_mode,
    resolve_model_role,
)

settings = get_settings()
logger = logging.getLogger(__name__)

# Bedrock Application Inference Profile ARNs. These ensure all calls are tagged/tracked.
# These are used only when provider == "aws-bedrock".
PROFILE_SONNET = "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/5c511xksna83"
PROFILE_HAIKU = "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/wpnvchycfust"
PROFILE_OPUS = "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/m4hvzo6r2exy"
PROFILE_EMBED = "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/4xgakngy389z"


def _resolve_model_arn(model_id: str) -> str:
    """Resolve a Bedrock model identifier to an Application Inference Profile ARN."""
    if model_id and model_id.startswith("arn:aws:bedrock:"):
        return model_id
    mapping = {
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": PROFILE_SONNET,
        "us.anthropic.claude-sonnet-4-20250514-v1:0": PROFILE_SONNET,
        "anthropic.claude-3-5-sonnet-20241022-v2:0": PROFILE_SONNET,
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": PROFILE_HAIKU,
        "anthropic.claude-3-5-haiku-20241022-v1:0": PROFILE_HAIKU,
        "anthropic.claude-3-haiku-20240307-v1:0": PROFILE_HAIKU,
        "anthropic.claude-opus-4-20250514-v1:0": PROFILE_OPUS,
        "us.anthropic.claude-opus-4-20250514-v1:0": PROFILE_OPUS,
        "anthropic.claude-3-opus-20240229-v1:0": PROFILE_OPUS,
        "cohere.embed-multilingual-v3": PROFILE_EMBED,
        "amazon.titan-embed-text-v1": PROFILE_EMBED,
    }
    resolved = mapping.get(model_id)
    if resolved:
        logger.warning("Direct Bedrock model ID %r mapped to a profile ARN. Update config to use the ARN directly.", model_id)
        return resolved
    raise ValueError(f"Unknown Bedrock model ID {model_id!r}. Use an Application Inference Profile ARN or known alias.")


# Role paths use tuple keys: provider, role, model, base URL, context length, max output.
# Legacy string buckets remain only for backward-compatible callers before migration.
_model_cache: dict[Any, dict[str, Any]] = {}
_AWS_CREDENTIALS_PATH = os.environ.get("AWS_SHARED_CREDENTIALS_FILE", os.path.expanduser("~/.aws/credentials"))
_refresh_task = None
_refresh_lock = asyncio.Lock()
_task_started = False


def _read_aws_credentials_metadata() -> dict | None:
    try:
        if not os.path.exists(_AWS_CREDENTIALS_PATH):
            return None
        mtime = os.path.getmtime(_AWS_CREDENTIALS_PATH)
        parser = configparser.RawConfigParser()
        parser.read(_AWS_CREDENTIALS_PATH)
        if not parser.has_section("default"):
            return None
        access_key_id = parser.get("default", "aws_access_key_id", fallback=None)
        expiration_raw = parser.get("default", "expiration", fallback=None)
        expiration_dt = None
        if expiration_raw:
            exp = expiration_raw.strip()
            if exp.endswith("Z"):
                exp = exp[:-1] + "+00:00"
            try:
                expiration_dt = datetime.fromisoformat(exp)
                if expiration_dt.tzinfo is not None:
                    expiration_dt = expiration_dt.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                logger.warning("Could not parse AWS credential expiration metadata")
        return {"access_key_id": access_key_id, "expiration_dt": expiration_dt, "mtime": mtime, "fingerprint": (access_key_id, expiration_raw, mtime)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading AWS credentials metadata: %s", exc)
        return None


def _get_next_token_expiry() -> datetime:
    meta = _read_aws_credentials_metadata()
    if meta and meta.get("expiration_dt"):
        return meta["expiration_dt"] - timedelta(minutes=10)
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        botocore_session = getattr(session, "_session", None)
        creds = getattr(botocore_session, "_credentials", None) if credentials and botocore_session else None
        expiry = getattr(creds, "_expiry_time", None) if creds else None
        if expiry:
            if expiry.tzinfo is not None:
                expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
            return expiry - timedelta(minutes=10)
    except Exception:
        pass
    return datetime.utcnow() + timedelta(minutes=50)


_SECGATEWAY_PATH = os.path.expanduser("~/secgateway/bin/secgateway.py")
_SECGATEWAY_PYTHON = "/usr/bin/python3"


def _run_secgateway() -> bool:
    if not os.path.exists(_SECGATEWAY_PATH):
        logger.warning("secgateway not found at configured path, skipping")
        return False
    try:
        result = subprocess.run([_SECGATEWAY_PYTHON, _SECGATEWAY_PATH], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("secgateway ran successfully")
            return True
        logger.error("secgateway exited %s", result.returncode)
        return False
    except subprocess.TimeoutExpired:
        logger.error("secgateway timed out after 30s")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to run secgateway: %s", exc, exc_info=True)
        return False


def _clear_boto3_credential_cache() -> None:
    try:
        if boto3.DEFAULT_SESSION is not None:
            boto3.DEFAULT_SESSION = None
            logger.info("Cleared boto3 DEFAULT_SESSION credential cache")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error clearing boto3 credential cache: %s", exc)


def _create_ollama_model(role_config: ModelRoleConfig) -> BaseChatModel:
    """Create a ChatOllama instance for a resolved model role."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError("Ollama provider selected but langchain-ollama is not installed. Run: pip install -U langchain-ollama") from exc
    kwargs: dict[str, Any] = {
        "model": role_config.model_name,
        "num_predict": role_config.max_output_tokens,
        "validate_model_on_init": False,
        "disable_streaming": False,
    }
    if role_config.context_length:
        kwargs["num_ctx"] = role_config.context_length
    if role_config.base_url:
        kwargs["base_url"] = role_config.base_url
    if role_config.temperature is not None:
        kwargs["temperature"] = role_config.temperature
    logger.info(
        "Creating Ollama model role=%s model=%r ctx=%s output=%s tool_capable=%s",
        role_config.role,
        role_config.model_name,
        role_config.context_length,
        role_config.max_output_tokens,
        role_config.tool_capable,
    )
    if role_config.keep_alive:
        kwargs["keep_alive"] = role_config.keep_alive
    return ChatOllama(**kwargs)


def _create_bedrock_model(role_config: ModelRoleConfig) -> BaseChatModel:
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:
        raise ImportError("aws-bedrock provider selected but langchain-aws is not installed. Run: pip install -U langchain-aws") from exc
    _clear_boto3_credential_cache()
    resolved_model = _resolve_model_arn(role_config.model_name)
    logger.info("Creating Bedrock model role=%s via profile ARN", role_config.role)
    return ChatBedrockConverse(
        model=resolved_model,
        max_tokens=role_config.max_output_tokens,
        region_name=settings.AWS_REGION,
        disable_streaming=False,
        provider="anthropic",
    )


def _create_model_for_role(role_config: ModelRoleConfig) -> BaseChatModel:
    if role_config.provider == "ollama":
        return _create_ollama_model(role_config)
    if role_config.provider == "aws-bedrock":
        return _create_bedrock_model(role_config)
    raise NotImplementedError(f"Provider {role_config.provider!r} not implemented in active model factory")


def _legacy_role(efficient: bool, use_opus: bool, role: str | None) -> str:
    if role:
        return role
    if use_opus:
        return "complex"
    if efficient:
        return "efficient"
    return "primary"


async def _refresh_models_periodically() -> None:
    """Refresh only cached AWS-backed models before credential expiry."""
    while True:
        try:
            if pure_ollama_chat_mode(settings):
                logger.info("Skipping AWS refresh loop in pure Ollama chat mode")
                await asyncio.sleep(300)
                continue
            refresh_time = _get_next_token_expiry()
            sleep_seconds = (refresh_time - datetime.utcnow()).total_seconds()
            if sleep_seconds > 0:
                logger.info("Next AWS token refresh in %.1f min", sleep_seconds / 60)
                await asyncio.sleep(sleep_seconds)
            _run_secgateway()
            creds_meta = _read_aws_credentials_metadata()
            for cache_key, entry in list(_model_cache.items()):
                role_config: ModelRoleConfig | None = entry.get("role_config")
                if not role_config or not is_aws_provider(role_config.provider) or entry.get("model") is None:
                    continue
                model = _create_model_for_role(role_config)
                _model_cache[cache_key] = {
                    "model": model,
                    "expires_at": _get_next_token_expiry(),
                    "cred_fingerprint": creds_meta["fingerprint"] if creds_meta else None,
                    "role_config": role_config,
                }
                logger.info("Refreshed AWS model for role=%s", role_config.role)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error in token refresh background task: %s", exc, exc_info=True)
            await asyncio.sleep(300)


async def start_token_refresh_task() -> None:
    """Start AWS credential refresh only when AWS-backed chat providers are active."""
    global _refresh_task, _task_started
    if pure_ollama_chat_mode(settings):
        logger.info("AWS token refresh task skipped: pure Ollama chat mode")
        return
    async with _refresh_lock:
        if _task_started:
            return
        if _refresh_task is None or _refresh_task.done():
            _refresh_task = asyncio.create_task(_refresh_models_periodically())
            _task_started = True
            logger.info("Started AWS token refresh background task")


def _get_cached_or_create(role_config: ModelRoleConfig, force_refresh: bool = False) -> BaseChatModel:
    cache_key = role_config.cache_key
    cache_entry = _model_cache.get(cache_key)
    current_fingerprint = None
    if is_aws_provider(role_config.provider):
        creds_meta = _read_aws_credentials_metadata()
        current_fingerprint = creds_meta["fingerprint"] if creds_meta else None
    if not force_refresh and cache_entry and cache_entry.get("model") is not None:
        cached_fingerprint = cache_entry.get("cred_fingerprint")
        expires_at = cache_entry.get("expires_at")
        fingerprint_changed = (
            is_aws_provider(role_config.provider)
            and current_fingerprint is not None
            and cached_fingerprint is not None
            and current_fingerprint != cached_fingerprint
        )
        if not fingerprint_changed and (expires_at is None or datetime.utcnow() < expires_at):
            return cache_entry["model"]
    model = _create_model_for_role(role_config)
    _model_cache[cache_key] = {
        "model": model,
        "expires_at": _get_next_token_expiry() if is_aws_provider(role_config.provider) else None,
        "cred_fingerprint": current_fingerprint,
        "role_config": role_config,
    }
    if is_aws_provider(role_config.provider):
        try:
            asyncio.create_task(start_token_refresh_task())
        except RuntimeError:
            pass
    return model


def get_model(
    efficient: bool = False,
    force_refresh: bool = False,
    use_opus: bool = False,
    model_arn: Optional[str] = None,
    role: str | None = None,
) -> BaseChatModel:
    """Return a chat model by provider-neutral role.

    `efficient` and `use_opus` remain compatibility shims. `model_arn` is a
    Bedrock-only override and is ignored for Ollama roles.
    """
    selected_role = _legacy_role(efficient=efficient, use_opus=use_opus, role=role)
    role_config = resolve_model_role(selected_role, settings=settings)
    if model_arn:
        if role_config.provider == "aws-bedrock":
            role_config = ModelRoleConfig(
                role=role_config.role,
                provider=role_config.provider,
                model_name=model_arn,
                base_url=role_config.base_url,
                context_length=role_config.context_length,
                max_output_tokens=role_config.max_output_tokens,
                temperature=role_config.temperature,
                reasoning=role_config.reasoning,
                tool_capable=role_config.tool_capable,
            )
        else:
            logger.warning("Ignoring Bedrock-specific model_arn for non-Bedrock role=%s", role_config.role)
    return _get_cached_or_create(role_config, force_refresh=force_refresh)


def get_tool_model(
    efficient: bool = False,
    force_refresh: bool = False,
    use_opus: bool = False,
    role: str | None = None,
) -> BaseChatModel:
    """Return a tool-capable model.

    For Ollama this resolves to the `tool_worker` role by default. For providers
    whose normal model supports tools, callers may configure the same role to a
    provider-native model.
    """
    selected_role = role or "tool_worker"
    role_config = resolve_model_role(selected_role, settings=settings)
    if role_config.provider != "ollama" and selected_role == "tool_worker":
        selected_role = _legacy_role(efficient=efficient, use_opus=use_opus, role=None)
        role_config = resolve_model_role(selected_role, settings=settings)
    return _get_cached_or_create(role_config, force_refresh=force_refresh)


async def invoke_with_retry(model: BaseChatModel, messages, efficient: bool = False, role: str | None = None, max_retries: int = 2) -> Any:
    """Invoke model with automatic retry on AWS ExpiredTokenException."""
    last_exception = None
    selected_role = _legacy_role(efficient=efficient, use_opus=False, role=role)
    for attempt in range(max_retries + 1):
        try:
            return await model.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            error_code = ""
            try:
                from botocore.exceptions import ClientError
                if isinstance(exc, ClientError):
                    error_code = exc.response.get("Error", {}).get("Code", "")
            except ImportError:
                pass
            if error_code == "ExpiredTokenException" and attempt < max_retries:
                logger.warning("AWS token expired during invocation; refreshing role=%s", selected_role)
                model = get_model(role=selected_role, force_refresh=True)
                await asyncio.sleep(1)
                continue
            raise
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in invoke_with_retry")


async def stream_with_retry(model: BaseChatModel, messages, efficient: bool = False, role: str | None = None, max_retries: int = 2):
    """Stream model response with automatic retry on AWS ExpiredTokenException."""
    last_exception = None
    selected_role = _legacy_role(efficient=efficient, use_opus=False, role=role)
    for attempt in range(max_retries + 1):
        try:
            async for chunk in model.astream(messages):
                yield chunk
            return
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            error_code = ""
            try:
                from botocore.exceptions import ClientError
                if isinstance(exc, ClientError):
                    error_code = exc.response.get("Error", {}).get("Code", "")
            except ImportError:
                pass
            if error_code == "ExpiredTokenException" and attempt < max_retries:
                logger.warning("AWS token expired during streaming; refreshing role=%s", selected_role)
                model = get_model(role=selected_role, force_refresh=True)
                await asyncio.sleep(1)
                continue
            raise
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in stream_with_retry")


__all__ = ["MODEL_ROLES", "get_model", "get_tool_model", "invoke_with_retry", "stream_with_retry", "start_token_refresh_task"]
