"""LLM factory exports with minimal-environment import tolerance."""
try:
    from .chat_models import get_model, get_tool_model, invoke_with_retry, stream_with_retry  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - stripped CI/runtime snapshots
    _missing = exc.name

    def get_model(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            f"LLM runtime dependency {_missing!r} is not installed; install backend requirements before binding models."
        ) from exc

    def get_tool_model(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            f"LLM runtime dependency {_missing!r} is not installed; install backend requirements before binding tool models."
        ) from exc

    async def invoke_with_retry(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            f"LLM runtime dependency {_missing!r} is not installed; install backend requirements before invoking models."
        ) from exc

    async def stream_with_retry(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            f"LLM runtime dependency {_missing!r} is not installed; install backend requirements before streaming models."
        ) from exc
