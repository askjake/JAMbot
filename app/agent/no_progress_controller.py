"""Bounded semantic no-progress controller for parent orchestration.

The controller recognizes authoritative tool-result envelopes only.  It never
uses assistant prose or tool-like JSON in an AI message as execution evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

BLOCKED_MISSING_CAPABILITY = "BLOCKED_MISSING_CAPABILITY"
NO_PROGRESS_LIMIT_REACHED = "BLOCKED_NO_PROGRESS_LIMIT"
logger = logging.getLogger(__name__)

MAX_NO_PROGRESS_ATTEMPTS = 2
"""Fallback ceiling on consecutive failed/blocked tool attempts.

Retained as the safe default. The effective value is resolved at call time by
``_resolve_max_no_progress_attempts()`` from ``settings.MAX_CONSECUTIVE_TOOL_ERRORS``.
"""

# D3B-FIX(2026-08-18): plausible range for the consecutive-tool-error ceiling.
# settings.MAX_CONSECUTIVE_TOOL_ERRORS shipped as 1555550, which would have
# silently disabled this safety net entirely had it been wired in. Any value
# outside this band is treated as a configuration error, not as intent.
_MIN_NO_PROGRESS_ATTEMPTS = 1
_MAX_NO_PROGRESS_ATTEMPTS_CEILING = 100


def _resolve_max_no_progress_attempts() -> int:
    """Resolve the consecutive failed/blocked tool-attempt ceiling.

    D3B-FIX(2026-08-18): ``settings.MAX_CONSECUTIVE_TOOL_ERRORS`` was declared in
    app/config.py but had **zero references** anywhere in the codebase, while the
    real enforcement point here was hardcoded to ``MAX_NO_PROGRESS_ATTEMPTS``.
    This honours the setting, but refuses values outside a plausible band so a
    mistyped config can never silently disable the loop guard (the shipped value
    was 1555550).
    """
    try:
        from app.config import get_settings

        configured = int(
            getattr(get_settings(), "MAX_CONSECUTIVE_TOOL_ERRORS", MAX_NO_PROGRESS_ATTEMPTS)
        )
    except Exception:
        return MAX_NO_PROGRESS_ATTEMPTS
    if (
        configured < _MIN_NO_PROGRESS_ATTEMPTS
        or configured > _MAX_NO_PROGRESS_ATTEMPTS_CEILING
    ):
        logger.warning(
            "MAX_CONSECUTIVE_TOOL_ERRORS=%s is outside the plausible range "
            "[%s, %s]; falling back to %s so the no-progress guard stays active.",
            configured,
            _MIN_NO_PROGRESS_ATTEMPTS,
            _MAX_NO_PROGRESS_ATTEMPTS_CEILING,
            MAX_NO_PROGRESS_ATTEMPTS,
        )
        return MAX_NO_PROGRESS_ATTEMPTS
    return configured

_NON_CAPABILITY_OWNERS = frozenset({
    "mailto", "http", "https", "ftp", "ftps", "ssh", "git", "file",
    "data", "toolset", "key", "value", "owner", "name",
})


def current_parent_required_tools(
    *,
    prompt_requested: Iterable[str] = (),
    management_requested: Iterable[str] = (),
    operational_required: Iterable[str] = (),
    first_tool: str = "",
) -> set[str]:
    """Build the no-progress required set from current parent-turn intent only.

    Checkpointed sticky requests remain eligible/bound, but they do not become
    current task requirements merely because they survived from an earlier
    domain or child run.
    """

    values = [
        *list(prompt_requested or ()),
        *list(management_requested or ()),
        *list(operational_required or ()),
    ]
    if str(first_tool or "").strip():
        values.append(str(first_tool))
    return _normalize_required_tools(values)


def _normalize_required_tools(required_tools: Iterable[str]) -> set[str]:
    """Return real tool identities, excluding incidental colon-token suffixes."""
    required: set[str] = set()
    for item in required_tools or ():
        text = str(item or "").strip()
        if not text:
            continue
        if ":" in text:
            owner, raw = text.split(":", 1)
            owner = owner.strip().lower()
            raw = raw.strip()
            if owner in _NON_CAPABILITY_OWNERS or not raw:
                continue
            # The required set is supplied by the normalized current-turn tool
            # plan, so any non-phantom owner may be a real family (for example
            # viewership or beta_report, neither of which ends in ``_mcp``).
            text = raw
        required.add(text)
    return required


def _normalize_attempted_tool_name(value: str) -> str:
    """Normalize an executed parent tool name without reviving phantom suffixes."""
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" in text:
        owner, raw = text.split(":", 1)
        owner = owner.strip().lower()
        raw = raw.strip()
        if owner in _NON_CAPABILITY_OWNERS or not raw:
            return ""
        text = raw
    return text


_REQUIRED_TOOL_BLOCK_CODES = {
    "TOOL_NOT_IN_LAST_BINDING",
    "BOUND_TOOL_NOT_EXECUTABLE",
    "BLOCKED_UPSTREAM_UNAVAILABLE",
    "TOOL_AUTHORIZATION_REQUIRED",
    "BLOCKED_CHILD_MISSING_CAPABILITY",
}
_NO_PROGRESS_CODES = _REQUIRED_TOOL_BLOCK_CODES | {
    # Identifier validation is an argument/provenance failure, not proof that
    # the capability is missing. Permit one bounded recovery attempt so the
    # model can correct the argument or use an explicitly allowed fallback.
    "TOOL_IDENTIFIER_VALIDATION_FAILED",
    "TOOL_EXECUTION_ERROR",
    "BLOCKED_REMOTE_TRANSPORT",
    "UPSTREAM_TOOL_ERROR",
}


@dataclass(frozen=True)
class NoProgressDecision:
    stop: bool
    result_code: str = "CONTINUE"
    no_progress_attempts: int = 0
    missing_tools: tuple[str, ...] = ()
    activation_allowed: bool = False
    functional_success: bool = False

    def render_terminal_message(self) -> str:
        named = ", ".join(self.missing_tools) if self.missing_tools else "required capability"
        if self.result_code == NO_PROGRESS_LIMIT_REACHED:
            reason = (
                f"Repeated tool attempts made no functional progress. "
                f"Failed or blocked tools: {named}."
            )
        else:
            reason = f"Missing or nonfunctional capability: {named}."
        return (
            f"{self.result_code}\n"
            f"{reason}\n"
            f"Completed evidence is preserved. No child, unrelated tool family, public search, "
            f"cluster inspection, or additional fallback will be substituted."
        )[:1199]


def _message_payload(message: Any) -> tuple[dict[str, Any] | None, str]:
    # Accept ToolMessage-like objects and compact test stand-ins.  A mapping is
    # accepted only when explicitly typed as a tool record.
    name = str(getattr(message, "name", "") or "")
    content = getattr(message, "content", None)
    if isinstance(message, Mapping):
        msg_type = str(message.get("type") or message.get("role") or "").lower()
        if msg_type not in {"tool", "toolmessage", "tool_result"}:
            return None, ""
        name = str(message.get("name") or "")
        content = message.get("content")
    elif type(message).__name__ not in {"ToolMessage", "Message", "FakeToolMessage"} and not name:
        return None, ""
    if isinstance(content, Mapping):
        payload = dict(content)
    elif isinstance(content, str):
        try:
            parsed = json.loads(content)
        except Exception:
            return None, name
        payload = dict(parsed) if isinstance(parsed, Mapping) else None
    else:
        payload = None
    return payload, name


def _is_human_turn_boundary(message: Any) -> bool:
    if isinstance(message, Mapping):
        msg_type = str(message.get("type") or message.get("role") or "").lower()
        return msg_type in {"human", "humanmessage", "user"}
    return type(message).__name__ == "HumanMessage"


def _tail_tool_records(messages: Sequence[Any]) -> list[tuple[dict[str, Any], str]]:
    """Return only tool records belonging to the current user turn.

    A new HumanMessage is a hard provenance boundary.  Without this boundary,
    a failed tool result from a prior turn can poison a later resume turn and
    trigger a synthetic terminal result before the model is called.
    """
    records: list[tuple[dict[str, Any], str]] = []
    for message in reversed(list(messages)):
        if _is_human_turn_boundary(message):
            break
        payload, name = _message_payload(message)
        if payload is None:
            # AI orchestration messages normally separate sequential tool calls
            # inside one user turn.  They are not evidence, but they are also not
            # a provenance boundary; only the HumanMessage above is.
            continue
        records.append((payload, name))
    records.reverse()
    return records


def _result_has_evidence(payload: Mapping[str, Any]) -> bool:
    if not bool(payload.get("ok")):
        return False
    result = payload.get("result")
    if result is None:
        return False
    if isinstance(result, Mapping):
        # Service-family health or registry presence is not requested-operation
        # success and must not satisfy a functional gate.
        keys = {str(k).lower() for k in result}
        if keys <= {"family", "health", "status", "tool_count", "inventory_source", "source"}:
            return False
        return bool(result)
    if isinstance(result, (list, tuple, set, str, bytes)):
        return len(result) > 0
    return True


def evaluate_no_progress(
    messages: Sequence[Any],
    *,
    required_tools: Iterable[str] = (),
    exact_activation_path_available: bool = False,
    activation_attempted: bool = False,
) -> NoProgressDecision:
    required = _normalize_required_tools(required_tools)
    records = _tail_tool_records(messages)
    if not records:
        return NoProgressDecision(stop=False)

    functional_success = False
    attempts = 0
    missing: set[str] = set()
    failed_parent_tools: set[str] = set()
    required_blocked = False

    for payload, fallback_name in records:
        scope = str(
            payload.get("scope")
            or (payload.get("audit") or {}).get("scope")
            or ""
        ).lower()
        if scope in {"mcop_child", "child", "child_turn"}:
            continue
        code = str(payload.get("result_code") or payload.get("status") or "")
        tool_name = _normalize_attempted_tool_name(
            str(payload.get("tool_name") or fallback_name or "")
        )
        if not tool_name:
            continue
        is_required = not required or tool_name in required
        ok_present = "ok" in payload
        ok_value = payload.get("ok") if ok_present else None
        if ok_value is True and _result_has_evidence(payload):
            # Only evidence from the requested/required path completes or resets
            # a turn that has an explicit required capability.  A successful
            # unrelated fallback must not erase prior failed attempts.
            if is_required:
                functional_success = True
                attempts = 0
            elif not required:
                attempts = 0
            continue

        # Legacy/neutral tool payloads may omit ``ok`` entirely (for example
        # agent_check_tasks historically returned {"status": "no_tasks"}).
        # Absence of an explicit success flag is not evidence of failure. Count
        # only a known no-progress code or an explicit ``ok: false`` result.
        explicit_failure = code in _NO_PROGRESS_CODES or ok_value is False
        if explicit_failure:
            # Any real parent-scope failed tool attempt counts toward the bounded
            # fallback limit, even when it is not the nominal first_tool.  This
            # closes sequential fallback thrash while keeping child/phantom
            # evidence excluded above.  Recursive MCOP spawn tools remain
            # excluded when they are unrelated to the current required path.
            if (
                tool_name not in {"agent_spawn_task", "agent_spawn_parallel"}
                or is_required
            ):
                attempts += 1
                failed_parent_tools.add(tool_name)
            if is_required and code in _REQUIRED_TOOL_BLOCK_CODES:
                required_blocked = True
                missing.add(tool_name)
            continue

        # A neutral response neither proves functional progress nor erases an
        # earlier failed attempt.  This matters when legacy tools omit ``ok`` or
        # when a modern tool returns an informational result with no evidence.
        continue

    if functional_success:
        return NoProgressDecision(stop=False, functional_success=True)

    if required_blocked:
        activation_allowed = bool(exact_activation_path_available and not activation_attempted)
        if activation_allowed:
            return NoProgressDecision(
                stop=False,
                no_progress_attempts=attempts,
                missing_tools=tuple(sorted(missing)),
                activation_allowed=True,
            )
        return NoProgressDecision(
            stop=True,
            result_code=BLOCKED_MISSING_CAPABILITY,
            no_progress_attempts=max(1, attempts),
            missing_tools=tuple(sorted(missing or required)),
        )

    if attempts >= _resolve_max_no_progress_attempts():
        return NoProgressDecision(
            stop=True,
            result_code=NO_PROGRESS_LIMIT_REACHED,
            no_progress_attempts=attempts,
            missing_tools=tuple(sorted(failed_parent_tools or required)),
        )
    return NoProgressDecision(stop=False, no_progress_attempts=attempts)
