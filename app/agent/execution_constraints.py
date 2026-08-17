"""Current-task execution-topology constraints.

The parent-only flag is a server-owned boolean.  It is inferred only from
explicit current-turn directives, may be explicitly released, and is enforced
both at model-facing binding time and again at the execution gate.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from app.agent.directive_text import sanitize_current_turn_directives

FORBID = "FORBID"
ALLOW = "ALLOW"
UNCHANGED = "UNCHANGED"
MCOP_SPAWN_TOOL_NAMES = frozenset({"agent_spawn_task", "agent_spawn_parallel"})

_FORBID_RE = re.compile(
    r"\b(?:work\s+(?:entirely\s+)?in\s+(?:the\s+)?parent\s+thread\s+only|"
    r"parent\s+thread\s+only|no\s+(?:mcop\s+)?(?:children|child\s+agents)|"
    r"do\s+not\s+(?:spawn|use|launch)\s+(?:mcop\s+)?(?:children|child\s+agents)|"
    r"do\s+not\s+use\s+agent_spawn_(?:task|parallel))\b",
    re.IGNORECASE,
)
_ALLOW_RE = re.compile(
    r"\b(?:allow|permit|you\s+may\s+use|you\s+may\s+spawn|enable)\s+"
    r"(?:mcop\s+)?(?:children|child\s+agents)\b|"
    r"\bmcop\s+children\s+(?:are\s+)?allowed\b",
    re.IGNORECASE,
)
_DISCUSSION_RE = re.compile(
    r"\b(?:how\s+do|what\s+is|explain|describe|document|summarize)\b.{0,40}"
    r"\b(?:mcop|child\s+agents?|agent_spawn_(?:task|parallel))\b",
    re.IGNORECASE,
)


def parse_mcop_constraint_delta(text: str | None) -> str:
    authoritative = sanitize_current_turn_directives(text)
    if _DISCUSSION_RE.search(authoritative):
        return UNCHANGED
    if _FORBID_RE.search(authoritative):
        return FORBID
    if _ALLOW_RE.search(authoritative):
        return ALLOW
    return UNCHANGED


def merge_mcop_children_forbidden(previous: Any, delta: str) -> bool:
    if delta == FORBID:
        return True
    if delta == ALLOW:
        return False
    return bool(previous)


def filter_mcop_spawn_tools(tool_names: Iterable[str], forbidden: bool) -> list[str]:
    if not forbidden:
        return [str(name) for name in tool_names]
    return [str(name) for name in tool_names if str(name) not in MCOP_SPAWN_TOOL_NAMES]


def constraints_mapping(mcop_children_forbidden: bool) -> dict[str, bool]:
    return {"mcop_children_forbidden": bool(mcop_children_forbidden)}


def spawn_blocked_by_constraints(
    tool_name: str, constraints: Mapping[str, Any] | None
) -> bool:
    raw = str(tool_name or "").split(":", 1)[-1]
    return bool((constraints or {}).get("mcop_children_forbidden")) and raw in MCOP_SPAWN_TOOL_NAMES
