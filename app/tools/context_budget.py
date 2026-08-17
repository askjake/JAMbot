"""
Dynamic context budget tracking for tool-result compression.

Budgets are maintained per session so browse-heavy conversations progressively
receive smaller tool-result budgets as history grows and repeated tool calls add
less marginal value.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from app.message.compression import count_message_tokens, effective_context_budget

logger = logging.getLogger(__name__)

def _default_budget_details(role: str = "primary") -> dict:
    return effective_context_budget(role)

def _default_usable_context(role: str = "primary") -> int:
    return int(_default_budget_details(role)["history_budget"])


def context_budget_selftest(role: str = "primary") -> dict:
    details = _default_budget_details(role)
    required = ["model_role", "model_name", "configured_context", "reserved_system_tool_output_budget", "history_budget", "tool_result_budget"]
    missing = [key for key in required if key not in details]
    return {"status": "pass" if not missing and details["history_budget"] > 0 else "fail", "missing": missing, "budget": details}

_DEFAULT_SESSION_ID = "__default__"


@dataclass
class ContextBudget:
    """Per-session dynamic context budget tracker."""

    total_budget: int = field(default_factory=_default_usable_context)
    tokens_used_by_history: int = 0
    tool_calls_this_session: int = 0
    complexity_score: int = 3
    tool_budgets: Dict[str, int] = field(default_factory=dict)

    def remaining_budget(self) -> int:
        """Tokens still available for message history and tool results."""
        return max(0, self.total_budget - self.tokens_used_by_history)

    def get_tool_budget(self, tool_name: str) -> int:
        """Calculate a dynamic token budget for one tool result."""
        remaining = self.remaining_budget()

        if remaining > 100_000:
            base = 6000
        elif remaining > 50_000:
            base = 3000
        elif remaining > 20_000:
            base = 1500
        else:
            base = 800

        calls_beyond_3 = max(0, self.tool_calls_this_session - 3)
        call_decay = max(0.3, 1.0 - (calls_beyond_3 * 0.1))
        adjusted = int(base * call_decay)

        if self.complexity_score >= 10:
            complexity_multiplier = 1.5
        elif self.complexity_score >= 7:
            complexity_multiplier = 1.2
        elif self.complexity_score < 3:
            complexity_multiplier = 0.7
        else:
            complexity_multiplier = 1.0

        final_budget = int(adjusted * complexity_multiplier)
        final_budget = max(500, min(final_budget, 8000))

        logger.debug(
            "Tool budget for %s: %s tokens (remaining=%s, calls=%s, complexity=%s)",
            tool_name,
            final_budget,
            remaining,
            self.tool_calls_this_session,
            self.complexity_score,
        )

        self.tool_budgets[tool_name] = final_budget
        self.tool_calls_this_session += 1
        return final_budget

    def update_history_tokens(self, messages: Optional[Iterable]) -> None:
        """Recalculate tokens used by the current conversation history."""
        if messages is None:
            return
        self.tokens_used_by_history = count_message_tokens(list(messages))

    def report(self) -> str:
        """Human-readable budget report for backend logs."""
        pct = (self.tokens_used_by_history / self.total_budget) * 100 if self.total_budget else 0
        return (
            f"Context: {self.tokens_used_by_history:,}/{self.total_budget:,} "
            f"({pct:.0f}%) | Remaining: {self.remaining_budget():,} | "
            f"Tools called: {self.tool_calls_this_session} | "
            f"Complexity: {self.complexity_score}"
        )


_session_budgets: Dict[str, ContextBudget] = {}
_current_budget_var: ContextVar[ContextBudget | None] = ContextVar("current_context_budget", default=None)
_fallback_current_budget: ContextBudget | None = None


def _normalize_session_id(session_id: Optional[str]) -> str:
    return str(session_id or _DEFAULT_SESSION_ID)


def get_session_budget(session_id: Optional[str] = None, messages=None, complexity_score: int = 3) -> ContextBudget:
    """Get or create the budget tracker for a session."""
    key = _normalize_session_id(session_id)
    if key not in _session_budgets:
        _session_budgets[key] = ContextBudget(complexity_score=complexity_score)

    budget = _session_budgets[key]
    budget.complexity_score = complexity_score
    if messages is not None:
        budget.update_history_tokens(messages)
    return budget


def clear_session_budget(session_id: Optional[str]) -> None:
    """Remove a session budget tracker."""
    _session_budgets.pop(_normalize_session_id(session_id), None)


def set_current_budget(budget: ContextBudget | None) -> Token:
    """Expose the active session budget to tool execution code."""
    global _fallback_current_budget
    _fallback_current_budget = budget
    return _current_budget_var.set(budget)


def reset_current_budget(token: Token) -> None:
    """Reset the ContextVar token for callers that have a bounded scope."""
    try:
        _current_budget_var.reset(token)
    except Exception:
        logger.debug("Unable to reset current context budget", exc_info=True)


def get_current_budget(tool_name: str = "web_browse", default: int = 3000) -> int:
    """Return the active dynamic budget, or default when no session context exists."""
    budget = _current_budget_var.get() or _fallback_current_budget
    if budget is None:
        return default
    try:
        return budget.get_tool_budget(tool_name)
    except Exception:
        logger.debug("Falling back to default budget for %s", tool_name, exc_info=True)
        return default
