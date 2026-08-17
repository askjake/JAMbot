"""
Rolling conversation summarizer with tiered memory.

This module keeps recent turns verbatim and turns older context into structured,
rule-based summaries.  The implementation is deterministic by default so it can
run safely in request paths without extra model calls, while preserving the API
shape needed to plug in a cheap async summarization model later.
"""

import logging
import re
from typing import Any

from .token_counter import count_tokens

logger = logging.getLogger(__name__)

_DEFAULT_PRESERVE_PATTERNS = [
    r"R\d{10}",
    r"[A-Z]{2,}-\d+",
    r"\d+\.\d+\.\d+",
    r"/[\w./-]+",
    r"https?://\S+",
    r"\b[A-Z]{2,}[-_]\d+\b",
    r"\b(?:ERROR|WARN|FAIL|PASS|Exception|Traceback)\b[^\n]*",
]

_IMPORTANT_KEYWORDS = (
    "decided",
    "decision",
    "resolved",
    "fix",
    "fixed",
    "error",
    "failed",
    "failure",
    "pass",
    "root cause",
    "blocker",
    "path",
    "file",
    "command",
    "test",
    "verify",
    "receiver",
    "token",
    "budget",
    "must",
    "should",
    "do not",
    "implement",
)

_FILLER_PATTERNS = [
    re.compile(r"^\s*(sure|ok|okay|got it|thanks|thank you|sounds good)[.!\s]*$", re.I),
    re.compile(r"^\s*(i'll|i will|let me)\b.*$", re.I),
]


class RollingSummarizer:
    """
    Implement a three-tier memory system for chat history.

    Args:
        tier1_turns: Number of most-recent turns kept verbatim.
        tier2_turns: Number of turns before tier 1 summarized into detailed
            memory.
        tier2_budget_tokens: Maximum token budget for detailed summary text.
        tier3_budget_tokens: Maximum token budget for abstract summary text.
        preserve_patterns: Regex patterns that must survive summaries.
    """

    def __init__(
        self,
        tier1_turns: int = 5,
        tier2_turns: int = 10,
        tier2_budget_tokens: int = 1500,
        tier3_budget_tokens: int = 500,
        preserve_patterns: list[str] | None = None,
    ):
        if tier1_turns <= 0:
            raise ValueError("tier1_turns must be positive")
        if tier2_turns < 0:
            raise ValueError("tier2_turns cannot be negative")
        if tier2_budget_tokens <= 0 or tier3_budget_tokens <= 0:
            raise ValueError("summary budgets must be positive")

        self.tier1_turns = tier1_turns
        self.tier2_turns = tier2_turns
        self.tier2_budget_tokens = tier2_budget_tokens
        self.tier3_budget_tokens = tier3_budget_tokens
        self.preserve_patterns = preserve_patterns or list(_DEFAULT_PRESERVE_PATTERNS)

        self.tiers: dict[int, Any] = {1: [], 2: "", 3: ""}
        self.full_history: list[dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> None:
        """
        Add a conversation turn and refresh tier summaries.

        Args:
            role: Message role, typically ``"user"`` or ``"assistant"``.
            content: Message content.  Non-string content should be serialized
                by the caller before ingestion.
        """
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        self.full_history.append({"role": role.strip(), "content": content})
        self._rebuild_tiers(self.full_history)

    def summarize_for_tier2(self, turns: list[dict]) -> str:
        """
        Compress turns into a detailed, structured summary.

        The default implementation is rule-based and preserves named entities,
        file paths, commands, error lines, code blocks, decisions, and technical
        facts.  It deliberately avoids an LLM call so request-path compression
        remains cheap and deterministic.
        """
        if not turns:
            return ""
        try:
            lines: list[str] = ["[TIER 2 DETAILED SUMMARY]"]
            for index, turn in enumerate(turns, start=1):
                role = str(turn.get("role", "unknown"))
                content = str(turn.get("content", ""))
                extracted = self._extract_summary_lines(content, detailed=True)
                if not extracted:
                    continue
                for item in extracted:
                    lines.append(f"- {role} turn {index}: {item}")
            summary = self._dedupe_lines(lines)
            return self._trim_to_budget("\n".join(summary), self.tier2_budget_tokens)
        except Exception as exc:  # pragma: no cover - defensive, not expected
            logger.warning("Tier 2 summarization failed: %s", exc)
            return self._fallback_summary(turns, self.tier2_budget_tokens, label="TIER 2 FALLBACK SUMMARY")

    async def summarize_for_tier2_async(self, turns: list[dict]) -> str:
        """
        Async-compatible wrapper for detailed summarization.

        A future LLM-backed summarizer can be plugged in here without changing
        the public calling pattern.
        """
        return self.summarize_for_tier2(turns)

    def summarize_for_tier3(self, current_tier2: str, demoted_tier2: str) -> str:
        """
        Merge detailed summaries into an abstract long-term memory block.
        """
        combined = "\n".join(part for part in (current_tier2, demoted_tier2) if part and part.strip())
        if not combined.strip():
            return ""
        try:
            lines = ["[TIER 3 ABSTRACT SUMMARY]"]
            entities = self._extract_protected(combined)
            if entities:
                lines.append("- Preserved entities: " + ", ".join(sorted(set(entities))))
            for line in combined.splitlines():
                clean = self._clean_line(line)
                if not clean or clean.startswith("["):
                    continue
                if self._is_important(clean):
                    lines.append(f"- {clean[:500]}")
            summary = self._dedupe_lines(lines)
            return self._trim_to_budget("\n".join(summary), self.tier3_budget_tokens)
        except Exception as exc:  # pragma: no cover - defensive, not expected
            logger.warning("Tier 3 summarization failed: %s", exc)
            return self._trim_to_budget(combined, self.tier3_budget_tokens)

    async def summarize_for_tier3_async(self, current_tier2: str, demoted_tier2: str) -> str:
        """
        Async-compatible wrapper for abstract summarization.
        """
        return self.summarize_for_tier3(current_tier2, demoted_tier2)

    def render_context(self) -> list[dict]:
        """
        Assemble summary memory plus tier-1 verbatim messages.

        Returns:
            Message dictionaries ready to be merged with a system prompt and the
            current user message by the context assembly engine.
        """
        messages: list[dict] = []
        memory_parts = [part for part in (self.tiers.get(3, ""), self.tiers.get(2, "")) if part]
        if memory_parts:
            messages.append({"role": "system", "content": "\n\n".join(memory_parts)})
        messages.extend(dict(message) for message in self.tiers.get(1, []))
        return messages

    def estimate_tokens(self) -> dict:
        """
        Return token usage by memory tier.
        """
        tier1_messages = self.tiers.get(1, [])
        tier2 = str(self.tiers.get(2, ""))
        tier3 = str(self.tiers.get(3, ""))
        return {
            "tier1_tokens": count_tokens(tier1_messages),
            "tier2_tokens": count_tokens(tier2),
            "tier3_tokens": count_tokens(tier3),
            "total_tokens": count_tokens(tier1_messages) + count_tokens(tier2) + count_tokens(tier3),
            "full_history_tokens": count_tokens(self.full_history),
            "history_turns": len(self.full_history),
        }

    def partition_history(self, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """
        Partition a history list into tier sources and rendered summaries.

        Args:
            history: Optional history override.  If omitted, full history is used.
        """
        source = list(self.full_history if history is None else history)
        tier1 = source[-self.tier1_turns :] if self.tier1_turns else []
        before_tier1 = source[: max(len(source) - len(tier1), 0)]
        tier2_source = before_tier1[-self.tier2_turns :] if self.tier2_turns else []
        tier3_source = before_tier1[: max(len(before_tier1) - len(tier2_source), 0)]
        tier2_summary = self.summarize_for_tier2(tier2_source)
        tier3_summary = self.summarize_for_tier3("", self.summarize_for_tier2(tier3_source))
        return {
            "tier1": tier1,
            "tier2_source": tier2_source,
            "tier3_source": tier3_source,
            "tier2": tier2_summary,
            "tier3": tier3_summary,
        }

    def _rebuild_tiers(self, history: list[dict[str, str]]) -> None:
        partition = self.partition_history(history)
        self.tiers[1] = partition["tier1"]
        self.tiers[2] = partition["tier2"]
        self.tiers[3] = partition["tier3"]

    def _extract_summary_lines(self, content: str, detailed: bool) -> list[str]:
        lines: list[str] = []
        protected = self._extract_protected(content)
        if protected:
            lines.append("entities=" + ", ".join(sorted(set(protected))))

        for code in re.findall(r"```[\s\S]*?```", content):
            code_lines = code.strip().splitlines()
            preview = " | ".join(code_lines[:6])
            lines.append(f"code preserved: {preview[:700]}")

        for raw_line in re.split(r"[\n\r]+", content):
            clean = self._clean_line(raw_line)
            if not clean or self._is_filler(clean):
                continue
            if self._is_command(clean) or self._is_important(clean):
                lines.append(clean[:700])

        if detailed and not lines:
            sentences = re.split(r"(?<=[.!?])\s+", content.strip())
            for sentence in sentences[:2]:
                clean = self._clean_line(sentence)
                if clean and not self._is_filler(clean):
                    lines.append(clean[:400])
        return self._dedupe_lines(lines)

    def _extract_protected(self, content: str) -> list[str]:
        found: list[str] = []
        for pattern in self.preserve_patterns:
            try:
                found.extend(match.group(0) for match in re.finditer(pattern, content))
            except re.error as exc:
                logger.warning("Invalid preserve pattern %r: %s", pattern, exc)
        return found

    def _is_important(self, text: str) -> bool:
        lowered = text.lower()
        if any(keyword in lowered for keyword in _IMPORTANT_KEYWORDS):
            return True
        return bool(self._extract_protected(text))

    @staticmethod
    def _is_command(text: str) -> bool:
        return bool(re.match(r"^\s*(?:[$#>]\s*)?(?:python|pytest|bash|sh|kubectl|aws|git|tar|zip|curl|npm|yarn|pip)\b", text))

    @staticmethod
    def _clean_line(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_filler(text: str) -> bool:
        return any(pattern.match(text) for pattern in _FILLER_PATTERNS)

    @staticmethod
    def _dedupe_lines(lines: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for line in lines:
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(line)
        return deduped

    def _trim_to_budget(self, text: str, budget_tokens: int) -> str:
        if count_tokens(text) <= budget_tokens:
            return text
        lines = text.splitlines()
        kept: list[str] = []
        for line in lines:
            candidate = "\n".join(kept + [line])
            if count_tokens(candidate) > budget_tokens:
                break
            kept.append(line)
        if kept:
            return "\n".join(kept)

        words = text.split()
        result: list[str] = []
        for word in words:
            candidate = " ".join(result + [word])
            if count_tokens(candidate) > budget_tokens:
                break
            result.append(word)
        return " ".join(result)

    def _fallback_summary(self, turns: list[dict], budget_tokens: int, label: str) -> str:
        lines = [f"[{label}]"]
        for index, turn in enumerate(turns, start=1):
            role = str(turn.get("role", "unknown"))
            content = self._clean_line(str(turn.get("content", "")))
            if content:
                lines.append(f"- {role} turn {index}: {content[:300]}")
        return self._trim_to_budget("\n".join(lines), budget_tokens)
