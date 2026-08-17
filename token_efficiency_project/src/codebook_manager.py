"""
Dynamic codebook manager for repeated conversation phrases.

The manager observes historical conversation text, promotes repeated long phrases
into compact Unicode sigils, and can encode/decode historical context while
tracking token savings.  The current user message should not be passed through
``encode`` by callers; that guard belongs in the orchestration layer.
"""

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from .token_counter import count_tokens

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:#@-]*|[^\s\w]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class CodebookEntry:
    """Active sigil mapping and accounting metadata."""

    sigil: str
    definition: str
    created_turn: int
    last_used_turn: int
    usage_count: int = 0
    tokens_saved: int = 0


@dataclass(slots=True)
class CandidatePhrase:
    """Observed phrase before it becomes profitable enough to promote."""

    phrase: str
    count: int = 0
    first_seen_turn: int = 0
    last_seen_turn: int = 0


class CodebookManager:
    """
    Manage a dynamic codebook of sigils mapped to verbose definitions.

    Args:
        max_entries: Maximum active codebook entries.  Keeping this small avoids
            spending more prompt tokens on definitions than the replacement saves.
        sigil_prefix: Unicode prefix used to generate visually distinct sigils.
        min_occurrences_to_promote: Minimum observed phrase count before a
            phrase can be promoted.
        min_phrase_tokens: Minimum phrase length, measured by the project token
            counter, before a phrase can be considered for promotion.
        expiry_turns: Active entries expire after this many observed turns
            without use.
        match_mode: Matching strategy.  ``"exact"`` performs normalized literal
            matching.  ``"semantic"`` is accepted for configuration compatibility
            and currently falls back to exact matching.
    """

    def __init__(
        self,
        max_entries: int = 20,
        sigil_prefix: str = "Σ",
        min_occurrences_to_promote: int = 3,
        min_phrase_tokens: int = 8,
        expiry_turns: int = 10,
        match_mode: str = "exact",
    ):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if min_occurrences_to_promote <= 0:
            raise ValueError("min_occurrences_to_promote must be positive")
        if min_phrase_tokens <= 0:
            raise ValueError("min_phrase_tokens must be positive")
        if expiry_turns <= 0:
            raise ValueError("expiry_turns must be positive")

        self.max_entries = max_entries
        self.sigil_prefix = sigil_prefix or "Σ"
        self.min_occurrences_to_promote = min_occurrences_to_promote
        self.min_phrase_tokens = min_phrase_tokens
        self.expiry_turns = expiry_turns
        self.match_mode = match_mode if match_mode in {"exact", "semantic"} else "exact"

        self.entries: dict[str, str] = {}
        self.usage_count: dict[str, int] = {}
        self.candidate_pool: dict[str, int] = {}

        self._active: dict[str, CodebookEntry] = {}
        self._phrase_to_sigil: dict[str, str] = {}
        self._candidates: dict[str, CandidatePhrase] = {}
        self._turn_index = 0
        self._next_id = 1
        self._total_tokens_saved = 0

    def observe(self, text: str) -> None:
        """
        Scan text for repeated n-grams that are candidates for promotion.

        Candidate extraction is intentionally deterministic and lightweight.  It
        looks for 3- to 15-word spans, requires a useful token length, ignores
        phrases that are already active codebook definitions, and records counts
        across observed turns.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._turn_index += 1
        if not text.strip():
            self._garbage_collect()
            return

        try:
            for sigil, entry in self._active.items():
                if self._contains_phrase(text, entry.definition):
                    entry.last_used_turn = self._turn_index
                    self._sync_public_state(sigil)

            for phrase, occurrences in self._extract_candidate_counts(text).items():
                key = self._normalize_key(phrase)
                if key in self._phrase_to_sigil:
                    continue
                candidate = self._candidates.get(key)
                if candidate is None:
                    candidate = CandidatePhrase(
                        phrase=phrase,
                        count=0,
                        first_seen_turn=self._turn_index,
                        last_seen_turn=self._turn_index,
                    )
                    self._candidates[key] = candidate
                candidate.count += occurrences
                candidate.last_seen_turn = self._turn_index
                self.candidate_pool[key] = candidate.count
        except Exception as exc:  # pragma: no cover - defensive, not expected
            logger.warning("Codebook observation failed: %s", exc)
        finally:
            self._garbage_collect()

    def promote_candidates(self) -> list[str]:
        """
        Promote profitable candidates to active codebook entries.

        Returns:
            List of newly created sigils ordered by expected net savings.
        """
        promoted: list[str] = []
        if len(self._active) >= self.max_entries:
            return promoted

        candidates = sorted(
            self._candidates.items(),
            key=lambda item: self._estimated_net_savings(item[1].phrase, item[1].count),
            reverse=True,
        )

        for key, candidate in candidates:
            if len(self._active) >= self.max_entries:
                break
            if key in self._phrase_to_sigil:
                continue
            if candidate.count < self.min_occurrences_to_promote:
                continue
            if count_tokens(candidate.phrase) < self.min_phrase_tokens:
                continue
            if self._overlaps_active_definition(candidate.phrase):
                continue
            if self._estimated_net_savings(candidate.phrase, candidate.count) <= 0:
                continue

            sigil = self._next_sigil()
            entry = CodebookEntry(
                sigil=sigil,
                definition=candidate.phrase,
                created_turn=self._turn_index,
                last_used_turn=self._turn_index,
            )
            self._active[sigil] = entry
            self._phrase_to_sigil[key] = sigil
            self._sync_public_state(sigil)
            promoted.append(sigil)

        return promoted

    def encode(self, text: str) -> str:
        """
        Replace active codebook definitions in text with their sigils.

        If encoding fails, the original text is returned.  Callers should only
        encode historical context, never the current user message.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text or not self._active:
            return text

        encoded = text
        try:
            for sigil, entry in sorted(
                self._active.items(), key=lambda item: len(item[1].definition), reverse=True
            ):
                encoded, replacements = self._replace_phrase(encoded, entry.definition, sigil)
                if replacements:
                    entry.usage_count += replacements
                    entry.last_used_turn = self._turn_index
                    saved_each = max(count_tokens(entry.definition) - count_tokens(sigil), 0)
                    saved = saved_each * replacements
                    entry.tokens_saved += saved
                    self._total_tokens_saved += saved
                    self._sync_public_state(sigil)
        except Exception as exc:  # pragma: no cover - defensive, not expected
            logger.warning("Codebook encoding failed: %s", exc)
            return text
        return encoded

    def decode(self, text: str) -> str:
        """
        Expand active sigils back to full definitions for human-readable output.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text or not self._active:
            return text

        decoded = text
        try:
            for sigil, entry in sorted(self._active.items(), key=lambda item: len(item[0]), reverse=True):
                decoded = decoded.replace(sigil, entry.definition)
        except Exception as exc:  # pragma: no cover - defensive, not expected
            logger.warning("Codebook decoding failed: %s", exc)
            return text
        return decoded

    def render_codebook_prompt(self) -> str:
        """
        Render the system-prompt block defining all active sigils.
        """
        if not self._active:
            return ""
        lines = ["[CODEBOOK]"]
        for sigil in sorted(self._active, key=self._sigil_sort_key):
            definition = self._active[sigil].definition.replace('"', '\\"')
            lines.append(f'{sigil} = "{definition}"')
        lines.append("[/CODEBOOK]")
        return "\n".join(lines)

    def estimate_savings(self) -> dict:
        """
        Return current token-savings statistics for monitoring.
        """
        overhead = count_tokens(self.render_codebook_prompt()) if self._active else 0
        estimated_saved = self._total_tokens_saved
        profitable = 0
        for sigil, entry in self._active.items():
            saved_each = max(count_tokens(entry.definition) - count_tokens(sigil), 0)
            entry_overhead = count_tokens(f'{sigil} = "{entry.definition}"')
            if entry.usage_count * saved_each > entry_overhead:
                profitable += 1
        return {
            "codebook_overhead_tokens": overhead,
            "total_tokens_saved": estimated_saved,
            "net_savings": estimated_saved - overhead,
            "entries_active": len(self._active),
            "entries_profitable": profitable,
        }

    def garbage_collect(self) -> list[str]:
        """
        Expire entries unused for ``expiry_turns`` observed turns.

        Returns:
            List of expired sigils.
        """
        return self._garbage_collect()

    def _extract_candidate_counts(self, text: str) -> dict[str, int]:
        words = [token for token in _WORD_RE.findall(text) if re.search(r"[A-Za-z0-9]", token)]
        if len(words) < 3:
            return {}
        words = words[:2500]
        counts: dict[str, int] = {}
        max_n = min(15, len(words))
        for n in range(3, max_n + 1):
            for index in range(0, len(words) - n + 1):
                phrase_words = words[index : index + n]
                phrase = " ".join(phrase_words)
                if count_tokens(phrase) < self.min_phrase_tokens:
                    continue
                if not self._is_candidate_phrase(phrase_words):
                    continue
                normalized = self._normalize_phrase(phrase)
                counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    def _is_candidate_phrase(self, words: Iterable[str]) -> bool:
        phrase_words = list(words)
        if len(phrase_words) < 3:
            return False
        alpha_numeric = sum(1 for word in phrase_words if re.search(r"[A-Za-z0-9]", word))
        if alpha_numeric < 3:
            return False
        phrase = " ".join(phrase_words).lower()
        low_value_prefixes = (
            "i think",
            "it seems",
            "as discussed",
            "let me",
            "i will",
            "we need",
        )
        return not phrase.startswith(low_value_prefixes)

    def _estimated_net_savings(self, phrase: str, occurrences: int) -> int:
        sigil = f"{self.sigil_prefix}{self._next_id}"
        saved_each = max(count_tokens(phrase) - count_tokens(sigil), 0)
        overhead = count_tokens(f'{sigil} = "{phrase}"') + 4
        return occurrences * saved_each - overhead

    def _next_sigil(self) -> str:
        while True:
            sigil = f"{self.sigil_prefix}{self._next_id}"
            self._next_id += 1
            if sigil not in self._active:
                return sigil

    def _garbage_collect(self) -> list[str]:
        expired: list[str] = []
        for sigil, entry in list(self._active.items()):
            if self._turn_index - entry.last_used_turn >= self.expiry_turns:
                expired.append(sigil)
                self._active.pop(sigil, None)
                self.entries.pop(sigil, None)
                self.usage_count.pop(sigil, None)
                self._phrase_to_sigil.pop(self._normalize_key(entry.definition), None)
        return expired

    def _overlaps_active_definition(self, phrase: str) -> bool:
        phrase_key = self._normalize_key(phrase)
        phrase_words = set(re.findall(r"[a-z0-9]+", phrase_key))
        for entry in self._active.values():
            active_key = self._normalize_key(entry.definition)
            if phrase_key in active_key or active_key in phrase_key:
                return True
            active_words = set(re.findall(r"[a-z0-9]+", active_key))
            if phrase_words and active_words:
                overlap = len(phrase_words & active_words) / min(len(phrase_words), len(active_words))
                if overlap >= 0.55:
                    return True
        return False

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        return self._normalize_key(phrase) in self._normalize_key(text)

    def _replace_phrase(self, text: str, phrase: str, sigil: str) -> tuple[str, int]:
        pattern = re.compile(re.escape(phrase))
        new_text, replacements = pattern.subn(sigil, text)
        if replacements == 0 and self.match_mode == "semantic":
            # Semantic matching is intentionally conservative until an embedding
            # backend is configured.  Normalized whitespace matching catches the
            # most common formatting differences without risking false positives.
            whitespace_pattern = re.compile(r"\s+".join(map(re.escape, phrase.split())))
            new_text, replacements = whitespace_pattern.subn(sigil, text)
        return new_text, replacements

    def _sync_public_state(self, sigil: str) -> None:
        entry = self._active[sigil]
        self.entries[sigil] = entry.definition
        self.usage_count[sigil] = entry.usage_count

    @staticmethod
    def _normalize_phrase(phrase: str) -> str:
        normalized = _SPACE_RE.sub(" ", phrase).strip()
        return normalized.strip(" \t\r\n.,;:!?")

    @classmethod
    def _normalize_key(cls, phrase: str) -> str:
        return cls._normalize_phrase(phrase).casefold()

    @staticmethod
    def _sigil_sort_key(sigil: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", sigil)
        if match:
            return int(match.group(1)), sigil
        return 0, sigil
