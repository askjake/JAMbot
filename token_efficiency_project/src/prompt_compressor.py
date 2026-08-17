"""
Heuristic prompt compressor inspired by LLMLingua.

The compressor removes low-information wording from historical context while
protecting exact spans such as code blocks, IDs, paths, URLs, error codes, and
quoted strings.  It is intentionally lightweight and dependency-optional: if a
perplexity or embedding backend is unavailable, safe heuristics are used.
"""

import logging
import math
import re
from dataclasses import dataclass

from .token_counter import count_tokens

logger = logging.getLogger(__name__)

_DEFAULT_PROTECTED_PATTERNS = [
    r"```[\s\S]*?```",
    r"R\d{10}",
    r"/[\w/.-]+",
    r"https?://\S+",
    r"\b[A-Z]{2,}[-_]\d+\b",
    r'"[^"\n]*"',
]

_DEFAULT_STOP_PHRASES = [
    "I see",
    "Got it",
    "Sure thing",
    "That makes sense",
    "As I mentioned",
    "As we discussed",
    "I think that",
    "It seems like",
    "basically",
    "essentially",
]

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "it",
    "this",
    "these",
    "those",
    "there",
    "here",
    "very",
    "really",
    "just",
    "also",
    "then",
    "than",
    "so",
}

_TOKEN_RE = re.compile(r"\s+|[^\s]+", re.UNICODE)


@dataclass(slots=True)
class ProtectedSpan:
    """Exact text span temporarily replaced during compression."""

    placeholder: str
    text: str


class PromptCompressor:
    """
    Compress historical prompt text while preserving protected content.

    Args:
        target_ratio: Fraction of lexical tokens to keep.  ``1.0`` disables
            compression; ``0.6`` keeps roughly 60% of eligible tokens.
        small_model: Placeholder name for optional perplexity backend.  The
            current implementation uses deterministic heuristics.
        protected_patterns: Regex patterns whose matches must never be modified.
        stop_phrases: Low-information phrases removed before token scoring.
        method: ``"heuristic"`` or ``"perplexity"``.  Perplexity currently
            falls back to heuristics unless a backend is added.
    """

    def __init__(
        self,
        target_ratio: float = 0.6,
        small_model: str = "gpt2",
        protected_patterns: list[str] | None = None,
        stop_phrases: list[str] | None = None,
        method: str = "heuristic",
    ):
        if not 0 < target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")
        self.target_ratio = target_ratio
        self.small_model = small_model
        self.protected_patterns = protected_patterns or list(_DEFAULT_PROTECTED_PATTERNS)
        self.stop_phrases = stop_phrases or list(_DEFAULT_STOP_PHRASES)
        self.method = method if method in {"heuristic", "perplexity"} else "heuristic"

    def compress(self, text: str, target_ratio: float | None = None) -> str:
        """
        Compress text to the requested ratio while preserving meaning.

        Protected spans are extracted before compression and restored exactly.
        On any failure, the original text is returned.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return text

        ratio = self.target_ratio if target_ratio is None else target_ratio
        if not 0 < ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")
        if ratio >= 0.99:
            return text

        original_tokens = count_tokens(text)
        try:
            protected_text, spans = self._extract_protected_spans(text)
            protected_text = self._remove_stop_phrases(protected_text)
            lexical_tokens = [token for token in _TOKEN_RE.findall(protected_text) if token]
            scored = [(token, self._score_token(token)) for token in lexical_tokens]
            eligible_count = sum(1 for token, _ in scored if not token.isspace())
            keep_count = max(1, math.ceil(eligible_count * ratio))

            ranked = sorted(
                ((index, token, score) for index, (token, score) in enumerate(scored) if not token.isspace()),
                key=lambda item: item[2],
                reverse=True,
            )
            keep_indices = {index for index, token, _score in ranked[:keep_count]}
            for index, (token, _score) in enumerate(scored):
                if self._is_placeholder(token) or self._must_keep_token(token):
                    keep_indices.add(index)

            compressed_parts: list[str] = []
            previous_was_space = False
            for index, (token, _score) in enumerate(scored):
                if token.isspace():
                    if compressed_parts and not previous_was_space:
                        compressed_parts.append(" ")
                        previous_was_space = True
                    continue
                if index not in keep_indices:
                    continue
                compressed_parts.append(token)
                previous_was_space = False

            compressed = "".join(compressed_parts).strip()
            compressed = re.sub(r"\s+([,.;:!?])", r"\1", compressed)
            compressed = re.sub(r"\s+", " ", compressed).strip()
            compressed = self._restore_protected_spans(compressed, spans)

            if not compressed:
                return text
            if count_tokens(compressed) >= original_tokens:
                return text
            if not self._protected_spans_preserved(text, compressed):
                return text
            return compressed
        except Exception as exc:  # pragma: no cover - defensive, not expected
            logger.warning("Prompt compression failed: %s", exc)
            return text

    def compute_token_importance(self, text: str) -> list[tuple[str, float]]:
        """
        Score each non-whitespace token by heuristic information content.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [(token, self._score_token(token)) for token in re.findall(r"\S+", text)]

    def evaluate_fidelity(self, original: str, compressed: str) -> float:
        """
        Estimate semantic preservation between original and compressed text.

        The score is a conservative blend of content-word overlap and protected
        span preservation.  It returns 1.0 for identical text and 0.0 when no
        signal remains.
        """
        if not isinstance(original, str) or not isinstance(compressed, str):
            raise TypeError("original and compressed must be strings")
        if original == compressed:
            return 1.0
        original_terms = self._content_terms(original)
        compressed_terms = self._content_terms(compressed)
        if not original_terms:
            overlap = 1.0 if compressed.strip() else 0.0
        else:
            overlap = len(original_terms & compressed_terms) / len(original_terms)

        protected = self._find_protected_text(original)
        if not protected:
            protected_score = 1.0
        else:
            preserved = sum(1 for span in protected if span in compressed)
            protected_score = preserved / len(protected)
        return max(0.0, min(1.0, (0.75 * overlap) + (0.25 * protected_score)))

    def _score_token(self, token: str) -> float:
        stripped = token.strip()
        if not stripped:
            return 0.0
        lowered = stripped.casefold().strip(".,;:!?()[]{}")
        if self._is_placeholder(stripped):
            return 10.0
        if self._must_keep_token(stripped):
            return 5.0
        if lowered in _STOP_WORDS:
            return 0.05
        if re.search(r"\d", stripped):
            return 1.0
        if re.search(r"[/_.:@#-]", stripped):
            return 0.95
        if stripped.isupper() and len(stripped) > 1:
            return 0.9
        if len(stripped) >= 10:
            return 0.8
        if len(stripped) >= 6:
            return 0.55
        return 0.35

    def _extract_protected_spans(self, text: str) -> tuple[str, list[ProtectedSpan]]:
        spans: list[ProtectedSpan] = []
        occupied = self._protected_ranges(text)

        result: list[str] = []
        last = 0
        for index, (start, end) in enumerate(occupied):
            placeholder = f"⟦PROTECTED_{index}⟧"
            result.append(text[last:start])
            result.append(placeholder)
            spans.append(ProtectedSpan(placeholder=placeholder, text=text[start:end]))
            last = end
        result.append(text[last:])
        return "".join(result), spans

    def _restore_protected_spans(self, text: str, spans: list[ProtectedSpan]) -> str:
        restored = text
        for span in spans:
            restored = restored.replace(span.placeholder, span.text)
        return restored

    def _remove_stop_phrases(self, text: str) -> str:
        cleaned = text
        for phrase in self.stop_phrases:
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _protected_spans_preserved(self, original: str, compressed: str) -> bool:
        return all(span in compressed for span in self._find_protected_text(original))

    def _find_protected_text(self, text: str) -> list[str]:
        return [text[start:end] for start, end in self._protected_ranges(text)]

    def _protected_ranges(self, text: str) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        for pattern in self.protected_patterns:
            try:
                for match in re.finditer(pattern, text, flags=re.MULTILINE):
                    start, end = match.span()
                    candidates.append((start, end))
            except re.error as exc:
                logger.warning("Invalid protected pattern %r: %s", pattern, exc)
        # Prefer the longest protected span when patterns overlap, such as a URL
        # that also contains a path-like //host/path substring.
        selected: list[tuple[int, int]] = []
        for start, end in sorted(candidates, key=lambda span: (-(span[1] - span[0]), span[0])):
            if any(start < used_end and end > used_start for used_start, used_end in selected):
                continue
            selected.append((start, end))
        return sorted(selected)

    @staticmethod
    def _is_placeholder(token: str) -> bool:
        return bool(re.fullmatch(r"⟦PROTECTED_\d+⟧", token))

    @staticmethod
    def _must_keep_token(token: str) -> bool:
        return bool(
            re.search(r"R\d{10}", token)
            or re.search(r"https?://", token)
            or re.search(r"/[\w/.-]+", token)
            or re.search(r"\b[A-Z]{2,}[-_]\d+\b", token)
            or token.startswith("```")
        )

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        terms = set()
        for token in re.findall(r"[A-Za-z0-9_./:#@-]+", text.casefold()):
            if len(token) <= 2 or token in _STOP_WORDS:
                continue
            terms.add(token)
        return terms
