"""
Context assembly engine for the Token Efficiency Layer.

The engine orchestrates codebook encoding, rolling summarization, and prompt
compression into a messages list that fits a target token budget while keeping
the current user message and system instructions verbatim.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .codebook_manager import CodebookManager
from .prompt_compressor import PromptCompressor
from .rolling_summarizer import RollingSummarizer
from .token_counter import count_tokens, estimate_cost

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Preserve exact IDs, file paths, URLs, error "
    "messages, code, and user intent."
)


class ContextAssemblyEngine:
    """
    Assemble optimized LLM prompts from raw conversation history.

    Args:
        max_context_tokens: Total model context window.
        target_response_tokens: Reserved tokens for the model response.
        safety_margin: Extra buffer to avoid overflow.
        system_prompt: System instructions to keep verbatim at the top.
        codebook: Optional preconfigured ``CodebookManager``.
        summarizer: Optional preconfigured ``RollingSummarizer``.
        compressor: Optional preconfigured ``PromptCompressor``.
        codebook_enabled: Enable codebook observation and encoding.
        summarizer_enabled: Enable tiered summaries.
        compressor_enabled: Enable final compression when over budget.
        input_cost_per_1k: Cost rate used for savings metrics.
        output_cost_per_1k: Output cost rate used for savings metrics.
    """

    def __init__(
        self,
        max_context_tokens: int = 128000,
        target_response_tokens: int = 4000,
        safety_margin: int = 500,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        codebook: CodebookManager | None = None,
        summarizer: RollingSummarizer | None = None,
        compressor: PromptCompressor | None = None,
        codebook_enabled: bool = True,
        summarizer_enabled: bool = True,
        compressor_enabled: bool = True,
        input_cost_per_1k: float = 0.003,
        output_cost_per_1k: float = 0.015,
    ):
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if target_response_tokens < 0 or safety_margin < 0:
            raise ValueError("target_response_tokens and safety_margin cannot be negative")
        self.max_context_tokens = max_context_tokens
        self.target_response_tokens = target_response_tokens
        self.safety_margin = safety_margin
        self.budget = max(max_context_tokens - target_response_tokens - safety_margin, 1)
        self.system_prompt = system_prompt

        self.codebook = codebook or CodebookManager()
        self.summarizer = summarizer or RollingSummarizer()
        self.compressor = compressor or PromptCompressor()
        self.codebook_enabled = codebook_enabled
        self.summarizer_enabled = summarizer_enabled
        self.compressor_enabled = compressor_enabled
        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k

        self._last_compressor_savings = 0
        self._last_metrics: dict[str, Any] = {
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compression_ratio": 1.0,
            "token_savings": 0,
            "cost_savings_usd": 0.0,
            "components": {
                "codebook_savings": 0,
                "summarizer_savings": 0,
                "compressor_savings": self._last_compressor_savings,
            },
        }

    @classmethod
    def from_config(cls, path: str | Path) -> "ContextAssemblyEngine":
        """
        Create an engine from a YAML or JSON configuration file.

        If YAML support is unavailable or parsing fails, the error is raised with
        context rather than silently producing a surprising configuration.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        data = cls._load_config_file(config_path)
        root = data.get("token_efficiency", data)
        if not isinstance(root, dict):
            raise ValueError("Config root must be a mapping")

        codebook_config = root.get("codebook", {}) or {}
        summarizer_config = root.get("summarizer", {}) or {}
        compressor_config = root.get("compressor", {}) or {}
        metrics_config = root.get("metrics", {}) or {}

        codebook = CodebookManager(
            max_entries=int(codebook_config.get("max_entries", 20)),
            sigil_prefix=str(codebook_config.get("sigil_prefix", "Σ")),
            min_occurrences_to_promote=int(codebook_config.get("min_occurrences_to_promote", 3)),
            min_phrase_tokens=int(codebook_config.get("min_phrase_tokens", 8)),
            expiry_turns=int(codebook_config.get("expiry_turns", 10)),
            match_mode=str(codebook_config.get("match_mode", "exact")),
        )
        summarizer = RollingSummarizer(
            tier1_turns=int(summarizer_config.get("tier1_verbatim_turns", 5)),
            tier2_turns=int(summarizer_config.get("tier2_turns", 10)),
            tier2_budget_tokens=int(summarizer_config.get("tier2_budget_tokens", 1500)),
            tier3_budget_tokens=int(summarizer_config.get("tier3_budget_tokens", 500)),
            preserve_patterns=list(summarizer_config.get("preserve_patterns", []) or []),
        )
        compressor = PromptCompressor(
            target_ratio=float(compressor_config.get("target_ratio", 0.65)),
            protected_patterns=list(compressor_config.get("protected_patterns", []) or []),
            stop_phrases=list(compressor_config.get("stop_phrases", []) or []),
            method=str(compressor_config.get("method", "heuristic")),
        )

        return cls(
            max_context_tokens=int(root.get("context_window", 128000)),
            target_response_tokens=int(root.get("target_response_tokens", 4000)),
            safety_margin=int(root.get("safety_margin", 500)),
            codebook=codebook,
            summarizer=summarizer,
            compressor=compressor,
            codebook_enabled=bool(codebook_config.get("enabled", True)),
            summarizer_enabled=bool(summarizer_config.get("enabled", True)),
            compressor_enabled=bool(compressor_config.get("enabled", True)),
            input_cost_per_1k=float(metrics_config.get("input_cost_per_1k", 0.003)),
            output_cost_per_1k=float(metrics_config.get("output_cost_per_1k", 0.015)),
        )

    def process_new_turn(self, role: str, content: str) -> None:
        """
        Ingest a new conversation turn.

        The text is observed for codebook candidates, then added to the rolling
        summarizer.  If a component fails, the turn is still kept in full history
        so correctness is preserved.
        """
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        if self.codebook_enabled:
            try:
                self.codebook.observe(content)
                self.codebook.promote_candidates()
            except Exception as exc:  # pragma: no cover - defensive, not expected
                logger.warning("Codebook processing failed; continuing unencoded: %s", exc)

        try:
            self.summarizer.add_turn(role, content)
        except Exception as exc:  # pragma: no cover - defensive, not expected
            logger.warning("Summarizer ingestion failed; appending raw history: %s", exc)
            self.summarizer.full_history.append({"role": role, "content": content})

    def build_optimized_prompt(self, current_message: str) -> list[dict]:
        """
        Build a messages list ready for an LLM API call.

        The current user message is appended exactly once and is never compressed
        or codebook-encoded, even when it has already been processed by
        ``process_new_turn``.
        """
        if not isinstance(current_message, str):
            raise TypeError("current_message must be a string")

        history = self._history_excluding_current(current_message)
        original_messages = self._build_uncompressed_messages(history, current_message)
        original_tokens = count_tokens(original_messages)

        try:
            messages = self._build_compressed_messages(history, current_message)
        except Exception as exc:  # pragma: no cover - defensive, not expected
            logger.warning("Optimized prompt assembly failed; falling back to raw prompt: %s", exc)
            messages = original_messages

        self._last_compressor_savings = 0
        messages = self._enforce_budget(messages, current_message)
        compressed_tokens = count_tokens(messages)
        codebook_stats = self.codebook.estimate_savings() if self.codebook_enabled else {}
        summarizer_stats = self.summarizer.estimate_tokens()
        self._last_metrics = {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compressed_tokens / original_tokens if original_tokens else 1.0,
            "token_savings": max(original_tokens - compressed_tokens, 0),
            "cost_savings_usd": estimate_cost(
                max(original_tokens - compressed_tokens, 0),
                0,
                input_rate=self.input_cost_per_1k,
                output_rate=self.output_cost_per_1k,
            ),
            "components": {
                "codebook_savings": int(codebook_stats.get("net_savings", 0)),
                "summarizer_savings": max(
                    int(summarizer_stats.get("full_history_tokens", 0))
                    - int(summarizer_stats.get("total_tokens", 0)),
                    0,
                ),
                "compressor_savings": self._last_compressor_savings,
            },
        }
        return messages

    def get_metrics(self) -> dict:
        """
        Return metrics from the most recent prompt build.
        """
        return dict(self._last_metrics)

    def _build_compressed_messages(self, history: list[dict[str, str]], current_message: str) -> list[dict]:
        partition = self.summarizer.partition_history(history) if self.summarizer_enabled else {
            "tier1": history,
            "tier2": "",
            "tier3": "",
        }
        codebook_prompt = self.codebook.render_codebook_prompt() if self.codebook_enabled else ""
        memory_parts = [part for part in (partition.get("tier3", ""), partition.get("tier2", "")) if part]
        system_content = self._join_blocks([codebook_prompt, self.system_prompt, *memory_parts])
        messages: list[dict] = [{"role": "system", "content": system_content}]

        for message in partition.get("tier1", []):
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if self.codebook_enabled:
                content = self.codebook.encode(content)
            messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": current_message})
        return messages

    def _build_uncompressed_messages(self, history: list[dict[str, str]], current_message: str) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))} for msg in history)
        messages.append({"role": "user", "content": current_message})
        return messages

    def _enforce_budget(self, messages: list[dict], current_message: str) -> list[dict]:
        if count_tokens(messages) <= self.budget:
            return messages

        adjusted = [dict(message) for message in messages]
        compressor_savings = 0
        if self.compressor_enabled:
            for index, message in enumerate(adjusted):
                if index == 0:
                    # Do not compress system instructions or codebook definitions.
                    continue
                if index == len(adjusted) - 1 and message.get("role") == "user" and message.get("content") == current_message:
                    continue
                before = count_tokens(str(message.get("content", "")))
                message["content"] = self.compressor.compress(str(message.get("content", "")))
                compressor_savings += max(before - count_tokens(str(message.get("content", ""))), 0)
                if count_tokens(adjusted) <= self.budget:
                    self._last_compressor_savings = compressor_savings
                    return adjusted

        # If still over budget, drop oldest non-system, non-current history first.
        while len(adjusted) > 2 and count_tokens(adjusted) > self.budget:
            adjusted.pop(1)

        # As a final guard, trim summary memory from the system block but keep the
        # system prompt and current user message intact.
        if count_tokens(adjusted) > self.budget and adjusted:
            adjusted[0]["content"] = self.system_prompt
        self._last_compressor_savings = compressor_savings
        return adjusted

    def _history_excluding_current(self, current_message: str) -> list[dict[str, str]]:
        history = list(self.summarizer.full_history)
        if history and history[-1].get("role") == "user" and history[-1].get("content") == current_message:
            return history[:-1]
        return history

    @staticmethod
    def _join_blocks(blocks: list[str]) -> str:
        return "\n\n".join(block.strip() for block in blocks if block and block.strip())

    @staticmethod
    def _load_config_file(path: Path) -> dict[str, Any]:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(raw)
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("PyYAML is required to load YAML config files") from exc
        loaded = yaml.safe_load(raw)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a mapping")
        return loaded
