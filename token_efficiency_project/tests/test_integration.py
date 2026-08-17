"""
Integration and component tests for the Token Efficiency Layer.
"""

import pytest

from src.codebook_manager import CodebookManager
from src.context_assembly import ContextAssemblyEngine
from src.prompt_compressor import PromptCompressor
from src.rolling_summarizer import RollingSummarizer
from src.token_counter import count_tokens

REPEATED_PHRASE = "Guide error 1031 popup appearing after channel change on Hopper receiver"
PROTECTED_ID = "R1955245852"
PROTECTED_PATH = "/tmp/superset_cookies.json"
PROTECTED_URL = "https://example.com/dashboard?id=1031"
PROTECTED_CODE = "ATVDI-1031"


def make_engine(**overrides):
    """Create a small-budget test engine with deterministic components."""
    defaults = {
        "max_context_tokens": 900,
        "target_response_tokens": 100,
        "safety_margin": 25,
        "codebook": CodebookManager(
            min_occurrences_to_promote=3,
            min_phrase_tokens=6,
            expiry_turns=100,
        ),
        "summarizer": RollingSummarizer(
            tier1_turns=5,
            tier2_turns=8,
            tier2_budget_tokens=180,
            tier3_budget_tokens=100,
        ),
        "compressor": PromptCompressor(target_ratio=0.55),
    }
    defaults.update(overrides)
    return ContextAssemblyEngine(**defaults)


def long_turn(index: int) -> tuple[str, str]:
    """Return a realistic user/assistant turn pair with repeated entities."""
    filler = (
        "This is explanatory background that repeats operational context and can be summarized. "
        "The system should keep technical facts but remove redundant narration. "
    ) * 4
    user = (
        f"Turn {index}: please investigate {REPEATED_PHRASE} for receiver {PROTECTED_ID}. "
        f"Use script {PROTECTED_PATH} and compare against {PROTECTED_CODE}. {filler}"
    )
    assistant = (
        f"Turn {index}: decided to preserve {PROTECTED_ID}, run python {PROTECTED_PATH}, "
        f"and verify {PROTECTED_CODE}. Root cause candidate remains {REPEATED_PHRASE}. {filler}"
    )
    return user, assistant


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_basic_compression(self):
        """Verify basic compression on a 10-turn conversation."""
        engine = make_engine()
        raw_messages = []
        for index in range(10):
            user, assistant = long_turn(index)
            engine.process_new_turn("user", user)
            engine.process_new_turn("assistant", assistant)
            raw_messages.extend([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ])

        current = f"Current request: summarize the failure for {PROTECTED_ID} without changing this message."
        optimized = engine.build_optimized_prompt(current)
        raw_with_current = [{"role": "system", "content": engine.system_prompt}, *raw_messages, {"role": "user", "content": current}]

        assert isinstance(optimized, list)
        assert optimized[-1] == {"role": "user", "content": current}
        assert count_tokens(optimized) < count_tokens(raw_with_current)
        assert engine.get_metrics()["token_savings"] > 0


    def test_30_turn_pipeline_meets_project_acceptance(self):
        """Verify a 30-turn conversation compresses strongly and keeps recent facts."""
        engine = make_engine(max_context_tokens=1400, target_response_tokens=120, safety_margin=40)
        raw_messages = []
        for index in range(30):
            user, assistant = long_turn(index)
            engine.process_new_turn("user", user)
            engine.process_new_turn("assistant", assistant)
            raw_messages.extend([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ])

        current = f"Final request: preserve {PROTECTED_ID}, {PROTECTED_PATH}, and {PROTECTED_CODE} exactly."
        optimized = engine.build_optimized_prompt(current)
        raw_with_current = [{"role": "system", "content": engine.system_prompt}, *raw_messages, {"role": "user", "content": current}]
        savings_ratio = 1 - (count_tokens(optimized) / count_tokens(raw_with_current))
        rendered = "\n".join(message["content"] for message in optimized)

        assert count_tokens(optimized) <= engine.budget
        assert savings_ratio > 0.40
        assert engine.codebook.entries
        assert "[CODEBOOK]" in optimized[0]["content"]
        for protected in [PROTECTED_ID, PROTECTED_PATH, PROTECTED_CODE]:
            assert protected in rendered
        assert optimized[-1]["content"] == current

    def test_entity_preservation(self):
        """Verify that receiver IDs, paths, codes survive compression."""
        engine = make_engine()
        for index in range(12):
            user, assistant = long_turn(index)
            engine.process_new_turn("user", user)
            engine.process_new_turn("assistant", assistant)

        current = f"Use {PROTECTED_ID}, {PROTECTED_PATH}, {PROTECTED_CODE}, and {PROTECTED_URL} exactly."
        optimized = engine.build_optimized_prompt(current)
        rendered = "\n".join(message["content"] for message in optimized)

        for protected in [PROTECTED_ID, PROTECTED_PATH, PROTECTED_CODE, PROTECTED_URL]:
            assert protected in rendered
        assert optimized[-1]["content"] == current

    def test_codebook_promotion(self):
        """Verify phrases are promoted to sigils after 3 occurrences."""
        manager = CodebookManager(min_occurrences_to_promote=3, min_phrase_tokens=6, expiry_turns=100)
        for _ in range(3):
            manager.observe(f"The active issue is {REPEATED_PHRASE}. Please keep tracking {REPEATED_PHRASE}.")
        promoted = manager.promote_candidates()

        assert promoted
        prompt = manager.render_codebook_prompt()
        assert "[CODEBOOK]" in prompt
        encoded = manager.encode(f"Historical note: {REPEATED_PHRASE} is still active.")
        assert promoted[0] in encoded
        assert REPEATED_PHRASE not in encoded
        assert REPEATED_PHRASE in manager.decode(encoded)

    def test_tier_promotion(self):
        """Verify turns move from tier1 → tier2 → tier3 correctly."""
        summarizer = RollingSummarizer(tier1_turns=3, tier2_turns=4, tier2_budget_tokens=120, tier3_budget_tokens=80)
        for index in range(10):
            summarizer.add_turn("user", f"Turn {index} decision: preserve {PROTECTED_ID} and error {PROTECTED_CODE}.")

        assert len(summarizer.tiers[1]) == 3
        assert "[TIER 2 DETAILED SUMMARY]" in summarizer.tiers[2]
        assert "[TIER 3 ABSTRACT SUMMARY]" in summarizer.tiers[3]
        assert PROTECTED_ID in summarizer.tiers[2] or PROTECTED_ID in summarizer.tiers[3]
        assert PROTECTED_CODE in summarizer.tiers[2] or PROTECTED_CODE in summarizer.tiers[3]

    def test_budget_enforcement(self):
        """Verify output never exceeds configured token budget."""
        engine = make_engine(max_context_tokens=260, target_response_tokens=50, safety_margin=20)
        for index in range(20):
            user, assistant = long_turn(index)
            engine.process_new_turn("user", user)
            engine.process_new_turn("assistant", assistant)

        current = "Current user message must remain verbatim with R1955245852."
        optimized = engine.build_optimized_prompt(current)

        assert count_tokens(optimized) <= engine.budget
        assert optimized[-1]["content"] == current

    def test_graceful_degradation(self, monkeypatch):
        """Verify system falls back to uncompressed on component failure."""
        engine = make_engine()
        engine.process_new_turn("user", f"Please inspect {REPEATED_PHRASE} on {PROTECTED_ID}.")
        engine.process_new_turn("assistant", "Acknowledged and preserving facts.")

        def explode(_history):
            raise RuntimeError("forced summarizer failure")

        monkeypatch.setattr(engine.summarizer, "partition_history", explode)
        current = "Current message should still be delivered exactly."
        optimized = engine.build_optimized_prompt(current)

        assert optimized[-1]["content"] == current
        assert any(REPEATED_PHRASE in message["content"] for message in optimized)

    def test_protected_patterns(self):
        """Verify regex-protected content is never modified."""
        compressor = PromptCompressor(target_ratio=0.35)
        text = (
            f"Got it. The receiver {PROTECTED_ID} failed with {PROTECTED_CODE}. "
            f"Read {PROTECTED_PATH} and open {PROTECTED_URL}. "
            "```python\nprint('keep exact code')\n``` "
            "This explanatory filler is intentionally verbose and redundant. " * 8
        )
        compressed = compressor.compress(text)

        assert PROTECTED_ID in compressed
        assert PROTECTED_CODE in compressed
        assert PROTECTED_PATH in compressed
        assert PROTECTED_URL in compressed
        assert "```python\nprint('keep exact code')\n```" in compressed
        assert count_tokens(compressed) < count_tokens(text)
