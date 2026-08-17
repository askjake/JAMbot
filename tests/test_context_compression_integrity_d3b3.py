"""Phase D3B3 - context-compression integrity invariants.

These tests pin three correctness invariants for the active parent-path
compression stack used by ``app.agent.agents.agentic_rag.call_model``:

1. Compression must be a pure function of its input.  Cross-call learned state
   makes the model-visible prefix churn and defeats provider prompt caching.
2. Compression must never emit an encoded sigil without the legend needed to
   decode it.
3. Compression must never strip negations out of narrative text, because that
   inverts factual and safety-relevant claims.

It also pins the provider-repair invariant that a duplicate tool result for one
tool-call id is not provider-valid and must be reduced to a single result.
"""

from __future__ import annotations

import copy
import hashlib

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.agents.agentic_rag import sanitize_tool_messages
from app.message.token_efficiency_adapter import apply_token_efficiency_layer

SIGIL_PREFIX = "\u03a3"


def _render(messages) -> str:
    return "\n".join(f"{type(m).__name__}: {m.content}" for m in messages)


def _digest(messages) -> str:
    return hashlib.sha256(_render(messages).encode("utf-8")).hexdigest()


def _long_conversation(phrase: str, turns: int = 14) -> list:
    """A conversation large enough to activate the token efficiency layer.

    The layer only engages at >= 15 messages AND >= 40000 estimated tokens, so
    the ToolMessage payloads must be genuinely large.
    """
    messages: list = []
    for i in range(turns):
        messages.append(
            HumanMessage(
                content=(
                    f"Step {i}: please review the {phrase} and report whether anything "
                    f"unusual is present in the captured diagnostic material."
                )
            )
        )
        messages.append(
            AIMessage(
                content=(
                    f"Step {i} result: I reviewed the {phrase}. "
                    f"No cross-child tool state was shared between the workers. "
                    f"The unresolved question is whether the gap is ingestion lag "
                    f"or genuine absence, and it is not a confirmed root cause."
                )
            )
        )
        messages.append(
            ToolMessage(
                content=("diagnostic payload line\n" * 700),
                tool_call_id=f"call_{i:03d}",
                name="search_logs",
            )
        )
    messages.append(HumanMessage(content="Summarise the investigation."))
    return messages


PHRASE = "Ravenwood substation telemetry anomaly register"


def test_token_efficiency_layer_is_deterministic():
    """Identical input must produce identical model-visible output regardless of
    how many prior invocations the process has served."""
    digests = []
    for _ in range(6):
        out = apply_token_efficiency_layer(copy.deepcopy(_long_conversation(PHRASE)))
        digests.append(_digest(out))
    assert len(set(digests)) == 1, (
        "apply_token_efficiency_layer is not a pure function of its input; "
        f"observed {len(set(digests))} distinct outputs across 6 identical calls: {digests}"
    )


def _narrative_text(messages) -> str:
    """Model-visible narrative text, excluding any injected codebook legend."""
    parts = []
    for msg in messages:
        content = str(getattr(msg, "content", ""))
        if content.startswith("[CODEBOOK"):
            continue
        parts.append(f"{type(msg).__name__}: {content}")
    return "\n".join(parts)


def test_token_efficiency_layer_never_emits_undecodable_sigils():
    """If codebook sigils are substituted into model-visible text, the legend
    that defines them must also be present in that same context."""
    for call in range(8):
        out = apply_token_efficiency_layer(copy.deepcopy(_long_conversation(PHRASE)))
        text = _render(out)
        if SIGIL_PREFIX in _narrative_text(out):
            assert "[CODEBOOK" in text, (
                f"call {call}: codebook sigils were substituted into model-visible "
                "context without the decoding legend"
            )


def test_token_efficiency_layer_is_semantically_lossless():
    """The layer must not silently destroy narrative content.

    Codebook substitution is acceptable because it is reversible when the legend
    is present.  Dropping words out of narrative prose is not reversible, and it
    can invert negations such as "no cross-child tool state was shared" or
    "not a confirmed root cause".  This test requires that every AI narrative
    message is either unchanged or fully recoverable by decoding the codebook.
    """
    import app.message.token_efficiency_adapter as tea

    for call in range(8):
        original = _long_conversation(PHRASE)
        originals = {
            id_: str(m.content)
            for id_, m in enumerate(original)
            if isinstance(m, AIMessage)
        }
        out = apply_token_efficiency_layer(copy.deepcopy(original))
        codebook = tea._codebook

        # Align AI messages by order; the layer preserves order and count of
        # non-injected messages.
        produced = [m for m in out if isinstance(m, AIMessage)]
        expected = [m for m in original if isinstance(m, AIMessage)]
        assert len(produced) == len(expected)

        for idx, (got, want) in enumerate(zip(produced, expected)):
            got_text = str(got.content)
            want_text = str(want.content)
            if got_text == want_text:
                continue
            decoded = codebook.decode(got_text) if codebook is not None else got_text
            assert decoded == want_text, (
                f"call {call} message {idx}: narrative content was irreversibly "
                f"altered by the token efficiency layer.\n"
                f"  original: {want_text!r}\n"
                f"  decoded : {decoded!r}"
            )


def test_sanitize_drops_duplicate_tool_messages():
    """Two tool results for one tool-call id are not provider-valid."""
    messages = [
        HumanMessage(content="run a search"),
        AIMessage(content="", tool_calls=[{"name": "search_logs", "id": "dup_1", "args": {}}]),
        ToolMessage(content="first result", tool_call_id="dup_1", name="search_logs"),
        ToolMessage(content="second result", tool_call_id="dup_1", name="search_logs"),
        AIMessage(content="done"),
    ]
    cleaned = sanitize_tool_messages(list(messages))
    ids = [m.tool_call_id for m in cleaned if isinstance(m, ToolMessage)]
    assert ids.count("dup_1") == 1, (
        f"duplicate ToolMessage for one tool_call_id survived repair: {ids}"
    )
    # The retained result must be the original first result, never fabricated.
    retained = [m for m in cleaned if isinstance(m, ToolMessage)]
    assert retained and retained[0].content == "first result"


def test_sanitize_preserves_valid_single_pairs():
    """Regression guard: normal pairs must be untouched."""
    messages = [
        HumanMessage(content="run a search"),
        AIMessage(content="", tool_calls=[{"name": "search_logs", "id": "ok_1", "args": {}}]),
        ToolMessage(content="only result", tool_call_id="ok_1", name="search_logs"),
        AIMessage(content="done"),
    ]
    cleaned = sanitize_tool_messages(list(messages))
    assert len(cleaned) == 4
    assert [type(m).__name__ for m in cleaned] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"
    ]
