import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.message.compression import count_message_tokens
from app.message.message_tiering import apply_tiered_compression
from app.message.ai_message_compressor import compress_ai_message
from app.tools.progressive_tool_memory import generate_progressive_levels
from app.agent.agents.agentic_rag import sanitize_tool_messages, ensure_bedrock_converse_message_shape


def make_turn(i, lines=45):
    tc_id = f'tool-{i}'
    human = HumanMessage(content=f'Turn {i}: run it and verify /tmp/script_{i}.py')
    code_block = chr(96)*3 + 'python\n' + '\n'.join(f'print({j})' for j in range(30)) + '\n' + chr(96)*3
    ai = AIMessage(
        content=(
            'Let me run that now.\n\n'
            + '\n'.join(f'Verbose explanation line {j} with implementation details and verification result passed for turn {i}.' for j in range(40))
            + '\n\n' + code_block + '\n\nFinal result: implemented and verified pass.'
        ),
        tool_calls=[{'id': tc_id, 'name': 'run_ssh_command', 'args': {'command': f'pytest tests/test_{i}.py'}}],
    )
    tool = ToolMessage(
        content='command: pytest tests/test_context.py\nexit code: 0\n' + '\n'.join(f'output line {j}' for j in range(lines)) + '\nPASS all tests',
        tool_call_id=tc_id,
        name='run_ssh_command',
    )
    return [human, ai, tool]


def test_tiered_compression_load():
    messages = []
    for i in range(1, 85):
        messages.extend(make_turn(i, lines=60))
    before = count_message_tokens(messages)
    start = time.perf_counter()
    compressed = apply_tiered_compression(messages, target_tokens=80000)
    elapsed = (time.perf_counter() - start) * 1000
    after = count_message_tokens(compressed)
    assert elapsed < 500, f'Too slow: {elapsed}ms'
    assert after < before * 0.65, (before, after)
    assert after <= 80000, f'After tokens {after} exceeds 80000'
    assert isinstance(compressed[0], HumanMessage)
    assert compressed[0].content.startswith('[CONVERSATION CONTEXT')
    # Last 3 user turns retained at full fidelity.
    assert any(getattr(m, 'content', '') == 'Turn 84: run it and verify /tmp/script_84.py' for m in compressed)
    # Complete tool-call pairs are still present for retained tool turns.
    cleaned = sanitize_tool_messages(compressed)
    assert len(cleaned) == len(compressed)
    assert isinstance(ensure_bedrock_converse_message_shape(cleaned)[0], HumanMessage)


def test_ai_message_compression_preserves_tool_calls():
    msg = AIMessage(
        content='Intro answer.\n\n' + '\n'.join('Decision: keep this important conclusion.' for _ in range(80)) + '\n\nFinal answer: done.',
        tool_calls=[{'id': 'abc', 'name': 'tool', 'args': {'x': 'y'}}]
    )
    out = compress_ai_message(msg, target_tokens=120)
    assert out.tool_calls == msg.tool_calls
    assert count_message_tokens([out]) < count_message_tokens([msg])
    assert 'COMPRESSED PRIOR AI RESPONSE' in out.content


def test_non_browse_progressive_memory():
    raw = 'command: ls -la /tmp\nexit code: 0\n' + '\n'.join(f'line {i}' for i in range(80))
    p = generate_progressive_levels('run_ssh_command', raw, tool_call_id='x', turn_created=1)
    near = p.get_content_for_turn(2)
    old = p.get_content_for_turn(7)
    assert 'first_20_lines' in near
    assert len(old) < len(near) < len(raw)
    assert 'Ran' in old or 'run_ssh_command' in old.lower()


def test_bedrock_shape_after_orphan_trim():
    orphan = ToolMessage(content='orphan', tool_call_id='missing')
    human = HumanMessage(content='hello')
    cleaned = sanitize_tool_messages([orphan, human])
    assert cleaned == [human]
    assert isinstance(ensure_bedrock_converse_message_shape(cleaned)[0], HumanMessage)


def test_agentic_truncate_integration():
    from app.agent.agents.agentic_rag import truncate_messages
    messages = []
    for i in range(1, 85):
        messages.extend(make_turn(i, lines=60))
    out = truncate_messages(messages, 100)
    assert isinstance(out[0], HumanMessage)
    assert count_message_tokens(out) <= 140000
    assert any(getattr(m, 'content', '') == 'Turn 84: run it and verify /tmp/script_84.py' for m in out)
