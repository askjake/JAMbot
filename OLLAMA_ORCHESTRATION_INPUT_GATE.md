# OLLAMA_ORCHESTRATION_INPUT_GATE — DishIP Actual Running Agent Bundle

## Scope and repo root

Repo root inspected: `/mnt/data/diship_work/jakebot` from `agent-bundle-diship-20260709_083730.tar.gz`.

Runtime appears to be the FastAPI backend in `app/`, with helper scripts at repo root, including `restart-dishchat.sh`.

## Runtime entrypoints

Primary runtime entrypoint: `app/main.py` creates the FastAPI app, initializes MCP tools during lifespan startup, sets up the LangGraph checkpointer, and includes backend routers.

Container/process entrypoints present: `Dockerfile`, `entrypoint.sh`, `restart-dishchat.sh`, and `app/main.py`.

## Active config file

Active config file: `app/config.py` via `from app.config import get_settings`.

Important finding: this actual running bundle already accepts `PLLM_PROVIDER=ollama` and `ELLM_PROVIDER=ollama`. It has an Ollama primary/efficient model and separate tool model fields. The configured API base points to an internal Ollama host, which is intentionally redacted from this report.

Initial config validation command:

```bash
PYTHONPATH=. PLLM_PROVIDER=ollama ELLM_PROVIDER=ollama python - <<'PY'
from app.config import Settings
s=Settings()
print('providers', s.PLLM_PROVIDER, s.ELLM_PROVIDER)
print('models', s.PLLM_MODEL, s.ELLM_MODEL)
print('bases', s.PLLM_API_BASE, s.ELLM_API_BASE)
PY
```

Result before edits: exit 0. Providers resolved as `ollama` / `ollama`. Model names resolved to the configured local model. API bases resolved to a configured internal URL and are redacted here.

## Active model factory

Active model factory: `app/core/llm/chat_models.py`, exported by `app/core/llm/__init__.py`.

Findings before edits:

- Active factory already imports `ChatOllama` lazily and supports `ollama` in `_create_model_for_cache_key()`.
- `requirements.txt` already includes `langchain-ollama>=0.3.0`.
- `get_tool_model()` exists and selects a tool-capable Ollama model for tool binding.
- Cache keys are legacy buckets (`primary`, `efficient`, `opus`, `tool_primary`, `tool_efficient`, `tool_opus`) rather than the required provider/role/model/base/context/output tuple.
- Active API still uses legacy `use_opus` and `model_arn` concepts, although comments state `model_arn` is ignored for Ollama.
- AWS credential refresh task only refreshes AWS-backed cached models, but `start_token_refresh_task()` does not explicitly skip pure Ollama mode before creating the background task.
- No provider-neutral role resolver was found.

## Active message / agent graph

Active chat graph: `app/agent/agents/agentic_rag.py`.

Findings before edits:

- Imports `get_model` and `get_tool_model`.
- Uses `should_use_opus_for_context()` and a boolean `use_opus`.
- Uses `get_tool_model(use_opus=use_opus)` when tools are present and `get_model(use_opus=use_opus)` otherwise.
- Logs model label as `Opus` or `Sonnet`.
- Complexity routing is not provider-neutral in active path.

## Active Agent Mode graph

Active Agent Mode graph: `app/agent_mode/agent.py`.

Findings before edits:

- `agent_mode_node()` still calls `get_model(model_arn=model_arn)`.
- Active `get_model()` accepts that parameter, but this remains a Bedrock-shaped interface and should not be part of the Ollama control path.
- Prompt builder is `app/agent_mode/adaptive_system_prompt.py`.

## Active MCOP child-conversation path

Active child runner: `app/agent_mode/child_conversation.py`.

Active MCOP tools: `app/agent_mode/mcop_tools.py`.

Findings before edits:

- `run_child_conversation()` accepts `max_iters`, but `_child_agent_node()` and `_child_route()` still use global `MCOP_CHILD_MAX_ITERS`, so per-child max iteration overrides are not honored.
- Child output is prose summary plus artifact list, not a structured evidence packet.
- Artifact discovery scans the entire parent workspace by mtime rather than a child task directory.
- Child tools exclude MCOP spawn/check/read result tools, which prevents recursive spawning in the usual path.
- No parent `read_packet` path was found.

## Tool registry structure

Active registry: `app/agent/agents/tools/registry.py`.

Findings before edits:

- Local tool factories live in `_TOOL_FACTORIES`.
- Async/MCP factories live in `_ASYNC_TOOL_FACTORIES` and load into `_ASYNC_TOOL_CACHE`.
- `get_tools_set(tool_type)` is the active lookup API.
- MCP tool-name sanitation applies Bedrock 64-character truncation regardless of active provider.
- No live tool inventory/selftest endpoint, duplicate report function, prompt/tool drift detector, or methodology output mapping was found.

## Prompt construction paths

Active paths:

- `app/agent/agents/prompts/chat_system_prompt.txt`, loaded by `app/agent/agents/utils.py:get_prompt()` and used by `app/agent/agents/agentic_rag.py`.
- `app/agent_mode/adaptive_system_prompt.py`, used by Agent Mode.
- `app/agent_mode/child_conversation.py:_build_child_system_prompt()`, used by MCOP children.
- `app/agent/agents/prompts/title_prompt.txt`, used by title generation.

Findings before edits:

- Active chat system prompt still identifies the assistant as a Claude/Sonnet/Anthropic-based assistant.
- Active prompt contains a legacy model-info block.
- Agent Mode prompt is mostly provider-neutral, but lacks the complete evidence/tool discipline contract.
- MCOP child prompt does not require structured evidence packets.

## Context / compression paths

Active paths:

- `app/agent/agents/agentic_rag.py` for message trimming and tool-message compression.
- `app/message/compression.py`, `app/message/message_tiering.py`, `app/message/tool_message_compressor.py`, and `app/message/token_efficiency_adapter.py`.
- `app/tools/context_budget.py`.

Findings before edits:

- Configured Ollama context is 32k in active config; protocol asks for local Ollama default target of at least 64k if hardware supports it.
- Prompt comments and compression code still contain legacy 200k managed-context assumptions.
- Context budget is not expressed per explicit model role.

## Health / selftest paths

Active health router: `app/health/router.py` provides only `/health`.

No `/rest/api/v1/health/ollama-agent-selftest` endpoint was found before edits.

## Existing tests

Existing root tests include token retry, import tests, visualization tests, kubectl security, and content-aware compression tests. Existing `tests/` contains content compression and related tests, but no Ollama model role, prompt identity, methodology selector, structured packet, verifier, or selftest endpoint tests.

## Exact gaps between active code and Ollama sidecar files

Sidecar files still exist at repo root and under `app/`: `chat_models.ollama_patched.py`, `config.ollama_patched.py`, `app/core/llm/chat_models.ollama_patched.py`, and `app/config.ollama_patched.py`.

The active running bundle has already merged some sidecar behavior:

- Active `app/config.py` accepts `ollama` and has Ollama model/base config.
- Active `app/core/llm/chat_models.py` supports `ChatOllama`.
- Active `requirements.txt` includes `langchain-ollama`.

Remaining gaps relative to protocol:

- Sidecar and active code still use primary/efficient/opus rather than explicit provider-neutral roles.
- Active Agent Mode still uses `model_arn` call shape.
- Active complexity routing still returns `use_opus` boolean.
- Active prompt still contains legacy identity text.
- Active MCOP lacks structured packet output and child max-iteration enforcement.
- Active tool registry lacks provider-specific sanitation and live inventory/report helpers.

## Static grep performed

Terms scanned across active `app`, `requirements.txt`, `restart-dishchat.sh`, and `tests`: `ollama`, `bedrock`, `sonnet`, `opus`, `anthropic`, `model_arn`, `get_model`, `use_opus`, `Claude`, `Sonnet`, `Opus`, `Anthropic`, `Bedrock`.

The raw grep output is stored in `reports/static_grep_initial_diship.txt`. This report intentionally does not reproduce internal URLs or bearer-like values.

## Edit gate

This report was written before code edits against the actual running DishIP bundle. Next patches should be small, preserve the already-working Ollama/tool-model path, avoid broad rewrite, avoid removing MCP tools, and add only the missing protocol components.
