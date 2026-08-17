import logging
import json
import re
from typing import Annotated, Any
from typing_extensions import TypedDict
from functools import cache

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage
from app.message.compression import count_message_tokens
from app.message.tool_message_compressor import compress_stb_log_tool_output, compress_tool_message_content
from app.message.message_tiering import apply_tiered_compression
from app.message.token_efficiency_adapter import apply_token_efficiency_layer
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from app.core.llm import get_model, get_tool_model
from app.core.llm.model_roles import resolve_model_role
from app.config import get_settings
from app.core.utils import get_datestr_now
from app.agent.complexity_detector import detect_prompt_complexity, choose_model_role_for_context, context_tracker

from ..db_utils import get_checkpointer
from ..utils import aggressive_cachept, cleanup_cachept, set_model_config
from .utils import get_prompt
from .tools import get_tools_set
from app.agent.tool_execution_policy import (
    append_tool_execution_failure,
    binding_audit,
    build_compact_tool_execution_system_prompt,
    build_tool_execution_failed_message,
    get_all_executor_tools,
    get_scoped_tools_for_prompt,
    rank_tools_for_retry,
    should_force_tool_retry,
)
from app.agent.audited_tool_node import make_audited_tool_node
from app.agent.no_progress_controller import current_parent_required_tools, evaluate_no_progress
from app.agent.operational_workflows import detect_operational_workflow
from app.agent.completion_contract import (
    completion_response_text,
    extract_completion_contract,
    render_completion_contract_system_prompt,
    render_incomplete_contract,
)
from app.agent.tool_execution_audit import record_model_facing_binding
from app.agent.tool_policy_transition import build_tool_policy_update
from app.agent.tool_policy_runtime import (
    build_policy_runtime_snapshot,
    publish_policy_runtime_snapshot,
)
from app.agent.tool_policy_state import (
    FIELD_BOUNDS,
    MAX_LAST_BOUND_TOOL_NAMES,
    TOOL_POLICY_VERSION,
    classify_activation_state,
    compute_policy_signature,
    current_registry_generation,
    load_tool_policy_state,
    make_ordered_set_reducer,
    make_replacement_reducer,
    merge_authorization_state,
    normalize_name_list,
    parse_authorization_delta,
    reduce_activation_status,
    reduce_authorization_flags,
    reduce_bool,
    reduce_revision,
    reduce_scalar,
    reduce_version,
)
from app.agent.tool_activation_intent import (
    RegistryToolIndex,
    ToolActivationIntent,
    activation_intent_from_checkpoint,
    activation_intent_from_management_messages,
    activation_intent_from_prompt,
    merge_activation_intents,
)
from app.agent_mode.thought_interceptor import interceptor

logger = logging.getLogger(__name__)


def _log_ollama_response_metadata(response, role: str = "unknown"):
    """Log Ollama response metadata."""
    try:
        meta = getattr(response, "response_metadata", {}) or {}
        load_ns = meta.get("load_duration", 0)
        total_ns = meta.get("total_duration", 0)
        model = meta.get("model", "unknown")
        done_reason = meta.get("done_reason", "unknown")
        load_ms = load_ns / 1_000_000 if load_ns else 0
        total_ms = total_ns / 1_000_000 if total_ns else 0
        if load_ms > 0 or total_ms > 0:
            logger.info(f"Ollama response metadata role={role} model={model} load_ms={load_ms:.0f} total_ms={total_ms:.0f} done_reason={done_reason}")
    except Exception:
        pass

settings = get_settings()

system_prompt = SystemMessage(
    content=get_prompt("chat_system").format(today=get_datestr_now())
)

# Maximum number of messages to keep in context
MAX_MESSAGES = 100

# Tool output compression settings.
# Raw log retrieval results can be enormous. The model should see a structured
# summary, not 100k-token raw log blobs.
TOOL_MESSAGE_COMPRESS_TOKEN_THRESHOLD = 8000
TOOL_COMPRESSED_MAX_CHARS = 12000
TOOL_FIRST_LINES = 40
TIERED_COMPRESSION_TARGET_TOKENS = 100_000


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model_config: dict[str, Any]
    # ---- Phase D1: checkpointed, thread-scoped tool-policy state ----
    # Every field is a bounded, JSON-serializable primitive.  No tool
    # objects, callables, connections, request objects, or tokens.
    # Sticky accumulating sets:
    active_toolsets: Annotated[list[str], make_replacement_reducer(FIELD_BOUNDS["active_toolsets"])]
    requested_toolsets: Annotated[list[str], make_ordered_set_reducer(FIELD_BOUNDS["requested_toolsets"])]
    requested_extra_tools: Annotated[list[str], make_ordered_set_reducer(FIELD_BOUNDS["requested_extra_tools"])]
    processed_activation_tool_call_ids: Annotated[
        list[str],
        make_ordered_set_reducer(FIELD_BOUNDS["processed_activation_tool_call_ids"]),
    ]
    # Derived sets: recomputed each turn, so they replace rather than grow.
    eligible_extra_tools: Annotated[list[str], make_replacement_reducer(FIELD_BOUNDS["eligible_extra_tools"])]
    pending_authorization_extra_tools: Annotated[list[str], make_replacement_reducer(FIELD_BOUNDS["pending_authorization_extra_tools"])]
    pending_registry_requests: Annotated[list[str], make_replacement_reducer(FIELD_BOUNDS["pending_registry_requests"])]
    unavailable_extra_tools: Annotated[list[str], make_replacement_reducer(FIELD_BOUNDS["unavailable_extra_tools"])]
    last_bound_tool_names: Annotated[list[str], make_replacement_reducer(MAX_LAST_BOUND_TOOL_NAMES)]
    authorization_flags: Annotated[dict[str, bool], reduce_authorization_flags]
    mcop_children_forbidden: Annotated[bool, reduce_bool]
    last_activation_status: Annotated[dict[str, str], reduce_activation_status]
    tool_profile_signature: Annotated[str, reduce_scalar]
    tool_registry_generation: Annotated[str, reduce_scalar]
    activation_request_revision: Annotated[int, reduce_revision]
    continuity_task_scope: Annotated[str, reduce_scalar]
    continuity_methodology: Annotated[str, reduce_scalar]
    continuity_environment: Annotated[str, reduce_scalar]
    continuity_revision: Annotated[int, reduce_revision]
    tool_policy_version: Annotated[int, reduce_version]


def _build_tool_policy_update(**kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper around the pure checkpoint transition."""
    # Older focused tests supplied ``prompt`` to the in-module implementation;
    # the pure transition correctly depends only on the normalized plan.
    kwargs.pop("prompt", None)
    if "activation_request_revision" not in kwargs:
        kwargs["activation_request_revision"] = int(
            getattr(kwargs.get("plan"), "activation_request_revision", 0) or 0
        )
    return build_tool_policy_update(**kwargs)


def _unprocessed_management_activation_messages(
    messages: list[BaseMessage],
    processed_tool_call_ids: list[str] | tuple[str, ...],
    *,
    revision: int = 0,
    registry_index: RegistryToolIndex | None = None,
) -> tuple[list[BaseMessage], list[str]]:
    """Return valid, not-yet-consumed management ToolMessages and new IDs."""
    processed = {str(value) for value in processed_tool_call_ids or () if str(value)}
    fresh_messages: list[BaseMessage] = []
    fresh_ids: list[str] = []
    for message in list(messages or ())[-64:]:
        if not isinstance(message, ToolMessage):
            continue
        call_id = str(getattr(message, "tool_call_id", "") or "")
        if call_id and call_id in processed:
            continue
        parsed = activation_intent_from_management_messages(
            [message],
            registry_index=registry_index,
            request_revision=revision,
        )
        if parsed.is_empty:
            continue
        fresh_messages.append(message)
        if call_id:
            fresh_ids.append(call_id)
    return fresh_messages, normalize_name_list(
        [*processed_tool_call_ids, *fresh_ids],
        FIELD_BOUNDS["processed_activation_tool_call_ids"],
    )


def _management_activation_intent(
    messages: list[BaseMessage],
    processed_tool_call_ids: list[str] | tuple[str, ...],
    *,
    revision: int = 0,
    registry_index: RegistryToolIndex | None = None,
) -> tuple[ToolActivationIntent, list[str]]:
    """Consume executed management activation results exactly once."""
    fresh_messages, processed = _unprocessed_management_activation_messages(
        messages,
        processed_tool_call_ids,
        revision=revision,
        registry_index=registry_index,
    )
    intent = activation_intent_from_management_messages(
        fresh_messages,
        registry_index=registry_index,
        request_revision=revision,
    )
    return intent, processed


def _log_tool_policy_state(update: dict[str, Any], *, methodology: str = "") -> None:
    """Emit one bounded, redacted line describing this turn's policy state.

    Authorization is recorded as booleans only.  No token, prompt, argument
    value, or tool result is logged.  This makes the D1 security state
    observable to operators and correlatable with the Phase C audit.
    """
    try:
        logger.info(
            "tool_policy_state %s",
            json.dumps(
                {
                    "methodology": methodology,
                    "active_toolsets": update.get("active_toolsets", [])[:24],
                    "requested_toolsets": update.get("requested_toolsets", [])[:24],
                    "requested_extra_tools": update.get("requested_extra_tools", [])[:24],
                    "eligible_extra_tools": update.get("eligible_extra_tools", [])[:24],
                    "pending_authorization_extra_tools": update.get(
                        "pending_authorization_extra_tools", []
                    )[:24],
                    "unavailable_extra_tools": update.get("unavailable_extra_tools", [])[:24],
                    "authorization_flags": update.get("authorization_flags", {}),
                    "mcop_children_forbidden": bool(
                        update.get("mcop_children_forbidden", False)
                    ),
                    "last_bound_tool_count": len(update.get("last_bound_tool_names", [])),
                    "last_activation_status": update.get("last_activation_status", {}),
                    "tool_profile_signature": update.get("tool_profile_signature", ""),
                    "tool_registry_generation": update.get("tool_registry_generation", ""),
                    "activation_request_revision": update.get("activation_request_revision", 0),
                    "continuity_task_scope": update.get("continuity_task_scope", ""),
                    "continuity_methodology": update.get("continuity_methodology", ""),
                    "continuity_environment": update.get("continuity_environment", ""),
                    "continuity_revision": update.get("continuity_revision", 0),
                    "tool_policy_version": update.get("tool_policy_version"),
                    "authorization_material_logged": False,
                },
                sort_keys=True,
                default=str,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - observability is never fatal
        logger.warning("tool_policy_state log failed: %s", type(exc).__name__)


def _current_environment_key(config: Any) -> str:
    configurable = {}
    if isinstance(config, dict):
        configurable = config.get("configurable", {}) or {}
    elif config is not None:
        configurable = getattr(config, "configurable", {}) or {}
    if not isinstance(configurable, dict):
        return ""
    explicit = str(configurable.get("authoritative_environment") or "")
    if explicit:
        return explicit[:128]
    identity = configurable.get("operational_identity") or {}
    if isinstance(identity, dict):
        target = str(identity.get("user_pinned_target") or "")
        hostname = str(identity.get("runtime_hostname") or identity.get("agent_execution_host") or "")
        if target or hostname:
            return f"{target}|{hostname}"[:128]
    return ""


def _record_parent_binding(
    bound_tool_names: list[str],
    *,
    authorization_flags: dict[str, bool],
    thread_id: Any,
    mcop_children_forbidden: bool = False,
) -> None:
    """Record the exact model-facing binding for the execution gate."""
    try:
        record_model_facing_binding(
            bound_tool_names=bound_tool_names,
            authorization_flags=authorization_flags,
            thread_id=thread_id,
            execution_constraints={
                "mcop_children_forbidden": bool(mcop_children_forbidden)
            },
        )
    except Exception as exc:  # noqa: BLE001 - audit correlation is never fatal
        logger.warning(
            'tool_execution_audit binding record failed: %s', type(exc).__name__
        )


def _authorization_flags_for_window(messages: list[BaseMessage]) -> dict[str, bool]:
    """Derive server-side authorization booleans from operator text only.

    Grants are parsed by ``app.agent.tool_authorization``, which records only
    booleans and never reads, stores, hashes, or echoes token material.  Flags
    accumulate across the retained human turns so an earlier grant remains
    effective for the conversation window.  Durable thread-sticky authorization
    state belongs to the checkpoint layer and is out of scope for this layer.
    """
    from app.agent.tool_authorization import authorization_state_for_turn

    flags: dict[str, Any] = {}
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        text = message.content if isinstance(message.content, str) else _tool_content_to_text(message.content)
        flags = authorization_state_for_turn(flags, text)["authorization_flags"]
    if not flags:
        flags = authorization_state_for_turn(None, "")["authorization_flags"]
    return {key: bool(value) for key, value in flags.items()}


_PATH_RE = re.compile(r"(s3://[^\s'\"<>]+|https?://[^\s'\"<>]+|/[A-Za-z0-9._/\-]+)")
_RECEIVER_RE = re.compile(r"\bR\d{6,}\b", re.IGNORECASE)


def _tool_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content or "")


def _maybe_json_loads(text: str):
    stripped = text.strip()
    if not stripped:
        return None
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _walk_dict_records(obj, out: list[dict], limit: int = 40) -> None:
    if len(out) >= limit:
        return

    interesting_keys = {
        "receiver_id", "receiver", "rx_id", "stb", "smartcard_id",
        "log_name", "log", "name", "filename", "file_name", "type",
        "request_status", "status", "state", "result", "success",
        "s3_path", "artifact_path", "path", "url", "location",
        "byte_size", "bytes", "size", "size_bytes", "content_length",
        "error", "errors", "exception", "stderr", "message",
    }

    if isinstance(obj, dict):
        lowered = {str(k).lower() for k in obj.keys()}
        if lowered & interesting_keys:
            out.append(obj)

        for v in obj.values():
            _walk_dict_records(v, out, limit)
            if len(out) >= limit:
                return

    elif isinstance(obj, list):
        for item in obj[: limit * 3]:
            _walk_dict_records(item, out, limit)
            if len(out) >= limit:
                return


def _pick(record: dict, *names: str):
    lowered = {str(k).lower(): v for k, v in record.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, "", [], {}):
            return value
    return None


def _strings_from_obj(obj, out: list[str], limit: int = 200) -> None:
    if len(out) >= limit:
        return

    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings_from_obj(v, out, limit)
            if len(out) >= limit:
                return
    elif isinstance(obj, list):
        for item in obj:
            _strings_from_obj(item, out, limit)
            if len(out) >= limit:
                return
    elif obj is not None:
        out.append(str(obj))


def _extract_paths_from_text(text: str, limit: int = 10) -> list[str]:
    seen = []
    for match in _PATH_RE.findall(text):
        clean = match.rstrip(".,);]")
        if clean not in seen:
            seen.append(clean)
        if len(seen) >= limit:
            break
    return seen


def _extract_error_lines(text: str, limit: int = 8) -> list[str]:
    error_terms = ("error", "exception", "traceback", "failed", "failure", "denied", "timeout")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(term in stripped.lower() for term in error_terms):
            lines.append(stripped[:500])
        if len(lines) >= limit:
            break
    return lines


def compress_log_tool_output(
    content: Any,
    tool_call_id: str | None = None,
    max_chars: int = TOOL_COMPRESSED_MAX_CHARS,
    first_lines: int = TOOL_FIRST_LINES,
    tool_name: str = "",
) -> str:
    """
    Compress STB-log-shaped tool output into a stable, model-readable summary.

    This compatibility wrapper intentionally falls back to plain head/tail
    truncation when the payload is not actually log-shaped.  That prevents JSON
    metadata tools from being flattened into fake ``receiver_id: unknown`` log
    records when callers use the old helper directly.
    """

    return compress_stb_log_tool_output(
        content=content,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        max_chars=max_chars,
        first_lines=first_lines,
    )


def sanitize_tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Remove orphaned tool calls and tool results from the message list.

    An orphaned tool call is an AIMessage with tool_calls where one or more
    corresponding ToolMessages are missing. An orphaned tool result is a
    ToolMessage whose parent AIMessage was dropped.

    This must be called both before and after any truncation to ensure
    the message list is always valid for the Bedrock Converse API.
    """
    # Strip any leading ToolMessages (can appear after truncation)
    while messages and isinstance(messages[0], ToolMessage):
        messages = messages[1:]

    # Pass 1: Collect all tool result IDs present in the message list
    tool_results_present: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, 'tool_call_id', None)
            if tool_call_id:
                tool_results_present.add(tool_call_id)

    # Pass 2: Walk messages, keeping only complete tool call/result pairs
    kept_tool_call_ids: set[str] = set()
    emitted_tool_result_ids: set[str] = set()
    cleaned: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            # Only keep this AIMessage if ALL of its tool calls have results
            if all(tc.get('id') in tool_results_present for tc in msg.tool_calls):
                for tc in msg.tool_calls:
                    kept_tool_call_ids.add(tc.get('id'))
                cleaned.append(msg)
            else:
                logger.warning(
                    f"Dropping AIMessage with incomplete tool calls: "
                    f"{[tc.get('id') for tc in msg.tool_calls]}"
                )
        elif isinstance(msg, ToolMessage):
            # Only keep if its parent AIMessage was kept
            tool_call_id = getattr(msg, 'tool_call_id', None)
            if tool_call_id in emitted_tool_result_ids:
                # D3B3: two results for one tool-call id are not provider-valid.
                # Keep the first result and drop the duplicate; never fabricate
                # or merge tool output.
                logger.warning(
                    f"Dropping duplicate ToolMessage: tool_call_id={tool_call_id}"
                )
            elif tool_call_id in kept_tool_call_ids:
                emitted_tool_result_ids.add(tool_call_id)
                cleaned.append(msg)
            else:
                logger.warning(
                    f"Dropping orphaned ToolMessage: tool_call_id={tool_call_id}"
                )
        else:
            cleaned.append(msg)

    return cleaned


def ensure_bedrock_converse_message_shape(messages: list[BaseMessage]) -> list[BaseMessage]:
    '''Ensure a Bedrock-compatible user-first prefix without deleting valid history.

    Upstream trimming can leave leading AI/tool fragments.  Remove only that
    invalid prefix: preserve from the *earliest surviving* HumanMessage rather
    than jumping to the latest HumanMessage and silently deleting prior task
    context.
    '''
    messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    if not messages:
        logger.warning(
            "Bedrock message repair: message list was empty; injecting recovery HumanMessage."
        )
        return [
            HumanMessage(
                content=(
                    "Continue from the available conversation context. "
                    "If tool results were just returned, summarize the useful result. "
                    "If required context is missing, ask a concise clarification."
                )
            )
        ]

    if isinstance(messages[0], HumanMessage):
        return messages

    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            repaired = sanitize_tool_messages(messages[idx:])
            if repaired and isinstance(repaired[0], HumanMessage):
                logger.warning(
                    "Bedrock message repair: removed invalid prefix through earliest "
                    "HumanMessage at index %s; preserved subsequent user history.",
                    idx,
                )
                return repaired

    logger.warning(
        "Bedrock message repair: first message was %s and no HumanMessage remained; "
        "prepending recovery HumanMessage.",
        type(messages[0]).__name__,
    )
    return [
        HumanMessage(
            content="Continue the conversation using the available tool results and context."
        ),
        *messages,
    ]



def _copy_tool_message_with_content(message: ToolMessage, content: str) -> ToolMessage:
    """Clone a ToolMessage while preserving tool_call_id/name/metadata."""
    try:
        return message.model_copy(update={"content": content})
    except Exception:
        pass
    try:
        return message.copy(update={"content": content})
    except Exception:
        pass

    kwargs = {"content": content, "tool_call_id": getattr(message, "tool_call_id", None)}
    for attr in ("name", "id", "additional_kwargs", "response_metadata", "status", "artifact"):
        value = getattr(message, attr, None)
        if value not in (None, [], {}):
            kwargs[attr] = value
    try:
        return ToolMessage(**kwargs)
    except TypeError:
        kwargs.pop("artifact", None)
        kwargs.pop("status", None)
        return ToolMessage(**kwargs)



def _sanitize_messages_for_ollama(messages: list) -> list:
    """Flatten list-typed message content to plain strings for Ollama.

    langchain_ollama only accepts content that is a str or a list of dicts
    whose 'type' is 'text' or 'image_url'.  AIMessages from tool-calling
    passes can carry content blocks of type 'tool_use', and Bedrock-style
    ToolMessages may carry 'tool_result' wrapper dicts.  This helper converts
    all such non-standard content blocks to plain strings before they reach
    the Ollama converter, preventing ValueError.
    """
    import json as _json
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

    def _blocks_to_text(content) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype in ("tool_use", "tool_result"):
                    # tool_use is already captured in msg.tool_calls;
                    # skip to avoid Ollama ValueError.
                    pass
                else:
                    try:
                        parts.append(_json.dumps(block, ensure_ascii=False, default=str))
                    except Exception:
                        pass
        return "\n".join(p for p in parts if p)

    sanitized = []
    for msg in messages:
        if isinstance(msg.content, list):
            flat = _blocks_to_text(msg.content)
            if isinstance(msg, AIMessage):
                new_msg = AIMessage(
                    content=flat,
                    tool_calls=getattr(msg, "tool_calls", []) or [],
                    id=getattr(msg, "id", None),
                )
            elif isinstance(msg, ToolMessage):
                new_msg = _copy_tool_message_with_content(msg, flat)
            elif isinstance(msg, HumanMessage):
                new_msg = HumanMessage(content=flat, id=getattr(msg, "id", None))
            elif isinstance(msg, SystemMessage):
                new_msg = SystemMessage(content=flat)
            else:
                new_msg = msg.__class__(content=flat)
            sanitized.append(new_msg)
        else:
            sanitized.append(msg)
    return sanitized


def truncate_large_messages(messages: list[BaseMessage], max_tokens_per_message: int = 50000) -> list[BaseMessage]:
    """
    Truncate individual messages that exceed a token limit.
    This prevents a single huge ToolMessage from causing API rejection.

    ToolMessages additionally degrade by age: full content on the turn created,
    key facts for recent old turns, then one-line provenance.  This now applies
    to browse, SSH/shell, file, search, and generic tools.
    """
    from app.tools.progressive_tool_memory import generate_progressive_levels

    current_turn = sum(1 for message in messages if isinstance(message, HumanMessage))
    turn_cursor = 0
    tool_call_names: dict[str, str] = {}
    tool_call_turns: dict[str, int] = {}
    truncated = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            turn_cursor += 1

        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            for tool_call in msg.tool_calls:
                tool_call_id = tool_call.get('id')
                tool_name = tool_call.get('name')
                if tool_call_id:
                    if tool_name:
                        tool_call_names[tool_call_id] = tool_name
                    tool_call_turns[tool_call_id] = turn_cursor

        if isinstance(msg.content, str):
            msg_tokens = count_message_tokens([msg])

            if isinstance(msg, ToolMessage):
                tool_call_id = getattr(msg, "tool_call_id", None)
                tool_name = getattr(msg, "name", None) or tool_call_names.get(tool_call_id, "")
                turn_created = tool_call_turns.get(tool_call_id, turn_cursor)

                if tool_name and turn_created < current_turn:
                    progressive = generate_progressive_levels(
                        tool_name=tool_name,
                        raw_result=msg.content,
                        tool_call_id=tool_call_id or "",
                        turn_created=turn_created,
                    )
                    progressive_content = progressive.get_content_for_turn(current_turn)
                    new_msg = _copy_tool_message_with_content(msg, progressive_content)
                    logger.debug(
                        "Progressive tool memory: %s turn=%s current=%s %s -> %s chars",
                        tool_name,
                        turn_created,
                        current_turn,
                        len(msg.content),
                        len(progressive_content),
                    )
                    truncated.append(new_msg)
                    continue

                if msg_tokens > TOOL_MESSAGE_COMPRESS_TOKEN_THRESHOLD:
                    compression = compress_tool_message_content(
                        tool_name=tool_name or "unknown",
                        content=msg.content,
                        tool_call_id=tool_call_id,
                        max_chars=TOOL_COMPRESSED_MAX_CHARS,
                        first_lines=TOOL_FIRST_LINES,
                    )
                    new_msg = _copy_tool_message_with_content(msg, compression.content)
                    logger.warning(
                        "Compressed large ToolMessage via %s/%s: %s tokens, %s chars -> %s chars",
                        compression.content_type.value,
                        compression.strategy,
                        msg_tokens,
                        len(msg.content),
                        len(compression.content),
                    )
                    # Emit compression telemetry to viz server for Token Savings tracking
                    try:
                        interceptor.metric_update("tool_result_compression", {
                            "tool": tool_name or "unknown",
                            "original_tokens": msg_tokens,
                            "original_chars": len(msg.content),
                            "compressed_chars": len(compression.content),
                            "savings_chars": len(msg.content) - len(compression.content),
                            "savings_tokens_estimate": msg_tokens - (len(compression.content) // 4),
                            "strategy": compression.strategy,
                            "content_type": compression.content_type.value,
                        })
                    except Exception:
                        pass  # Never let telemetry break the agent loop
                    truncated.append(new_msg)
                    continue

            if msg_tokens > max_tokens_per_message:
                # Calculate how much content to keep (rough estimate)
                # 1 token ≈ 4 characters for English text
                max_chars = max_tokens_per_message * 4
                truncated_content = msg.content[:max_chars] + f"\n\n[... truncated {len(msg.content) - max_chars} characters ({msg_tokens - max_tokens_per_message} tokens) ...]"
                
                # Create a new message with truncated content
                if isinstance(msg, HumanMessage):
                    new_msg = HumanMessage(content=truncated_content)
                elif isinstance(msg, AIMessage):
                    new_msg = AIMessage(
                        content=truncated_content,
                        tool_calls=getattr(msg, 'tool_calls', None)
                    )
                elif isinstance(msg, ToolMessage):
                    new_msg = _copy_tool_message_with_content(msg, truncated_content)
                else:
                    new_msg = msg  # Unknown message type, keep as-is
                
                logger.warning(
                    f"Truncated large {type(msg).__name__}: {msg_tokens} -> ~{max_tokens_per_message} tokens"
                )
                truncated.append(new_msg)
            else:
                truncated.append(msg)
        else:
            # Complex content (lists, etc.) - keep as-is for now
            truncated.append(msg)
    
    return truncated


def _active_history_budget_tokens() -> int:
    if settings.PLLM_PROVIDER == "ollama" or settings.ELLM_PROVIDER == "ollama":
        configured_context = int(getattr(settings, "OLLAMA_CTX_LEN", settings.PLLM_CTX_LEN))
        reserved = int(getattr(settings, "OLLAMA_MAX_OUTPUT_TOKENS", settings.MAX_OUTPUT_COUNT)) + 4096
        return max(4096, configured_context - reserved)
    return 140000


def _active_provider_requires_bedrock_shape() -> bool:
    return settings.PLLM_PROVIDER == "aws-bedrock" or settings.ELLM_PROVIDER == "aws-bedrock"


def truncate_messages(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    '''Bound history while treating the latest HumanMessage as a hard invariant.'''
    messages = list(messages)

    latest_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_human = msg
            break

    if len(messages) > max_messages:
        logger.info(f"Truncating message history from {len(messages)} to {max_messages}")
        messages = messages[-max_messages:]
        if latest_human is not None and not any(
            msg is latest_human for msg in messages
        ):
            messages = [latest_human, *messages[-max(0, max_messages - 1):]]

    try:
        messages = apply_tiered_compression(
            messages,
            target_tokens=TIERED_COMPRESSION_TARGET_TOKENS,
        )
    except Exception as e:
        logger.warning(
            f"Tiered conversation compression failed; continuing without it: {e}",
            exc_info=True,
        )

    try:
        messages = apply_token_efficiency_layer(messages)
    except Exception as e:
        logger.warning(
            f"Token efficiency layer failed; continuing without it: {e}",
            exc_info=True,
        )

    messages = truncate_large_messages(messages, max_tokens_per_message=50000)

    try:
        from langchain_core.messages import trim_messages

        messages = trim_messages(
            messages,
            max_tokens=_active_history_budget_tokens(),
            strategy="last",
            token_counter=count_message_tokens,
        )
        logger.info(
            f"Token-based truncation applied, final message count: {len(messages)}"
        )
    except Exception as e:
        logger.warning(f"Token-based truncation failed: {e}")

    messages = sanitize_tool_messages(messages)

    if latest_human is not None:
        latest_id = getattr(latest_human, "id", None)
        latest_content = getattr(latest_human, "content", None)

        def _contains_latest_human(items: list[BaseMessage]) -> bool:
            for item in items:
                if item is latest_human:
                    return True
                if not isinstance(item, HumanMessage):
                    continue
                if latest_id is not None and getattr(item, "id", None) == latest_id:
                    return True
                if getattr(item, "content", None) == latest_content:
                    return True
            return False

        if not _contains_latest_human(messages):
            logger.error(
                "Token/context truncation removed the latest HumanMessage; "
                "restoring the active user turn instead of synthetic continuation."
            )
            # Keep any already-bounded recent suffix only when it can coexist
            # behind the restored current turn without invalid tool fragments.
            messages = sanitize_tool_messages([latest_human, *messages])
            if not messages or not isinstance(messages[0], HumanMessage):
                messages = [latest_human]

    if _active_provider_requires_bedrock_shape():
        return ensure_bedrock_converse_message_shape(messages)
    return messages



def _get_last_human_message(messages: list[BaseMessage]) -> str:
    """Extract the text content of the last human message for complexity detection."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                return " ".join(text_parts)
            elif isinstance(content, dict) and "text" in content:
                return content["text"]
            return str(content) if content else ""
    return ""


### Nodes
async def call_model(state: AgentState, config=None):
    """
    Call the model with tools enabled.
    The model can choose to use tools or respond directly.
    Includes intelligent Opus routing based on prompt complexity.
    """
    messages = list(state["messages"])

    # Step 1: Sanitize - remove orphaned tool calls and results
    messages = sanitize_tool_messages(messages)

    # Step 2: Truncate to max context window, then re-sanitize the boundary
    messages = truncate_messages(messages, MAX_MESSAGES)

    # Step 2b: Stash pre-invocation token estimate for calibration feedback.
    # The usage tracking layer will compare this against Bedrock's actual count
    # to adaptively tune the offline tokenizer's accuracy over time.
    try:
        from app.message.token_counter import stash_pre_invocation_estimate
        stash_pre_invocation_estimate(count_message_tokens(messages))
    except ImportError:
        pass

    # Step 3: Handle cachepoints
    cleanup_cachept(messages)
    messages = aggressive_cachept(messages, settings.MAX_CACHEPOINT_CNT)

    # Step 4: Final provider-specific message repair happens after role selection.

    # Step 5: Intelligent model-role selection with complexity detection
    # Extract session_id from config for context tracking
    session_id = None
    if config and hasattr(config, 'get'):
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        session_id = configurable.get("thread_id") or configurable.get("session_id")
    elif config and hasattr(config, 'configurable'):
        session_id = getattr(config.configurable, 'thread_id', None)
    
    # Get last human message for complexity analysis
    last_human_content = _get_last_human_message(messages)
    
    # Determine prompt complexity for routing and context budgeting.
    raw_complexity_score = detect_prompt_complexity(last_human_content)
    context_bonus = context_tracker.get_context_bonus(session_id, raw_complexity_score) if session_id else 0
    effective_score = raw_complexity_score + context_bonus

    # Determine provider-neutral role for this turn.
    selected_role = choose_model_role_for_context(last_human_content, session_id=session_id)
    role_config = resolve_model_role(selected_role)
    if role_config.provider == "aws-bedrock":
        messages = ensure_bedrock_converse_message_shape(messages)

    # ── Dual-model routing with scoped tool binding ───────────────────────────
    # Local Ollama models are much less reliable when handed the entire MCP
    # catalog.  Bind only the methodology-relevant tools to the tool-worker.
    # ToolMessages no longer force a tool-free synthesis pass for data
    # investigations: popup and QoS workflows need chained tool calls.
    _has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    _is_ollama = (settings.PLLM_PROVIDER == "ollama")
    # ── Phase D1: checkpointed tool policy for this turn ──────────────────
    # Order is deliberate: load stored state, then apply ONLY the current
    # user message's authorization delta.  Nothing is re-derived from the
    # retained or compressed message window, so compressing an old grant
    # cannot revoke authorization and an old quoted grant cannot re-grant it.
    _stored_policy = load_tool_policy_state(state)
    _auth_delta = parse_authorization_delta(last_human_content)
    _authorization_flags = merge_authorization_state(
        _stored_policy["authorization_flags"], _auth_delta
    )
    # Resolve one bounded registry snapshot, then merge three explicit sources:
    # checkpointed intent, the current user prompt, and executed management-tool
    # results. Assistant-authored tool-like JSON is not an accepted source.
    _registry_index = RegistryToolIndex.live()
    _checkpoint_activation = activation_intent_from_checkpoint(
        _stored_policy, registry_index=_registry_index
    )
    _prompt_activation = activation_intent_from_prompt(
        last_human_content,
        registry_index=_registry_index,
        request_revision=_stored_policy.get("activation_request_revision", 0),
    )
    _fresh_management_messages, _processed_activation_ids = (
        _unprocessed_management_activation_messages(
            messages,
            _stored_policy.get("processed_activation_tool_call_ids", []),
            revision=_stored_policy.get("activation_request_revision", 0),
            registry_index=_registry_index,
        )
    )
    _management_activation = activation_intent_from_management_messages(
        _fresh_management_messages,
        registry_index=_registry_index,
        request_revision=_stored_policy.get("activation_request_revision", 0),
    )
    _activation_intent = merge_activation_intents(
        _checkpoint_activation, _prompt_activation, _management_activation
    )

    tools, tool_plan = get_scoped_tools_for_prompt(
        last_human_content,
        has_prior_tool_results=_has_tool_results,
        prior_active_toolsets=_stored_policy["active_toolsets"],
        prior_extra_tools=_checkpoint_activation.requested_exact_tools,
        prior_requested_toolsets=_stored_policy["requested_toolsets"],
        authorization_flags=_authorization_flags,
        requested_extra_tools=_activation_intent.requested_exact_tools,
        requested_toolsets=_activation_intent.requested_toolsets,
        prior_activation_request_revision=_stored_policy.get(
            "activation_request_revision", 0
        ),
        activation_request_revision=_activation_intent.request_revision,
        registry_index=_registry_index,
        prior_continuity_methodology=_stored_policy.get("continuity_methodology", ""),
        prior_continuity_task_scope=_stored_policy.get("continuity_task_scope", ""),
        prior_continuity_environment=_stored_policy.get("continuity_environment", ""),
        prior_continuity_revision=_stored_policy.get("continuity_revision", 0),
        current_environment=_current_environment_key(config),
        prior_mcop_children_forbidden=bool(
            _stored_policy.get("mcop_children_forbidden", False)
        ),
    )
    _investigation_needs_tools = bool(tools) and tool_plan.data_investigation

    # Stop boundedly when the authoritative tool-result tail proves that a
    # required capability is unavailable, or when two consecutive attempts
    # make no measurable progress.  Health-only inventory responses do not
    # satisfy functional success for a requested exact tool.
    _current_operational = detect_operational_workflow(last_human_content)
    _required_tools = current_parent_required_tools(
        prompt_requested=_prompt_activation.requested_exact_tools,
        management_requested=_management_activation.requested_exact_tools,
        operational_required=(
            _current_operational.required_exact_tools
            if _current_operational is not None else ()
        ),
        first_tool=str(getattr(tool_plan, "first_tool", "") or ""),
    )
    _no_progress = evaluate_no_progress(
        messages,
        required_tools=_required_tools,
        exact_activation_path_available=False,
        activation_attempted=bool(_management_activation.requested_exact_tools),
    )
    if _no_progress.stop:
        logger.warning(
            "NO_PROGRESS_TERMINAL result_code=%s attempts=%d missing_tools=%s current_turn=%s",
            _no_progress.result_code,
            _no_progress.no_progress_attempts,
            list(_no_progress.missing_tools),
            bool(last_human_content),
        )
        _policy_update = _build_tool_policy_update(
            stored=_stored_policy,
            plan=tool_plan,
            authorization_flags=_authorization_flags,
            bound_tool_names=[],
            activation_request_revision=_activation_intent.request_revision,
            processed_activation_tool_call_ids=_processed_activation_ids,
        )
        publish_policy_runtime_snapshot(
            build_policy_runtime_snapshot(
                _policy_update, methodology=getattr(tool_plan, "methodology", "")
            )
        )
        _log_tool_policy_state(
            _policy_update, methodology=getattr(tool_plan, "methodology", "")
        )
        _record_parent_binding(
            [], authorization_flags=_authorization_flags, thread_id=session_id,
            mcop_children_forbidden=bool(tool_plan.mcop_children_forbidden),
        )
        return {
            "messages": [AIMessage(content=_no_progress.render_terminal_message())],
            **_policy_update,
        }

    if _is_ollama and _has_tool_results and not _investigation_needs_tools:
        # Synthesis phase for non-investigation flows.
        model = get_model(role=selected_role)
        _bind_tools = False
        logger.info("Dual-model: synthesis pass -> role=%s (no tools bound)", selected_role)
    elif _is_ollama and tools:
        # Orchestration phase: use tool-capable model with scoped tools.
        model = get_tool_model(role="tool_worker")
        _bind_tools = True
        logger.info(
            "Dual-model: orchestration pass -> role=tool_worker methodology=%s scoped_toolsets=%s bound_tools=%d",
            tool_plan.methodology,
            list(tool_plan.candidate_toolsets),
            len(tools),
        )
    elif _is_ollama:
        model = get_model(role=selected_role)
        _bind_tools = False
        logger.warning(
            "No scoped tools available for methodology=%s selected_toolsets=%s; model must report tool unavailability.",
            tool_plan.methodology,
            list(tool_plan.candidate_toolsets),
        )
    else:
        # Provider-native tool models still receive the scoped binding.
        model = get_model(role=selected_role)
        _bind_tools = bool(tools)
    set_model_config(model, state["model_config"])

    try:
        from app.tools.context_budget import get_session_budget, set_current_budget
        budget = get_session_budget(session_id, messages, complexity_score=effective_score)
        set_current_budget(budget)
        logger.info(f"Context budget: {budget.report()}")
    except (ImportError, ModuleNotFoundError):
        logger.debug("context_budget module not available, skipping budget tracking")

    model_with_tools = model.bind_tools(tools) if _bind_tools else model

    # Record the exact model-facing binding for authoritative tool-execution
    # audit correlation.  This is server-owned state and is never exposed as a
    # model-editable tool argument.
    _bound_tool_names = [
        _name
        for _name in (getattr(_t, 'name', '') for _t in (tools if _bind_tools else []))
        if _name
    ]
    _policy_update = _build_tool_policy_update(
        stored=_stored_policy,
        plan=tool_plan,
        authorization_flags=_authorization_flags,
        bound_tool_names=_bound_tool_names,
        activation_request_revision=_activation_intent.request_revision,
        processed_activation_tool_call_ids=_processed_activation_ids,
    )
    # Publish for the read-only management facades, which execute in the
    # tools node and must report actual state rather than a recommendation.
    publish_policy_runtime_snapshot(
        build_policy_runtime_snapshot(
            _policy_update, methodology=getattr(tool_plan, 'methodology', '')
        )
    )
    _log_tool_policy_state(
        _policy_update, methodology=getattr(tool_plan, 'methodology', '')
    )
    _record_parent_binding(
        _bound_tool_names,
        authorization_flags=_authorization_flags,
        thread_id=session_id,
        mcop_children_forbidden=bool(tool_plan.mcop_children_forbidden),
    )

    # Step 6: Log message state (consolidated)
    active_role_config = resolve_model_role("tool_worker" if (_is_ollama and _bind_tools) else selected_role)
    msg_types = {}
    for msg in messages:
        t = type(msg).__name__
        msg_types[t] = msg_types.get(t, 0) + 1
    type_summary = " ".join(f"{k}={v}" for k, v in msg_types.items())
    logger.info(
        "Calling model role=%s provider=%s model=%s ctx=%s output=%s | %s msgs [%s]",
        active_role_config.role,
        active_role_config.provider,
        active_role_config.model_name,
        active_role_config.context_length,
        active_role_config.max_output_tokens,
        len(messages),
        type_summary,
    )

    bind_audit = binding_audit(
        role=active_role_config.role,
        model_name=active_role_config.model_name,
        tools=tools if _bind_tools else [],
        plan=tool_plan,
        bound_model=model_with_tools if _bind_tools else None,
    )
    #logger.info("Tool binding audit: %s", json.dumps(bind_audit, sort_keys=True, default=str))

    # Step 7: Invoke the model
    # Invoke with ExpiredTokenException retry (token refresh fix)
    from botocore.exceptions import ClientError

    # Ollama message sanitization: flatten list-typed content blocks
    # (tool_use, tool_result, unknown types) to plain strings so
    # langchain_ollama does not raise ValueError on the content-type check.
    _invoke_messages = (
        _sanitize_messages_for_ollama(messages) if _is_ollama else messages
    )
    policy_prompt = (
        SystemMessage(content=build_compact_tool_execution_system_prompt(tool_plan))
        if (_bind_tools or tool_plan.data_investigation) else None
    )
    _completion_contract = extract_completion_contract(last_human_content)
    completion_policy_prompt = (
        SystemMessage(content=render_completion_contract_system_prompt(_completion_contract))
        if _completion_contract.active else None
    )
    if _completion_contract.active:
        logger.info(
            "COMPLETION_CONTRACT_DETECTED required_fields=%s",
            list(_completion_contract.required_fields),
        )
    invoke_prefix = [system_prompt]
    if policy_prompt:
        invoke_prefix.append(policy_prompt)
    if completion_policy_prompt:
        invoke_prefix.append(completion_policy_prompt)

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = await model_with_tools.ainvoke([*invoke_prefix, *_invoke_messages], config=config)
            break
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ExpiredTokenException" and attempt < max_retries:
                logger.warning(
                    "AWS token expired during ainvoke (attempt %d/%d), refreshing...",
                    attempt + 1,
                    max_retries + 1,
                )
                import asyncio as _aio
                if _is_ollama and _bind_tools:
                    model = get_tool_model(force_refresh=True, role="tool_worker")
                else:
                    model = get_model(force_refresh=True, role=selected_role)
                model_with_tools = model.bind_tools(tools) if _bind_tools else model
                await _aio.sleep(1)
                continue
            raise

    _log_ollama_response_metadata(response, role=active_role_config.role)
    response_audit = binding_audit(
        role=active_role_config.role,
        model_name=active_role_config.model_name,
        tools=tools if _bind_tools else [],
        plan=tool_plan,
        bound_model=model_with_tools if _bind_tools else None,
        response=response,
    )
    #logger.info("Tool response audit: %s", json.dumps(response_audit, sort_keys=True, default=str))

    if should_force_tool_retry(
        prompt=last_human_content,
        plan=tool_plan,
        tools=tools,
        messages=messages,
        response=response,
    ):
        logger.error(
            "FAILED_TOOL_EXECUTION: investigation prompt reached first response with zero tool calls; methodology=%s tools=%d",
            tool_plan.methodology,
            len(tools),
        )
        ledger_path = append_tool_execution_failure(
            session_id=session_id,
            prompt=last_human_content,
            plan=tool_plan,
            response_text=str(getattr(response, "content", "")),
            reason="zero_tool_calls_initial",
            audit=response_audit,
        )
        retry_tools = rank_tools_for_retry(tool_plan.methodology, tools, limit=3)
        retry_model_with_tools = model.bind_tools(retry_tools)
        # The retry response is the one that will be executed, so the
        # enforced binding must be the narrower retry binding, not the
        # original one.  Without this the gate would block legitimately
        # emitted retry calls, or enforce a stale wider binding.
        _bound_tool_names = normalize_name_list(
            [getattr(_t, 'name', '') for _t in retry_tools], MAX_LAST_BOUND_TOOL_NAMES
        )
        _policy_update = _build_tool_policy_update(
            stored=_stored_policy,
            prompt=last_human_content,
            plan=tool_plan,
            authorization_flags=_authorization_flags,
            bound_tool_names=_bound_tool_names,
            activation_request_revision=_activation_intent.request_revision,
            processed_activation_tool_call_ids=_processed_activation_ids,
        )
        publish_policy_runtime_snapshot(
            build_policy_runtime_snapshot(
                _policy_update, methodology=getattr(tool_plan, 'methodology', '')
            )
        )
        _log_tool_policy_state(
            _policy_update, methodology=getattr(tool_plan, 'methodology', '')
        )
        _record_parent_binding(
            _bound_tool_names,
            authorization_flags=_authorization_flags,
            thread_id=session_id,
            mcop_children_forbidden=bool(tool_plan.mcop_children_forbidden),
        )
        retry_prompt = SystemMessage(content=build_compact_tool_execution_system_prompt(tool_plan, retry=True))
        try:
            retry_response = await retry_model_with_tools.ainvoke([system_prompt, retry_prompt, *_invoke_messages], config=config)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Strict tool-forcing retry failed: %s", exc)
            append_tool_execution_failure(
                session_id=session_id,
                prompt=last_human_content,
                plan=tool_plan,
                response_text=f"retry exception: {type(exc).__name__}: {exc}",
                reason="strict_retry_exception",
                audit=response_audit,
            )
            retry_response = AIMessage(content=build_tool_execution_failed_message(tool_plan, ledger_path))
        if not getattr(retry_response, "tool_calls", None):
            logger.error(
                "FAILED_TOOL_EXECUTION: strict retry also produced zero tool calls; methodology=%s",
                tool_plan.methodology,
            )
            ledger_path = append_tool_execution_failure(
                session_id=session_id,
                prompt=last_human_content,
                plan=tool_plan,
                response_text=str(getattr(retry_response, "content", "")),
                reason="zero_tool_calls_retry",
                audit=response_audit,
            )
            retry_response = AIMessage(content=build_tool_execution_failed_message(tool_plan, ledger_path))
        response = retry_response

    # Explicit structured completion contracts are a terminal-response gate,
    # not prompt advice. Permit one bounded self-repair; if the model still
    # omits required fields, return an honest deterministic incomplete state.
    if _completion_contract.active and not getattr(response, "tool_calls", None):
        _missing_completion_fields = _completion_contract.missing_from(
            completion_response_text(getattr(response, "content", ""))
        )
        if _missing_completion_fields:
            logger.warning(
                "COMPLETION_CONTRACT_REPAIR missing_fields=%s",
                list(_missing_completion_fields),
            )
            # A local, non-persisted Human follow-up keeps provider message
            # ordering valid (System messages must remain in the prefix).
            repair_prompt = HumanMessage(content=(
                "The previous response attempted to terminate before satisfying "
                "the user's explicit completion contract. Continue the required "
                "work with the currently bound tools, or report a genuine external "
                "blocker. Required missing fields: "
                + ", ".join(_missing_completion_fields)
            ))
            try:
                repaired = await model_with_tools.ainvoke(
                    [*invoke_prefix, *_invoke_messages, response, repair_prompt],
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "completion-contract repair invocation failed: %s",
                    type(exc).__name__,
                )
                repaired = None
            if repaired is not None:
                response = repaired
            if not getattr(response, "tool_calls", None):
                still_missing = _completion_contract.missing_from(
                    completion_response_text(getattr(response, "content", ""))
                )
                if still_missing:
                    logger.error(
                        "COMPLETION_CONTRACT_TERMINAL missing_fields=%s",
                        list(still_missing),
                    )
                    response = AIMessage(
                        content=render_incomplete_contract(still_missing)
                    )

    # Step 8: Clean up cachepoints so they're not stored persistently
    cleanup_cachept(messages)

    return {"messages": [response], **_policy_update}


def should_continue(state: AgentState) -> str:
    """
    Determine if we should continue to tools or end.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


### Graph
@cache
def get_graph():
    # Executor inventory can remain broad because ToolNode only executes tool
    # calls already emitted by the model. The model-facing bind_tools() path is
    # scoped dynamically in call_model().
    tools = get_all_executor_tools()

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    # AuditedToolNode enforces the execution gate and emits exactly one
    # redacted audit event per emitted call.  ToolMessage content shape is
    # unchanged because execution still runs through the real tool.
    workflow.add_node("tools", make_audited_tool_node(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=get_checkpointer())
