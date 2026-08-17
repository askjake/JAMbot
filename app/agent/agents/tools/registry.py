"""
Updated Tool Registry with Grasshopper & S3 STB Logs MCP Integration
=====================================================================

MCP tool sets registered:
- beta_report: Beta report analysis tools
- viewership: Viewership measurement tools
- netra_mcp: Netra internal tools
- jira_mcp: JIRA integration tools
- confluence_mcp: Confluence integration tools
- grasshopper_mcp: Grasshopper STB log upload via Lambda MCP server
- s3_stb_logs: S3 STB diagnostic log reader via Lambda MCP server

Changes (2026-05-18):
- Added per-tool timeout (15s) in initialize_mcp_tools() to prevent
  one slow/broken MCP from blocking all subsequent tool loading
- Added grasshopper_mcp and s3_stb_logs to async factories
"""

import asyncio
import hashlib

from app.agent import registry_canonical as canonical
from app.agent import mcp_registry_health as registry_health
import json
import logging
import os
import re
from datetime import datetime, timezone
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.agent.agents.utils import get_mcp_tools
from app.tools.web_search import public_web_search
from app.tools.internal_search import internal_search
from app.tools.cluster_inspect import cluster_inspect
from app.tools.aws_tools import (
    bedrock_list_models,
    bedrock_invoke_model,
    s3_list_buckets,
    s3_list_objects,
    s3_get_object,
    athena_list_databases,
    athena_list_tables,
    athena_execute_query,
    glue_get_table_schema,
)
from app.agent_mode.tools import (
    agent_git_clone,
    agent_create_venv,
    agent_run_python,
    agent_list_artifacts,
    agent_run_shell,
)
from app.agent_mode.mcop_tools import (
    agent_spawn_task,
    agent_spawn_parallel,
    agent_check_tasks,
    agent_read_task_result,
    agent_read_packet,
)
from app.tools.log_assist_gateway import (
    logassist_web_search,
    logassist_get_journal_files,
    logassist_append_journal,
    logassist_trigger_workflow,
    logassist_embed_content,
)
from app.agent.agents.tools.management import BACKEND_MANAGEMENT_FACADES


# Import internal tools
from app.tools.internal_tools import (
    dish_internal_tool,
    google_drive_search,
)

# Web browsing tools
# Web browsing tools: local Playwright (can access internal/local IPs)
# Tool names: local_web_browse, local_web_browse_interact, local_web_browse_api
from app.tools.web_browse import web_browse, web_browse_interact, web_browse_api, local_web_browse_manual_login, local_web_browse_clear_session



logger = logging.getLogger(__name__)
settings = get_settings()

# Per-tool timeout for MCP initialization (seconds).
# Prevents one slow/broken MCP server from blocking all subsequent tools.
MCP_PER_TOOL_TIMEOUT = 20  # 20s per tool; outer timeout in main.py is 180s

# In-memory cache of asynchronously-initialised tool sets (MCP clients, etc.)
_ASYNC_TOOL_CACHE: Dict[str, List[BaseTool]] = {}

# Factories that produce tool lists on demand.
_TOOL_FACTORIES: Dict[str, Callable[[], List[BaseTool]]] = {}

# Factories that may require async initialisation (MCP clients).
_ASYNC_TOOL_FACTORIES: Dict[str, Callable[[], Awaitable[List[BaseTool]] | List[BaseTool]]] = {}

# Refreshable registry state.  Values deliberately contain no raw endpoint,
# header, or credential material.
_MCP_REGISTRY_LOCK = asyncio.Lock()
_MCP_REGISTRY_GENERATION = 0
_MCP_REGISTRY_INITIALIZED_AT = ""
_MCP_TOOLSET_STATUS: Dict[str, Dict[str, Any]] = {}

# D3B1: content identity, health, and lifecycle are separate concerns.
#
#   _MCP_REGISTRY_REFRESH_EPOCH  process-local monotonic refresh counter
#   _FAMILY_CONTENT_SIGNATURE    semantic identity of each family inventory
#   _FAMILY_LKG_MATERIALS        schema-only last-known-good material
#
# Only content signatures may drive profile identity.  Refresh epoch, loaded
# timestamps, latency, and transient errors never do.
_MCP_REGISTRY_REFRESH_EPOCH = 0
_FAMILY_CONTENT_SIGNATURE: Dict[str, str] = {}
_FAMILY_LKG_MATERIALS: Dict[str, list] = {}
_LKG_PRELOADED = False
MCP_INIT_CONCURRENCY = max(1, int(os.environ.get("MCP_INIT_CONCURRENCY", "6")))

# Only enable MCP tool sets that are configured
if settings.BETAREPORT_MCP_CONFIG:
    _ASYNC_TOOL_FACTORIES["beta_report"] = lambda: get_mcp_tools(
        settings.BETAREPORT_MCP_CONFIG
    )
if settings.VIEWERSHIP_MCP_CONFIG:
    _ASYNC_TOOL_FACTORIES["viewership"] = lambda: get_mcp_tools(
        settings.VIEWERSHIP_MCP_CONFIG
    )

if getattr(settings, "ENABLE_INTERNAL_TOOLS_MCP", False):
    _ASYNC_TOOL_FACTORIES["netra_mcp"] = lambda: get_mcp_tools(
        settings.INTERNAL_TOOLS_MCP_CONFIG
    )

# Individual MCP tool sets for JIRA and Confluence
if getattr(settings, "JIRA_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["jira_mcp"] = lambda: get_mcp_tools(
        settings.JIRA_MCP_CONFIG
    )

if getattr(settings, "CONFLUENCE_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["confluence_mcp"] = lambda: get_mcp_tools(
        settings.CONFLUENCE_MCP_CONFIG
    )

# Grasshopper MCP - STB log upload via Lambda MCP server
if getattr(settings, "GRASSHOPPER_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["grasshopper_mcp"] = lambda: get_mcp_tools(
        settings.GRASSHOPPER_MCP_CONFIG
    )

# S3 STB Logs MCP - Read STB diagnostic logs from S3
if getattr(settings, "S3_STB_LOGS_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["s3_stb_logs"] = lambda: get_mcp_tools(
        settings.S3_STB_LOGS_MCP_CONFIG
    )

# S3 STB Logs MCP (PROD) - Production variant, tools prefixed "prod_" to avoid
# name collisions with the dev s3_stb_logs set when Bedrock builds toolConfig.
async def _get_s3_stb_logs_prod_tools():
    tools = await get_mcp_tools(settings.S3_STB_LOGS_PROD_MCP_CONFIG)
    for t in tools:
        if not t.name.startswith("prod_"):
            t.name = f"prod_{t.name}"
    return tools

if getattr(settings, "S3_STB_LOGS_PROD_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["s3_stb_logs_prod"] = _get_s3_stb_logs_prod_tools
# STB Health MCP - STB Health data access
if getattr(settings, "STBHEALTH_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["stbhealth_mcp"] = lambda: get_mcp_tools(
        settings.STBHEALTH_MCP_CONFIG
    )

# STB Health Popups MCP - STB Health popups descriptions
if getattr(settings, "STBHEALTH_POPUPS_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["stbhealth_popups_mcp"] = lambda: get_mcp_tools(
        settings.STBHEALTH_POPUPS_MCP_CONFIG
    )

# RCA MCP - Root Cause Analysis pipelines for device log investigation
if getattr(settings, "RCA_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["rca_mcp"] = lambda: get_mcp_tools(
        settings.RCA_MCP_CONFIG
    )

# RTR Alerts MCP - RTR alert data, definitions, and anomalies from Elasticsearch
if getattr(settings, "RTR_ALERTS_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["rtr_alerts_mcp"] = lambda: get_mcp_tools(
        settings.RTR_ALERTS_MCP_CONFIG
    )


# QoS MCP - QoS session analysis and OTA switchback diagnostics
if getattr(settings, "QOS_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["qos_mcp"] = lambda: get_mcp_tools(
        settings.QOS_MCP_CONFIG
    )

# EPG MCP - EPG schedule data, STB-delivered EPG, and channel/service metadata
if getattr(settings, "EPG_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["epg_mcp"] = lambda: get_mcp_tools(
        settings.EPG_MCP_CONFIG
    )

# Net Detective MCP - ML-powered Netra data analysis for Dish STBs
if getattr(settings, "NET_DETECTIVE_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["net_detective_mcp"] = lambda: get_mcp_tools(
        settings.NET_DETECTIVE_MCP_CONFIG
    )

# Qodo Context Retriever MCP - Semantic code search across indexed DISH repos
if getattr(settings, "QODO_CONTEXT_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["qodo_context_mcp"] = lambda: get_mcp_tools(
        settings.QODO_CONTEXT_MCP_CONFIG
    )

if getattr(settings, "HEADLESS_BROWSER_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["headless_browser_mcp"] = lambda: get_mcp_tools(
        settings.HEADLESS_BROWSER_MCP_CONFIG
    )




# dish-code-tools MCP - Structural code-intelligence (browse, grep, symbols, git)
if getattr(settings, "DISH_CODE_TOOLS_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["dish_code_tools"] = lambda: get_mcp_tools(
        settings.DISH_CODE_TOOLS_MCP_CONFIG
    )

# DVA MCP - STB software jamming via JAMboreeLite on engacc network
if getattr(settings, "DVA_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["dva_mcp"] = lambda: get_mcp_tools(
        settings.DVA_MCP_CONFIG
    )

# Google Drive MCP - Read/search Google Drive files and documents
if getattr(settings, "GDRIVE_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["gdrive_mcp"] = lambda: get_mcp_tools(
        settings.GDRIVE_MCP_CONFIG
    )

# ServiceNow ITSM MCP - Incident, change, and user search
if getattr(settings, "SERVICENOW_MCP_CONFIG", None):
    _ASYNC_TOOL_FACTORIES["servicenow_mcp"] = lambda: get_mcp_tools(
        settings.SERVICENOW_MCP_CONFIG
    )

# Pure-Python tools that don't need MCP-style startup.
_TOOL_FACTORIES.update(
    {
        # Web + internal search tools used by the chat agent.
        "search": lambda: [public_web_search, internal_search, cluster_inspect],

        # Coverity / Log Assist tools via the Flask gateway (HTTP, not MCP).
        "log_assist": lambda: [
            logassist_web_search,
            logassist_get_journal_files,
            logassist_append_journal,
            logassist_trigger_workflow,
            logassist_embed_content,
        ],

        # AWS tools for Bedrock, S3, Athena, and Glue (read-focused)
        "aws": lambda: [
            bedrock_list_models,
            bedrock_invoke_model,
            s3_list_buckets,
            s3_list_objects,
            s3_get_object,
            athena_list_databases,
            athena_list_tables,
            athena_execute_query,
            glue_get_table_schema,
        ],
        
        # Filesystem + process sandbox used in agent mode.
        # MCOP tools added 2026-06-23: spawn/parallel/check/read child convs.
        "agent_mode": lambda: [
            agent_git_clone,
            agent_create_venv,
            agent_run_python,
            agent_list_artifacts,
            agent_run_shell,
            # Local web browsing tools (Playwright - accesses internal/local IPs)
            web_browse,
            web_browse_interact,
            web_browse_api,
            local_web_browse_manual_login,
            local_web_browse_clear_session,
            # Multi-Conversation Orchestration Protocol (MCOP)
            agent_spawn_task,
            agent_spawn_parallel,
            agent_check_tasks,
            agent_read_task_result,
            agent_read_packet,
        ],
        
        # Internal DISH tools (Google Drive, DISH Internal Tools)
        "dish_internal": lambda: [
            dish_internal_tool,
            google_drive_search,
        ],

        # Permanently bound, read-only registry/binding/activation-request facades.
        "backend_facades": lambda: list(BACKEND_MANAGEMENT_FACADES),
        
    }
)

# Bedrock toolSpec.name must be <= 64 characters. The limit is provider-specific;
# Ollama/native paths preserve stable names unless deterministic collision handling is needed.
_BEDROCK_MAX_NAME = 64

def _active_provider_requires_bedrock_tool_limit() -> bool:
    return settings.PLLM_PROVIDER == "aws-bedrock" or settings.ELLM_PROVIDER == "aws-bedrock"

def _stable_collision_name(toolset_name: str, tool_name: str, *, max_len: int | None = None) -> str:
    candidate = toolset_name.replace("_mcp", "").replace("_", "")[:10] + "_" + tool_name
    if max_len and len(candidate) > max_len:
        h = hashlib.sha256(candidate.encode()).hexdigest()[:5]
        candidate = candidate[: max_len - 6] + "_" + h
    return candidate

def _sanitize_tool_names(tools: list, toolset_name: str) -> list:
    """Sanitize tool names according to the active provider."""
    seen: dict = {}
    apply_bedrock_limit = _active_provider_requires_bedrock_tool_limit()
    for tool in tools:
        original_name = tool.name
        if apply_bedrock_limit and len(tool.name) > _BEDROCK_MAX_NAME:
            h = hashlib.sha256(tool.name.encode()).hexdigest()[:5]
            short = tool.name[:58] + "_" + h
            logger.warning(
                "Tool name exceeds Bedrock 64-char limit (len=%d, set=%s): %r -> %r",
                len(tool.name), toolset_name, tool.name, short,
            )
            original_desc = tool.description or ""
            tool.description = f"[original name: {tool.name}] {original_desc}".strip()
            tool.name = short
        if tool.name in seen:
            new_name = _stable_collision_name(toolset_name, original_name, max_len=_BEDROCK_MAX_NAME if apply_bedrock_limit else None)
            original_desc = tool.description or ""
            tool.description = f"[original name: {tool.name}] {original_desc}".strip()
            tool.name = new_name
        seen[tool.name] = True
    return tools


def _redact_registry_error_text(value) -> str:
    """Bound and redact provider/transport error text before status exposure."""
    text = re.sub(r"https?://[^\s]+", "<redacted-url>", str(value or ""))
    text = re.sub(
        r"(?i)(token|password|authorization|secret)=?[^\s,;]+",
        r"\1=<redacted>",
        text,
    )
    return text.strip()


def _exception_leaf_details(exc, limit: int = 4) -> list[dict[str, str]]:
    """Flatten ExceptionGroup leaves without leaking raw connection material."""
    leaves: list[dict[str, str]] = []

    def walk(error) -> None:
        if len(leaves) >= limit:
            return
        if isinstance(error, BaseExceptionGroup):
            for child in error.exceptions:
                walk(child)
                if len(leaves) >= limit:
                    break
            return
        leaves.append({
            "code": type(error).__name__,
            "message": _redact_registry_error_text(error)[:240],
        })

    walk(exc)
    return leaves


def _registry_exception_details(exc) -> dict[str, Any]:
    """Return a safe outer code plus useful redacted leaf diagnostics."""
    leaves = _exception_leaf_details(exc)
    leaf_codes: list[str] = []
    summaries: list[str] = []
    for leaf in leaves:
        code = str(leaf.get("code") or "Exception")
        message = str(leaf.get("message") or "")
        if code not in leaf_codes:
            leaf_codes.append(code)
        summaries.append(f"{code}: {message}" if message else code)
    if not summaries:
        summaries = [f"{type(exc).__name__}: {_redact_registry_error_text(exc)}"]
    return {
        "error_code": type(exc).__name__,
        "error_leaf_codes": leaf_codes,
        "error_summary": "; ".join(summaries)[:480],
    }


async def _load_async_toolset(
    name: str,
    factory: Callable[[], Awaitable[List[BaseTool]] | List[BaseTool]],
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[BaseTool], dict[str, Any]]:
    started = datetime.now(timezone.utc)
    async with semaphore:
        try:
            tools = factory()
            if asyncio.iscoroutine(tools) or isinstance(tools, Awaitable):
                tools = await asyncio.wait_for(tools, timeout=MCP_PER_TOOL_TIMEOUT)
            loaded = _sanitize_tool_names(list(tools), name)
            return name, loaded, {
                "status": "loaded",
                "error_code": "",
                "duration_ms": int((datetime.now(timezone.utc) - started).total_seconds() * 1000),
            }
        except asyncio.TimeoutError:
            return name, [], {
                "status": "timeout",
                "error_code": "MCP_TOOLSET_INIT_TIMEOUT",
                "duration_ms": int((datetime.now(timezone.utc) - started).total_seconds() * 1000),
            }
        except Exception as exc:  # noqa: BLE001
            # Preserve the useful leaf cause of TaskGroup/ExceptionGroup failures
            # while retaining the existing no-secret/no-endpoint exposure rule.
            details = _registry_exception_details(exc)
            return name, [], {
                "status": "failed",
                **details,
                "duration_ms": int((datetime.now(timezone.utc) - started).total_seconds() * 1000),
            }


def _tool_schema_material(tool: Any) -> dict[str, Any]:
    """Semantic material for one tool, with no arbitrary object repr.

    D3B1: an args schema that cannot be represented deterministically becomes an
    explicit token instead of str(args_schema), which previously embedded a
    class repr into semantic identity.
    """
    return registry_health.tool_material(tool)


def _toolset_schema_signature(tools: list[Any]) -> str:
    """Deterministic content signature for one family inventory.

    Uses the versioned canonicalizer rather than the JSON default-coercion hook,
    so a callable, connection object, or unexpected type raises a safe
    deterministic error instead of silently entering the signature.
    """
    materials = registry_health.family_materials(tools)
    signature, _error = registry_health.safe_materials_signature(materials)
    if signature:
        return signature
    # Canonicalization failure is reported as an explicit deterministic token.
    # It never falls back to stringifying the offending value.
    fallback, _err = canonical.safe_content_signature(
        {"__uncanonicalizable_inventory__": True, "tool_count": len(list(tools or ()))}
    )
    return fallback


def preload_persistent_lkg(families: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    """Load validated persisted last-known-good snapshots before live discovery.

    This establishes trusted semantic content identity for every family that has
    a baseline, so a transient startup timeout or an outer cancellation cannot
    silently convert a known family into empty or absent semantic content.

    Persisted material is schema-only.  It is never converted into executable
    tools, so a degraded family contributes semantic identity without ever
    allowing a blind call to an upstream that is not currently proven healthy.
    """
    global _LKG_PRELOADED
    names = list(families or _ASYNC_TOOL_FACTORIES.keys())
    outcome: dict[str, str] = {}
    for name in names:
        snapshot, error = registry_health.load_snapshot(name)
        if snapshot is not None and not error:
            materials = [dict(item) for item in snapshot.get("tools") or ()]
            signature = str(snapshot.get("content_signature") or "")
            _FAMILY_LKG_MATERIALS[name] = materials
            _FAMILY_CONTENT_SIGNATURE[name] = signature
            registry_health.set_family_state(
                name,
                health=registry_health.FAMILY_DEGRADED_LAST_KNOWN_GOOD,
                source=registry_health.SOURCE_PERSISTED_LKG,
                content_signature_value=signature,
                refresh_epoch=_MCP_REGISTRY_REFRESH_EPOCH,
                tool_count=len(materials),
                last_success_at=str(snapshot.get("fetched_at") or ""),
            )
            outcome[name] = registry_health.SOURCE_PERSISTED_LKG
            continue
        registry_health.set_family_state(
            name,
            health=registry_health.FAMILY_UNAVAILABLE_NO_BASELINE,
            source=registry_health.SOURCE_NONE,
            refresh_epoch=_MCP_REGISTRY_REFRESH_EPOCH,
            error_class=error,
        )
        outcome[name] = registry_health.ERROR_CACHE_CORRUPT if error else "NO_BASELINE"
    _LKG_PRELOADED = True
    return outcome


def _degrade_family(name: str, epoch: int, error_class: str) -> None:
    """Retain last-known-good semantic content and mark the family degraded."""
    materials = _FAMILY_LKG_MATERIALS.get(name) or []
    signature = _FAMILY_CONTENT_SIGNATURE.get(name, "")
    in_memory = bool(_ASYNC_TOOL_CACHE.get(name))
    previous = registry_health.get_family_state(name)
    registry_health.set_family_state(
        name,
        health=registry_health.FAMILY_DEGRADED_LAST_KNOWN_GOOD,
        source=registry_health.SOURCE_IN_MEMORY_LKG if in_memory else registry_health.SOURCE_PERSISTED_LKG,
        content_signature_value=signature,
        refresh_epoch=epoch,
        tool_count=len(materials) or len(_ASYNC_TOOL_CACHE.get(name) or []),
        error_class=error_class,
        last_success_at=str(previous.get("last_success_at") or ""),
    )


def _classify_unprocessed_family(name: str, epoch: int) -> None:
    """Classify a family the outer scope never committed.

    Before D3B1 an outer cancellation left such a family absent from
    _TOOL_FACTORIES entirely, which silently changed semantic registry identity
    even though the upstream inventory had not changed.
    """
    has_baseline = bool(_FAMILY_LKG_MATERIALS.get(name)) or bool(_ASYNC_TOOL_CACHE.get(name))
    if has_baseline:
        _degrade_family(name, epoch, registry_health.ERROR_CANCELLED)
        return
    registry_health.set_family_state(
        name,
        health=registry_health.FAMILY_UNAVAILABLE_NO_BASELINE,
        source=registry_health.SOURCE_NONE,
        refresh_epoch=epoch,
        error_class=registry_health.ERROR_CANCELLED,
    )


def _recompute_live_family_content(epoch: int) -> None:
    """Refresh content signatures for healthy families after deduplication.

    Collision-safe names are part of the real model and executor schema, so the
    persisted baseline must record the deduplicated inventory.
    """
    for name, tools in _ASYNC_TOOL_CACHE.items():
        state = registry_health.get_family_state(name)
        if str(state.get("health") or "") != registry_health.FAMILY_HEALTHY:
            continue
        materials = registry_health.family_materials(tools, name)
        signature, _error = registry_health.safe_materials_signature(materials)
        if not signature:
            continue
        if signature != _FAMILY_CONTENT_SIGNATURE.get(name):
            registry_health.write_snapshot(name, materials)
        _FAMILY_LKG_MATERIALS[name] = materials
        _FAMILY_CONTENT_SIGNATURE[name] = signature
        registry_health.set_family_state(
            name,
            health=registry_health.FAMILY_HEALTHY,
            source=registry_health.SOURCE_LIVE_DISCOVERY,
            content_signature_value=signature,
            refresh_epoch=epoch,
            tool_count=len(materials),
        )
        if name in _MCP_TOOLSET_STATUS:
            _MCP_TOOLSET_STATUS[name]["tool_count"] = len(tools)
            _MCP_TOOLSET_STATUS[name]["schema_signature"] = signature
            _MCP_TOOLSET_STATUS[name]["source_identity"] = "schema:" + signature.removeprefix("sha256:")


def owning_mcp_family(tool_name: str) -> str:
    """Return the MCP family that owns one fully qualified tool name."""
    target = str(tool_name or "")
    if not target:
        return ""
    for family, tools in _ASYNC_TOOL_CACHE.items():
        for tool in tools:
            if str(getattr(tool, "name", "") or "") == target:
                return family
    for family, materials in _FAMILY_LKG_MATERIALS.items():
        for item in materials:
            if str(item.get("name") or "") == target:
                return family
    return ""


async def initialize_mcp_tools(
    *,
    force: bool = False,
    toolsets: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Initialise independent MCP families concurrently with bounded concurrency.

    Completed families are committed to the cache immediately, so cancellation by
    an outer startup timeout does not hide tools already discovered.  A single
    optional MCP failure never blocks other families.

    D3B1 health semantics:

    * a successful discovery replaces the last-known-good snapshot atomically
      and marks the family HEALTHY;
    * a timeout, cancellation, or transport failure with a valid baseline retains
      the last-known-good semantic content, preserves the family content
      signature, and marks the family DEGRADED_LAST_KNOWN_GOOD;
    * a failure with no baseline fails closed as UNAVAILABLE_NO_BASELINE and
      binds no tools;
    * a newly discovered inventory that cannot be canonicalized never overwrites
      a valid baseline.
    """
    global _ASYNC_TOOL_CACHE, _TOOL_FACTORIES
    global _MCP_REGISTRY_GENERATION, _MCP_REGISTRY_INITIALIZED_AT
    global _MCP_REGISTRY_REFRESH_EPOCH

    requested = list(toolsets or _ASYNC_TOOL_FACTORIES.keys())
    unknown = sorted(set(requested) - set(_ASYNC_TOOL_FACTORIES))
    if unknown:
        raise ValueError(f"Unknown MCP toolsets requested for initialization: {unknown}")

    if not _LKG_PRELOADED:
        # Establish trusted baselines before any live discovery can fail.
        preload_persistent_lkg()

    async with _MCP_REGISTRY_LOCK:
        pending = [name for name in requested if force or name not in _ASYNC_TOOL_CACHE]
        if not pending:
            return
        _MCP_REGISTRY_REFRESH_EPOCH += 1
        epoch = _MCP_REGISTRY_REFRESH_EPOCH
        semaphore = asyncio.Semaphore(MCP_INIT_CONCURRENCY)
        tasks = [
            asyncio.create_task(_load_async_toolset(name, _ASYNC_TOOL_FACTORIES[name], semaphore))
            for name in pending
        ]
        outstanding = set(pending)
        completed = 0
        try:
            for future in asyncio.as_completed(tasks):
                name, discovered_tools, status = await future
                outstanding.discard(name)
                status = dict(status)
                previous_tools = _ASYNC_TOOL_CACHE.get(name)
                discovery_status = str(status.get("status") or "failed")
                has_baseline = bool(_FAMILY_LKG_MATERIALS.get(name)) or previous_tools is not None

                if discovery_status == "loaded":
                    materials = registry_health.family_materials(discovered_tools, name)
                    signature, _canon_error = registry_health.safe_materials_signature(materials)
                    if not signature:
                        # An invalid new schema must never overwrite a trusted
                        # baseline, and the offending value is never surfaced.
                        effective_tools = previous_tools if previous_tools is not None else []
                        status.update({
                            "status": "invalid_schema",
                            "error_code": registry_health.ERROR_INVALID_SCHEMA,
                            "using_last_known_good": has_baseline,
                        })
                        if has_baseline:
                            _degrade_family(name, epoch, registry_health.ERROR_INVALID_SCHEMA)
                        else:
                            registry_health.set_family_state(
                                name,
                                health=registry_health.FAMILY_INVALID_SCHEMA,
                                source=registry_health.SOURCE_NONE,
                                refresh_epoch=epoch,
                                error_class=registry_health.ERROR_INVALID_SCHEMA,
                            )
                    else:
                        effective_tools = discovered_tools
                        _FAMILY_LKG_MATERIALS[name] = materials
                        _FAMILY_CONTENT_SIGNATURE[name] = signature
                        registry_health.write_snapshot(name, materials)
                        registry_health.set_family_state(
                            name,
                            health=registry_health.FAMILY_HEALTHY,
                            source=registry_health.SOURCE_LIVE_DISCOVERY,
                            content_signature_value=signature,
                            refresh_epoch=epoch,
                            tool_count=len(materials),
                        )
                        status["using_last_known_good"] = False
                elif has_baseline:
                    # A transient refresh outage must not erase a previously
                    # verified executor schema.  Keep the last-known-good
                    # inventory and expose the failed refresh explicitly.
                    effective_tools = previous_tools if previous_tools is not None else []
                    error_class = (
                        registry_health.ERROR_TIMEOUT
                        if discovery_status == "timeout"
                        else registry_health.ERROR_TRANSPORT
                    )
                    _degrade_family(name, epoch, error_class)
                    status.update({
                        "status": "refresh_failed_using_cached",
                        "refresh_failure_status": discovery_status,
                        "using_last_known_good": True,
                    })
                else:
                    effective_tools = []
                    error_class = (
                        registry_health.ERROR_TIMEOUT
                        if discovery_status == "timeout"
                        else registry_health.ERROR_TRANSPORT
                    )
                    registry_health.set_family_state(
                        name,
                        health=registry_health.FAMILY_UNAVAILABLE_NO_BASELINE,
                        source=registry_health.SOURCE_NONE,
                        refresh_epoch=epoch,
                        error_class=error_class,
                    )
                    status["using_last_known_good"] = False

                _ASYNC_TOOL_CACHE[name] = effective_tools
                _TOOL_FACTORIES[name] = lambda n=name: _ASYNC_TOOL_CACHE.get(n, [])
                family_signature = _FAMILY_CONTENT_SIGNATURE.get(name) or _toolset_schema_signature(effective_tools)
                status.update({
                    "tool_count": len(effective_tools),
                    "schema_signature": family_signature,
                    "source_identity": "schema:" + family_signature.removeprefix("sha256:"),
                    "source_version": None,
                    "source_version_status": "UNVERIFIED_SCHEMA_ONLY",
                    "health": registry_health.family_health(name),
                    "refresh_epoch": epoch,
                })
                _MCP_TOOLSET_STATUS[name] = status
                completed += 1
                if status["status"] == "loaded":
                    logger.info("Loaded MCP tool set %r with %d tools", name, len(effective_tools))
                else:
                    logger.warning(
                        "MCP tool set %r initialization status=%s health=%s code=%s",
                        name, status["status"], status.get("health"), status.get("error_code"),
                    )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # D3B1: a family the outer scope never committed is explicitly
            # classified instead of disappearing from registry structures.
            for name in sorted(outstanding):
                _classify_unprocessed_family(name, epoch)
            # Collision handling occurs after every refresh and remains
            # deterministic in registered toolset order.  Recalculate each status
            # signature afterward because collision-safe names are part of the
            # actual model/executor schema.
            _deduplicate_tool_names_across_sets(_ASYNC_TOOL_CACHE)
            _recompute_live_family_content(epoch)
            _MCP_REGISTRY_GENERATION = epoch
            if completed:
                _MCP_REGISTRY_INITIALIZED_AT = datetime.now(timezone.utc).isoformat()


async def refresh_mcp_tools(toolsets: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    """Explicitly rediscover selected upstream schemas and return redacted status."""
    await initialize_mcp_tools(force=True, toolsets=toolsets)
    return get_mcp_registry_status()


def invalidate_mcp_tool_cache(toolsets: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    """Invalidate selected schema caches without contacting an upstream MCP."""
    names = list(toolsets or _ASYNC_TOOL_CACHE.keys())
    removed = []
    for name in names:
        if name in _ASYNC_TOOL_CACHE:
            _ASYNC_TOOL_CACHE.pop(name, None)
            _MCP_TOOLSET_STATUS.pop(name, None)
            removed.append(name)
    return {"ok": True, "invalidated_toolsets": sorted(removed), "refresh_required": bool(removed)}



def _deduplicate_tool_names_across_sets(cache: dict) -> None:
    """Rename tools whose names collide across different MCP tool sets.

    Bedrock raises ValidationException if the same tool name appears twice in
    toolConfig.  When two separate MCP servers expose a tool with the same name
    (e.g. query_popups appearing in both stbhealth_popups_mcp and rtr_alerts_mcp)
    we prefix the later-loading set's tools with a short tag derived from the
    toolset key so that every name in the flat list is unique.

    Preserves original name in tool.description so the LLM still knows intent.
    """
    # Build a flat name -> first_seen_toolset index map
    seen: dict = {}  # name -> toolset_name
    for toolset_name, tools in cache.items():
        for tool in tools:
            if tool.name in seen:
                first = seen[tool.name]
                # Derive a short prefix from the toolset name (up to 8 chars)
                prefix = toolset_name.replace('_mcp', '').replace('_', '')[:8] + '_'
                limit = _BEDROCK_MAX_NAME if _active_provider_requires_bedrock_tool_limit() else None
                new_name = _stable_collision_name(toolset_name, tool.name, max_len=limit)
                logger.warning(
                    "Tool name collision across sets (name=%r in %r and %r) -> renaming to %r",
                    tool.name, first, toolset_name, new_name,
                )
                original_desc = tool.description or ''
                tool.description = f'[original name: {tool.name}] {original_desc}'.strip()
                tool.name = new_name
            else:
                seen[tool.name] = toolset_name


def get_tools_set(tool_type: str) -> List[BaseTool]:
    """Return the configured tool set for a given logical type.

    Known types as of now:
      - "search":         public web search + internal_search + cluster_inspect
      - "beta_report":    MCP tools for the beta-report agent
      - "log_assist":     MCP tools for Coverity/Log Assist
      - "internal_tools": Generic internal MCP utilities
      - "agent_mode":     Filesystem + process sandbox tools
      - "dish_internal":  Internal DISH tools (Google Drive, DISH Internal Tools)
      - "grasshopper_mcp": Grasshopper STB log upload via MCP Lambda
      - "s3_stb_logs":          S3 STB diagnostic log reader via MCP Lambda (dev)
      - "s3_stb_logs_prod":     S3 STB diagnostic log reader via MCP Lambda (production)
      - "stbhealth_mcp":        STB Health data access
      - "stbhealth_popups_mcp": STB Health popups descriptions
      - "rca_mcp":              Root Cause Analysis pipelines for device log investigation
      - "rtr_alerts_mcp":       RTR alert data, definitions and anomalies from Elasticsearch
      - "qos_mcp":              QoS session analysis and OTA switchback diagnostics
      - "epg_mcp":              EPG schedule data, STB-delivered EPG, channel metadata
      - "net_detective_mcp":    ML-powered Netra data analysis for Dish STBs
      - "qodo_context_mcp":     Semantic code search across indexed DISH git repos (Qodo)
      - "gdrive_mcp":            Google Drive file search, read, and folder listing
      - "servicenow_mcp":       ServiceNow ITSM — incidents, changes, user lookup

    Unknown types return an empty list.
    """
    factory = _TOOL_FACTORIES.get(tool_type)
    if not factory:
        logger.warning("Unknown tool_type %r requested; returning empty tool list.", tool_type)
        return []
    return list(factory())


def _matches_requested_tool_name(tool: Any, requested: str) -> bool:
    actual = str(getattr(tool, "name", ""))
    if actual == requested:
        return True
    description = str(getattr(tool, "description", "") or "")
    if f"[original name: {requested}]" in description:
        return True
    return actual.endswith("_" + requested)


def get_tools_set_filtered(tool_type: str, allowed_names: list[str] | tuple[str, ...] | set[str]) -> List[BaseTool]:
    """Return only explicitly allowed names from one tool family."""
    requested = tuple(str(name) for name in allowed_names if str(name))
    if not requested:
        return []
    tools = get_tools_set(tool_type)
    return [tool for tool in tools if any(_matches_requested_tool_name(tool, name) for name in requested)]


def _family_content_material(name: str) -> dict[str, Any]:
    """Content contribution of one configured MCP family.

    A family with no trusted baseline contributes an explicit unavailable token.
    That keeps it distinguishable from a healthy family that genuinely exposes
    zero tools, and prevents a transient outage from looking like a semantic
    inventory change.
    """
    state = registry_health.get_family_state(name)
    health_value = str(state.get("health") or registry_health.FAMILY_UNAVAILABLE_NO_BASELINE)
    if health_value in (registry_health.FAMILY_UNAVAILABLE_NO_BASELINE, registry_health.FAMILY_INVALID_SCHEMA):
        return {"family": name, "content": {"__unavailable_no_baseline__": True}}
    if health_value == registry_health.FAMILY_DISABLED:
        return {"family": name, "content": {"__disabled__": True}}
    return {"family": name, "content": _FAMILY_CONTENT_SIGNATURE.get(name, "")}


def get_registry_content_signature() -> str:
    """Stable identity of the semantic effective tool inventory.

    Deliberately independent of refresh epoch, load timestamps, latency,
    transient error state, process identity, and discovery ordering.  A family
    holding a validated last-known-good baseline contributes the same content as
    when it was last discovered successfully.
    """
    material: list[dict[str, Any]] = []
    async_names = set(_ASYNC_TOOL_FACTORIES) | set(_FAMILY_CONTENT_SIGNATURE)
    for name in sorted(set(_TOOL_FACTORIES) - async_names):
        try:
            tools = list(_TOOL_FACTORIES[name]())
        except Exception:  # noqa: BLE001
            tools = []
        material.append({"family": name, "content": _toolset_schema_signature(tools)})
    for name in sorted(async_names):
        material.append(_family_content_material(name))
    signature, _error = canonical.safe_content_signature(material)
    return signature


def get_registry_refresh_epoch() -> int:
    """Process-local monotonic refresh counter.  Never a semantic identity."""
    return int(_MCP_REGISTRY_REFRESH_EPOCH)


def get_registry_health_signature() -> str:
    """Deterministic summary of current per-family health states."""
    return registry_health.registry_health_signature()


def get_tool_inventory_signature() -> str:
    """Backward-compatible alias for the semantic content signature."""
    return get_registry_content_signature()


def get_mcp_registry_status() -> dict[str, Any]:
    toolsets = {}
    for name in sorted(_ASYNC_TOOL_FACTORIES):
        state = dict(_MCP_TOOLSET_STATUS.get(name) or {})
        state.setdefault("status", "uninitialized")
        state.setdefault("tool_count", len(_ASYNC_TOOL_CACHE.get(name, [])))
        state.setdefault("schema_signature", _toolset_schema_signature(_ASYNC_TOOL_CACHE.get(name, [])))
        state["refreshable"] = True
        toolsets[name] = state
    attempted = set(_ASYNC_TOOL_CACHE)
    configured = set(_ASYNC_TOOL_FACTORIES)
    loaded = sorted(name for name, state in toolsets.items() if state.get("status") == "loaded")
    degraded = sorted(name for name, state in toolsets.items() if state.get("status") == "refresh_failed_using_cached")
    unavailable = sorted(name for name, state in toolsets.items() if state.get("status") in {"failed", "timeout"})
    families = registry_health.effective_registry_model()
    healthy = sorted(n for n, s in families.items() if s.get("health") == registry_health.FAMILY_HEALTHY)
    degraded_lkg = sorted(
        n for n, s in families.items()
        if s.get("health") == registry_health.FAMILY_DEGRADED_LAST_KNOWN_GOOD
    )
    no_baseline = sorted(
        n for n, s in families.items()
        if s.get("health") in (
            registry_health.FAMILY_UNAVAILABLE_NO_BASELINE,
            registry_health.FAMILY_INVALID_SCHEMA,
        )
    )
    return {
        "schema": "diship_mcp_registry_status.v2",
        "initialized": bool(attempted),
        "initialization_complete": configured.issubset(attempted),
        # D3B1: content identity, lifecycle, and health are reported separately.
        "content_signature": get_registry_content_signature(),
        "refresh_epoch": get_registry_refresh_epoch(),
        "health_signature": get_registry_health_signature(),
        "canonicalization_version": canonical.CANONICALIZATION_VERSION,
        "lkg_schema_version": registry_health.LKG_SCHEMA_VERSION,
        "families": families,
        "healthy_families": healthy,
        "degraded_last_known_good_families": degraded_lkg,
        "unavailable_no_baseline_families": no_baseline,
        # Retained for compatibility with existing readers.
        "generation": _MCP_REGISTRY_GENERATION,
        "initialized_at": _MCP_REGISTRY_INITIALIZED_AT or None,
        "bounded_concurrency": MCP_INIT_CONCURRENCY,
        "inventory_signature": get_registry_content_signature(),
        "source_version_status": "UNVERIFIED_SCHEMA_ONLY",
        "loaded_toolsets": loaded,
        "degraded_toolsets": degraded,
        "unavailable_toolsets": unavailable,
        "toolsets": toolsets,
    }


def _tool_to_inventory_entry(tool: Any, toolset_name: str, *, enabled: bool, source: str) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    schema: Any = None
    if args_schema is not None:
        try:
            schema = args_schema.model_json_schema()
        except Exception:
            try:
                schema = args_schema.schema()
            except Exception:
                schema = str(args_schema)
    return {
        "toolset": toolset_name,
        "tool_name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", "") or "",
        "args_schema": schema,
        "handler_source": source,
        "enabled": enabled,
        "provider_server": "mcp" if toolset_name in _ASYNC_TOOL_FACTORIES or toolset_name in _ASYNC_TOOL_CACHE else "local",
    }


def get_tool_inventory(include_uninitialized: bool = True) -> list[dict[str, Any]]:
    """Return live inventory from registered local factories and MCP cache state."""
    inventory: list[dict[str, Any]] = []
    for toolset_name, factory in sorted(_TOOL_FACTORIES.items()):
        try:
            for tool in factory():
                inventory.append(_tool_to_inventory_entry(tool, toolset_name, enabled=True, source=repr(factory)))
        except Exception as exc:
            inventory.append({"toolset": toolset_name, "tool_name": "<factory-error>", "description": str(exc), "args_schema": None, "handler_source": repr(factory), "enabled": False, "provider_server": "local"})
    if include_uninitialized:
        for toolset_name, factory in sorted(_ASYNC_TOOL_FACTORIES.items()):
            if toolset_name in _ASYNC_TOOL_CACHE or toolset_name in _TOOL_FACTORIES:
                continue
            inventory.append({"toolset": toolset_name, "tool_name": "<uninitialized-mcp-toolset>", "description": "MCP schemas not loaded yet; startup initialize_mcp_tools() discovers them.", "args_schema": None, "handler_source": repr(factory), "enabled": False, "provider_server": "mcp"})
    return inventory


def detect_duplicate_tools(inventory: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    inventory = inventory or get_tool_inventory()
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for item in inventory:
        name = item.get("tool_name")
        if not name or str(name).startswith("<"):
            continue
        toolset = item.get("toolset", "")
        if name in seen:
            duplicates.append({"tool_name": name, "first_toolset": seen[name], "duplicate_toolset": toolset})
        else:
            seen[name] = toolset
    return {"status": "pass" if not duplicates else "fail", "duplicates": duplicates}


def detect_prompt_tool_drift(prompt_text: str = "", inventory: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    inventory = inventory or get_tool_inventory()
    active_names = {item["tool_name"] for item in inventory if item.get("enabled") and not str(item.get("tool_name", "")).startswith("<")}
    advertised = {name for name in active_names if prompt_text and name in prompt_text}
    missing = sorted({tok.strip('`,.;:') for tok in prompt_text.split() if tok.startswith("agent_") and tok.strip('`,.;:') not in active_names}) if prompt_text else []
    return {"status": "pass" if not missing else "fail", "tools_advertised_in_prompt": sorted(advertised), "tools_advertised_but_missing": missing, "tools_present_but_not_advertised_count": len(active_names - advertised) if prompt_text else len(active_names)}


def get_tool_registry_selftest(prompt_text: str = "") -> dict[str, Any]:
    inventory = get_tool_inventory()
    return {
        "inventory_count": len(inventory),
        "duplicate_status": detect_duplicate_tools(inventory),
        "prompt_drift_status": detect_prompt_tool_drift(prompt_text, inventory),
        "bedrock_name_limit_active": _active_provider_requires_bedrock_tool_limit(),
        "provider_name_policy": "bedrock_64_char_limit" if _active_provider_requires_bedrock_tool_limit() else "ollama_preserve_stable_names",
    }
