from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from app.config import get_settings
from app.core.llm.model_roles import pure_ollama_chat_mode, resolve_all_model_roles
from .schemas import Health

router = APIRouter()


@router.get("/health", tags=["health"])
async def health_check() -> Health:
    return Health()


def tool_registry_identity_payload() -> dict[str, Any]:
    """Report semantic registry identity, health, and lifecycle separately.

    D3B1 requires health and selftest surfaces to distinguish the semantic
    content identity of the effective tool inventory, the current per-family
    health, the process-local refresh epoch, and the source of each family
    inventory.  Conflating them is what previously let a transient upstream
    outage look like a real inventory change.

    The payload is bounded and contains no transport URL, header, token,
    exception body, latency, or process identity.  Error information is limited
    to a bounded classification vocabulary.
    """
    try:
        from app.agent import registry_canonical as canonical
        from app.agent.agents.tools.registry import (
            get_mcp_registry_status,
            get_tool_inventory,
        )
        from app.agent.tool_policy_state import compute_policy_signature
    except Exception as exc:  # noqa: BLE001
        return {"status": "fail", "error_type": type(exc).__name__}

    try:
        status = get_mcp_registry_status()
        content_signature = str(status.get("content_signature") or "")

        families: dict[str, Any] = {}
        for name, state in sorted((status.get("families") or {}).items()):
            families[name] = {
                "health": state.get("health"),
                "source": state.get("source"),
                "content_signature": state.get("content_signature"),
                "tool_count": state.get("tool_count"),
                "error_class": state.get("error_class"),
                "refresh_epoch": state.get("refresh_epoch"),
            }

        # A fixed policy input set makes the profile signature comparable across
        # restarts.  Only the registry content signature varies with inventory.
        fixed_profile_signature = compute_policy_signature(
            active_toolsets=["fixed_probe_family"],
            eligible_extra_tools=[],
            registry_generation=content_signature,
            authorization_flags={},
        )

        model_surface = []
        for entry in get_tool_inventory(include_uninitialized=False):
            if not entry.get("enabled"):
                continue
            name = str(entry.get("tool_name") or "")
            if not name or name.startswith("<"):
                continue
            model_surface.append({
                "toolset": str(entry.get("toolset") or ""),
                "tool_name": name,
                "description": str(entry.get("description") or ""),
                "args_schema": entry.get("args_schema"),
            })
        model_surface.sort(key=lambda item: (item["toolset"], item["tool_name"]))
        model_schema_digest, digest_error = canonical.safe_content_signature(model_surface)

        return {
            "status": "pass",
            "schema": "diship_tool_registry_identity.v1",
            "canonicalization_version": status.get("canonicalization_version"),
            "lkg_schema_version": status.get("lkg_schema_version"),
            "registry_content_signature": content_signature,
            "registry_refresh_epoch": status.get("refresh_epoch"),
            "registry_health_signature": status.get("health_signature"),
            "fixed_policy_profile_signature": fixed_profile_signature,
            "model_facing_schema_digest": model_schema_digest,
            "model_facing_tool_count": len(model_surface),
            "model_facing_digest_error": digest_error or None,
            "healthy_families": status.get("healthy_families"),
            "degraded_last_known_good_families": status.get("degraded_last_known_good_families"),
            "unavailable_no_baseline_families": status.get("unavailable_no_baseline_families"),
            "families": families,
            "write_performed": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "fail", "error_type": type(exc).__name__}


@router.get("/health/tool-registry-identity", tags=["health"])
async def tool_registry_identity() -> dict[str, Any]:
    """Read-only registry identity, health, and lifecycle report."""
    return tool_registry_identity_payload()


async def _ollama_reachability(base_url: str) -> dict[str, Any]:
    try:
        import httpx
    except Exception as exc:
        return {"reachable": False, "models": [], "running_models": [], "error": f"httpx unavailable: {exc}"}
    root = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            tags_response = await client.get(root + "/api/tags")
            ps_response = await client.get(root + "/api/ps")
        tags_data = tags_response.json() if tags_response.headers.get("content-type", "").startswith("application/json") else {}
        ps_data = ps_response.json() if ps_response.headers.get("content-type", "").startswith("application/json") else {}
        models = [
            m.get("name") or m.get("model")
            for m in tags_data.get("models", [])
            if isinstance(m, dict) and (m.get("name") or m.get("model"))
        ]
        running = []
        for m in ps_data.get("models", []):
            if not isinstance(m, dict):
                continue
            running.append({
                "name": m.get("name") or m.get("model"),
                "model": m.get("model") or m.get("name"),
                "expires_at": m.get("expires_at"),
                "size_vram": m.get("size_vram"),
                "context_length": m.get("context_length"),
            })
        return {
            "reachable": tags_response.status_code < 500,
            "tags_status_code": tags_response.status_code,
            "ps_status_code": ps_response.status_code,
            "models": models[:50],
            "running_models": running[:50],
        }
    except Exception as exc:
        return {"reachable": False, "models": [], "running_models": [], "error": str(exc)}


def _import_status(module_name: str) -> dict[str, Any]:
    return {"module": module_name, "available": importlib.util.find_spec(module_name) is not None}


def _prompt_identity_status() -> dict[str, Any]:
    prompt_path = Path(__file__).resolve().parents[1] / "agent" / "agents" / "prompts" / "chat_system_prompt.txt"
    forbidden = ["Claude", "Sonnet", "Opus", "Anthropic", "Bedrock"]
    text = prompt_path.read_text() if prompt_path.exists() else ""
    leaks = [term for term in forbidden if term in text]
    contract = all(marker in text.lower() for marker in ("evidence discipline", "tool discipline", "verifier"))
    return {"status": "pass" if not leaks and contract else "fail", "identity_leaks": leaks, "contract_present": contract}


def _safe_call(label: str, func) -> dict[str, Any]:
    try:
        result = func()
        return result if isinstance(result, dict) else {"status": "pass", "result": result}
    except Exception as exc:
        return {"status": "unverified", "error": f"{label}: {exc}"}


@router.get("/health/ollama-agent-selftest", tags=["health"])
async def ollama_agent_selftest() -> dict[str, Any]:
    settings = get_settings()
    resolved_roles = resolve_all_model_roles(settings=settings)
    role_map = {role: cfg.to_dict() for role, cfg in resolved_roles.items()}
    ollama_status = await _ollama_reachability(settings.PLLM_API_BASE or settings.ELLM_API_BASE or "http://127.0.0.1:11434")
    local_models = set(ollama_status.get("models", []))
    running_models = {m.get("name") or m.get("model") for m in ollama_status.get("running_models", []) if isinstance(m, dict)}
    expected_models = sorted({cfg.model_name for cfg in resolved_roles.values() if cfg.provider == "ollama"})
    missing_expected_models = [name for name in expected_models if name not in local_models]
    not_running_expected_models = [name for name in expected_models if name not in running_models]
    langchain_ollama = _import_status("langchain_ollama")

    def _model_bind_status(role: str) -> dict[str, Any]:
        if not langchain_ollama["available"]:
            return {"role": role, "status": "skipped", "reason": "langchain-ollama import unavailable"}
        try:
            from app.core.llm.chat_models import get_model, get_tool_model
            model = get_tool_model(role=role) if role == "tool_worker" else get_model(role=role)
            return {"role": role, "status": "pass", "model_type": type(model).__name__}
        except Exception as exc:
            return {"role": role, "status": "fail", "error": str(exc)}

    tool_registry = _safe_call("tool registry selftest", lambda: __import__("app.agent.agents.tools.registry", fromlist=["get_tool_registry_selftest"]).get_tool_registry_selftest())
    methodology = _safe_call("methodology selftest", lambda: __import__("app.agent.methodology", fromlist=["methodology_selftest"]).methodology_selftest())
    tool_policy = _safe_call("tool execution policy selftest", lambda: __import__("app.agent.tool_execution_policy", fromlist=["tool_execution_policy_selftest"]).tool_execution_policy_selftest())
    packets = _safe_call("packet selftest", lambda: __import__("app.agent_mode.orchestration_packets", fromlist=["orchestration_packet_selftest"]).orchestration_packet_selftest(Path(tempfile.mkdtemp(prefix="ollama_packet_selftest_"))))
    context_budget = _safe_call("context budget selftest", lambda: __import__("app.tools.context_budget", fromlist=["context_budget_selftest"]).context_budget_selftest("primary"))
    verifier = _safe_call("verifier status", lambda: __import__("app.agent_mode.orchestration_packets", fromlist=["verify_final_answer_against_packets"]).verify_final_answer_against_packets("packet write/read works", [{"facts": [{"claim": "packet write/read works", "source": "selftest", "reference": "selftest", "confidence": "high"}]}]).__dict__)

    pure_ollama = pure_ollama_chat_mode(settings)
    return {
        "provider_config": {"PLLM_PROVIDER": settings.PLLM_PROVIDER, "ELLM_PROVIDER": settings.ELLM_PROVIDER, "pure_ollama_chat_mode": pure_ollama},
        "ollama_base_url_reachability": ollama_status,
        "available_local_models": ollama_status.get("models", []),
        "running_ollama_models": ollama_status.get("running_models", []),
        "expected_role_models": expected_models,
        "missing_expected_role_models": missing_expected_models,
        "expected_role_models_not_currently_running": not_running_expected_models,
        "role_provider_model_mapping": role_map,
        "langchain_ollama_import_status": langchain_ollama,
        "primary_model_bind_status": _model_bind_status("primary"),
        "tool_worker_model_bind_status": _model_bind_status("tool_worker"),
        "verifier_structured_output_status": verifier,
        "tool_registry_duplicate_status": tool_registry.get("duplicate_status", tool_registry),
        # D3B1: content identity, health, and refresh epoch are reported as
        # distinct selftest fields rather than one conflated generation value.
        "tool_registry_identity_status": _safe_call(
            "tool registry identity", tool_registry_identity_payload
        ),
        "prompt_identity_status": _prompt_identity_status(),
        "methodology_selector_status": methodology,
        "tool_execution_policy_status": tool_policy,
        "mcop_packet_status": packets,
        "evidence_ledger_write_read_status": packets,
        "context_budget_status": context_budget,
        "aws_refresh_disabled_or_skipped_for_pure_ollama": pure_ollama,
        "backend_startup_status": "selftest_route_loaded",
    }
