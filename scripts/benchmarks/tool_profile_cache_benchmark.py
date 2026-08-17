#!/usr/bin/env python3
"""D3B2 tool-profile, model-facing schema, and cache-stability benchmark.

What this measures
------------------
This harness measures the *real* dynamic-tool-binding pipeline:

* ``app.agent.tool_execution_policy.build_profile_for_prompt`` -> the real
  ``ToolProfile`` and its real ``profile_signature``;
* ``app.agent.tool_execution_policy.get_tools_for_toolsets(model_facing=True)``
  -> the real model-facing tool objects, including the real
  ``prepare_model_facing_tool`` authorization-argument sanitizer and the real
  D3B0 code-execution withholding;
* ``langchain_aws.chat_models.bedrock_converse._format_tools`` -> the real
  provider-side Bedrock Converse ``toolSpec`` wire representation that
  ``ChatBedrockConverse.bind_tools`` sends.

It does not hand-write an approximate schema. If the provider serializer cannot
be imported the harness records ``serializer=unavailable`` and fails rather than
silently substituting an approximation.

Safety
------
* No production database access, no real user chat, no checkpoint read.
* No tool is ever executed. Only schemas are serialized.
* The broad-inventory baseline is serialized in isolation and is never bound to
  a live model.
* Registry content is supplied by a deterministic synthetic fixture
  (``--registry fixture``) or read from the live in-process registry
  (``--registry live``). Fixture mode is offline and clean-room safe.

Cache honesty
-------------
Matching signatures prove *cache eligibility and stability*, not provider cache
usage. This harness never claims a provider cache hit. Provider cache telemetry
is reported only by ``--probe-provider-cache``, and only if the SDK actually
exposes it.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_SCHEMA = "d3b2_tool_profile_cache_benchmark.v1"

# Fixed synthetic identities. Real system prompt text and real model ARNs are
# deliberately NOT embedded; only a stable identity token is needed for a cache
# key, and an ARN is deployment configuration.
SYNTHETIC_SYSTEM_PROMPT_ID = "d3b2_synthetic_system_prompt.v1"
SYNTHETIC_MODEL_ID = "d3b2_synthetic_bedrock_anthropic_role.v1"
SYNTHETIC_MODEL_CONFIG = {"temperature": 0.0, "max_output_tokens": 8192}


# ---------------------------------------------------------------------------
# Token measurement (see D3B2_TOKEN_MEASUREMENT_METHOD.md)
# ---------------------------------------------------------------------------
@dataclass
class TokenMeter:
    name: str
    version: str
    exact: bool
    method: str
    model_mapping: str
    _encode: Callable[[str], int] | None = None

    def count(self, text: str) -> int:
        if self._encode is not None:
            return self._encode(text)
        # Documented fallback: 4 characters per token.
        return (len(text) + 3) // 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer": self.name,
            "tokenizer_version": self.version,
            "token_measurement": "exact" if self.exact else "estimated",
            "method": self.method,
            "model_mapping": self.model_mapping,
        }


def resolve_token_meter(provider: str) -> TokenMeter:
    """Resolve the most authoritative *locally available* tokenizer.

    The active provider is aws-bedrock serving Anthropic Claude models. Anthropic
    does not ship a local tokenizer in the installed SDK, and Bedrock exposes no
    local tokenizer. There is therefore no authoritative tokenizer available
    offline, and any count reported here is an explicitly labelled ESTIMATE.
    """
    authoritative_available = False
    try:  # pragma: no cover - probe only
        import anthropic  # noqa: F401

        authoritative_available = hasattr(getattr(anthropic, "Anthropic", object), "count_tokens")
    except Exception:  # noqa: BLE001
        authoritative_available = False

    if authoritative_available:  # pragma: no cover - not reachable on this build
        import anthropic

        return TokenMeter(
            name="anthropic.count_tokens",
            version=getattr(anthropic, "__version__", "unknown"),
            exact=True,
            method="anthropic sdk local tokenizer",
            model_mapping="anthropic claude via aws-bedrock",
        )

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return TokenMeter(
            name="tiktoken/cl100k_base (PROXY, not authoritative for Claude)",
            version=getattr(importlib.import_module("tiktoken"), "__version__", "unknown"),
            exact=False,
            method="cl100k_base BPE proxy over the serialized toolConfig JSON",
            model_mapping="NO authoritative Claude/Bedrock tokenizer available locally; "
                          "cl100k_base used as a stable relative proxy only",
            _encode=lambda text: len(enc.encode(text)),
        )
    except Exception:  # noqa: BLE001
        return TokenMeter(
            name="chars_div_4_heuristic",
            version="n/a",
            exact=False,
            method="ceil(len(text)/4)",
            model_mapping="no tokenizer available",
        )


# ---------------------------------------------------------------------------
# Provider-side model-facing serialization
# ---------------------------------------------------------------------------
def _provider_serializer() -> tuple[Callable[[Sequence[Any]], list[dict]], str]:
    from langchain_aws.chat_models.bedrock_converse import _format_tools

    return _format_tools, "langchain_aws.chat_models.bedrock_converse._format_tools"


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class SchemaMeasure:
    serializer: str
    tool_count: int
    tool_names: tuple[str, ...]
    schema_bytes: int
    schema_chars: int
    schema_tokens: int
    schema_digest: str
    per_tool: tuple[dict[str, Any], ...] = ()

    def to_dict(self, include_per_tool: bool = False) -> dict[str, Any]:
        out = {
            "serializer": self.serializer,
            "tool_count": self.tool_count,
            "schema_bytes": self.schema_bytes,
            "schema_chars": self.schema_chars,
            "schema_tokens": self.schema_tokens,
            "schema_digest": self.schema_digest,
        }
        if include_per_tool:
            out["per_tool"] = list(self.per_tool)
        return out


def serialize_model_facing(tools: Sequence[Any], meter: TokenMeter) -> SchemaMeasure:
    """Serialize tools exactly as ChatBedrockConverse would put them on the wire."""
    from app.agent import registry_canonical as canonical

    fmt, serializer_id = _provider_serializer()
    specs = fmt(list(tools))
    blob = _dumps(specs)

    per_tool: list[dict[str, Any]] = []
    for spec in specs:
        inner = spec.get("toolSpec", {}) if isinstance(spec, Mapping) else {}
        name = str(inner.get("name") or "")
        desc = inner.get("description") or ""
        schema = inner.get("inputSchema") or {}
        whole = _dumps(spec)
        d_blob = _dumps(desc)
        s_blob = _dumps(schema)
        per_tool.append({
            "name": name,
            "bytes": len(whole.encode("utf-8")),
            "tokens": meter.count(whole),
            "description_bytes": len(d_blob.encode("utf-8")),
            "description_tokens": meter.count(d_blob),
            "args_schema_bytes": len(s_blob.encode("utf-8")),
            "args_schema_tokens": meter.count(s_blob),
        })

    digest = canonical.content_signature(specs)
    return SchemaMeasure(
        serializer=serializer_id,
        tool_count=len(specs),
        tool_names=tuple(str((s.get("toolSpec") or {}).get("name") or "") for s in specs),
        schema_bytes=len(blob.encode("utf-8")),
        schema_chars=len(blob),
        schema_tokens=meter.count(blob),
        schema_digest=digest,
        per_tool=tuple(per_tool),
    )


# ---------------------------------------------------------------------------
# Controlled registry
# ---------------------------------------------------------------------------
_JSON_TYPE_TO_PY = {
    "string": str,
    "integer": int,
    "boolean": bool,
    "number": float,
}


def _resolve_structured_tool() -> Any:
    """Resolve the real ``StructuredTool`` even under test module pollution.

    ``tests/test_dynamic_registry_runtime_v1.py`` deliberately replaces
    ``sys.modules["langchain_core.tools"]`` with a minimal stub exposing only
    ``BaseTool`` and ``tool``.  A later lazy ``from langchain_core.tools import
    StructuredTool`` then fails with "unknown location".  The concrete
    ``langchain_core.tools.structured`` submodule is not replaced, so it is
    resolved first.  This is a benchmark-side robustness fix only; no
    production module and no other test is mutated.
    """
    submodule = sys.modules.get("langchain_core.tools.structured")
    candidate = getattr(submodule, "StructuredTool", None)
    if candidate is not None:
        return candidate
    try:
        from langchain_core.tools.structured import StructuredTool  # type: ignore

        return StructuredTool
    except Exception:  # noqa: BLE001
        pass
    from langchain_core.tools import StructuredTool  # type: ignore

    return StructuredTool


def _tool_from_descriptor(descriptor: Mapping[str, Any]) -> Any:
    StructuredTool = _resolve_structured_tool()
    from pydantic import Field, create_model

    definitions: dict[str, Any] = {}
    for arg in descriptor.get("args") or ():
        name = str(arg["name"])
        kind = str(arg.get("type") or "string")
        if kind == "array_string":
            annotation: Any = list[str]
        else:
            annotation = _JSON_TYPE_TO_PY.get(kind, str)
        desc = str(arg.get("description") or "")
        if arg.get("required"):
            definitions[name] = (annotation, Field(description=desc))
        else:
            default: Any = arg.get("default")
            if default is None:
                default = {"string": "", "integer": 0, "boolean": False,
                           "number": 0.0, "array_string": []}.get(kind, "")
            if kind == "array_string":
                definitions[name] = (annotation, Field(default_factory=list, description=desc))
            else:
                definitions[name] = (annotation, Field(default=default, description=desc))

    name = str(descriptor["name"])
    args_model = create_model("BenchArgs_" + name, **definitions) if definitions else None

    def _never_called(**_kwargs: Any) -> str:
        raise RuntimeError("benchmark tool objects are never executed")

    return StructuredTool.from_function(
        func=_never_called,
        name=name,
        description=str(descriptor.get("description") or ""),
        args_schema=args_model,
    )


def _semantic_material(families: Mapping[str, Any]) -> dict[str, Any]:
    """Semantic content material: names, descriptions, argument shape only.

    Deliberately excludes refresh epoch, loaded_at, health, and source so an
    equivalent refresh cannot change the content signature.
    """
    out: dict[str, Any] = {}
    for family in sorted(families):
        tools = families[family].get("tools") or ()
        out[family] = [
            {
                "name": str(t.get("name") or ""),
                "description": str(t.get("description") or ""),
                "args": [
                    {
                        "name": str(a.get("name") or ""),
                        "type": str(a.get("type") or ""),
                        "required": bool(a.get("required")),
                        "description": str(a.get("description") or ""),
                    }
                    for a in (t.get("args") or ())
                ],
            }
            for t in tools
        ]
    return out


@dataclass
class ControlledRegistry:
    fixture_id: str
    families: dict[str, Any]
    refresh_epoch: int
    loaded_at: str
    content_signature: str = ""
    tools_by_family: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_fixture(cls, fixture: Mapping[str, Any]) -> "ControlledRegistry":
        from app.agent import registry_canonical as canonical

        families = copy.deepcopy(dict(fixture["families"]))
        reg = cls(
            fixture_id=str(fixture.get("fixture_id") or "fixture"),
            families=families,
            refresh_epoch=int(fixture.get("refresh_epoch") or 0),
            loaded_at=str(fixture.get("loaded_at") or ""),
        )
        reg.content_signature = canonical.content_signature(_semantic_material(families))
        reg.tools_by_family = {
            name: [_tool_from_descriptor(t) for t in spec.get("tools") or ()]
            for name, spec in families.items()
        }
        return reg

    def mutated(self, *, fixture_id: str, refresh_epoch: int | None = None,
                loaded_at: str | None = None,
                mutate: Callable[[dict[str, Any]], None] | None = None) -> "ControlledRegistry":
        payload = {
            "fixture_id": fixture_id,
            "refresh_epoch": self.refresh_epoch if refresh_epoch is None else refresh_epoch,
            "loaded_at": self.loaded_at if loaded_at is None else loaded_at,
            "families": copy.deepcopy(self.families),
        }
        if mutate is not None:
            mutate(payload["families"])
        return ControlledRegistry.from_fixture(payload)

    # -- patch installation -------------------------------------------------
    def install(self) -> Callable[[], None]:
        import app.agent.agents.tools as tools_pkg
        import app.agent.agents.tools.registry as registry_mod

        saved: list[tuple[Any, str, Any]] = []

        def patch(module: Any, attr: str, value: Any) -> None:
            saved.append((module, attr, getattr(module, attr, None)))
            setattr(module, attr, value)

        families = self.tools_by_family

        patch(tools_pkg, "get_tools_set", lambda name: list(families.get(name, [])))
        patch(
            tools_pkg,
            "get_tools_set_filtered",
            lambda name, allowed: [t for t in families.get(name, []) if t.name in set(allowed)],
        )
        patch(registry_mod, "get_tool_inventory_signature", lambda: self.content_signature)
        patch(registry_mod, "get_registry_content_signature", lambda: self.content_signature)
        patch(registry_mod, "get_registry_refresh_epoch", lambda: self.refresh_epoch)

        def restore() -> None:
            for module, attr, previous in reversed(saved):
                setattr(module, attr, previous)

        return restore


# ---------------------------------------------------------------------------
# Synthetic thread state (never a real checkpoint)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThreadState:
    active_toolsets: tuple[str, ...] = ()
    requested_extra_tools: tuple[str, ...] = ()
    authorization_flags: tuple[tuple[str, bool], ...] = ()
    previously_eligible: tuple[str, ...] = ()

    @property
    def flags(self) -> dict[str, bool]:
        return {k: bool(v) for k, v in self.authorization_flags}


def apply_authorization_delta(state: ThreadState, message: str) -> tuple[ThreadState, dict[str, str]]:
    """Apply the real D1 current-turn authorization delta parser."""
    from app.agent.tool_policy_state import merge_authorization_state, parse_authorization_delta

    delta = parse_authorization_delta(message)
    merged = merge_authorization_state(state.flags, delta)
    return replace(state, authorization_flags=tuple(sorted(merged.items()))), delta.as_dict()


# ---------------------------------------------------------------------------
# Cache-key identity (see D3B2_CACHE_KEY_INPUTS.md)
# ---------------------------------------------------------------------------
def cache_key_digest(*, profile: Any, schema_digest: str, registry_content_signature: str) -> str:
    from app.agent import registry_canonical as canonical

    material = {
        "schema": "d3b2_cache_key.v1",
        "system_prompt_identity": SYNTHETIC_SYSTEM_PROMPT_ID,
        "model_identity": SYNTHETIC_MODEL_ID,
        "model_configuration": dict(SYNTHETIC_MODEL_CONFIG),
        "methodology": profile.methodology,
        "active_toolsets": list(profile.active_toolsets),
        "eligible_extra_tools": list(profile.eligible_extra_tools),
        "authorization_flags": dict(profile.authorization_flags),
        "registry_content_signature": registry_content_signature,
        "tool_profile_signature": profile.signature,
        "model_facing_schema_digest": schema_digest,
    }
    return canonical.content_signature(material)


# ---------------------------------------------------------------------------
# One measured turn
# ---------------------------------------------------------------------------
def measure_turn(
    *,
    scenario_id: str,
    description: str,
    prompt: str,
    state: ThreadState,
    registry: ControlledRegistry,
    meter: TokenMeter,
    has_prior_tool_results: bool = False,
    requested_extra_tools: Sequence[str] = (),
    keep_per_tool: bool = False,
) -> dict[str, Any]:
    from app.agent.tool_execution_policy import build_profile_for_prompt, get_tools_for_toolsets
    from app.agent.tool_policy_state import classify_activation_state, resolve_upstream_availability

    profile = build_profile_for_prompt(
        prompt,
        has_prior_tool_results=has_prior_tool_results,
        prior_active_toolsets=state.active_toolsets,
        prior_extra_tools=state.requested_extra_tools,
        authorization_flags=state.flags,
        requested_extra_tools=list(requested_extra_tools),
    )
    tools = get_tools_for_toolsets(
        profile.active_toolsets,
        curated_tools_by_toolset=profile.curated_tools_by_toolset,
        model_facing=True,
        authorization_flags=dict(profile.authorization_flags or {}),
    )
    measure = serialize_model_facing(tools, meter)
    bound_names = set(measure.tool_names)

    available, unavailable = resolve_upstream_availability(list(profile.requested_extra_tools))
    statuses = {
        extra: classify_activation_state(
            extra,
            available=available,
            eligible=profile.eligible_extra_tools,
            pending=profile.pending_authorization_extra_tools,
            bound_tool_names=bound_names,
            previously_eligible=state.previously_eligible,
        )
        for extra in profile.requested_extra_tools
    }

    next_state = ThreadState(
        active_toolsets=tuple(profile.active_toolsets),
        requested_extra_tools=tuple(profile.requested_extra_tools),
        authorization_flags=tuple(sorted(dict(profile.authorization_flags).items())),
        previously_eligible=tuple(sorted(set(state.previously_eligible) | set(profile.eligible_extra_tools))),
    )

    record = {
        "scenario_id": scenario_id,
        "description": description,
        "registry_fixture_id": registry.fixture_id,
        "registry_content_signature": registry.content_signature,
        "registry_refresh_epoch": registry.refresh_epoch,
        "methodology": profile.methodology,
        "active_toolsets": list(profile.active_toolsets),
        "requested_extra_tools": list(profile.requested_extra_tools),
        "eligible_extra_tools": list(profile.eligible_extra_tools),
        "pending_authorization_extra_tools": list(profile.pending_authorization_extra_tools),
        "unavailable_upstream_extra_tools": list(unavailable),
        "activation_status": statuses,
        "authorization_flags": dict(profile.authorization_flags),
        "curated_tools_by_toolset": {k: list(v) for k, v in sorted(profile.curated_tools_by_toolset.items())},
        "bound_tool_names": list(measure.tool_names),
        "bound_tool_count": measure.tool_count,
        "profile_signature": profile.signature,
        "inventory_signature": profile.inventory_signature,
        "model_schema": measure.to_dict(include_per_tool=keep_per_tool),
        "cache_key_digest": cache_key_digest(
            profile=profile,
            schema_digest=measure.schema_digest,
            registry_content_signature=registry.content_signature,
        ),
    }
    return {"record": record, "state": next_state, "measure": measure, "profile": profile}


def diff_records(prior: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "methodology", "active_toolsets", "requested_extra_tools", "eligible_extra_tools",
        "pending_authorization_extra_tools", "authorization_flags", "bound_tool_names",
        "bound_tool_count", "profile_signature", "registry_content_signature",
        "cache_key_digest",
    )
    changed = {k: {"prior": prior.get(k), "current": current.get(k)}
               for k in keys if prior.get(k) != current.get(k)}
    ps, cs = prior.get("model_schema") or {}, current.get("model_schema") or {}
    return {
        "changed_fields": sorted(changed),
        "changed_detail": changed,
        "schema_digest_changed": ps.get("schema_digest") != cs.get("schema_digest"),
        "delta_bytes": int(cs.get("schema_bytes", 0)) - int(ps.get("schema_bytes", 0)),
        "delta_chars": int(cs.get("schema_chars", 0)) - int(ps.get("schema_chars", 0)),
        "delta_tokens": int(cs.get("schema_tokens", 0)) - int(ps.get("schema_tokens", 0)),
        "delta_tool_count": int(cs.get("tool_count", 0)) - int(ps.get("tool_count", 0)),
        "added_tools": sorted(set(current.get("bound_tool_names") or ()) - set(prior.get("bound_tool_names") or ())),
        "removed_tools": sorted(set(prior.get("bound_tool_names") or ()) - set(current.get("bound_tool_names") or ())),
    }


# ---------------------------------------------------------------------------
# Broad-inventory baseline (serialized in isolation; never bound to a model)
# ---------------------------------------------------------------------------
ALL_AUTHORIZED = {
    "operator_authorized": True,
    "heavy_tools_authorized": True,
    "persistence_authorized": True,
    "mutation_authorized": True,
}


def measure_broad_inventory(registry: ControlledRegistry, meter: TokenMeter) -> dict[str, Any]:
    """Serialize the full inventory of every family as a comparison baseline.

    This is an ISOLATED serialization only. The result is never handed to
    ``bind_tools`` against a live production model.
    """
    from app.agent.tool_execution_policy import BROAD_TOOLSETS, get_tools_for_toolsets

    families = [f for f in BROAD_TOOLSETS if f in registry.tools_by_family]
    extra = sorted(set(registry.tools_by_family) - set(BROAD_TOOLSETS))
    all_families = list(families) + extra

    privileged = get_tools_for_toolsets(
        all_families, model_facing=True, authorization_flags=dict(ALL_AUTHORIZED)
    )
    gated = get_tools_for_toolsets(
        all_families, model_facing=True, authorization_flags={}
    )
    m_priv = serialize_model_facing(privileged, meter)
    m_gated = serialize_model_facing(gated, meter)
    return {
        "families_considered": all_families,
        "families_present": [f for f in all_families if registry.tools_by_family.get(f)],
        "broad_all_tools": m_priv.to_dict(include_per_tool=True),
        "broad_default_gated": m_gated.to_dict(),
        "note": "broad_all_tools is the true bind-everything baseline (privileged "
                "code-execution tools included). Serialized in isolation only.",
    }


def contributor_report(measure_per_tool: Sequence[Mapping[str, Any]],
                       registry: ControlledRegistry, top_n: int = 15) -> dict[str, Any]:
    owner: dict[str, str] = {}
    for family, tools in registry.tools_by_family.items():
        for tool in tools:
            owner.setdefault(getattr(tool, "name", ""), family)

    by_family: dict[str, dict[str, int]] = {}
    for entry in measure_per_tool:
        fam = owner.get(entry["name"], "unknown")
        acc = by_family.setdefault(fam, {"tools": 0, "bytes": 0, "tokens": 0,
                                         "description_bytes": 0, "args_schema_bytes": 0})
        acc["tools"] += 1
        acc["bytes"] += int(entry["bytes"])
        acc["tokens"] += int(entry["tokens"])
        acc["description_bytes"] += int(entry["description_bytes"])
        acc["args_schema_bytes"] += int(entry["args_schema_bytes"])

    total_bytes = sum(int(e["bytes"]) for e in measure_per_tool) or 1
    total_desc = sum(int(e["description_bytes"]) for e in measure_per_tool)
    total_args = sum(int(e["args_schema_bytes"]) for e in measure_per_tool)

    fams = sorted(by_family.items(), key=lambda kv: -kv[1]["bytes"])
    tools = sorted(measure_per_tool, key=lambda e: -int(e["bytes"]))
    return {
        "total_bytes": total_bytes,
        "description_share_pct": round(100.0 * total_desc / total_bytes, 2),
        "args_schema_share_pct": round(100.0 * total_args / total_bytes, 2),
        "top_families": [
            {"family": name, **vals, "pct_of_bytes": round(100.0 * vals["bytes"] / total_bytes, 2)}
            for name, vals in fams[:top_n]
        ],
        "top_tools": [
            {"name": e["name"], "family": owner.get(e["name"], "unknown"),
             "bytes": e["bytes"], "tokens": e["tokens"],
             "description_bytes": e["description_bytes"],
             "args_schema_bytes": e["args_schema_bytes"],
             "pct_of_bytes": round(100.0 * int(e["bytes"]) / total_bytes, 2)}
            for e in tools[:top_n]
        ],
    }


# ---------------------------------------------------------------------------
# Provider cache telemetry probe (capability only; never fabricated)
# ---------------------------------------------------------------------------
def probe_provider_cache_telemetry() -> dict[str, Any]:
    findings: dict[str, Any] = {
        "schema": "d3b2_provider_cache_telemetry.v1",
        "telemetry_captured": False,
        "provider_cache_telemetry": "unavailable",
        "reason": "",
        "sdk_capability": {},
    }
    try:
        import inspect

        from langchain_aws.chat_models import bedrock_converse as bc

        src = inspect.getsource(bc)
        findings["sdk_capability"] = {
            "supports_cache_point_blocks": "_is_cache_point" in src or "cachePoint" in src,
            "mentions_cache_read_input_tokens": "cacheReadInputTokens" in src
                                                or "cache_read_input_tokens" in src,
            "mentions_cache_write_input_tokens": "cacheWriteInputTokens" in src
                                                 or "cache_creation_input_tokens" in src,
        }
    except Exception as exc:  # noqa: BLE001
        findings["sdk_capability"] = {"probe_error": type(exc).__name__}

    findings["reason"] = (
        "Capturing provider prompt-cache creation/read token counts requires a real "
        "Bedrock InvokeModel/Converse call and the returned usage metadata. D3B2 "
        "performs no billed live inference call for benchmarking, so no provider "
        "cache hit/miss counter was observed. Signature and schema-digest equality "
        "proves cache ELIGIBILITY and STABILITY only, never a provider cache hit."
    )
    return findings


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------
def load_fixture() -> tuple[dict[str, Any], str]:
    """Build the committed deterministic fixture in memory and verify its digest."""
    import hashlib

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmarks"))
    from generate_tool_profile_fixture import build_fixture  # type: ignore

    payload = build_fixture()
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return payload, digest


def fixture_lock_path() -> Path:
    return REPO_ROOT / "tests" / "fixtures" / "tool_profile_benchmark" / "registry_baseline.sha256"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
GENERIC_PROMPT_1 = "Give me a one-sentence greeting."
GENERIC_PROMPT_2 = "Thanks. Give me another short greeting."
S3_PROMPT_1 = (
    "Investigate the receiver reboot and DVR playback problem on this set-top box "
    "using the receiver log evidence that is already available. Read-only."
)
S3_PROMPT_2 = (
    "Continue the same receiver-log investigation using the currently available "
    "read-only evidence."
)
HEAVY_EXACT = "s3_stb_logs:build_log_capsule"
HEAVY_PERSIST_EXACT = "s3_stb_logs:build_complete_log_capsule"
SCENE_EXACT = "s3_stb_logs:build_incident_scene"


def run_scenarios(registry: ControlledRegistry, meter: TokenMeter) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    checks: dict[str, Any] = {}

    fresh = ThreadState(authorization_flags=tuple(sorted(
        {k: False for k in ("operator_authorized", "heavy_tools_authorized",
                            "persistence_authorized", "mutation_authorized")}.items())))

    # --- S01 / S02: generic stable core ---------------------------------
    s01 = measure_turn(scenario_id="S01_generic_initial",
                       description="Fresh synthetic thread, all authorization false, generic greeting.",
                       prompt=GENERIC_PROMPT_1, state=fresh, registry=registry, meter=meter,
                       keep_per_tool=True)
    scenarios["S01_generic_initial"] = s01["record"]

    s02 = measure_turn(scenario_id="S02_generic_followup",
                       description="Checkpointed state from S01, second generic greeting.",
                       prompt=GENERIC_PROMPT_2, state=s01["state"], registry=registry, meter=meter)
    scenarios["S02_generic_followup"] = s02["record"]
    d12 = diff_records(s01["record"], s02["record"])
    checks["S02_generic_followup_stable"] = {
        "profile_signature_stable": s01["record"]["profile_signature"] == s02["record"]["profile_signature"],
        "schema_digest_stable": not d12["schema_digest_changed"],
        "tool_count_stable": d12["delta_tool_count"] == 0,
        "cache_key_stable": s01["record"]["cache_key_digest"] == s02["record"]["cache_key_digest"],
        "diff": d12,
    }

    # --- S03 / S04: S3 read-only investigation ---------------------------
    s03 = measure_turn(scenario_id="S03_s3_initial",
                       description="Dedicated synthetic state; receiver-log investigation, no heavy authorization.",
                       prompt=S3_PROMPT_1, state=fresh, registry=registry, meter=meter,
                       keep_per_tool=True)
    scenarios["S03_s3_initial"] = s03["record"]

    s04 = measure_turn(scenario_id="S04_s3_followup",
                       description="Same checkpointed state; explicit read-only continuation.",
                       prompt=S3_PROMPT_2, state=s03["state"], registry=registry, meter=meter)
    scenarios["S04_s3_followup"] = s04["record"]
    d34 = diff_records(s03["record"], s04["record"])

    s04b = measure_turn(scenario_id="S04b_s3_followup_with_prior_tool_results",
                        description="Same continuation but has_prior_tool_results=True, which "
                                    "intentionally activates the documented follow-up family expansion.",
                        prompt=S3_PROMPT_2, state=s03["state"], registry=registry, meter=meter,
                        has_prior_tool_results=True)
    scenarios["S04b_s3_followup_with_prior_tool_results"] = s04b["record"]
    d34b = diff_records(s03["record"], s04b["record"])
    checks["S04_s3_followup_stable"] = {
        "profile_signature_stable": s03["record"]["profile_signature"] == s04["record"]["profile_signature"],
        "schema_digest_stable": not d34["schema_digest_changed"],
        "cache_key_stable": s03["record"]["cache_key_digest"] == s04["record"]["cache_key_digest"],
        "no_family_oscillation": set(s03["record"]["active_toolsets"]).issubset(
            set(s04["record"]["active_toolsets"])),
        "diff": d34,
        "followup_expansion_variant": {
            "additive_only": set(s03["record"]["active_toolsets"]).issubset(
                set(s04b["record"]["active_toolsets"])),
            "removed_tools": d34b["removed_tools"],
            "added_families": sorted(set(s04b["record"]["active_toolsets"])
                                     - set(s03["record"]["active_toolsets"])),
            "diff": d34b,
        },
    }

    # --- S04c / S04d: characterize the generic-fallback churn -------------
    # S04c: a SECOND consecutive generic-fallback continuation. If the churn is
    # one-time (converging) the profile and schema must now be stable.
    s04c = measure_turn(scenario_id="S04c_s3_followup_second_continuation",
                        description="Second consecutive read-only continuation from the S04 state; "
                                    "tests whether generic-fallback family injection converges.",
                        prompt=S3_PROMPT_2, state=s04["state"], registry=registry, meter=meter)
    scenarios["S04c_s3_followup_second_continuation"] = s04c["record"]
    d44c = diff_records(s04["record"], s04c["record"])

    # S04d: the same continuation intent, but worded so methodology selection
    # still resolves the receiver-log methodology.
    s04d = measure_turn(scenario_id="S04d_s3_followup_domain_worded",
                        description="Read-only continuation worded so the receiver-log methodology "
                                    "is re-selected instead of falling back to generic.",
                        prompt="Continue the receiver reboot and DVR playback log investigation on "
                               "this set-top box using the read-only evidence already available.",
                        state=s03["state"], registry=registry, meter=meter)
    scenarios["S04d_s3_followup_domain_worded"] = s04d["record"]
    d34d = diff_records(s03["record"], s04d["record"])
    checks["S04_generic_fallback_churn_characterization"] = {
        "finding": "The prescribed read-only continuation prompt re-selects the generic "
                   "fallback methodology, whose default family set contributes dish_internal "
                   "on top of an established domain profile. The profile stays additive and "
                   "never oscillates, but the model-facing schema changes once.",
        "second_continuation_converged": (
            s04["record"]["profile_signature"] == s04c["record"]["profile_signature"]
            and not d44c["schema_digest_changed"]
            and d44c["delta_bytes"] == 0
        ),
        "second_continuation_diff_bytes": d44c["delta_bytes"],
        "second_continuation_added_tools": d44c["added_tools"],
        "domain_worded_profile_stable":
            s03["record"]["profile_signature"] == s04d["record"]["profile_signature"],
        "domain_worded_schema_stable": not d34d["schema_digest_changed"],
        "domain_worded_delta_bytes": d34d["delta_bytes"],
        "monotone_never_removes": d34["removed_tools"] == [] and d44c["removed_tools"] == [],
        "root_cause": "app.agent.tool_execution_policy.GENERIC_INITIAL_TOOLSETS contains "
                      "dish_internal, and build_tool_profile unions the current-turn "
                      "methodology family set with prior_active_toolsets.",
    }

    # --- S05: pending heavy exact, no authorization ----------------------
    s05 = measure_turn(scenario_id="S05_pending_heavy_exact",
                       description="Exact heavy tool requested with heavy=false and persistence=false.",
                       prompt=S3_PROMPT_2, state=s03["state"], registry=registry, meter=meter,
                       requested_extra_tools=[HEAVY_EXACT])
    scenarios["S05_pending_heavy_exact"] = s05["record"]
    d45 = diff_records(s04["record"], s05["record"])
    checks["S05_pending_adds_no_schema"] = {
        "status": s05["record"]["activation_status"].get(HEAVY_EXACT),
        "request_retained": HEAVY_EXACT in s05["record"]["requested_extra_tools"],
        "not_eligible": HEAVY_EXACT not in s05["record"]["eligible_extra_tools"],
        "not_model_bound": "build_log_capsule" not in s05["record"]["bound_tool_names"],
        "delta_bytes": d45["delta_bytes"],
        "delta_tokens": d45["delta_tokens"],
        "delta_tool_count": d45["delta_tool_count"],
        "zero_schema_delta": d45["delta_bytes"] == 0 and d45["delta_tokens"] == 0,
        "profile_signature_changed": s04["record"]["profile_signature"] != s05["record"]["profile_signature"],
    }

    # --- S06: later authorization ----------------------------------------
    auth_state, delta = apply_authorization_delta(s05["state"], "Heavy tools authorized.")
    s06 = measure_turn(scenario_id="S06_heavy_authorized",
                       description="Heavy authorization granted on a later turn; tool NOT executed.",
                       prompt=S3_PROMPT_2, state=auth_state, registry=registry, meter=meter)
    scenarios["S06_heavy_authorized"] = s06["record"]
    d56 = diff_records(s05["record"], s06["record"])
    checks["S06_authorization_adds_only_necessary"] = {
        "authorization_delta": delta,
        "status": s06["record"]["activation_status"].get(HEAVY_EXACT),
        "eligible": HEAVY_EXACT in s06["record"]["eligible_extra_tools"],
        "no_repeat_request_needed": HEAVY_EXACT in s06["record"]["requested_extra_tools"],
        "added_tools": d56["added_tools"],
        "removed_tools": d56["removed_tools"],
        "delta_bytes": d56["delta_bytes"],
        "delta_tokens": d56["delta_tokens"],
        "delta_tool_count": d56["delta_tool_count"],
        "only_requested_tool_added": d56["added_tools"] == ["build_log_capsule"],
        "profile_signature_changed": s05["record"]["profile_signature"] != s06["record"]["profile_signature"],
    }

    # --- S06b/S06c: staged two-requirement transition --------------------
    st_b, _ = apply_authorization_delta(fresh, "Heavy tools authorized.")
    s06b = measure_turn(scenario_id="S06b_two_requirement_heavy_only",
                        description="build_complete_log_capsule requires heavy AND persistence; heavy only.",
                        prompt=S3_PROMPT_1, state=st_b, registry=registry, meter=meter,
                        requested_extra_tools=[HEAVY_PERSIST_EXACT])
    scenarios["S06b_two_requirement_heavy_only"] = s06b["record"]
    st_c, _ = apply_authorization_delta(s06b["state"], "Persistence authorized.")
    s06c = measure_turn(scenario_id="S06c_two_requirement_heavy_plus_persistence",
                        description="Both requirements satisfied; tool becomes eligible. Not executed.",
                        prompt=S3_PROMPT_1, state=st_c, registry=registry, meter=meter)
    scenarios["S06c_two_requirement_heavy_plus_persistence"] = s06c["record"]
    d_bc = diff_records(s06b["record"], s06c["record"])
    checks["S06_staged_two_requirement_transition"] = {
        "heavy_only_status": s06b["record"]["activation_status"].get(HEAVY_PERSIST_EXACT),
        "heavy_only_bound": "build_complete_log_capsule" in s06b["record"]["bound_tool_names"],
        "both_status": s06c["record"]["activation_status"].get(HEAVY_PERSIST_EXACT),
        "both_bound": "build_complete_log_capsule" in s06c["record"]["bound_tool_names"],
        "added_tools": d_bc["added_tools"],
        "delta_bytes": d_bc["delta_bytes"],
        "delta_tokens": d_bc["delta_tokens"],
    }

    # --- S07: revocation --------------------------------------------------
    rev_state, rev_delta = apply_authorization_delta(s06["state"], "Heavy tool authorization is revoked.")
    s07 = measure_turn(scenario_id="S07_revocation",
                       description="Heavy authorization explicitly revoked.",
                       prompt=S3_PROMPT_2, state=rev_state, registry=registry, meter=meter)
    scenarios["S07_revocation"] = s07["record"]
    d67 = diff_records(s06["record"], s07["record"])
    checks["S07_revocation_removes_schema"] = {
        "authorization_delta": rev_delta,
        "status": s07["record"]["activation_status"].get(HEAVY_EXACT),
        "request_retained": HEAVY_EXACT in s07["record"]["requested_extra_tools"],
        "not_eligible": HEAVY_EXACT not in s07["record"]["eligible_extra_tools"],
        "not_bound": "build_log_capsule" not in s07["record"]["bound_tool_names"],
        "removed_tools": d67["removed_tools"],
        "removal_bytes": -d67["delta_bytes"],
        "removal_tokens": -d67["delta_tokens"],
        "profile_signature_changed": s06["record"]["profile_signature"] != s07["record"]["profile_signature"],
        "returns_to_pending_baseline_digest":
            s07["record"]["model_schema"]["schema_digest"] == s05["record"]["model_schema"]["schema_digest"],
    }

    # --- S08: unavailable Incident Scene tool -----------------------------
    s08 = measure_turn(scenario_id="S08_unavailable_incident_scene",
                       description="build_incident_scene requested; absent from the current S3 runtime.",
                       prompt=S3_PROMPT_2, state=s04["state"], registry=registry, meter=meter,
                       requested_extra_tools=[SCENE_EXACT])
    scenarios["S08_unavailable_incident_scene"] = s08["record"]
    d48 = diff_records(s04["record"], s08["record"])
    auth_scene_state, _ = apply_authorization_delta(s08["state"], "Heavy tools authorized.")
    s08b = measure_turn(scenario_id="S08b_unavailable_even_when_authorized",
                        description="Same absent tool with heavy authorization granted.",
                        prompt=S3_PROMPT_2, state=auth_scene_state, registry=registry, meter=meter)
    scenarios["S08b_unavailable_even_when_authorized"] = s08b["record"]
    checks["S08_unavailable_adds_no_schema"] = {
        "status": s08["record"]["activation_status"].get(SCENE_EXACT),
        "status_when_authorized": s08b["record"]["activation_status"].get(SCENE_EXACT),
        "not_pending": SCENE_EXACT not in s08["record"]["pending_authorization_extra_tools"]
                       or s08["record"]["activation_status"].get(SCENE_EXACT) == "UNAVAILABLE_UPSTREAM",
        "reported_unavailable": SCENE_EXACT in s08["record"]["unavailable_upstream_extra_tools"],
        "not_bound": "build_incident_scene" not in s08["record"]["bound_tool_names"],
        "not_bound_when_authorized": "build_incident_scene" not in s08b["record"]["bound_tool_names"],
        "delta_bytes": d48["delta_bytes"],
        "delta_tokens": d48["delta_tokens"],
        "zero_schema_delta": d48["delta_bytes"] == 0 and d48["delta_tokens"] == 0,
        "profile_signature_changed": s04["record"]["profile_signature"] != s08["record"]["profile_signature"],
    }
    return {"scenarios": scenarios, "checks": checks,
            "states": {"s03": s03, "s04": s04, "s01": s01}}


# ---------------------------------------------------------------------------
# S09 / S10: registry refresh and semantic change
# ---------------------------------------------------------------------------
def _remove_tool(families: dict[str, Any], family: str, name: str) -> None:
    families[family]["tools"] = [t for t in families[family]["tools"] if t["name"] != name]


def _add_tool(families: dict[str, Any], family: str, name: str) -> None:
    families[family]["tools"] = list(families[family]["tools"]) + [{
        "name": name,
        "description": "Read-only benchmark tool added to exercise semantic invalidation.",
        "args": [{"name": "primary_id", "type": "string", "required": True,
                  "description": "Bounded input."}],
    }]


def _change_schema(families: dict[str, Any], family: str, name: str) -> None:
    for tool in families[family]["tools"]:
        if tool["name"] == name:
            tool["args"] = list(tool["args"]) + [{
                "name": "added_bound", "type": "integer", "required": False,
                "description": "Newly added bounded argument.",
            }]


def _change_capability_metadata(families: dict[str, Any], family: str, name: str) -> None:
    """Registry-side capability change: add a server-controlled authorization arg."""
    for tool in families[family]["tools"]:
        if tool["name"] == name:
            tool["args"] = list(tool["args"]) + [{
                "name": "allow_heavy", "type": "boolean", "required": False,
                "description": "Server-controlled authorization switch.",
            }]


def run_registry_scenarios(baseline: ControlledRegistry, meter: TokenMeter) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    checks: dict[str, Any] = {}

    fresh = ThreadState(authorization_flags=tuple(sorted(
        {k: False for k in ("operator_authorized", "heavy_tools_authorized",
                            "persistence_authorized", "mutation_authorized")}.items())))

    def measure_pair(registry: ControlledRegistry, suffix: str) -> dict[str, Any]:
        restore = registry.install()
        try:
            g = measure_turn(scenario_id="generic" + suffix, description="generic profile",
                             prompt=GENERIC_PROMPT_1, state=fresh, registry=registry, meter=meter)
            s = measure_turn(scenario_id="s3" + suffix, description="s3 read-only profile",
                             prompt=S3_PROMPT_1, state=fresh, registry=registry, meter=meter)
        finally:
            restore()
        return {"generic": g["record"], "s3": s["record"]}

    base = measure_pair(baseline, "_baseline")
    scenarios["S09_baseline"] = base

    # --- S09: equivalent refresh (epoch + loaded_at only) ----------------
    equivalent = baseline.mutated(fixture_id="equivalent_refresh",
                                  refresh_epoch=baseline.refresh_epoch + 7,
                                  loaded_at="2026-01-02T03:04:05Z")
    eq = measure_pair(equivalent, "_equivalent_refresh")
    scenarios["S09_equivalent_refresh"] = eq
    checks["S09_equivalent_refresh_no_churn"] = {
        "refresh_epoch_differs": equivalent.refresh_epoch != baseline.refresh_epoch,
        "loaded_at_differs": equivalent.loaded_at != baseline.loaded_at,
        "content_signature_identical": equivalent.content_signature == baseline.content_signature,
        "generic_profile_signature_stable":
            eq["generic"]["profile_signature"] == base["generic"]["profile_signature"],
        "generic_schema_digest_stable":
            eq["generic"]["model_schema"]["schema_digest"] == base["generic"]["model_schema"]["schema_digest"],
        "generic_cache_key_stable": eq["generic"]["cache_key_digest"] == base["generic"]["cache_key_digest"],
        "s3_profile_signature_stable":
            eq["s3"]["profile_signature"] == base["s3"]["profile_signature"],
        "s3_schema_digest_stable":
            eq["s3"]["model_schema"]["schema_digest"] == base["s3"]["model_schema"]["schema_digest"],
        "s3_cache_key_stable": eq["s3"]["cache_key_digest"] == base["s3"]["cache_key_digest"],
    }

    # --- S10: semantic changes -------------------------------------------
    mutations = {
        "tool_added_unrelated_family": lambda f: _add_tool(f, "qos_mcp", "qos_benchmark_added_tool"),
        "tool_removed_relevant_family": lambda f: _remove_tool(f, "s3_stb_logs", "list_files"),
        "argument_schema_changed_relevant": lambda f: _change_schema(f, "s3_stb_logs", "search_logs"),
        "capability_metadata_changed_relevant":
            lambda f: _change_capability_metadata(f, "s3_stb_logs", "get_summary"),
    }
    per_change: dict[str, Any] = {}
    for label, mutate in mutations.items():
        changed = baseline.mutated(fixture_id="semantic_" + label, mutate=mutate)
        got = measure_pair(changed, "_" + label)
        per_change[label] = {
            "content_signature_changed": changed.content_signature != baseline.content_signature,
            "generic_bound_names_changed":
                got["generic"]["bound_tool_names"] != base["generic"]["bound_tool_names"],
            "generic_schema_digest_changed":
                got["generic"]["model_schema"]["schema_digest"]
                != base["generic"]["model_schema"]["schema_digest"],
            "generic_profile_signature_changed":
                got["generic"]["profile_signature"] != base["generic"]["profile_signature"],
            "s3_bound_names_changed":
                got["s3"]["bound_tool_names"] != base["s3"]["bound_tool_names"],
            "s3_schema_digest_changed":
                got["s3"]["model_schema"]["schema_digest"]
                != base["s3"]["model_schema"]["schema_digest"],
            "s3_profile_signature_changed":
                got["s3"]["profile_signature"] != base["s3"]["profile_signature"],
            "s3_schema_delta_bytes": int(got["s3"]["model_schema"]["schema_bytes"])
                                     - int(base["s3"]["model_schema"]["schema_bytes"]),
            "s3_bound_tool_count": got["s3"]["bound_tool_count"],
        }
        scenarios["S10_" + label] = got
    checks["S10_semantic_change_invalidation"] = {
        "per_change": per_change,
        "every_semantic_change_invalidated_content_signature":
            all(v["content_signature_changed"] for v in per_change.values()),
        "unrelated_change_left_generic_binding_unchanged":
            per_change["tool_added_unrelated_family"]["generic_bound_names_changed"] is False
            and per_change["tool_added_unrelated_family"]["generic_schema_digest_changed"] is False,
        "relevant_change_altered_s3_surface":
            per_change["tool_removed_relevant_family"]["s3_schema_digest_changed"] is True,
        "generic_model_schema_digest_stable_for_every_change":
            all(v["generic_schema_digest_changed"] is False for v in per_change.values()),
        "generic_profile_signature_invalidated_by_every_change":
            all(v["generic_profile_signature_changed"] is True for v in per_change.values()),
        "coarse_invalidation_note":
            "registry content signature participates in profile_signature via "
            "ToolProfile.inventory_signature, so any semantic registry change "
            "invalidates every profile signature. Model-facing schema digests remain "
            "stable when the binding is unaffected, so provider prompt-prefix cache "
            "eligibility is preserved; only application-level profile identity is "
            "invalidated more coarsely than strictly necessary.",
    }
    return {"scenarios": scenarios, "checks": checks}


# ---------------------------------------------------------------------------
# S11: restart / fresh-process determinism
# ---------------------------------------------------------------------------
def emit_restart_signature() -> dict[str, Any]:
    """Emit signatures from a completely fresh interpreter (restart analogue)."""
    fixture, digest = load_fixture()
    meter = resolve_token_meter("aws-bedrock")
    registry = ControlledRegistry.from_fixture(fixture)
    restore = registry.install()
    try:
        fresh = ThreadState(authorization_flags=tuple(sorted(
            {k: False for k in ("operator_authorized", "heavy_tools_authorized",
                                "persistence_authorized", "mutation_authorized")}.items())))
        g = measure_turn(scenario_id="restart_generic", description="restart probe",
                         prompt=GENERIC_PROMPT_1, state=fresh, registry=registry, meter=meter)
        s = measure_turn(scenario_id="restart_s3", description="restart probe",
                         prompt=S3_PROMPT_1, state=fresh, registry=registry, meter=meter)
    finally:
        restore()
    return {
        "fixture_digest": digest,
        "registry_content_signature": registry.content_signature,
        "generic_profile_signature": g["record"]["profile_signature"],
        "generic_schema_digest": g["record"]["model_schema"]["schema_digest"],
        "generic_cache_key_digest": g["record"]["cache_key_digest"],
        "s3_profile_signature": s["record"]["profile_signature"],
        "s3_schema_digest": s["record"]["model_schema"]["schema_digest"],
        "s3_cache_key_digest": s["record"]["cache_key_digest"],
        "pid": os.getpid(),
        "python": platform.python_version(),
    }


def run_restart_scenario(restarts: int = 3) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for _ in range(restarts):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--emit-restart-signature"],
            capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        if proc.returncode != 0:
            return {"status": "fail", "error": proc.stderr[-2000:]}
        runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    keys = [k for k in runs[0] if k not in {"pid", "python"}]
    stable = {k: len({json.dumps(r[k], sort_keys=True) for r in runs}) == 1 for k in keys}
    return {
        "status": "pass" if all(stable.values()) else "fail",
        "restarts": restarts,
        "distinct_pids": sorted({r["pid"] for r in runs}),
        "stability": stable,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# S12: child read-only narrowing
# ---------------------------------------------------------------------------
def run_child_scenario(registry: ControlledRegistry, meter: TokenMeter) -> dict[str, Any]:
    from app.agent_mode.child_tool_policy import (
        ChildToolPolicy,
        child_safe_tools,
        narrow_policy_for_task,
        recursive_spawn_tool_names,
    )
    from app.agent.tool_execution_policy import get_tools_for_toolsets
    from app.agent.tool_profiles import build_tool_profile, is_code_execution_tool

    restore = registry.install()
    try:
        parent_flags = {
            "operator_authorized": False,
            "heavy_tools_authorized": True,
            "persistence_authorized": False,
            "mutation_authorized": False,
        }
        parent_profile = build_tool_profile(
            methodology="receiver_reboot_dvr_playback",
            methodology_toolsets=("s3_stb_logs", "rtr_alerts_mcp", "qos_mcp"),
            authorization_flags=parent_flags,
            requested_extra_tools=[HEAVY_EXACT],
            inventory_signature=registry.content_signature,
        )
        parent_tools = get_tools_for_toolsets(
            parent_profile.active_toolsets,
            curated_tools_by_toolset=parent_profile.curated_tools_by_toolset,
            model_facing=True,
            authorization_flags=dict(parent_profile.authorization_flags),
        )
        parent_measure = serialize_model_facing(parent_tools, meter)

        policy = ChildToolPolicy(
            authorization_flags=tuple(sorted(parent_flags.items())),
            eligible_toolsets=tuple(parent_profile.active_toolsets),
            eligible_extra_tools=tuple(parent_profile.eligible_extra_tools),
            registry_content_signature=registry.content_signature,
            registry_generation=registry.content_signature,
            registry_refresh_epoch=registry.refresh_epoch,
            profile_signature=parent_profile.signature,
            child_run_id="d3b2-bench-child",
        )
        narrowed = narrow_policy_for_task(policy, task_required_toolsets=("s3_stb_logs",))
        child_tools = child_safe_tools(parent_tools, narrowed)
        child_measure = serialize_model_facing(child_tools, meter)

        # Elevation attempt: a task asking for a family the parent never permitted.
        elevated = narrow_policy_for_task(
            policy, task_required_toolsets=("qos_mcp", "dish_code_tools", "agent_mode"))
        elevation_families = list(elevated.eligible_toolsets)

        spawn = recursive_spawn_tool_names()
        child_names = set(child_measure.tool_names)
        return {
            "parent": {
                "active_toolsets": list(parent_profile.active_toolsets),
                "profile_signature": parent_profile.signature,
                "eligible_extra_tools": list(parent_profile.eligible_extra_tools),
                "authorization_flags": dict(parent_profile.authorization_flags),
                "model_schema": parent_measure.to_dict(),
                "bound_tool_names": list(parent_measure.tool_names),
            },
            "child": {
                "task_required_toolsets": ["s3_stb_logs"],
                "permitted_toolsets": list(narrowed.eligible_toolsets),
                "permitted_extra_tools": list(narrowed.eligible_extra_tools),
                "profile_signature": narrowed.profile_signature,
                "snapshot_status": narrowed.snapshot_status,
                "authorization_flags": narrowed.flags,
                "model_schema": child_measure.to_dict(),
                "bound_tool_names": list(child_measure.tool_names),
            },
            "checks": {
                "child_tool_count_le_parent": child_measure.tool_count <= parent_measure.tool_count,
                "child_bytes_le_parent": child_measure.schema_bytes <= parent_measure.schema_bytes,
                "child_tokens_le_parent": child_measure.schema_tokens <= parent_measure.schema_tokens,
                "child_names_subset_of_parent": child_names.issubset(set(parent_measure.tool_names)),
                "only_task_family_retained": list(narrowed.eligible_toolsets) == ["s3_stb_logs"],
                "recursive_spawn_tools_absent": not (child_names & set(spawn)),
                "code_execution_tools_absent": not any(is_code_execution_tool(n) for n in child_names),
                "no_authorization_elevation": narrowed.flags == parent_flags,
                "elevation_attempt_families": elevation_families,
                "elevation_attempt_did_not_broaden":
                    set(elevation_families).issubset(set(parent_profile.active_toolsets)),
            },
        }
    finally:
        restore()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D3B2 tool-profile cache benchmark")
    parser.add_argument("--out", default="", help="output directory for JSON/Markdown artifacts")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--emit-restart-signature", action="store_true")
    parser.add_argument("--verify-fixture-lock", action="store_true")
    ns = parser.parse_args(list(argv) if argv is not None else None)

    if ns.emit_restart_signature:
        print(json.dumps(emit_restart_signature(), sort_keys=True))
        return 0

    fixture, fixture_digest = load_fixture()
    lock = fixture_lock_path()
    lock_expected = lock.read_text(encoding="utf-8").split()[0].strip() if lock.exists() else ""
    lock_ok = bool(lock_expected) and lock_expected == fixture_digest

    if ns.verify_fixture_lock:
        print(json.dumps({"fixture_digest": fixture_digest,
                          "lock_expected": lock_expected, "lock_ok": lock_ok}, sort_keys=True))
        return 0 if lock_ok else 1

    meter = resolve_token_meter("aws-bedrock")
    registry = ControlledRegistry.from_fixture(fixture)

    restore = registry.install()
    try:
        core = run_scenarios(registry, meter)
        broad = measure_broad_inventory(registry, meter)
        contributors = contributor_report(broad["broad_all_tools"]["per_tool"], registry, ns.top)
    finally:
        restore()

    reg_scen = run_registry_scenarios(registry, meter)
    child = run_child_scenario(registry, meter)
    restart = run_restart_scenario(ns.restarts)
    telemetry = probe_provider_cache_telemetry()

    generic = core["scenarios"]["S01_generic_initial"]
    s3 = core["scenarios"]["S03_s3_initial"]
    auth = core["scenarios"]["S06_heavy_authorized"]
    broad_all = broad["broad_all_tools"]

    def pct(base_v: int, small_v: int) -> float:
        return round(100.0 * (base_v - small_v) / base_v, 2) if base_v else 0.0

    reductions = {
        target: {
            "tool_count": int(rec["model_schema"]["tool_count"]),
            "schema_bytes": int(rec["model_schema"]["schema_bytes"]),
            "schema_chars": int(rec["model_schema"]["schema_chars"]),
            "schema_tokens": int(rec["model_schema"]["schema_tokens"]),
        }
        for target, rec in {
            "generic_stable_core": generic,
            "s3_read_only": s3,
            "authorized_heavy_exact": auth,
            "child_read_only": child["child"],
        }.items()
    }
    for target, vals in reductions.items():
        vals["tool_count_reduction_pct"] = pct(broad_all["tool_count"], vals["tool_count"])
        vals["schema_byte_reduction_pct"] = pct(broad_all["schema_bytes"], vals["schema_bytes"])
        vals["schema_char_reduction_pct"] = pct(broad_all["schema_chars"], vals["schema_chars"])
        vals["schema_token_reduction_pct"] = pct(broad_all["schema_tokens"], vals["schema_tokens"])

    checks = {**core["checks"], **reg_scen["checks"]}
    checks["S11_restart_determinism"] = restart
    checks["S12_child_narrowing"] = child["checks"]

    results = {
        "schema": BENCHMARK_SCHEMA,
        "fixture": {"digest": fixture_digest, "lock_expected": lock_expected, "lock_ok": lock_ok,
                    "families": len(fixture["families"]),
                    "tools": sum(len(f["tools"]) for f in fixture["families"].values())},
        "token_measurement": meter.to_dict(),
        "serializer": broad_all["serializer"],
        "registry_content_signature": registry.content_signature,
        "scenarios": {**core["scenarios"], **reg_scen["scenarios"], "S12_child": child},
        "checks": checks,
        "broad_inventory": {k: v for k, v in broad.items() if k != "broad_all_tools"}
                           | {"broad_all_tools": {k: v for k, v in broad_all.items() if k != "per_tool"}},
        "dynamic_vs_broad": reductions,
        "model_schema_contributors": contributors,
        "provider_cache_telemetry": telemetry,
    }

    if ns.out:
        outdir = Path(ns.out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "D3B2_SCENARIO_RESULTS.json").write_text(
            json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote", outdir / "D3B2_SCENARIO_RESULTS.json")
    else:
        print(json.dumps(results, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
