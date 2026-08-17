#!/usr/bin/env python3
"""Generate the deterministic controlled-registry fixture used by D3B2 benchmarks.

The fixture is a *synthetic* projection of the real tool-registry shape.  It
reproduces the real family names and per-family tool counts so schema-size
benchmarks are structurally realistic, but every description and argument name
is generated deterministically from the family and tool name.  No real tool
description, no real argument default, no credential, and no customer or
receiver identifier is ever written into the fixture.

Determinism: output depends only on the constants in this file.  Re-running
regenerates a byte-identical fixture.

Usage:
    python scripts/benchmarks/generate_tool_profile_fixture.py \
        --out tests/fixtures/tool_profile_benchmark/registry_baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURE_SCHEMA = "d3b2_registry_fixture.v1"

# Real family names and real per-family tool counts (registry shape only).
FAMILY_TOOL_COUNTS: dict[str, int] = {
    "agent_mode": 5,
    "backend_facades": 3,
    "beta_report": 10,
    "confluence_mcp": 3,
    "dish_internal": 2,
    "epg_mcp": 16,
    "grasshopper_mcp": 13,
    "headless_browser_mcp": 5,
    "jira_mcp": 7,
    "log_assist": 5,
    "net_detective_mcp": 6,
    "netra_mcp": 10,
    "qos_mcp": 27,
    "rca_mcp": 7,
    "rtr_alerts_mcp": 12,
    "s3_stb_logs": 171,
    "s3_stb_logs_prod": 171,
    "search": 2,
    "stbhealth_popups_mcp": 1,
    "viewership": 9,
}

# Named tools that must exist under the exact real names, because policy code
# references them by name.  Everything else is generated as family_tool_NN.
NAMED_TOOLS: dict[str, tuple[str, ...]] = {
    "agent_mode": (
        "agent_run_python",
        "agent_run_shell",
        "agent_git_clone",
        "agent_create_venv",
        "agent_list_artifacts",
    ),
    "backend_facades": (
        "diship_backend_tool_inventory_status",
        "diship_backend_tool_binding_status",
        "diship_backend_activate_tool_binding",
    ),
    "search": ("public_web_search_enhanced", "internal_search"),
    # Curated S3 read-only core.  ``list_incident_scenes`` is deliberately
    # ABSENT: the current S3 runtime does not expose Incident Scene tools, and
    # the benchmark must reproduce that upstream gap rather than assume it away.
    "s3_stb_logs": (
        "get_tool_info",
        "get_heavy_auth_status",
        "list_dates",
        "list_files",
        "search_logs",
        "get_summary",
        "list_log_capsules",
        # heavy / gated exact tools that DO exist upstream
        "build_log_capsule",
        "build_complete_log_capsule",
        "get_timeline",
        "summarize_log_patterns",
        "filter_log_lines",
        "compare_log_capsules",
        "create_log_bundle",
    ),
}
NAMED_TOOLS["s3_stb_logs_prod"] = NAMED_TOOLS["s3_stb_logs"]

# Tools that carry server-controlled authorization arguments upstream.  The
# benchmark must exercise the model-facing sanitizer on real argument names.
HIDDEN_ARG_TOOLS: frozenset[str] = frozenset({
    "build_log_capsule",
    "build_complete_log_capsule",
    "get_timeline",
    "summarize_log_patterns",
    "filter_log_lines",
    "compare_log_capsules",
    "create_log_bundle",
})

ARG_TYPES = ("string", "integer", "boolean", "number", "array_string")

# Generic, non-proprietary description vocabulary.
PHRASES = (
    "Return a bounded, read-only projection of the requested records.",
    "Results are ordered deterministically and truncated to the configured limit.",
    "Supply an explicit window; an unbounded window is rejected.",
    "Missing or inaccessible inputs are reported, never inferred.",
    "This call performs no mutation and writes no persistent artifact.",
    "Use the discovery call first to confirm which inputs are available.",
    "Output is machine readable and safe to page through.",
    "Errors are classified into a bounded vocabulary without transport detail.",
)


def _h(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _description(family: str, tool: str) -> str:
    seed = _h(family, tool)
    count = 2 + (seed % 4)
    picked = [PHRASES[(seed + i * 7) % len(PHRASES)] for i in range(count)]
    head = "Read-only " + family.replace("_", " ") + " operation " + tool + "."
    return " ".join([head, *picked])


def _args(family: str, tool: str) -> list[dict[str, Any]]:
    seed = _h("args", family, tool)
    count = 2 + (seed % 6)
    args: list[dict[str, Any]] = []
    for i in range(count):
        s = _h("arg", family, tool, str(i))
        args.append({
            "name": "param_%d" % i if i else "primary_id",
            "type": ARG_TYPES[s % len(ARG_TYPES)],
            "required": i == 0,
            "description": "Bounded input %d for %s; validated before use." % (i, tool),
        })
    if tool in HIDDEN_ARG_TOOLS:
        args.append({
            "name": "allow_heavy",
            "type": "boolean",
            "required": False,
            "description": "Server-controlled authorization switch.",
        })
        args.append({
            "name": "heavy_auth_token",
            "type": "string",
            "required": False,
            "description": "Server-controlled authorization material.",
        })
    return args


def build_fixture() -> dict[str, Any]:
    families: dict[str, Any] = {}
    for family in sorted(FAMILY_TOOL_COUNTS):
        total = FAMILY_TOOL_COUNTS[family]
        named = list(NAMED_TOOLS.get(family, ()))
        names = list(named)
        idx = 0
        while len(names) < total:
            candidate = "%s_tool_%02d" % (family, idx)
            idx += 1
            if candidate not in names:
                names.append(candidate)
        names = names[:total]
        tools = [
            {"name": n, "description": _description(family, n), "args": _args(family, n)}
            for n in names
        ]
        families[family] = {"health": "HEALTHY", "source": "fixture", "tools": tools}
    return {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": "baseline",
        "note": "Synthetic controlled registry. Real family/tool-count shape only; "
                "no real descriptions, defaults, credentials, or customer data.",
        "refresh_epoch": 41,
        "loaded_at": "1970-01-01T00:00:00Z",
        "families": families,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    ns = parser.parse_args()
    payload = build_fixture()
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    total = sum(len(f["tools"]) for f in payload["families"].values())
    print("wrote", out, "families=%d" % len(payload["families"]), "tools=%d" % total,
          "bytes=%d" % len(text.encode()),
          "sha256=" + hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
