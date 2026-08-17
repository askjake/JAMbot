#!/usr/bin/env python3
"""Operator-only, read-only tool-execution audit inspector.

This CLI is intentionally *not* model-facing.  It is the lowest-exposure
inspection path for ``tool_execution_audit.v1`` evidence: no HTTP surface is
added, so no authenticated endpoint can be reached by a prompt.

It cannot return tool argument values or tool-result bodies, because those are
never written to the audit log in the first place.

Examples
--------
  python scripts/active/query_tool_execution_audit.py --limit 20
  python scripts/active/query_tool_execution_audit.py --request-id req-<hex>
  python scripts/active/query_tool_execution_audit.py --decision BLOCKED
  python scripts/active/query_tool_execution_audit.py --validate --limit 100
  python scripts/active/query_tool_execution_audit.py --storage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent.tool_execution_audit import (  # noqa: E402
    DECISIONS,
    DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT,
    RESULT_CODES,
    TOOL_EXECUTION_AUDIT_SCHEMA,
    audit_storage_summary,
    query_events,
    validate_event,
)

# Defence in depth: even though these keys are never written, refuse to print a
# record that somehow contains them.
_FORBIDDEN_OUTPUT_KEYS = {
    "args",
    "arguments",
    "argument_values",
    "result",
    "tool_result",
    "tool_result_body",
    "content",
    "prompt",
    "prompt_text",
    "assistant_text",
    "heavy_auth_token",
    "operator_auth_token",
    "authorization_token",
    "mutation_auth_token",
    "thread_id",
    "token_hash",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request-id", default="", help="Exact server-owned request ID.")
    parser.add_argument("--thread-digest", default="", help="Exact thread digest (thr-...).")
    parser.add_argument("--tool-call-id", default="", help="Exact emitted tool-call ID.")
    parser.add_argument("--tool-name", default="", help="Exact tool name.")
    parser.add_argument("--decision", default="", choices=("", *sorted(DECISIONS)), help="Gate decision.")
    parser.add_argument("--result-code", default="", help="Exact result code.")
    parser.add_argument("--since", default="", help="Inclusive ISO-8601 UTC lower bound (YYYY-MM-DDTHH:MM:SSZ).")
    parser.add_argument("--until", default="", help="Inclusive ISO-8601 UTC upper bound.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_QUERY_LIMIT,
        help=f"Maximum records (default {DEFAULT_QUERY_LIMIT}, hard maximum {MAX_QUERY_LIMIT}).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--validate", action="store_true", help="Validate records against the schema and exit non-zero on failure.")
    parser.add_argument("--storage", action="store_true", help="Print storage/rotation/retention configuration and exit.")
    return parser


def _redact_for_output(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in _FORBIDDEN_OUTPUT_KEYS}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.storage:
        print(json.dumps(audit_storage_summary(), indent=2, sort_keys=True))
        return 0

    if args.result_code and args.result_code not in RESULT_CODES:
        print(f"unknown --result-code {args.result_code!r}; known: {sorted(RESULT_CODES)}", file=sys.stderr)
        return 2

    limit = max(1, min(int(args.limit), MAX_QUERY_LIMIT))
    records = [
        _redact_for_output(record)
        for record in query_events(
            request_id=args.request_id,
            thread_id_digest=args.thread_digest,
            tool_call_id=args.tool_call_id,
            tool_name=args.tool_name,
            decision=args.decision,
            result_code=args.result_code,
            since=args.since,
            until=args.until,
            limit=limit,
        )
    ]

    if args.validate:
        failures = []
        for record in records:
            problems = validate_event(record)
            if problems:
                failures.append({"event_id": record.get("event_id", "<none>"), "problems": problems})
        report = {
            "schema_version": TOOL_EXECUTION_AUDIT_SCHEMA,
            "records_checked": len(records),
            "invalid_records": len(failures),
            "status": "PASS" if not failures else "FAIL",
            "failures": failures[:20],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if not failures else 1

    if args.json:
        print(json.dumps({"records": records, "count": len(records), "limit": limit}, indent=2, sort_keys=True))
        return 0

    if not records:
        print("no audit records matched (evidence is bounded and rotated)")
        return 0

    header = f"{'TIMESTAMP':21} {'DECISION':10} {'RESULT_CODE':40} {'EXEC':5} {'PAIR':5} {'MS':>6} TOOL"
    print(header)
    print("-" * len(header))
    for record in records:
        print(
            f"{str(record.get('timestamp', '')):21} "
            f"{str(record.get('decision', '')):10} "
            f"{str(record.get('result_code', '')):40} "
            f"{str(record.get('executed', '')):5} "
            f"{str(record.get('paired_tool_result', '')):5} "
            f"{str(record.get('duration_ms', '')):>6} "
            f"{str(record.get('tool_name', ''))}"
        )
    print()
    print(f"{len(records)} record(s); limit={limit}; argument values and result bodies are never stored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
