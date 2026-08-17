"""Serial tool executor enforcing verified names, one attempt, role, and commit safety."""
from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
from typing import Any, Protocol

from .config import Settings
from .contracts import contract_for
from .state import RunState, RunStore, StepRecord, stable_hash, utc_now

log = logging.getLogger("nightly_rca.executor")


def _requests_live_write(tool: str, arguments: dict[str, Any], *, write_capable: bool, two_step_action: str | None) -> bool:
    """Return True only when this invocation can perform an external mutation."""
    if not write_capable:
        return False
    if two_step_action is not None:
        return arguments.get("dry_run") is False
    if tool == "grasshopper_upload_profile_logs":
        return arguments.get("dry_run") is False
    if tool == "human_review_materialize_engineer_contexts":
        return arguments.get("persist") is True
    if tool == "update_upload_tracker_from_s3":
        return arguments.get("dry_run") is not True
    return True


class ToolClient(Protocol):
    async def call(self, server: str, tool_name: str, arguments: dict[str, Any]) -> Any: ...


_SECRET_PATTERNS = (
    re.compile(r"(?i)(token|secret|password|authorization|api[_-]?key)\s*[:=]\s*[^\s,]+"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if any(marker in key.lower() for marker in ("token", "secret", "password", "api_key", "authorization")):
                out[key] = "<redacted>"
            else:
                out[key] = redact(val)
        return out
    if isinstance(value, list):
        return [redact(x) for x in value]
    if isinstance(value, str):
        text = value
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub("<redacted>", text)
        return text
    return value


class SerialExecutor:
    def __init__(self, client: ToolClient, settings: Settings, state: RunState, store: RunStore):
        self.client = client
        self.settings = settings
        self.state = state
        self.store = store
        self._lock = asyncio.Lock()
        self.max_in_flight = 0
        self._in_flight = 0

    async def call(
        self,
        *,
        phase: str,
        step: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        required: bool = False,
        tolerate_error: bool = False,
    ) -> dict[str, Any]:
        contract = contract_for(tool)
        args = copy.deepcopy(arguments or {})
        if contract.operator_required:
            args["role"] = self.settings.role
        started = utc_now()
        t0 = time.monotonic()
        status = "OK"
        response: Any = None
        error: str | None = None
        actual_call = False

        live_write = _requests_live_write(
            tool, args, write_capable=contract.write_capable, two_step_action=contract.two_step_action
        )
        if live_write and not self.settings.commit:
            status = "WRITE_BLOCKED"
            error = "dry-run safety policy blocked a live mutation"
        else:
            async with self._lock:
                self._in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self._in_flight)
                try:
                    actual_call = True
                    response = await self.client.call(contract.server, tool, args)
                    if isinstance(response, dict):
                        code = str(response.get("code") or response.get("status") or "")
                        if "ROLE_OPERATOR_REQUIRED" in code:
                            status = "ROLE_ERROR"
                            error = code
                        elif "TOOL_NOT_FOUND" in code:
                            status = "TOOL_NOT_FOUND"
                            error = code
                        elif response.get("ok") is False and response.get("error"):
                            status = "STEP_FAILED"
                            error = str(response.get("error"))
                except Exception as exc:  # one attempt only; never retry
                    error = f"{type(exc).__name__}: {exc}"
                    lowered = error.lower()
                    if "tool_not_found" in lowered or "unknown tool" in lowered:
                        status = "TOOL_NOT_FOUND"
                    elif "role_operator_required" in lowered:
                        status = "ROLE_ERROR"
                    else:
                        status = "STEP_FAILED"
                finally:
                    self._in_flight -= 1

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        record = StepRecord(
            phase=phase,
            step=step,
            tool=tool,
            server=contract.server,
            arguments=redact(args),
            status=status,
            started_at=started,
            completed_at=utc_now(),
            elapsed_ms=elapsed_ms,
            response=redact(response),
            error=redact(error),
        )
        rec = record.__dict__
        self.state.steps.append({
            "phase": phase, "step": step, "tool": tool, "server": contract.server,
            "status": status, "elapsed_ms": elapsed_ms, "attempt": 1,
            "error": redact(error), "response_hash": stable_hash(redact(response)),
        })
        if actual_call:
            self.state.metrics["tool_calls"] += 1
        if status == "WRITE_BLOCKED":
            self.state.metrics.setdefault("write_blocked", 0)
            self.state.metrics["write_blocked"] += 1
        elif status == "TOOL_NOT_FOUND":
            self.state.metrics["tool_not_found"] += 1
        elif status == "ROLE_ERROR":
            self.state.metrics["role_errors"] += 1
        elif status == "STEP_FAILED":
            if not tolerate_error:
                self.state.metrics["step_failed"] += 1
        self.store.append_event(rec)
        self.store.save(self.state)
        log.info("%s %s %s (%dms)", phase, step, status, elapsed_ms)

        return {
            "status": status,
            "response": response,
            "error": error,
            "required": required,
            "tool": tool,
            "step": step,
        }
