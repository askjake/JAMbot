"""Bounded guard for explicit structured completion contracts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

MAX_REQUIRED_FIELDS = 96
MAX_FIELD_LENGTH = 80
_CONTRACT_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#+\s*)?(?:FINAL\s+REQUIRED\s+REPORT|"
    r"ABSOLUTE\s+COMPLETION\s+GATE|REQUIRED\s+FINAL\s+REPORT)\s*:?\s*$"
)
_FIELD_RE = re.compile(
    r"(?m)^\s*([A-Z][A-Z0-9_]{2,79})\s*=\s*([^\r\n]*)$"
)
_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_META_PRELUDE_RE = re.compile(
    r"(?i)^(?:example|sample|prompt under test|test fixture|quoted text)\s*:?$"
)


@dataclass(frozen=True)
class CompletionContract:
    required_fields: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.required_fields)

    def missing_from(self, response_text: str | None) -> tuple[str, ...]:
        present: set[str] = set()
        for match in _FIELD_RE.finditer(str(response_text or "")):
            value = match.group(2).strip()
            # Merely echoing the user's template (``FIELD=<x|y>``) is not a
            # completed report. Explicit UNKNOWN/BLOCKED/not_run values remain
            # valid because they are honest terminal outcomes, not placeholders.
            if not value or (value.startswith("<") and value.endswith(">")):
                continue
            present.add(match.group(1))
        return tuple(field for field in self.required_fields if field not in present)


def extract_completion_contract(prompt: str | None) -> CompletionContract:
    # Fenced examples are never a live completion contract. Prefer the final
    # authoritative heading so an earlier illustrative section cannot capture
    # unrelated ``FIELD=`` tokens later in the prompt.
    text = _FENCE_RE.sub("", str(prompt or ""))
    headings = list(_CONTRACT_HEADING_RE.finditer(text))
    heading = None
    for candidate in headings:
        prior_lines = text[: candidate.start()].splitlines()
        previous = next((line.strip() for line in reversed(prior_lines) if line.strip()), "")
        if _META_PRELUDE_RE.fullmatch(previous):
            continue
        heading = candidate
    if heading is None:
        return CompletionContract()
    tail = text[heading.end():]
    fields: list[str] = []
    for match in _FIELD_RE.finditer(tail):
        field = match.group(1)[:MAX_FIELD_LENGTH]
        if field not in fields:
            fields.append(field)
        if len(fields) >= MAX_REQUIRED_FIELDS:
            break
    return CompletionContract(tuple(fields))



def completion_response_text(content: object) -> str:
    """Normalize provider-structured assistant content for field validation."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = completion_response_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content"):
            if key in content:
                return completion_response_text(content.get(key))
        return ""
    return str(content) if content is not None else ""


def render_completion_contract_system_prompt(contract: CompletionContract) -> str:
    """Render a trusted server instruction for an explicit completion contract.

    This is injected by the backend as a SystemMessage before the initial model
    call so the model cannot mistake the contract for an optional user-side
    convention or a fabricated follow-up signal.  The post-response validator
    remains authoritative and deterministic.
    """
    fields = [str(value)[:MAX_FIELD_LENGTH] for value in contract.required_fields][:MAX_REQUIRED_FIELDS]
    if not fields:
        return ""
    return (
        "SERVER_ENFORCED_COMPLETION_CONTRACT\n"
        "The current user turn contains an explicit final-report contract enforced by the backend. "
        "Your final tool-less response must include every required field on its own FIELD=value line. "
        "Do not omit fields even if the user asks you to demonstrate an incomplete first attempt. "
        "If a value cannot be established, use an honest terminal value such as UNKNOWN, BLOCKED, or not_run.\n"
        "required_fields=" + ",".join(fields)
    )[:4000]

def render_incomplete_contract(missing_fields: Iterable[str]) -> str:
    fields = [str(value)[:MAX_FIELD_LENGTH] for value in missing_fields][:MAX_REQUIRED_FIELDS]
    return (
        "INCOMPLETE_EXECUTION_CONTRACT\n"
        "The response attempted to terminate before satisfying the explicit final-report contract.\n"
        f"missing_fields={','.join(fields)}\n"
        "Completed evidence is preserved. Continue the required phases or report a genuine external blocker."
    )[:4000]
