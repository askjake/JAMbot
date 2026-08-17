"""Deterministic extraction of authoritative current-turn directives.

Prompts often contain examples, quoted transcripts, code fixtures, expected
outputs, and copied logs.  Those regions are data, not operator directives.
This module provides a small, framework-neutral sanitizer used by the
authorization, activation, and execution-constraint parsers.
"""
from __future__ import annotations

import re

_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_DOUBLE_QUOTE_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_SINGLE_QUOTE_RE = re.compile(r"'(?:\\.|[^'\\])*'")

_META_MARKERS = (
    "example",
    "test case",
    "test fixture",
    "expected output",
    "expected result",
    "must result",
    "should result",
    "acceptance test",
    "regression test",
    "prompt under test",
    "sample prompt",
    "quoted text",
    "literal text",
    "schema example",
    "report field",
    "final required report",
    "required final report",
    "absolute completion gate",
    "final report",
    "required output",
    "expected output",
    "observed output",
    "acceptance criteria",
    "pass criteria",
)


def _heading_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.endswith(":"):
        return True
    letters = [char for char in stripped if char.isalpha()]
    return bool(letters) and all(char.isupper() for char in letters)


def _meta_heading(line: str) -> bool:
    low = line.strip().lower()
    return any(marker in low for marker in _META_MARKERS) and (
        _heading_like(line)
        or low.startswith(("example", "test", "expected", "sample", "prompt under test"))
    )


def sanitize_current_turn_directives(text: str | None) -> str:
    """Return current-turn text with non-authoritative regions blanked.

    Newlines are retained so line-oriented canonical declarations remain
    deterministic.  The sanitizer intentionally does not interpret the
    surviving text; callers still apply their own narrow grammars.
    """

    source = _FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), str(text or ""))
    out: list[str] = []
    in_meta_section = False
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(">"):
            out.append("")
            continue
        if _meta_heading(raw_line):
            in_meta_section = True
            out.append("")
            continue
        if _heading_like(raw_line) and not _meta_heading(raw_line):
            in_meta_section = False
        if in_meta_section:
            out.append("")
            continue
        line = _INLINE_CODE_RE.sub("", raw_line)
        line = _DOUBLE_QUOTE_RE.sub("", line)
        line = _SINGLE_QUOTE_RE.sub("", line)
        # Same-line labels such as "Example: Bind ..." are data even when the
        # label is not a standalone heading.
        if any(marker in line.lower() for marker in _META_MARKERS):
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)
