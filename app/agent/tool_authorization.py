"""Server-side authorization-state parsing for the current Human turn only.

The parser records only boolean capability grants/revocations.  It never asks
for, validates, hashes, stores, or echoes authorization tokens or credentials.
Quoted examples, code fixtures, and copied report material are not directives.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from app.agent.directive_text import sanitize_current_turn_directives
from app.agent.tool_profiles import AUTHORIZATION_DEFAULTS, normalize_authorization_flags

AUTHORIZATION_STATE_SCHEMA = "diship_authorization_state.v1"
AUTHORIZATION_PARSER_VERSION = "authorization_phrase_parser.v2"

_CAPABILITY_TERMS: dict[str, tuple[str, ...]] = {
    "operator_authorized": ("operator", "operator actions", "operator mode"),
    "heavy_tools_authorized": (
        "heavy tools",
        "heavy tool",
        "heavy compute",
        "heavy authorization",
        "heavy tool authorization",
    ),
    "persistence_authorized": ("persistence", "persist", "artifact persistence"),
    "mutation_authorized": ("mutation", "mutations", "write actions", "writes"),
}

_POSITIVE_CONTEXT = re.compile(
    r"\b(?:authorize|authorized|authorization\s+(?:is\s+)?granted|"
    r"grant(?:ed)?|allow(?:ed)?|enable(?:d)?|approve(?:d)?)\b",
    re.IGNORECASE,
)
_NEGATIVE_CONTEXT = re.compile(
    r"\b(?:not\s+authorized|unauthorized|do\s+not\s+authorize|don't\s+authorize|"
    r"revoke(?:d)?|disable(?:d)?|deny|denied|no\s+authorization|"
    r"authorization\s+(?:is\s+)?revoked)\b",
    re.IGNORECASE,
)
_NEGATED_POSITIVE = re.compile(
    r"\b(?:not|never|no|without|lacking|absent)\s+(?:\w+[\s-]+){0,3}?"
    r"(?:authorize[ds]?|authorizing|authorization|authorised|"
    r"grant(?:ed|ing|s)?|allow(?:ed|ing|s)?|enable[ds]?|enabling|"
    r"approve[ds]?|approving|permit(?:ted|ting|s)?|permission)\b"
    r"|\b(?:isn't|aren't|wasn't|weren't|won't|don't|doesn't|didn't|cannot|can't)\s+"
    r"(?:\w+[\s-]+){0,3}?"
    r"(?:authorize[ds]?|authorization|grant(?:ed|s)?|allow(?:ed|s)?|enable[ds]?|"
    r"approve[ds]?|permit(?:ted|s)?)\b",
    re.IGNORECASE,
)
_ELLIPTICAL_NEGATIVE = re.compile(
    r"\b(?:is|are|was|were|do|does|did|will|would|remains?)\s+not\b[\s.!?;,]$"
    r"|\bnot\b[\s.!?;,]$",
    re.IGNORECASE,
)
_INQUIRY_CONTEXT = re.compile(
    r"\?|\b(?:whether|tell\s+me|show\s+me|report|inspect|check|confirm|verify|query|"
    r"describe|explain|status\s+of|what\s+is|what\s+are|which|are\s+you|do\s+you|"
    r"is\s+it|am\s+i|can\s+you|could\s+you)\b",
    re.IGNORECASE,
)
_CANONICAL_RE = re.compile(
    r"^\s*(operator_authorized|heavy_tools_authorized|mutation_authorized|"
    r"persistence_authorized)\s*=\s*(true|false)\s*[.;,]?\s*$",
    re.IGNORECASE,
)


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(
        r"\b" + re.escape(term).replace(r"\ ", r"[\s-]+") + r"\b",
        re.IGNORECASE,
    )


def _sentences(text: str) -> list[str]:
    return [
        part.strip(" ,")
        for part in re.split(
            r"(?<=[.!?;\n])\s*|\b(?:but|while)\b", str(text or ""), flags=re.IGNORECASE
        )
        if part.strip(" ,")
    ]


def parse_authorization_updates(text: str | None) -> dict[str, bool]:
    """Return explicit grants/revocations from authoritative current-turn text.

    Canonical declarations accept only bare ``true``/``false``.  Revocation
    wins over any grant for the same capability within one turn.
    """

    authoritative = sanitize_current_turn_directives(text)
    grants: dict[str, bool] = {}
    revoked: set[str] = set()

    for line in authoritative.splitlines():
        match = _CANONICAL_RE.fullmatch(line)
        if not match:
            continue
        key = match.group(1).lower()
        if match.group(2).lower() == "false":
            revoked.add(key)
        else:
            grants[key] = True

    for sentence in _sentences(authoritative):
        if _CANONICAL_RE.fullmatch(sentence) or _INQUIRY_CONTEXT.search(sentence):
            continue
        negative = bool(
            _NEGATIVE_CONTEXT.search(sentence)
            or _NEGATED_POSITIVE.search(sentence)
            or _ELLIPTICAL_NEGATIVE.search(sentence)
        )
        positive = bool(_POSITIVE_CONTEXT.search(sentence))
        if not negative and not positive:
            continue
        for key, terms in _CAPABILITY_TERMS.items():
            if not any(_term_pattern(term).search(sentence) for term in terms):
                continue
            if negative:
                revoked.add(key)
            else:
                grants[key] = True

    for key in revoked:
        grants[key] = False
    return {key: grants[key] for key in AUTHORIZATION_DEFAULTS if key in grants}


def merge_authorization_flags(
    current: Mapping[str, Any] | None,
    updates: Mapping[str, Any] | None,
) -> dict[str, bool]:
    """Apply explicit updates while keeping all other flags sticky."""
    merged = normalize_authorization_flags(current)
    for key in AUTHORIZATION_DEFAULTS:
        if updates and key in updates:
            merged[key] = bool(updates[key])
    return merged


def authorization_state_for_turn(
    current: Mapping[str, Any] | None,
    latest_user_text: str | None,
) -> dict[str, Any]:
    updates = parse_authorization_updates(latest_user_text)
    flags = merge_authorization_flags(current, updates)
    return {
        "schema": AUTHORIZATION_STATE_SCHEMA,
        "parser_version": AUTHORIZATION_PARSER_VERSION,
        "authorization_flags": flags,
        "explicit_updates": updates,
        "authorization_material_stored": False,
    }
