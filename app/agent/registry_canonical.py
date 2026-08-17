"""Deterministic canonicalization for semantic tool-registry identity.

Version: registry_signature.v2

D3B1 replaces ``json.dumps(..., the JSON default-coercion hook)`` in every semantic-hashing path.
``the JSON default-coercion hook`` silently stringifies arbitrary objects, so a callable, a
connection wrapper, a Pydantic model class, or a memory address could enter a
semantic signature and make registry identity depend on process identity.

This module canonicalizes only explicitly supported semantic shapes and raises
:class:`CanonicalizationError` for anything else.  A failure never embeds the
offending value, so an unexpected object cannot leak a URL, token, or header
into a signature or a log line.

Guarantees
----------
* mapping keys are sorted;
* unordered collections are normalized;
* semantically ordered sequences preserve order;
* non-finite floats use explicit deterministic tokens;
* output is compact, stable UTF-8 JSON;
* no timestamps, process ids, request ids, memory addresses, callables, or
  arbitrary object reprs are ever included.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from enum import Enum
from typing import Any, Callable, Mapping

CANONICALIZATION_VERSION = "registry_signature.v2"

MAX_DEPTH = 40
MAX_SEQUENCE_ITEMS = 5000
MAX_MAPPING_KEYS = 5000
MAX_STRING_CHARS = 200_000

NAN_TOKEN = {"__nonfinite_float__": "nan"}
POS_INF_TOKEN = {"__nonfinite_float__": "inf"}
NEG_INF_TOKEN = {"__nonfinite_float__": "-inf"}

UNAVAILABLE_SCHEMA_TOKEN = {"__schema_unavailable__": True}


class CanonicalizationError(TypeError):
    """Raised when a value cannot be canonicalized deterministically."""


_ADAPTERS: dict[type, Callable[[Any], Any]] = {}


def register_adapter(target: type, adapter: Callable[[Any], Any]) -> None:
    """Register an explicit semantic adapter for one type.

    Adapters exist so an unsupported object is handled by declared intent
    rather than by silent stringification.
    """
    if not isinstance(target, type):
        raise CanonicalizationError("adapter target must be a type")
    if not callable(adapter):
        raise CanonicalizationError("adapter must be callable")
    _ADAPTERS[target] = adapter


def _fail(kind: str) -> "CanonicalizationError":
    return CanonicalizationError("uncanonicalizable value of kind " + kind)


def _canonical_float(value: float) -> Any:
    if math.isnan(value):
        return dict(NAN_TOKEN)
    if math.isinf(value):
        return dict(POS_INF_TOKEN) if value > 0 else dict(NEG_INF_TOKEN)
    return float(value)


def _canonical_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(int(key))
    if key is None:
        return "null"
    if isinstance(key, Enum):
        return _canonical_key(key.value)
    raise _fail("mapping-key:" + type(key).__name__)


def _adapter_for(value: Any) -> Callable[[Any], Any] | None:
    for target, adapter in _ADAPTERS.items():
        if isinstance(value, target):
            return adapter
    return None


def canonicalize(value: Any, *, depth: int = 0) -> Any:
    """Return a JSON-safe deterministic projection of one semantic value."""
    if depth > MAX_DEPTH:
        raise _fail("max-depth-exceeded")

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            raise _fail("oversized-string")
        return value
    if isinstance(value, (bytes, bytearray)):
        # Never embed raw bytes; a stable digest keeps identity without leaking.
        return {"__bytes_sha256__": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, Enum):
        return {"__enum__": type(value).__name__, "value": canonicalize(value.value, depth=depth + 1)}

    adapter = _adapter_for(value)
    if adapter is not None:
        return canonicalize(adapter(value), depth=depth + 1)

    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_KEYS:
            raise _fail("oversized-mapping")
        items = []
        for raw_key, raw_value in value.items():
            items.append((_canonical_key(raw_key), canonicalize(raw_value, depth=depth + 1)))
        items.sort(key=lambda pair: pair[0])
        result: dict[str, Any] = {}
        for key, canonical_value in items:
            if key in result:
                raise _fail("duplicate-canonical-mapping-key")
            result[key] = canonical_value
        return result

    if isinstance(value, (set, frozenset)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise _fail("oversized-set")
        rendered = [canonicalize(item, depth=depth + 1) for item in value]
        rendered.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return {"__unordered__": rendered}

    if isinstance(value, (list, tuple)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise _fail("oversized-sequence")
        return [canonicalize(item, depth=depth + 1) for item in value]

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {}
        for field in dataclasses.fields(value):
            fields[field.name] = getattr(value, field.name, None)
        return canonicalize(fields, depth=depth + 1)

    # Pydantic v2 / v1 instances expose deterministic JSON-mode dumps.
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return canonicalize(dump(mode="json"), depth=depth + 1)
        except TypeError:
            return canonicalize(dump(), depth=depth + 1)
        except Exception as exc:
            raise _fail("pydantic-instance:" + type(value).__name__) from exc

    if isinstance(value, type):
        schema = json_schema_of(value)
        if schema is None:
            raise _fail("type-without-json-schema:" + value.__name__)
        return canonicalize(schema, depth=depth + 1)

    raise _fail(type(value).__name__)


def json_schema_of(candidate: Any) -> Any:
    """Return a JSON schema mapping for a Pydantic model class, else None."""
    for attribute in ("model_json_schema", "schema"):
        method = getattr(candidate, attribute, None)
        if callable(method):
            try:
                produced = method()
            except Exception:
                continue
            if isinstance(produced, Mapping):
                return produced
    return None


def canonical_json(value: Any) -> str:
    """Return compact canonical JSON text for one semantic value."""
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_signature(value: Any) -> str:
    """Return the versioned deterministic content signature of a value."""
    body = canonical_json({"canonicalization_version": CANONICALIZATION_VERSION, "content": value})
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def safe_content_signature(value: Any) -> tuple[str, str]:
    """Return ``(signature, error_kind)``; signature is empty when unsafe.

    The caller decides policy.  A canonicalization failure never overwrites a
    trusted baseline and never surfaces the offending value.
    """
    try:
        return content_signature(value), ""
    except CanonicalizationError as exc:
        return "", str(exc)[:200]
    except Exception:
        return "", "uncanonicalizable value of kind unknown"
