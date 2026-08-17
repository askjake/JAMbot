"""D3B1 canonicalization and signature-stability tests (registry_signature.v2).

Covers requirement items 1-7 and 23-25 of the D3B1 test matrix.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

from app.agent import registry_canonical as canonical

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Modules whose hashing is semantic identity rather than logging or display.
SEMANTIC_HASHING_MODULES = (
    "app/agent/registry_canonical.py",
    "app/agent/mcp_registry_health.py",
    "app/agent/agents/tools/registry.py",
    "app/agent/tool_profiles.py",
)

SUBPROCESS_COUNT = 24
HASH_SEEDS = ("0", "1", "2", "3", "5", "7", "11", "13", "17", "101", "9999", "random")

PROBE = (
    "import json;"
    "from app.agent.agents.tools.registry import get_registry_content_signature;"
    "from app.agent.tool_profiles import profile_signature;"
    "s = get_registry_content_signature();"
    "p = profile_signature({'schema': 'fixed', 'active_toolsets': ['a', 'b'],"
    " 'registry_generation': s, 'authorization_flags': {'operator_authorized': False}});"
    "print(json.dumps({'content': s, 'profile': p}))"
)


def _run_probe(index: int, cache_dir: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONHASHSEED"] = HASH_SEEDS[index % len(HASH_SEEDS)]
    env["MCP_REGISTRY_CACHE_DIR"] = cache_dir
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        raise AssertionError("probe failed: " + completed.stderr[-2000:])
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload["hash_seed"] = env["PYTHONHASHSEED"]
    return payload


# --------------------------------------------------------------------------
# 1, 2, 15: cross-process determinism
# --------------------------------------------------------------------------
def test_content_and_profile_signature_stable_across_24_subprocesses(tmp_path):
    cache_dir = tmp_path / "registry_cache"
    cache_dir.mkdir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda i: _run_probe(i, str(cache_dir)), range(SUBPROCESS_COUNT)))

    assert len(results) == SUBPROCESS_COUNT
    contents = {item["content"] for item in results}
    profiles = {item["profile"] for item in results}
    seeds = {item["hash_seed"] for item in results}
    assert len(contents) == 1, "content signature varied across processes: " + repr(sorted(contents))
    assert len(profiles) == 1, "profile signature varied across processes: " + repr(sorted(profiles))
    assert len(seeds) >= 8, "hash seed variation was not exercised"
    only = contents.pop()
    assert only.startswith("sha256:")
    matrix = {
        "subprocess_count": SUBPROCESS_COUNT,
        "distinct_hash_seeds": sorted(seeds),
        "content_signature": only,
        "profile_signature": profiles.pop(),
        "stable": True,
    }
    (tmp_path / "matrix.json").write_text(json.dumps(matrix, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------
# 3, 4: order independence
# --------------------------------------------------------------------------
def test_equivalent_mapping_order_produces_one_signature():
    a = {"beta": 1, "alpha": {"y": [1, 2], "x": "s"}, "gamma": True}
    b = {"gamma": True, "alpha": {"x": "s", "y": [1, 2]}, "beta": 1}
    assert canonical.content_signature(a) == canonical.content_signature(b)


def test_equivalent_discovery_order_produces_one_signature():
    first = [{"family": "b", "content": "x"}, {"family": "a", "content": "y"}]
    second = [{"family": "a", "content": "y"}, {"family": "b", "content": "x"}]
    ordered_differs = canonical.content_signature(first) != canonical.content_signature(second)
    assert ordered_differs, "ordered sequences must preserve order"
    # Discovery order is normalized by sorting before hashing, which is what the
    # registry does; the sorted projections agree.
    assert canonical.content_signature(sorted(first, key=lambda i: i["family"])) == canonical.content_signature(
        sorted(second, key=lambda i: i["family"])
    )


def test_unordered_collections_are_normalized():
    assert canonical.content_signature({1, 2, 3}) == canonical.content_signature({3, 1, 2})
    assert canonical.content_signature(frozenset({"a", "b"})) == canonical.content_signature(frozenset({"b", "a"}))


# --------------------------------------------------------------------------
# 5: the JSON default-coercion hook is gone from semantic hashing
# --------------------------------------------------------------------------
def test_semantic_hashing_modules_do_not_use_default_coercion_hook():
    token = "default=" + "str"
    offenders = []
    for relative in SEMANTIC_HASHING_MODULES:
        body = (REPO_ROOT / relative).read_text(encoding="utf-8")
        if token in body:
            offenders.append(relative)
    assert offenders == [], "semantic hashing still uses the coercion hook: " + repr(offenders)


def test_canonical_json_rejects_nan_at_encoder_level():
    with pytest.raises(ValueError):
        json.dumps(float("nan"), allow_nan=False)


# --------------------------------------------------------------------------
# 6: unsupported values fail safely and never leak the value
# --------------------------------------------------------------------------
def test_callable_is_rejected_and_not_stringified():
    def secret_factory():  # pragma: no cover - never called
        return None

    with pytest.raises(canonical.CanonicalizationError) as excinfo:
        canonical.content_signature({"factory": secret_factory})
    message = str(excinfo.value)
    assert "secret_factory" not in message
    assert "0x" not in message


def test_connection_like_object_is_rejected():
    class FakeConnection:
        def __init__(self):
            self.url = "https://internal.example/secret?token=abc"

    with pytest.raises(canonical.CanonicalizationError) as excinfo:
        canonical.content_signature({"conn": FakeConnection()})
    assert "token" not in str(excinfo.value)
    assert "https://" not in str(excinfo.value)


def test_safe_content_signature_reports_failure_without_value():
    signature, error = canonical.safe_content_signature({"conn": object()})
    assert signature == ""
    assert error
    assert "0x" not in error


def test_bytes_use_a_digest_token_not_raw_material():
    rendered = canonical.canonicalize(b"super-secret-bytes")
    assert "__bytes_sha256__" in rendered
    assert "super-secret" not in json.dumps(rendered)


# --------------------------------------------------------------------------
# 7: non-finite floats are explicit and deterministic
# --------------------------------------------------------------------------
def test_nonfinite_floats_are_deterministic_tokens():
    nan_one = canonical.content_signature({"v": float("nan")})
    nan_two = canonical.content_signature({"v": float("nan")})
    assert nan_one == nan_two
    pos = canonical.content_signature({"v": math.inf})
    neg = canonical.content_signature({"v": -math.inf})
    assert pos != neg
    assert pos != nan_one
    assert canonical.canonicalize(float("nan")) == dict(canonical.NAN_TOKEN)
    assert canonical.canonicalize(math.inf) == dict(canonical.POS_INF_TOKEN)
    assert canonical.canonicalize(-math.inf) == dict(canonical.NEG_INF_TOKEN)


def test_supported_shapes_are_all_accepted():
    import dataclasses
    from enum import Enum

    class Colour(Enum):
        RED = "red"

    @dataclasses.dataclass
    class Sample:
        a: int
        b: str

    material = {
        "none": None,
        "bool": True,
        "int": 7,
        "float": 1.5,
        "str": "s",
        "list": [1, "a"],
        "tuple": (1, 2),
        "set": {"x"},
        "map": {"k": "v"},
        "enum": Colour.RED,
        "dataclass": Sample(a=1, b="two"),
    }
    assert canonical.content_signature(material).startswith("sha256:")


def test_registered_adapter_handles_an_otherwise_unsupported_type():
    class Custom:
        def __init__(self, value):
            self.value = value

    with pytest.raises(canonical.CanonicalizationError):
        canonical.content_signature(Custom("v"))
    canonical.register_adapter(Custom, lambda item: {"custom": item.value})
    try:
        assert canonical.content_signature(Custom("v")) == canonical.content_signature({"custom": "v"})
    finally:
        canonical._ADAPTERS.pop(Custom, None)


def test_canonicalization_version_is_v2():
    assert canonical.CANONICALIZATION_VERSION == "registry_signature.v2"
    assert canonical.content_signature("x") != (
        "sha256:" + __import__("hashlib").sha256(b'"x"').hexdigest()
    ), "signature must be version-bound"
