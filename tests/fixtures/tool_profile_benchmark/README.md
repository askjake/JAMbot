# tool_profile_benchmark fixtures

Phase D3B2 measures tool-profile stability, model-facing schema churn, and
cache-key behaviour. Those measurements need a registry whose semantic content
we control exactly, so an "equivalent refresh" can be distinguished from a real
inventory change.

## What is committed

* `registry_baseline.sha256` — the digest lock for the deterministic fixture.

The fixture body itself is **not** committed. It is generated in memory by
`scripts/benchmarks/generate_tool_profile_fixture.py`, which is deterministic:
re-running it always produces byte-identical output. Committing the generator
plus a digest lock keeps the repository small while still detecting accidental
fixture drift — `tool_profile_cache_benchmark.py --verify-fixture-lock` and
`tests/test_tool_profile_cache_benchmark_d3b2.py` both fail if the generated
digest stops matching the lock.

## What the fixture contains

Real family names and real per-family tool counts, so schema-size measurements
are structurally realistic. Everything else is synthetic:

* every description is generated from the family and tool name;
* every argument name is `primary_id` or `param_N`, plus the real
  server-controlled authorization argument names where the sanitizer must be
  exercised (`allow_heavy`, `heavy_auth_token`);
* no real tool description, no real argument default, no credential, no
  receiver identifier, and no customer data is present.

`list_incident_scenes` and `build_incident_scene` are deliberately **absent**.
The current S3 runtime does not expose Incident Scene tools, and the benchmark
must reproduce that upstream gap rather than assume it away.

## Regenerating

    python scripts/benchmarks/generate_tool_profile_fixture.py \
        --out /tmp/registry_baseline.json

Then update `registry_baseline.sha256` only if the change is intended.
