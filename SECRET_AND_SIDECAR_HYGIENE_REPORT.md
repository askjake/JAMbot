# SECRET_AND_SIDECAR_HYGIENE_REPORT

Date: 2026-07-09

## Verdict
PASS_WITH_RISKS.

Active config and active Ollama sidecars are clean for this pass. Obsolete `.ollama_patched.py` sidecars and old prompt backups were quarantined out of the patched bundle. Some inactive config backup files in the original working tree contained bearer-token-looking literals; values were not printed, rotation is recommended, and the downloadable tarball excludes backup/env files.

## Active-code findings

| Check | Result |
|---|---|
| Active `app/config.py` contains hardcoded bearer token values | PASS: none found |
| Active `.ollama_patched.py` sidecars remain | PASS: none remain |
| Active prompt backup files remain in prompt directory | PASS: none remain |
| Reports print token values | PASS: no token values printed |

## Quarantined from patched working tree

See `SECRET_AND_SIDECAR_REMOVAL_MANIFEST.md` for path and sha256 only. Removed/quarantined classes:

- stale Ollama patch sidecars
- obsolete `app/opus_metrics_router.py`
- stale active-config backup `app/config.ollama_active.py.bak`
- old active-prompt backup files

## Remaining security-sensitive backup/doc findings

The redacted scanner produced `reports/hardening/bearer_literal_scan_redacted.json`.

- Active real literal suspects: 0
- Active placeholder examples: 2 documentation files with placeholder token examples
- Inactive backup literal suspects in source scan: 6 `app/config.py.backup*` files; excluded from downloadable tarball

Values were not copied into this report. Treat any real value in inactive backups as compromised and rotate it before redistributing this bundle outside the deployment boundary.

## Recommended next security step

Either quarantine all historical `app/config.py.backup*` files before external redistribution, or replace them with sanitized archival notes. Rotation is recommended for every token-like value found in those backups.

## Downloadable artifact sanitization

The patched tarball excludes local `.env*`, `internal_tools.env`, historical backup files, stale sidecars, selected historical docs/scripts with token examples, caches, and PID files. See `DISTRIBUTION_SANITIZATION_MANIFEST.md`.
