# DISTRIBUTION_SANITIZATION_MANIFEST

The downloadable patched tarball is a sanitized redistributable backend bundle. It excludes local environment files, historical backups, stale sidecars, caches, and PID files that are not required for source-code review or code deployment.

Excluded classes:

- `.env`, `.env.*`, and `internal_tools.env` files
- historical `*.backup*` / `*.bak*` files and backup directories
- obsolete `*.ollama_patched.py` sidecars
- obsolete `app/opus_metrics_router.py`
- old prompt backup files
- Python caches and pytest caches
- runtime PID files

Rationale:

- Active config reads secrets from environment variables.
- Active Ollama orchestration code is in normal package paths, not sidecars.
- Historical backup files contained token-looking literals; values were not printed. Rotate any real values before redistributing the original bundle.

Deployment note:

Use the target host's managed environment/secret store for required runtime values. Do not restore historical backup files into the active backend path unless they are sanitized first.

Additional documentation/script exclusions:

- `docs/deployment/AUTH_CONFIGURATION_REPORT.md`
- `docs/deployment/DEPLOYMENT_README.md`
- `scripts/archive/setup-dishchat*.sh`
- `scripts/archive/setup_sentry.sh`

These historical docs/scripts contained environment-token examples or defaults. They are not needed for the Ollama source patch and should be regenerated from sanitized templates if needed.

Additional high-signal secret-hygiene exclusions:

- `Token_Tracker/` and `Token_Tracker.bak_pre_v2/`
- `*.env` files anywhere in the tree
- historical deployment/implementation docs with environment-key examples
- `*.deprecated` files
