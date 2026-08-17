#!/usr/bin/env python3
"""Redacted source-tree credential scanner.

Reports only kind, path, line, and classification.  Matched values are never
written to stdout or the JSON artifact.  The scanner distinguishes test/public
placeholders and variableized URLs from high-confidence embedded credentials.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int
    classification: str


_BINARY_SUFFIXES = {
    ".zip", ".gz", ".tgz", ".tar", ".png", ".jpg", ".jpeg", ".gif",
    ".pdf", ".pyc", ".sqlite", ".db", ".whl", ".ico", ".woff", ".woff2",
}
_SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}

# Split sensitive literals so this scanner does not match its own source.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("google_chat_webhook", re.compile(r"https://chat\.googleapis\.com/v1/" + r"spaces/[A-Za-z0-9_\-/?.=&]+")),
    ("google_api_key", re.compile("AI" + r"za[0-9A-Za-z_-]{30,}")),
    ("gitlab_token", re.compile("gl" + r"pat-[A-Za-z0-9_.-]{15,}")),
    ("sentry_token", re.compile("sntry" + r"u_[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile("AK" + r"IA[0-9A-Z]{16}")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("embedded_url_credentials", re.compile(r"https?://[^\s:/]+:[^\s@]+@[^\s]+")),
)


def _iter_text_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_PARTS for part in path.parts):
            continue
        try:
            if not path.is_file() or path.is_symlink():
                continue
        except OSError:
            continue
        if path.suffix.lower() in _BINARY_SUFFIXES:
            continue
        yield path


def _classify(*, kind: str, relative_path: str, line: str) -> str:
    lower_path = relative_path.lower()
    if relative_path == "coverity_assist_gateway.py" and "user:pass@proxy" in line:
        return "DOCUMENTATION_EXAMPLE"
    if "tests/" in lower_path or lower_path.startswith("test"):
        return "TEST_FIXTURE"
    if ("AKIA" + "IOSFODNN7EXAMPLE") in line:
        return "PUBLIC_PLACEHOLDER"
    if any(marker in line for marker in ("<REDACTED", "<INJECT_", "<SET_", "example.com")):
        return "REDACTED_OR_DOCUMENTATION_PLACEHOLDER"
    if "${" in line or "$GITLAB_PAT" in line or "$CI_PUSH_TOKEN" in line:
        return "VARIABLEIZED_CREDENTIAL_REFERENCE"
    if kind == "embedded_url_credentials" and "user:pass@proxy" in line:
        return "DOCUMENTATION_EXAMPLE"
    return "LIVE_CREDENTIAL_CANDIDATE"


def scan(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path in _iter_text_files(root):
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except (OSError, UnicodeError):
            continue
        relative_path = path.relative_to(root).as_posix()
        for line_number, line in enumerate(lines, start=1):
            for kind, pattern in _PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(
                        kind=kind,
                        path=relative_path,
                        line=line_number,
                        classification=_classify(
                            kind=kind,
                            relative_path=relative_path,
                            line=line,
                        ),
                    ))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-live", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = scan(root)
    live = [item for item in findings if item.classification == "LIVE_CREDENTIAL_CANDIDATE"]
    payload = {
        "schema": "redacted_source_secret_scan.v1",
        "root_name": root.name,
        "finding_count": len(findings),
        "live_candidate_count": len(live),
        "status": "PASS" if not live else "FAIL",
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 1 if args.fail_on_live and live else 0


if __name__ == "__main__":
    raise SystemExit(main())
