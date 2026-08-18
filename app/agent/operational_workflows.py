"""Deterministic operational-workflow intent detection (Phase D3B2A).

Repository checkout and isolated local deployment are real operational
tasks that require privileged executor tools.  This module answers exactly
one question, deterministically: *which exact tools does the requested task
need?*

It never decides whether those tools are authorized.  Authorization remains
owned by :mod:`app.agent.tool_authorization` and enforced independently by
:mod:`app.agent.tool_execution_gate`.  Requesting a tool is not a grant, and
"use your SSH key" is permission to use the configured SSH identity once the
required capabilities are authorized -- never an implicit grant of arbitrary
code execution.

The module is intentionally dependency-free with respect to the rest of
``app.agent`` so it can be imported by the profile builder, the methodology
selector, and the management facades without creating an import cycle.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

OPERATIONAL_WORKFLOW_SCHEMA = "diship_operational_workflow.v1"

#: Semantic name of the repository checkout + isolated local deployment task.
REPO_CHECKOUT_LOCAL_DEPLOY = "repo_checkout_local_deploy"

#: Semantic name of a direct local command / filesystem-inspection task that
#: has NO repository target.
#:
#: D3B-FIX(2026-08-18): before this existed, detect_operational_workflow()
#: returned None for every prompt without a repository URL, so a legitimate
#: request like "run these diagnostic bash commands on this host" or "read
#: every module under src/orchestrator/ and produce a plan" could never
#: register an operational need. Under the two GENERIC_CODE_EXECUTION_
#: METHODOLOGIES ("generic_engineering", "no_tool_response") that made code
#: execution *unreachable*: build_tool_profile() builds an allowlist from
#: `eligible INTERSECT operational_requested`, which was necessarily empty,
#: and an empty-but-not-None allowlist withholds every code-execution tool
#: ahead of the authorization gate. Granting all four authorization flags
#: could not override it. Reproduced live: a short "Proceed with the ...
#: task" follow-up selects generic_engineering, parses 4/4 flags true, and
#: still binds zero operational tools.
#:
#: This is a *capability request*, never an authorization decision. The tools
#: it declares still have to clear extra_tool_eligibility(), which for
#: agent_run_shell requires all four of operator_authorized,
#: heavy_tools_authorized, mutation_authorized and persistence_authorized.
LOCAL_COMMAND_EXECUTION = "local_command_execution"

#: Owning tool family for each privileged executor tool.  Requesting an exact
#: tool must activate its owner family without exposing the whole family.
OPERATIONAL_TOOL_OWNERS: dict[str, str] = {
    "agent_git_clone": "agent_mode",
    "agent_run_shell": "agent_mode",
    "agent_create_venv": "agent_mode",
    "agent_run_python": "agent_mode",
}

#: Hosts for which a private HTTPS URL may be deterministically rewritten to
#: an SSH clone URL.  An unlisted host is reported, never silently rewritten.
TRUSTED_GIT_HOSTS: tuple[str, ...] = (
    "gitlab.com",
    "github.com",
    "git.dtc.dish.corp",
)

# URL status vocabulary -----------------------------------------------------
URL_SSH_DERIVED = "SSH_URL_DERIVED"
URL_UNTRUSTED_HOST = "UNTRUSTED_REPOSITORY_HOST"
URL_MALFORMED = "MALFORMED_REPOSITORY_URL"
URL_NO_TARGET = "NO_REPOSITORY_TARGET"

# SSH environment vocabulary ----------------------------------------------
SSH_IDENTITY_CONFIGURED = "SSH_IDENTITY_CONFIGURED"
SSH_IDENTITY_NOT_CONFIGURED = "SSH_IDENTITY_NOT_CONFIGURED"
SSH_ENVIRONMENT_UNKNOWN = "SSH_ENVIRONMENT_UNKNOWN"

# ---------------------------------------------------------------------------
# Deterministic intent triggers
# ---------------------------------------------------------------------------
_CLONE_VERB_RE = re.compile(
    r"\b(?:git\s+)?clone(?:d|s|ing)?\b|\bchecks?\s?out\b|\bcheckout\b|\bpull\s+down\b",
    re.I,
)
_SSH_IDENTITY_RE = re.compile(
    r"\buse\s+(?:your|the|my|our|configured|existing)\s+ssh\s+"
    r"(?:key|keys|identity|identities|credential|credentials)\b",
    re.I,
)
_LOCAL_DEPLOY_RE = re.compile(
    r"\b(?:deploy|run|start|launch|serve|spin\s+up|stand\s+up|bring\s+up)\b"
    r"(?:\s+\S+){0,6}?\s+\b(?:local|locally|isolated|sandbox|sandboxed|on\s+this\s+host)\b",
    re.I,
)
_ENV_SETUP_RE = re.compile(
    r"\b(?:create|set\s?up|build|make)\b(?:\s+\S+){0,4}?\s+"
    r"\b(?:virtual\s?env(?:ironment)?|venv|environment)\b"
    r"|\binstall\s+(?:the\s+)?(?:dependencies|requirements|packages|deps)\b",
    re.I,
)
_RUN_TESTS_RE = re.compile(
    r"\brun\s+(?:the\s+|its\s+|repository\s+)?(?:unit\s+|integration\s+)?tests?\b|\bpytest\b",
    re.I,
)
_EXPLICIT_PYTHON_RE = re.compile(
    r"\bagent_run_python\b|\brun\s+(?:a\s+|the\s+)?python\s+(?:script|code|snippet|program)\b",
    re.I,
)

# D3B-FIX(2026-08-18): a direct local command / filesystem-inspection request
# with no repository target. Deliberately narrow -- it must NOT fire on
# ordinary chat. Verified non-matching against the D3B2A/D3B0 invariant
# fixtures: "Please summarise the attached document.", "Summarize the release
# notes.", "Draft an email about the outage."
#
# Two independent triggers:
#   1. An explicit command-execution directive naming a command/shell object.
#   2. An explicit inspection verb aimed at a concrete local path or filename.
# Trigger 2 is what makes "read README.md and every module under
# src/orchestrator/" a real capability request instead of unanswerable chat.
_LOCAL_EXEC_VERB_RE = re.compile(
    r"\b(?:run|execute|invoke)\b(?:\s+\S+){0,4}?\s+"
    r"\b(?:command|commands|shell|bash|sh|script|scripts|diagnostics?|"
    r"one-?liner|cli)\b",
    re.I,
)
#: A concrete local path or filename -- NOT a URL. A repository URL is handled
#: by the repository-target path above and must never reach this branch.
_LOCAL_PATH_TOKEN = (
    r"(?:(?:~|\.{1,2})?/[\w.@~-]+(?:/[\w.@~-]+)*/?"
    r"|\b[\w-]+\.(?:md|py|ya?ml|json|txt|j2|sh|toml|cfg|ini|lock)\b)"
)
_LOCAL_INSPECT_RE = re.compile(
    r"\b(?:read|inspect|analy[sz]e|examine|audit|open|list|cat|grep|"
    r"walk|enumerate)\b"
    r"(?:\s+\S+){0,8}?\s+" + _LOCAL_PATH_TOKEN,
    re.I,
)


def _local_execution_intent(text: str) -> bool:
    """True when a no-repository prompt still needs a local executor tool."""
    return bool(_LOCAL_EXEC_VERB_RE.search(text) or _LOCAL_INSPECT_RE.search(text))

#: A purely interrogative message is discussion, not an operational request.
_DISCUSSION_LEAD_RE = re.compile(
    r"^\s*(?:how|what|why|when|which|who|whose|where|does|do|did|can|could|"
    r"should|would|is|are|was|were|explain|describe|summari[sz]e|review)\b",
    re.I,
)
_IMPERATIVE_ACTION_RE = re.compile(
    r"\b(?:clone|deploy|install|checkout|launch|provision)\b"
    r"|\b(?:run|start|serve|set\s?up|create|build)\b(?:\s+\S+){0,6}?\s+"
    r"\b(?:local|locally|isolated|sandbox|sandboxed)\b",
    re.I,
)

# Repository target patterns ------------------------------------------------
_HTTPS_REPO_RE = re.compile(
    r"\bhttps?://(?:www\.)?([A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9])(?::\d+)?/(\S+)",
    re.I,
)
_SCP_REPO_RE = re.compile(
    r"\bgit@([A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9]):(\S+)",
)

_TRAILING_PUNCT = ",;:!)]}>\"'"
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _clean_path(raw: str) -> str:
    """Strip trailing punctuation, query/fragment, and a .git suffix."""
    value = str(raw or "").strip()
    value = value.split("#", 1)[0].split("?", 1)[0]
    while value and value[-1] in _TRAILING_PUNCT:
        value = value[:-1]
    value = value.strip("/")
    if value.lower().endswith(".git"):
        value = value[: -len(".git")]
    return value


def _path_is_valid(path: str) -> bool:
    if not path or ".." in path or "//" in path:
        return False
    segments = path.split("/")
    if len(segments) < 2:
        return False
    return all(_PATH_SEGMENT_RE.match(segment) for segment in segments)


def derive_ssh_clone_url(host: str, path: str) -> tuple[str, str]:
    """Return ``(ssh_url, status)`` for a repository host and path.

    Rewriting only happens for a trusted host and a validated path.  A
    malformed or untrusted target yields an empty URL plus an explicit
    status so the caller can report the real reason instead of guessing.
    """
    hostname = str(host or "").strip().lower().rstrip(".")
    cleaned = _clean_path(path)
    if not hostname or not _path_is_valid(cleaned):
        return "", URL_MALFORMED
    if hostname not in TRUSTED_GIT_HOSTS:
        return "", URL_UNTRUSTED_HOST
    return f"git@{hostname}:{cleaned}.git", URL_SSH_DERIVED


def extract_repository_target(prompt: str | None) -> tuple[str, str, str]:
    """Return ``(original, host, path)`` for the first repository reference."""
    text = str(prompt or "")
    match = _HTTPS_REPO_RE.search(text)
    if match:
        host, raw_path = match.group(1), match.group(2)
        original = match.group(0)
        while original and original[-1] in _TRAILING_PUNCT:
            original = original[:-1]
        return original, host, _clean_path(raw_path)
    match = _SCP_REPO_RE.search(text)
    if match:
        host, raw_path = match.group(1), match.group(2)
        original = match.group(0)
        while original and original[-1] in _TRAILING_PUNCT:
            original = original[:-1]
        return original, host, _clean_path(raw_path)
    return "", "", ""


def is_discussion_only(prompt: str | None) -> bool:
    """True when the message is a question about tooling, not a task.

    "How does git clone work?" and "What is a virtual environment?" must
    never activate privileged operational tools.
    """
    text = str(prompt or "").strip()
    if not text:
        return True
    if not _DISCUSSION_LEAD_RE.match(text):
        return False
    # An interrogative lead is still a task if it contains an imperative
    # operational directive on another line ("...? Clone it and deploy it.").
    if _IMPERATIVE_ACTION_RE.search(text) and "\n" in text:
        return False
    return text.endswith("?") or not _IMPERATIVE_ACTION_RE.search(text)


# ---------------------------------------------------------------------------
# Tool-name helpers
# ---------------------------------------------------------------------------
def _raw_name(value: str) -> str:
    text = str(value or "").strip()
    return text.split(":", 1)[1] if ":" in text else text


def operational_tool_owner(name: str) -> str:
    """Owning family for an operational tool, or "" when not operational."""
    return OPERATIONAL_TOOL_OWNERS.get(_raw_name(name), "")


def is_operational_tool(name: str) -> bool:
    return bool(operational_tool_owner(name))


def qualified_operational_tool(name: str) -> str:
    owner = operational_tool_owner(name)
    return f"{owner}:{_raw_name(name)}" if owner else ""


def operational_tools_in(names: Iterable[str] | None) -> tuple[str, ...]:
    """Subset of ``names`` that are operational tools, in the given form."""
    result: list[str] = []
    for value in names or ():
        text = str(value or "").strip()
        if text and is_operational_tool(text) and text not in result:
            result.append(text)
    return tuple(result)


def workflow_from_requested_tools(names: Iterable[str] | None) -> str:
    """Sticky workflow identity implied by already-requested exact tools."""
    return REPO_CHECKOUT_LOCAL_DEPLOY if operational_tools_in(names) else ""


# ---------------------------------------------------------------------------
# Workflow request
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperationalWorkflowRequest:
    """A deterministic statement of what a task needs -- not an authorization."""

    schema: str = OPERATIONAL_WORKFLOW_SCHEMA
    workflow: str = REPO_CHECKOUT_LOCAL_DEPLOY
    methodology: str = REPO_CHECKOUT_LOCAL_DEPLOY
    triggers: tuple[str, ...] = ()
    repository_url: str = ""
    repository_host: str = ""
    repository_path: str = ""
    ssh_clone_url: str = ""
    url_status: str = URL_NO_TARGET
    deployment_scope: str = "isolated_local"
    required_exact_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_operational_workflow(prompt: str | None) -> OperationalWorkflowRequest | None:
    """Detect a repository checkout / isolated local deployment request.

    Returns ``None`` for discussion, review-only, and every prompt without a
    real repository target.  Detection is a *capability request*, not an
    authorization decision, and never implies execution.
    """
    text = str(prompt or "")
    if not text.strip():
        return None
    if is_discussion_only(text):
        return None

    original, host, path = extract_repository_target(text)
    if not original:
        # D3B-FIX(2026-08-18): no repository target. A direct local command or
        # filesystem-inspection request is still a real operational need, so it
        # must be able to declare the executor tools it requires -- otherwise
        # code execution is unreachable under generic_engineering /
        # no_tool_response regardless of authorization (see
        # LOCAL_COMMAND_EXECUTION above).
        #
        # This branch is deliberately placed INSIDE the `not original` guard.
        # A prompt that *does* carry a repository URL (e.g. review-only
        # "Review the code quality of https://gitlab.com/...") never reaches
        # here and keeps its existing behaviour of returning None unless a
        # clone/ssh/deploy intent is present.
        if not _local_execution_intent(text):
            return None
        required_local = ["agent_run_shell"]
        if _EXPLICIT_PYTHON_RE.search(text):
            required_local.append("agent_run_python")
        return OperationalWorkflowRequest(
            workflow=LOCAL_COMMAND_EXECUTION,
            methodology=LOCAL_COMMAND_EXECUTION,
            triggers=("local_command_execution",),
            url_status=URL_NO_TARGET,
            deployment_scope="isolated_local",
            required_exact_tools=tuple(sorted(
                qualified_operational_tool(name) for name in required_local
            )),
        )

    clone_intent = bool(_CLONE_VERB_RE.search(text))
    ssh_identity_intent = bool(_SSH_IDENTITY_RE.search(text))
    deploy_intent = bool(_LOCAL_DEPLOY_RE.search(text))
    env_intent = bool(_ENV_SETUP_RE.search(text))
    tests_intent = bool(_RUN_TESTS_RE.search(text))
    python_intent = bool(_EXPLICIT_PYTHON_RE.search(text))

    if not (clone_intent or ssh_identity_intent or deploy_intent):
        return None

    triggers: list[str] = []
    if clone_intent:
        triggers.append("repository_checkout")
    if ssh_identity_intent:
        triggers.append("configured_ssh_identity")
    if deploy_intent:
        triggers.append("local_deployment")
    if env_intent:
        triggers.append("environment_setup")
    if tests_intent:
        triggers.append("repository_tests")
    if python_intent:
        triggers.append("explicit_python_execution")

    ssh_url, url_status = derive_ssh_clone_url(host, path)

    # Minimal required set.  agent_run_python is deliberately excluded unless
    # the request names it: ordinary shell and repository commands suffice for
    # checkout, dependency setup, startup, and health verification.
    required: list[str] = ["agent_git_clone"]
    if deploy_intent or env_intent or tests_intent:
        required.append("agent_run_shell")
    if deploy_intent or env_intent:
        required.append("agent_create_venv")
    if python_intent:
        required.append("agent_run_python")

    qualified = tuple(sorted({qualified_operational_tool(name) for name in required}))

    return OperationalWorkflowRequest(
        triggers=tuple(triggers),
        repository_url=original,
        repository_host=host.lower(),
        repository_path=path,
        ssh_clone_url=ssh_url,
        url_status=url_status,
        required_exact_tools=qualified,
    )


# ---------------------------------------------------------------------------
# Server-owned SSH environment classification
# ---------------------------------------------------------------------------
_IDENTITY_CANDIDATES: tuple[str, ...] = (
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
)


def ssh_environment_status(home: str | None = None) -> dict[str, Any]:
    """Report whether an SSH identity exists, without reading key material.

    Only booleans and counts are returned.  Key contents, filenames of
    private keys, and known-host entries are never exposed.  The agent may
    only report SSH availability from this server-owned evidence -- never
    from whether a tool happens to be bound.
    """
    executable = shutil.which("ssh") or ""
    base = os.path.join(home or os.path.expanduser("~"), ".ssh")
    identity_count = 0
    agent_config_present = False
    try:
        for candidate in _IDENTITY_CANDIDATES:
            if os.path.isfile(os.path.join(base, candidate)):
                identity_count += 1
        agent_config_present = os.path.isfile(os.path.join(base, "config"))
    except OSError:
        identity_count = 0
    if not executable:
        status = SSH_ENVIRONMENT_UNKNOWN
    elif identity_count > 0:
        status = SSH_IDENTITY_CONFIGURED
    else:
        status = SSH_IDENTITY_NOT_CONFIGURED
    return {
        "schema": "diship_ssh_environment_status.v1",
        "ssh_executable_present": bool(executable),
        "identity_file_count": identity_count,
        "client_config_present": agent_config_present,
        "status": status,
        "key_material_exposed": False,
    }
