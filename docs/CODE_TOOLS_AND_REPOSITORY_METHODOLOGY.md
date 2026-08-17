---
document_type: methodology
protocol_id: code-tools-and-repos
version: "1.0"
status: active
priority: sub-protocol
date: 2026-06-24
owner: montjac
parent_protocol: mcop
triggers:
  - REPOSITORY_NOT_FOUND
  - repository not indexed
  - git tools failing
  - code search returns no results
  - dish-code-tools
  - need to search source code
  - code path reconstruction
  - find_symbol returns empty
  - browse_directory returns error
  - search_regex returns no matches
  - commit lineage investigation
  - source code analysis
  - Qodo coverage gap
  - QODO_COVERAGE_GAP
---

# CODE TOOLS AND REPOSITORY METHODOLOGY

**Protocol ID:** `code-tools-and-repos`
**Version:** 1.0
**Date:** 2026-06-24
**Owner:** montjac
**Status:** ACTIVE
**Parent Protocol:** `mcop`

---

## 1. PURPOSE

This protocol governs how to use the two complementary code intelligence systems
(Qodo semantic search and dish-code-tools deterministic search) and how to resolve
repository coverage gaps. It answers:

- What code tools are available and what each is best for
- What repositories are indexed and where
- How to diagnose and fix a REPOSITORY_NOT_FOUND condition
- How to add new repositories so future investigations succeed

---

## 2. TWO CODE INTELLIGENCE SYSTEMS

### Qodo Context MCP (semantic search)

- **MCP tool set:** `qodo_context_mcp` (4 tools)
- **Tools:** `list_repositories`, `get_context`, `ask`, `deep_research`
- **Strengths:** Semantic understanding, cross-file reasoning, call-path tracing
- **Weaknesses:** Index may be stale, cannot return exact line numbers reliably,
  coverage limited to what Qodo has indexed
- **Indexed repos (as of 2026-06-24):** DT-ENG/modules_stbhlive, DT-ENG/src_tree,
  DT-ENG/modules_pipeline_control, DT-ENG/DeviceManager, DT-ENG/stbctrl,
  DT-ENG/TvManager, DT-ENG/modules_media_player_api, DT-DEVOPS/ansible,
  DT-ENG/modules_player_health, DT-ENG/modules_qos
- **How to verify:** Call `list_repositories` — returns authoritative current list

### dish-code-tools MCP (deterministic search)

- **MCP tool set:** `dish_code_tools` (12 tools)
- **Tools:** `browse_directory`, `read_file`, `search_regex`, `find_symbol`,
  `find_references`, `get_repo_status`, `refresh_repository_history`,
  `fetch_change_ref`, `search_git_history`, `get_commit_details`, `get_log`, `get_diff`
- **Strengths:** Exact grep, full file reads, git history/blame/diff, symbol lookup,
  call-site discovery — deterministic and reproducible
- **Weaknesses:** Only works on repos that are cloned to disk
- **Repos on disk (as of 2026-06-24):** stbctrl, DeviceManager, TvManager, src_tree,
  gui_qt, atv_utils, ATV_Qt_UI, modules_sg_server, modules_stbhlive,
  modules_pipeline_control, modules_media_player_api, modules_player_health, modules_qos
- **Storage path:** `/mnt/tnas/public/repos/code_tools/`
- **Discovery:** Automatic — any git repo cloned to that path is available on next restart
- **How to verify:** Call `browse_directory` with the repo name

---

## 3. WHICH TOOL TO USE WHEN

| Task | Best tool | Fallback |
|------|-----------|----------|
| Understand a subsystem broadly | Qodo `get_context` | dish-code-tools `browse_directory` + `read_file` |
| Trace a call path across files | Qodo `ask` or `deep_research` | dish-code-tools `find_references` chain |
| Find exact error string / log message | dish-code-tools `search_regex` | Qodo `get_context` with the string |
| Find function definition | dish-code-tools `find_symbol` | Qodo `ask` |
| Find all callers of a function | dish-code-tools `find_references` | Qodo `ask` |
| Read a specific file in full | dish-code-tools `read_file` | Qodo `get_context` (returns excerpts only) |
| Search git history for a commit | dish-code-tools `search_git_history` | (no alternative) |
| Compare two revisions | dish-code-tools `get_diff` | (no alternative) |
| Find a fix by Jira key in commits | dish-code-tools `search_git_history` mode=message | (no alternative) |
| Find code that was added/removed | dish-code-tools `search_git_history` mode=pickaxe_string | (no alternative) |

**Always try dish-code-tools first for deterministic lookups.** Use Qodo when you need
semantic understanding or when the exact search terms are unknown.

---

## 4. DIAGNOSING REPOSITORY_NOT_FOUND

When a code search returns "Git repository not found" or "repository not indexed":

### Step 1 — Identify which system failed

- If `dish_code_tools` returned the error: the repo is not cloned to disk
- If Qodo `list_repositories` does not include the repo: the repo is not in Qodo's index
- Both can be true simultaneously

### Step 2 — Verify the repo exists on the git server

The DISH internal git server is `git.dtc.dish.corp` (GitHub Enterprise).
Repos are typically under the `DT-ENG/` organization.

Test with SSH: `ssh git@git.dtc.dish.corp` (should return "Hi <user>!")

### Step 3 — Clone the missing repo to TNAS

```bash
cd /mnt/tnas/public/repos/code_tools
git clone git@git.dtc.dish.corp:DT-ENG/<repo_name>.git
```

For very large repos (e.g., src_tree), use shallow clone:
```bash
git clone --depth=1 git@git.dtc.dish.corp:DT-ENG/<repo_name>.git
```

### Step 4 — Restart dish-code-tools (or full agent restart)

The dish-code-tools MCP server auto-discovers repos at startup by scanning
`REPOS_BASE` for directories containing `.git`. After cloning, restart:

```bash
bash /home/jakebot/Jakes-agent/restart-dishchat.sh
```

Or restart just the dish-code-tools process:
```bash
kill $(lsof -ti:8087)
cd /home/montjac/dish-code-tools
REPOS_BASE=/mnt/tnas/public/repos/code_tools nohup .venv/bin/python -m app \
    >> /home/jakebot/Jakes-agent/logs/dish-code-tools.log 2>&1 &
```

### Step 5 — Verify

Call `browse_directory` with the new repo name. If it returns a directory tree,
the repo is available.

---

## 5. QODO COVERAGE GAPS

If a repo IS in Qodo's `list_repositories` but returns no results for a query:

- The index may be stale (check `last_modified_date` on returned chunks)
- The code may be in a binary JAR/AAR (e.g., ndmapi.jar contains IpAddr.java)
- The file may be auto-generated (AIDL, protobuf, etc.)
- The branch indexed may not contain the fix yet

**Do not use Qodo absence as evidence the code does not exist.**
Always cross-check with dish-code-tools `search_regex` or `find_symbol`.

If BOTH Qodo and dish-code-tools fail to find a file that Jira references,
classify as `QODO_COVERAGE_GAP` and document:
- What was searched for
- Which tools were tried
- Whether the code is in a binary/generated artifact
- Whether a different repo might contain it

---

## 6. REPOSITORY INVENTORY

### Storage layout

```
/mnt/tnas/public/repos/code_tools/
├── ATV_Qt_UI/               (Java, Android Qt host APK)
├── DeviceManager/           (Java, Android device management)
├── TvManager/               (Java, TV management app)
├── atv_utils/               (Java, shared Android utilities)
├── gui_qt/                  (QML/JS/C++, Rigel UI layer)
├── modules_media_player_api/ (media player interface)
├── modules_pipeline_control/ (pipeline control)
├── modules_player_health/   (player health monitoring)
├── modules_qos/             (quality of service metrics)
├── modules_sg_server/       (SGS proxy, core_sgs, sgsproxy, sgs services)
├── modules_stbhlive/        (STB HLive module)
├── src_tree/                (C/Makefiles, STB firmware build system) [SHALLOW]
└── stbctrl/                 (C, STB controller application)
```

### Git remote pattern

All DT-ENG repos: `git@git.dtc.dish.corp:DT-ENG/<repo_name>.git`
DevOps repos: `git@git.dtc.dish.corp:DT-DEVOPS/<repo_name>.git` (restricted access)

### Adding a new repo (quick reference)

```bash
cd /mnt/tnas/public/repos/code_tools
git clone git@git.dtc.dish.corp:DT-ENG/<repo_name>.git
# Then restart the agent — discovery is automatic
bash /home/jakebot/Jakes-agent/restart-dishchat.sh
```

### Updating repos

```bash
cd /mnt/tnas/public/repos/code_tools/<repo_name>
git pull
```

For shallow repos that need full history (for pickaxe searches):
```bash
cd /mnt/tnas/public/repos/code_tools/src_tree
git fetch --unshallow
```

---

## 7. CONFIGURATION REFERENCE

| Setting | Location | Value |
|---------|----------|-------|
| REPOS_BASE | restart-dishchat.sh Phase 4d | `/mnt/tnas/public/repos/code_tools` |
| dish-code-tools source | disk | `/home/montjac/dish-code-tools/` |
| dish-code-tools port | restart-dishchat.sh | `localhost:8087` |
| dish-code-tools log | disk | `/home/jakebot/Jakes-agent/logs/dish-code-tools.log` |
| Qodo MCP port-forward | restart-dishchat.sh Phase 4c | `localhost:18443 → cluster svc/qodo-ssh-proxy:8443` |
| Qodo EKS context | restart-dishchat.sh | `arn:aws:eks:us-west-2:233532778289:cluster/apps-xx-eks-gltqp-2fq9x` |

---

## 8. CONSTRAINTS

- **Do not modify source repos on the TNAS.** They are read-only references.
- **Do not use dish-code-tools absence as evidence code does not exist** — the repo
  may simply not be cloned yet.
- **Do not assume Qodo index reflects the latest commit.** Check `last_modified_date`
  on returned chunks when freshness matters.
- **DT-DEVOPS/ansible is access-restricted** — do not attempt to clone without
  explicit authorization.
- **src_tree is shallow cloned** — `search_git_history` with pickaxe will not work
  until `git fetch --unshallow` is run.

---

## 9. RELATED DOCUMENTS

- `docs/MCOP_METHODOLOGY.md` — parent orchestration protocol
- `docs/S3_STB_LOG_ANALYSIS_METHODOLOGY.md` — runtime validation after code analysis
- `docs/STB_FIELD_RCA_AND_COHORT_METHODOLOGY.md` — full RCA workflow
- `/home/montjac/dish-code-tools/README.md` — dish-code-tools server documentation

---

## 10. REVISION HISTORY

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 1.0 | 2026-06-24 | montjac | Initial version — 13 repos, dual-system methodology |
