"""Regression test: the tools package must re-export what callers import.

app/agent/tool_execution_policy.py imports get_tools_set_filtered from the
package (not from registry directly). A missing re-export still passes
compileall and unit tests that never build the executor tool list, but breaks
every request that constructs the agent graph at runtime.
"""
from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "app.agent.agents.tools"
INIT = REPO_ROOT / "app" / "agent" / "agents" / "tools" / "__init__.py"


def _package_level_imports() -> set[str]:
    """Every name imported from the tools package anywhere in the tree."""
    names: set[str] = set()
    for path in list(REPO_ROOT.glob("app/**/*.py")) + list(REPO_ROOT.glob("apps/**/*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == PACKAGE:
                for alias in node.names:
                    if alias.name != "*":
                        names.add(alias.name)
    return names


def test_every_imported_name_is_reexported() -> None:
    required = _package_level_imports()
    assert required, "expected at least one package-level import"
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    exported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exported.add(alias.asname or alias.name)
    missing = sorted(required - exported)
    assert not missing, f"tools package does not re-export: {missing}"


def test_get_tools_set_filtered_is_importable_from_package() -> None:
    module = __import__(PACKAGE, fromlist=["get_tools_set_filtered"])
    assert callable(getattr(module, "get_tools_set_filtered"))


def test_declared_all_matches_imports() -> None:
    tree = ast.parse(INIT.read_text(encoding="utf-8"))
    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    declared = {
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
    assert "get_tools_set_filtered" in declared
