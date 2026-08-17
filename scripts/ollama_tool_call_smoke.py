#!/usr/bin/env python3
"""Minimal LangChain/Ollama structured tool-call smoke test.

Runs independently of MCP. It binds only a local echo_probe tool first, then can
optionally score several local Ollama model names with 1/5/20 bound tools.

Usage:
  PYTHONPATH=. python scripts/ollama_tool_call_smoke.py \
    --models llama3.2:latest qwen3 qwen2.5 \
    --base-url http://127.0.0.1:11434
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _make_probe_tools(count: int):
    from langchain_core.tools import StructuredTool

    def echo_probe(value: str) -> dict[str, str]:
        """Return the supplied value as {'echo': value}."""
        return {"echo": value}

    tools = [
        StructuredTool.from_function(
            func=echo_probe,
            name="echo_probe",
            description="Echoes a string value. Use this when asked to call echo_probe.",
        )
    ]
    for idx in range(2, count + 1):
        def dummy(value: str, _idx: int = idx) -> dict[str, str]:
            """Dummy distractor tool."""
            return {"dummy": str(_idx), "value": value}

        tools.append(
            StructuredTool.from_function(
                func=dummy,
                name=f"dummy_probe_{idx}",
                description=f"Distractor probe tool {idx}; do not use unless asked by exact name.",
            )
        )
    return tools


async def _run_one(model_name: str, base_url: str, tool_count: int) -> dict[str, Any]:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, ToolMessage
    from langgraph.prebuilt import ToolNode

    tools = _make_probe_tools(tool_count)
    model = ChatOllama(model=model_name, base_url=base_url, temperature=0)
    bound = model.bind_tools(tools)
    messages = [HumanMessage(content="Call the `echo_probe` tool with value `hello-tool-test`. Do not answer directly.")]
    ai = await bound.ainvoke(messages)
    tool_calls = getattr(ai, "tool_calls", None) or []
    tool_result_messages: list[Any] = []
    toolnode_executed = False
    final_response = ""

    if tool_calls:
        tool_result = await ToolNode(tools=tools).ainvoke({"messages": [ai]})
        tool_result_messages = tool_result.get("messages", []) if isinstance(tool_result, dict) else []
        toolnode_executed = any(isinstance(m, ToolMessage) for m in tool_result_messages)
        if tool_result_messages:
            final = await model.ainvoke([*messages, ai, *tool_result_messages])
            final_response = str(getattr(final, "content", final))

    correct_tool = bool(tool_calls and tool_calls[0].get("name") == "echo_probe")
    correct_args = bool(tool_calls and (tool_calls[0].get("args") or {}).get("value") == "hello-tool-test")
    return {
        "model_role": "tool_worker_smoke",
        "model_name": model_name,
        "bound_tool_count": tool_count,
        "raw_ai_message": repr(ai),
        "tool_calls": _jsonable(tool_calls),
        "emits_valid_tool_call": bool(tool_calls),
        "correct_tool_selected": correct_tool,
        "correct_args": correct_args,
        "no_prose_only_answer": bool(tool_calls),
        "toolnode_executed": toolnode_executed,
        "tool_result_messages": [repr(m) for m in tool_result_messages],
        "final_response": final_response,
        "acceptance": bool(tool_calls and correct_tool and correct_args and toolnode_executed and "hello-tool-test" in final_response),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("PLLM_API_BASE") or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434")
    parser.add_argument("--models", nargs="*", default=[os.getenv("PLLM_TOOL_MODEL", "llama3.2:latest"), "qwen3", "qwen2.5"])
    parser.add_argument("--tool-counts", nargs="*", type=int, default=[1, 5, 20])
    parser.add_argument("--output", default="reports/ollama_tool_call_smoke.json")
    args = parser.parse_args()

    seen = []
    for model in args.models:
        if model and model not in seen:
            seen.append(model)

    results = []
    for model in seen:
        for count in args.tool_counts:
            try:
                results.append(await _run_one(model, args.base_url, count))
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "model_role": "tool_worker_smoke",
                    "model_name": model,
                    "bound_tool_count": count,
                    "acceptance": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump({"base_url": args.base_url, "results": results}, fh, indent=2, sort_keys=True)
    print(json.dumps({"base_url": args.base_url, "results": results}, indent=2, sort_keys=True))
    return 0 if any(r.get("acceptance") for r in results if r.get("bound_tool_count") == 1) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
