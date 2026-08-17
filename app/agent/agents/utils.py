from typing import Any
import logging
from functools import cache
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import get_settings

settings = get_settings()


@cache
def get_prompt(name: str):
    current_dir = Path(__file__).parent
    prompt_path = current_dir / "prompts" / f"{name}_prompt.txt"
    with open(prompt_path) as f:
        prompt = f.read()
    return prompt

async def get_mcp_tools(config: dict) -> list[BaseTool]:
    client = MultiServerMCPClient(config)
    return await client.get_tools()
