import time
import functools
from collections import defaultdict
from typing import Callable

_CALL_TIMES: dict = defaultdict(list)
RATE_LIMITS = {
    "agent_run_shell": (20, 15),
    "agent_run_python": (20, 15),
    "agent_git_clone": (5, 60),
    "agent_create_venv": (5, 60),
}

def rate_limited(tool_name: str, user_key: str = "global") -> tuple:
    if tool_name not in RATE_LIMITS:
        return True, ""
    limit, window = RATE_LIMITS[tool_name]
    key = f"{tool_name}:{user_key}"
    now = time.time()
    calls = [t for t in _CALL_TIMES[key] if now - t < window]
    _CALL_TIMES[key] = calls
    if len(calls) >= limit:
        oldest = min(calls)
        wait = int(window - (now - oldest))
        return False, f"Rate limit: {limit}/{window}s. Wait {wait}s."
    _CALL_TIMES[key].append(now)
    return True, ""
