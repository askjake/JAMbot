from fastapi import APIRouter
from app.agent_mode.tools import agent_git_clone, agent_list_artifacts

router = APIRouter(prefix="/agent-mode/selftest", tags=["agent-mode-selftest"])


@router.post("/git-clone")
async def test_git_clone(url: str):
    """
    Test git clone functionality directly without going through the chat stack.
    Useful for verifying the sandbox environment is healthy.
    
    Example: POST /agent-mode/selftest/git-clone?url=https://github.com/user/repo.git
    """
    try:
        result = await agent_git_clone.ainvoke({"repo_url": url})
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/list-artifacts")
async def test_list_artifacts():
    """
    List all artifacts in the sandbox.
    """
    try:
        result = await agent_list_artifacts.ainvoke({})
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}
