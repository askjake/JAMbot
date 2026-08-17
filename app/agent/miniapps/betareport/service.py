from typing import Optional
import logging
import datetime
import json

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import get_settings

from .schemas import (
    BetaReport,
    IssueCandidate,
    IssueFeedback,
    PlatformEnum,
    Release,
    ReleaseShort,
)


setting = get_settings()
logger = logging.getLogger(__name__)


def get_betareport_service():
    return BetaReportService()


class BetaReportService:
    def __init__(self):
        self.mcp_client = MultiServerMCPClient(setting.BETAREPORT_MCP_CONFIG)

    async def _call_mcp_tool(self, tool_name: str, **kwargs):
        """Helper to call MCP tools"""
        tools = await self.mcp_client.get_tools()
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        return json.loads(await tool.with_retry().ainvoke(kwargs))

    async def list_available_platforms(self) -> list[str]:
        return [e.value for e in PlatformEnum]

    async def list_available_devices(self) -> list[str]:
        result = await self._call_mcp_tool("get_valid_device_models")
        return result.get("models", [])

    async def list_beta_releases(
        self,
        offset: int = 0,
        limit: int = 50,
        platform: Optional[str] = None,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> tuple[int, list[ReleaseShort]]:
        if not platform:
            raise ValueError("Platform is required")

        kwargs = {"platform": platform, "limit": limit, "offset": offset}
        if start_date or end_date:
            start = start_date.isoformat() if start_date else None
            end = end_date.isoformat() if end_date else None
            kwargs["date_range"] = (start, end)

        result = await self._call_mcp_tool("search_releases", **kwargs)
        releases = [Release(**r) for r in result.get("releases", [])]
        total = result.get("total", len(releases))

        results = [
            ReleaseShort(id=r.id, release=r.release, release_date=r.release_date)
            for r in releases
        ]

        return total, results

    async def list_beta_reports(
        self,
        offset: int = 0,
        limit: int = 50,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        release: Optional[int] = None,
        software: Optional[str] = None,
        platform: Optional[str] = None,
        device: Optional[str] = None,
    ) -> tuple[int, list[BetaReport]]:
        if not platform:
            raise ValueError("Platform is required")

        kwargs = {
            "platform": platform,
            "limit": limit,
            "offset": offset,
            "full_metadata": True,
        }
        # Default to last 30 days if no date range specified
        if not start_date and not end_date:
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=30)

        if start_date or end_date:
            start = start_date.isoformat() if start_date else None
            end = end_date.isoformat() if end_date else None
            kwargs["date_range"] = (start, end)

        if release is not None:
            kwargs["release_ids"] = [release]
        if software:
            kwargs["softwares"] = [software]
        if device:
            kwargs["devices"] = [device]

        result = await self._call_mcp_tool("search_reports", **kwargs)
        reports = [BetaReport(**r) for r in result.get("reports", [])]
        total = result.get("total", len(reports))

        return total, reports

    async def list_issue_candidates(
        self,
        email: str,
        offset: int = 0,
        limit: int = 50,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        release: Optional[int] = None,
        platform: Optional[str] = None,
        min_priority: Optional[int] = None,
        max_priority: Optional[int] = None,
    ) -> tuple[int, list[IssueCandidate]]:
        if not platform:
            raise ValueError("Platform is required")

        kwargs = {"platform": platform, "limit": limit, "offset": offset}
        # Default to last 30 days if no date range specified
        if not start_date and not end_date:
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=30)

        if start_date or end_date:
            start = start_date.isoformat() if start_date else None
            end = end_date.isoformat() if end_date else None
            kwargs["date_range"] = (start, end)

        if release is not None:
            kwargs["release_ids"] = [release]
        if min_priority is not None:
            kwargs["priority_min"] = min_priority
        if max_priority is not None:
            kwargs["priority_max"] = max_priority

        result = await self._call_mcp_tool("search_issues", **kwargs)
        issues = [IssueCandidate(**i) for i in result.get("issues", [])]
        total = result.get("total", len(issues))

        return total, issues

    async def process_issue_feedback(self, feedback: IssueFeedback, user_email: str):
        """Process feedback on an issue candidate"""
        feedback_text = feedback.comments or (
            "Accepted" if feedback.accept else "Rejected"
        )
        await self._call_mcp_tool(
            "send_feedback",
            feedback=feedback_text,
            positive=feedback.accept,
            issue_id=feedback.issue_id,
        )
