# Ported from dish-chat quick-wins
"""
Verification Transparency Layer for Dish-Chat Agent
====================================================

This module provides transparent verification of agent actions by:
1. Recording every verification command executed
2. Displaying verification results to users
3. Clearly distinguishing "I DID" vs "I PREPARED" vs "YOU SHOULD"
"""

from datetime import datetime
from typing import List, Dict, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field
import subprocess
import json


class VerificationStatus(str, Enum):
    """Status of a verification action"""
    VERIFIED = "verified"
    PENDING = "pending"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_PERFORMED = "not_performed"


class ActionType(str, Enum):
    """Type of action taken by agent"""
    DID = "I_DID"
    PREPARED = "I_PREPARED"
    SHOULD = "YOU_SHOULD"
    CANNOT = "I_CANNOT"


class VerificationRecord(BaseModel):
    """Individual verification record"""
    timestamp: datetime = Field(default_factory=datetime.now)
    command: str
    output: Optional[str] = None
    status: VerificationStatus
    action_type: ActionType
    description: str
    duration_ms: Optional[int] = None


class VerificationReport(BaseModel):
    """Complete verification report for an agent action"""
    action_description: str
    action_type: ActionType
    verifications: List[VerificationRecord] = Field(default_factory=list)
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_verification(self, 
                        command: str,
                        description: str,
                        action_type: ActionType,
                        status: VerificationStatus = VerificationStatus.PENDING) -> VerificationRecord:
        """Add a verification step"""
        record = VerificationRecord(
            command=command,
            description=description,
            status=status,
            action_type=action_type
        )
        self.verifications.append(record)
        return record
    
    def to_markdown(self) -> str:
        """Convert verification report to user-friendly markdown"""
        lines = []
        
        # Header with action type indicator
        icon_map = {
            ActionType.DID: "✅",
            ActionType.PREPARED: "📝",
            ActionType.SHOULD: "👉",
            ActionType.CANNOT: "❌"
        }
        icon = icon_map.get(self.action_type, "ℹ️")
        
        lines.append(f"## {icon} VERIFICATION REPORT\n")
        lines.append(f"**Action**: {self.action_description}\n")
        lines.append(f"**Type**: {self.action_type.value}\n")
        time_str = self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f"**Time**: {time_str}\n")
        lines.append("---\n")
        
        # Verification steps
        if self.verifications:
            lines.append("### Verification Steps:\n")
            for i, ver in enumerate(self.verifications, 1):
                status_map = {
                    VerificationStatus.VERIFIED: "✅",
                    VerificationStatus.PENDING: "⏳",
                    VerificationStatus.FAILED: "❌",
                    VerificationStatus.SKIPPED: "⏭️",
                    VerificationStatus.NOT_PERFORMED: "⚠️"
                }
                status_icon = status_map.get(ver.status, "•")
                
                lines.append(f"**{i}. {status_icon} {ver.description}**\n")
                lines.append(f"```bash\n{ver.command}\n```\n")
                
                if ver.output:
                    output_preview = ver.output[:500]
                    if len(ver.output) > 500:
                        output_preview += "\n... (truncated)"
                    lines.append(f"```\n{output_preview}\n```\n")
                
                if ver.duration_ms:
                    lines.append(f"*Duration: {ver.duration_ms}ms*\n")
                lines.append("")
        
        # Summary
        if self.summary:
            lines.append(f"### 📊 Summary\n{self.summary}\n")
        
        return "\n".join(lines)


class VerificationTransparencyLayer:
    """
    Main service for transparent verification
    """
    
    def __init__(self):
        self.reports: List[VerificationReport] = []
    
    def create_report(self, 
                     action_description: str,
                     action_type: ActionType) -> VerificationReport:
        """Create a new verification report"""
        report = VerificationReport(
            action_description=action_description,
            action_type=action_type
        )
        self.reports.append(report)
        return report
    
    def verify_command(self,
                      report: VerificationReport,
                      command: str,
                      description: str,
                      execute: bool = False,
                      timeout: int = 30) -> VerificationRecord:
        """
        Add verification command and optionally execute it
        """
        start_time = datetime.now()
        
        record = report.add_verification(
            command=command,
            description=description,
            action_type=report.action_type,
            status=VerificationStatus.PENDING if execute else VerificationStatus.NOT_PERFORMED
        )
        
        if execute:
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                record.output = result.stdout if result.returncode == 0 else result.stderr
                record.status = VerificationStatus.VERIFIED if result.returncode == 0 else VerificationStatus.FAILED
                
            except subprocess.TimeoutExpired:
                record.output = f"Command timed out after {timeout}s"
                record.status = VerificationStatus.FAILED
            except Exception as e:
                record.output = f"Error executing command: {str(e)}"
                record.status = VerificationStatus.FAILED
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            record.duration_ms = int(duration)
        
        return record
    
    def get_latest_report(self) -> Optional[VerificationReport]:
        """Get the most recent verification report"""
        return self.reports[-1] if self.reports else None
