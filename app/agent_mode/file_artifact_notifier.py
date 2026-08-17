# Ported from dish-chat quick-wins
"""
File Artifact Notification System for Dish-Chat Agent
======================================================

Automatically notifies users when files are created/modified by the agent,
with download links and previews.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import mimetypes
import hashlib


class FileArtifact(BaseModel):
    """Represents a file created by the agent"""
    filename: str
    filepath: str
    size_bytes: int
    mime_type: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    download_url: Optional[str] = None
    preview: Optional[str] = None
    checksum: Optional[str] = None
    
    @property
    def size_human(self) -> str:
        """Human-readable file size"""
        size = self.size_bytes
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    @property
    def file_icon(self) -> str:
        """Get emoji icon for file type"""
        if not self.mime_type:
            return "📄"
        
        if self.mime_type.startswith("text/"):
            return "📝"
        elif self.mime_type == "application/json":
            return "📊"
        elif self.mime_type == "application/pdf":
            return "📕"
        elif self.mime_type.startswith("image/"):
            return "🖼️"
        elif self.mime_type.startswith("video/"):
            return "🎬"
        elif self.mime_type.startswith("audio/"):
            return "🎵"
        elif self.mime_type in ["application/zip", "application/x-tar", "application/gzip"]:
            return "📦"
        
        return "📄"


class ArtifactNotification(BaseModel):
    """Notification about created/modified files"""
    title: str = "Files Created"
    artifacts: List[FileArtifact] = Field(default_factory=list)
    workspace_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    def to_markdown(self) -> str:
        """Convert notification to user-friendly markdown"""
        if not self.artifacts:
            return ""
        
        lines = []
        lines.append(f"### 📂 {self.title}\n")
        
        if len(self.artifacts) == 1:
            lines.append("I created **1 file** for you:\n")
        else:
            lines.append(f"I created **{len(self.artifacts)} files** for you:\n")
        
        for artifact in self.artifacts:
            lines.append(f"**{artifact.file_icon} {artifact.filename}** ({artifact.size_human})")
            
            if artifact.download_url:
                lines.append(f"  - [📥 Download]({artifact.download_url})")
            else:
                lines.append(f"  - Location: `{artifact.filepath}`")
            
            if artifact.mime_type:
                lines.append(f"  - Type: {artifact.mime_type}")
            
            if artifact.preview:
                preview_text = artifact.preview[:100]
                lines.append(f"  - Preview: {preview_text}...")
            
            lines.append("")
        
        if self.workspace_path:
            lines.append(f"*All files saved to: `{self.workspace_path}`*\n")
        
        return "\n".join(lines)


class FileArtifactNotifier:
    """
    Service for tracking and notifying about file artifacts
    """
    
    def __init__(self, workspace_path: str, base_download_url: str = "/rest/api/v1/agent-mode/artifacts"):
        self.workspace_path = Path(workspace_path)
        self.base_download_url = base_download_url
        self.tracked_files: List[FileArtifact] = []
    
    def track_file(self, 
                   filename: str,
                   include_preview: bool = True,
                   preview_lines: int = 5) -> FileArtifact:
        """Track a file that was created/modified"""
        filepath = self.workspace_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get file info
        size_bytes = filepath.stat().st_size
        mime_type, _ = mimetypes.guess_type(str(filepath))
        
        # Calculate checksum
        checksum = self._calculate_checksum(filepath)
        
        # Get preview for text files
        preview = None
        if include_preview and mime_type and mime_type.startswith("text/"):
            try:
                with open(filepath, 'r') as f:
                    lines = [f.readline() for _ in range(preview_lines)]
                    preview = ''.join(lines).strip()
            except Exception:
                preview = None
        
        # Build download URL
        workspace_name = self.workspace_path.name
        download_url = f"{self.base_download_url}/{workspace_name}/{filename}"
        
        artifact = FileArtifact(
            filename=filename,
            filepath=str(filepath),
            size_bytes=size_bytes,
            mime_type=mime_type,
            download_url=download_url,
            preview=preview,
            checksum=checksum
        )
        
        self.tracked_files.append(artifact)
        return artifact
    
    def track_directory(self, 
                       directory: str = ".",
                       pattern: str = "*",
                       recursive: bool = False) -> List[FileArtifact]:
        """Track all files in a directory"""
        dir_path = self.workspace_path / directory
        
        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)
        
        artifacts = []
        for file_path in files:
            if file_path.is_file():
                relative_path = file_path.relative_to(self.workspace_path)
                artifact = self.track_file(str(relative_path))
                artifacts.append(artifact)
        
        return artifacts
    
    def create_notification(self, title: str = "Files Created") -> ArtifactNotification:
        """Create notification from tracked files"""
        notification = ArtifactNotification(
            title=title,
            artifacts=self.tracked_files.copy(),
            workspace_path=str(self.workspace_path)
        )
        
        # Clear tracked files after notification
        self.tracked_files.clear()
        
        return notification
    
    def auto_notify_on_exit(self, title: str = "Task Complete - Files Created") -> Optional[str]:
        """Automatically create notification if files were tracked"""
        if not self.tracked_files:
            return None
        
        notification = self.create_notification(title)
        return notification.to_markdown()
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
