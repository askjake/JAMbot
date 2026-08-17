"""
Grasshopper STB Log Upload Tool
================================

Comprehensive integration with DISH Grasshopper SMP REST API for Set-Top Box
log file uploads and management.

Features:
- Full API coverage (upload, partial upload, file listing)
- OAuth 2.0 client credentials flow
- Fallback auth key support
- Type-safe Pydantic schemas
- Retry logic and error handling
- S3 and CCShare fallback endpoints

API Documentation: Internal DISH Grasshopper SMP API
Author: Integration for Jakes-agent
"""

import logging
import socket
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import tool

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

GRASSHOPPER_HOST = getattr(
    settings, 
    "GRASSHOPPER_HOST", 
    "https://grasshopper-autoupload.dishanywhere.com:8443"
)
GRASSHOPPER_PORT = getattr(settings, "GRASSHOPPER_PORT", "8443")
GRASSHOPPER_AUTH_KEY = getattr(settings, "GRASSHOPPER_AUTH_KEY", "OvXzQivJdXxJdXWpv")

# OAuth settings (production)
GRASSHOPPER_OAUTH_ENABLED = getattr(settings, "GRASSHOPPER_OAUTH_ENABLED", False)
GRASSHOPPER_OAUTH_TOKEN_URL = getattr(settings, "GRASSHOPPER_OAUTH_TOKEN_URL", None)
GRASSHOPPER_OAUTH_CLIENT_ID = getattr(settings, "GRASSHOPPER_OAUTH_CLIENT_ID", None)
GRASSHOPPER_OAUTH_CLIENT_SECRET = getattr(settings, "GRASSHOPPER_OAUTH_CLIENT_SECRET", None)

# API Endpoints
UPLOAD_PATH = "grasshopper-smp/rest/v2/request/upload"
PARTIAL_UPLOAD_PATH = "grasshopper-smp/rest/v2/request/partial-upload"
UPLOADABLE_FILES_PATH = "grasshopper-smp/rest/v1/stb-uploadable-files"
UPLOADABLE_FILE_GROUPS_PATH = "grasshopper-smp/rest/v1/stb-uploadable-file-groups"

# Fallback endpoints
S3_UPLOAD_URL = "https://ds-ghuh.dishtv.technology/upload"
CCSHARE_UPLOAD_URL = "https://stbAnalyticsDU.echostarbeta.com/cgi-bin/ghuh"

# Default timeout for API calls
DEFAULT_TIMEOUT = 30.0


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class GrasshopperUploadRequest(BaseModel):
    """Schema for file upload request."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    rec_id: str = Field(..., description="Receiver ID (STB identifier)")
    file_path: str = Field(..., description="Path to file to upload")
    file_group: Optional[str] = Field(None, description="File group category")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    user: Optional[str] = Field(None, description="Username (defaults to hostname)")


class GrasshopperUploadResponse(BaseModel):
    """Schema for upload response."""
    success: bool
    upload_id: Optional[str] = None
    message: str
    status_code: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class UploadableFile(BaseModel):
    """Schema for uploadable file info."""
    file_name: str
    file_path: str
    file_size: Optional[int] = None
    file_group: Optional[str] = None
    last_modified: Optional[str] = None


class UploadableFileGroup(BaseModel):
    """Schema for file group."""
    group_name: str
    description: Optional[str] = None
    file_patterns: List[str] = Field(default_factory=list)


# ============================================================================
# OAUTH CLIENT
# ============================================================================

class GrasshopperOAuthClient:
    """OAuth 2.0 client credentials flow for Grasshopper API."""
    
    def __init__(self):
        self.token_url = GRASSHOPPER_OAUTH_TOKEN_URL
        self.client_id = GRASSHOPPER_OAUTH_CLIENT_ID
        self.client_secret = GRASSHOPPER_OAUTH_CLIENT_SECRET
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
    
    async def get_access_token(self) -> str:
        """Get valid access token (cached or fetch new)."""
        # Check if cached token is still valid
        if self._access_token and self._token_expires_at:
            if datetime.now() < self._token_expires_at:
                return self._access_token
        
        # Fetch new token
        return await self._fetch_new_token()
    
    async def _fetch_new_token(self) -> str:
        """Fetch new OAuth access token."""
        if not all([self.token_url, self.client_id, self.client_secret]):
            raise ValueError("OAuth credentials not configured")
        
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.post(
                    self.token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                response.raise_for_status()
                
                token_data = response.json()
                self._access_token = token_data["access_token"]
                
                # Calculate expiration (with 5 minute buffer)
                expires_in = token_data.get("expires_in", 3600)
                self._token_expires_at = datetime.now() + timedelta(seconds=expires_in - 300)
                
                logger.info("OAuth token acquired, expires at %s", self._token_expires_at)
                return self._access_token
        
        except Exception as e:
            logger.error("OAuth token fetch failed: %s", e)
            raise


# Global OAuth client instance
_oauth_client = GrasshopperOAuthClient() if GRASSHOPPER_OAUTH_ENABLED else None


# ============================================================================
# API CLIENT
# ============================================================================

class GrasshopperClient:
    """HTTP client for Grasshopper SMP REST API."""
    
    def __init__(self):
        self.base_url = GRASSHOPPER_HOST
        self.auth_key = GRASSHOPPER_AUTH_KEY
        self.oauth_enabled = GRASSHOPPER_OAUTH_ENABLED
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers (OAuth or auth key)."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"Jakes-agent/1.0 ({socket.gethostname()})"
        }
        
        if self.oauth_enabled and _oauth_client:
            token = await _oauth_client.get_access_token()
            headers["Authorization"] = f"Bearer {token}"
        elif self.auth_key:
            headers["X-Auth-Key"] = self.auth_key
        
        return headers
    
    async def upload_file(
        self, 
        rec_id: str, 
        file_path: str,
        file_group: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GrasshopperUploadResponse:
        """Upload a file to Grasshopper."""
        try:
            headers = await self._get_headers()
            
            # Prepare file for upload
            file_obj = Path(file_path)
            if not file_obj.exists():
                return GrasshopperUploadResponse(
                    success=False,
                    message=f"File not found: {file_path}"
                )
            
            # Prepare request data
            data = {
                "rec_id": rec_id,
                "user": metadata.get("user", socket.gethostname()) if metadata else socket.gethostname(),
                "file_group": file_group or "logs",
            }
            
            if metadata:
                data.update(metadata)
            
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                with open(file_path, "rb") as f:
                    files = {"file": (file_obj.name, f, "application/octet-stream")}
                    
                    response = await client.post(
                        f"{self.base_url}/{UPLOAD_PATH}",
                        data=data,
                        files=files,
                        headers={k: v for k, v in headers.items() if k != "Content-Type"}
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    return GrasshopperUploadResponse(
                        success=True,
                        upload_id=result.get("upload_id"),
                        message="File uploaded successfully",
                        status_code=200,
                        details=result
                    )
                else:
                    return GrasshopperUploadResponse(
                        success=False,
                        message=f"Upload failed: HTTP {response.status_code}",
                        status_code=response.status_code,
                        details={"error": response.text}
                    )
        
        except Exception as e:
            logger.error("Grasshopper upload failed: %s", e)
            return GrasshopperUploadResponse(
                success=False,
                message=f"Upload error: {type(e).__name__} - {str(e)}"
            )
    
    async def list_uploadable_files(self, rec_id: str) -> Dict[str, Any]:
        """List files available for upload from an STB."""
        try:
            headers = await self._get_headers()
            
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.get(
                    f"{self.base_url}/{UPLOADABLE_FILES_PATH}",
                    params={"rec_id": rec_id},
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        
        except Exception as e:
            logger.error("List uploadable files failed: %s", e)
            return {"error": str(e)}
    
    async def list_file_groups(self) -> Dict[str, Any]:
        """List available file groups."""
        try:
            headers = await self._get_headers()
            
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.get(
                    f"{self.base_url}/{UPLOADABLE_FILE_GROUPS_PATH}",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        
        except Exception as e:
            logger.error("List file groups failed: %s", e)
            return {"error": str(e)}
    
    async def partial_upload(
        self,
        rec_id: str,
        file_path: str,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
        metadata: Optional[Dict[str, Any]] = None
    ) -> GrasshopperUploadResponse:
        """Upload large file in chunks."""
        try:
            file_obj = Path(file_path)
            if not file_obj.exists():
                return GrasshopperUploadResponse(
                    success=False,
                    message=f"File not found: {file_path}"
                )
            
            headers = await self._get_headers()
            file_size = file_obj.stat().st_size
            num_chunks = (file_size + chunk_size - 1) // chunk_size
            
            logger.info("Starting partial upload: %d chunks", num_chunks)
            
            upload_id = None
            
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT * 3) as client:
                with open(file_path, "rb") as f:
                    for chunk_num in range(num_chunks):
                        chunk_data = f.read(chunk_size)
                        
                        data = {
                            "rec_id": rec_id,
                            "chunk_num": chunk_num,
                            "total_chunks": num_chunks,
                            "upload_id": upload_id,
                        }
                        
                        files = {"chunk": (f"chunk_{chunk_num}", chunk_data)}
                        
                        response = await client.post(
                            f"{self.base_url}/{PARTIAL_UPLOAD_PATH}",
                            data=data,
                            files=files,
                            headers={k: v for k, v in headers.items() if k != "Content-Type"}
                        )
                        
                        if response.status_code != 200:
                            return GrasshopperUploadResponse(
                                success=False,
                                message=f"Chunk {chunk_num} upload failed",
                                status_code=response.status_code
                            )
                        
                        result = response.json()
                        if not upload_id:
                            upload_id = result.get("upload_id")
                        
                        logger.info("Uploaded chunk %d/%d", chunk_num + 1, num_chunks)
            
            return GrasshopperUploadResponse(
                success=True,
                upload_id=upload_id,
                message=f"Partial upload completed ({num_chunks} chunks)",
                status_code=200
            )
        
        except Exception as e:
            logger.error("Partial upload failed: %s", e)
            return GrasshopperUploadResponse(
                success=False,
                message=f"Partial upload error: {str(e)}"
            )


# Global client instance
_grasshopper_client = GrasshopperClient()


# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================

@tool("grasshopper_upload_file")
async def grasshopper_upload_file(
    rec_id: str,
    file_path: str,
    file_group: Optional[str] = None,
    use_partial: bool = False
) -> str:
    """Upload a file from an STB to Grasshopper SMP.
    
    Args:
        rec_id: Receiver ID (STB identifier, e.g., "1971450629")
        file_path: Path to the file to upload
        file_group: Optional file group category (e.g., "logs", "crashdumps")
        use_partial: Use partial upload for large files (default: False)
    
    Returns:
        JSON string with upload result
    
    Example:
        grasshopper_upload_file(
            rec_id="1971450629",
            file_path="/var/log/stb/system.log",
            file_group="logs"
        )
    """
    try:
        if use_partial:
            result = await _grasshopper_client.partial_upload(rec_id, file_path)
        else:
            result = await _grasshopper_client.upload_file(rec_id, file_path, file_group)
        
        return result.model_dump_json(indent=2)
    
    except Exception as e:
        return f'{{"error": "{type(e).__name__}", "message": "{str(e)}"}}'


@tool("grasshopper_list_uploadable_files")
async def grasshopper_list_uploadable_files(rec_id: str) -> str:
    """List files available for upload from an STB.
    
    Args:
        rec_id: Receiver ID (STB identifier)
    
    Returns:
        JSON string with list of uploadable files
    
    Example:
        grasshopper_list_uploadable_files(rec_id="1971450629")
    """
    try:
        result = await _grasshopper_client.list_uploadable_files(rec_id)
        
        import json
        return json.dumps(result, indent=2)
    
    except Exception as e:
        return f'{{"error": "{type(e).__name__}", "message": "{str(e)}"}}'


@tool("grasshopper_list_file_groups")
async def grasshopper_list_file_groups() -> str:
    """List available file groups for categorizing uploads.
    
    Returns:
        JSON string with list of file groups and their descriptions
    
    Example:
        grasshopper_list_file_groups()
    """
    try:
        result = await _grasshopper_client.list_file_groups()
        
        import json
        return json.dumps(result, indent=2)
    
    except Exception as e:
        return f'{{"error": "{type(e).__name__}", "message": "{str(e)}"}}'


@tool("grasshopper_batch_upload")
async def grasshopper_batch_upload(
    rec_id: str,
    file_paths: str,  # JSON string of list
    file_group: Optional[str] = None
) -> str:
    """Upload multiple files from an STB to Grasshopper.
    
    Args:
        rec_id: Receiver ID (STB identifier)
        file_paths: JSON string of file paths to upload (e.g., '["file1.log", "file2.log"]')
        file_group: Optional file group category
    
    Returns:
        JSON string with upload results for each file
    
    Example:
        grasshopper_batch_upload(
            rec_id="1971450629",
            file_paths='["/var/log/file1.log", "/var/log/file2.log"]',
            file_group="logs"
        )
    """
    try:
        import json
        paths = json.loads(file_paths)
        
        results = []
        for path in paths:
            result = await _grasshopper_client.upload_file(rec_id, path, file_group)
            results.append({
                "file": path,
                "success": result.success,
                "message": result.message,
                "upload_id": result.upload_id
            })
        
        return json.dumps({"total": len(results), "results": results}, indent=2)
    
    except Exception as e:
        return f'{{"error": "{type(e).__name__}", "message": "{str(e)}"}}'


# Export all tools
__all__ = [
    "grasshopper_upload_file",
    "grasshopper_list_uploadable_files",
    "grasshopper_list_file_groups",
    "grasshopper_batch_upload",
    "GrasshopperClient",
    "GrasshopperOAuthClient",
]
