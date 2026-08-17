#!/usr/bin/env python3
"""
Google Drive MCP Server
=======================
A FastMCP server that exposes Google Drive capabilities as MCP tools.
Integrates with Jakes-agent via the standard streamable_http transport.

Authentication options (in priority order):
  1. Service Account JSON key file  → GDRIVE_SERVICE_ACCOUNT_FILE env var
  2. Service Account JSON string    → GDRIVE_SERVICE_ACCOUNT_JSON env var
  3. OAuth2 credentials file        → GDRIVE_OAUTH_CREDENTIALS_FILE env var
     (requires one-time browser auth, then token stored at GDRIVE_TOKEN_FILE)

Scopes:
  - Read-only: https://www.googleapis.com/auth/drive.readonly
  - Full:      https://www.googleapis.com/auth/drive

Start:
  python apps/gdrive_mcp/server.py
  uvicorn apps.gdrive_mcp.server:mcp_app --host 0.0.0.0 --port 8090
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

# ── Configuration from environment ────────────────────────────────────────────
PORT = int(os.getenv("GDRIVE_MCP_PORT", "8090"))
HOST = os.getenv("GDRIVE_MCP_HOST", "127.0.0.1")

SERVICE_ACCOUNT_FILE   = os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE")
SERVICE_ACCOUNT_JSON   = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
OAUTH_CREDENTIALS_FILE = os.getenv("GDRIVE_OAUTH_CREDENTIALS_FILE")
TOKEN_FILE             = os.getenv("GDRIVE_TOKEN_FILE", "/home/jakebot/Jakes-agent/state/gdrive_token.pickle")

READONLY_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FULL_SCOPES     = ["https://www.googleapis.com/auth/drive"]

# Use readonly by default; override with GDRIVE_FULL_ACCESS=1 for write ops
SCOPES = FULL_SCOPES if os.getenv("GDRIVE_FULL_ACCESS", "0") == "1" else READONLY_SCOPES

# ── Auth helper ────────────────────────────────────────────────────────────────

def _get_credentials():
    """
    Build Google API credentials from the best available source.
    Returns a google.auth.credentials.Credentials object or raises.
    """
    # 1. Service Account JSON file
    if SERVICE_ACCOUNT_FILE and Path(SERVICE_ACCOUNT_FILE).exists():
        from google.oauth2 import service_account
        logger.info("Auth: service account file → %s", SERVICE_ACCOUNT_FILE)
        return service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )

    # 2. Service Account JSON string
    if SERVICE_ACCOUNT_JSON:
        from google.oauth2 import service_account
        logger.info("Auth: service account JSON (env var)")
        info = json.loads(SERVICE_ACCOUNT_JSON)
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    # 3. OAuth2 token file (previously authorised)
    token_path = Path(TOKEN_FILE)
    if token_path.exists():
        from google.auth.transport.requests import Request
        logger.info("Auth: cached OAuth2 token → %s", TOKEN_FILE)
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, "wb") as f:
                pickle.dump(creds, f)
        return creds

    # 4. OAuth2 interactive flow (headless-friendly: prints URL, reads code)
    if OAUTH_CREDENTIALS_FILE and Path(OAUTH_CREDENTIALS_FILE).exists():
        from google_auth_oauthlib.flow import InstalledAppFlow
        logger.info("Auth: OAuth2 interactive flow → %s", OAUTH_CREDENTIALS_FILE)
        flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CREDENTIALS_FILE, SCOPES)
        # console=True avoids needing a browser on the server
        creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)
        logger.info("Token saved → %s", TOKEN_FILE)
        return creds

    raise RuntimeError(
        "No Google credentials found. Set one of: "
        "GDRIVE_SERVICE_ACCOUNT_FILE, GDRIVE_SERVICE_ACCOUNT_JSON, "
        "or GDRIVE_OAUTH_CREDENTIALS_FILE"
    )


def _drive_service():
    """Return an authenticated Google Drive API v3 service."""
    from googleapiclient.discovery import build
    creds = _get_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ── MCP Server ────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "google-drive",
    instructions=(
        "Tools for reading and searching Google Drive files. "
        "Supports Docs, Sheets, Slides, PDFs, and any Drive file type."
    ),
)


@mcp.tool()
def gdrive_search(
    query: str,
    max_results: int = 20,
    file_type: Optional[str] = None,
    parent_folder_id: Optional[str] = None,
) -> str:
    """
    Search Google Drive for files matching a query.

    Args:
        query:            Text to search for in file names and full text.
        max_results:      Max files to return (default 20, max 100).
        file_type:        Optional filter — one of: document, spreadsheet,
                          presentation, pdf, folder, image, video.
        parent_folder_id: Restrict search to a specific folder ID.

    Returns:
        JSON list of matching files with id, name, mimeType, webViewLink,
        modifiedTime, size.
    """
    mime_map = {
        "document":     "application/vnd.google-apps.document",
        "spreadsheet":  "application/vnd.google-apps.spreadsheet",
        "presentation": "application/vnd.google-apps.presentation",
        "pdf":          "application/pdf",
        "folder":       "application/vnd.google-apps.folder",
        "image":        "image/",   # prefix match applied below
        "video":        "video/",
    }

    q_parts = ["trashed=false"]
    if query:
        safe = query.replace("'", "\\'")
        q_parts.append(f"(name contains '{safe}' or fullText contains '{safe}')")
    if file_type:
        ft = file_type.lower()
        if ft in mime_map:
            mime = mime_map[ft]
            if mime.endswith("/"):
                q_parts.append(f"mimeType contains '{mime}'")
            else:
                q_parts.append(f"mimeType='{mime}'")
    if parent_folder_id:
        q_parts.append(f"'{parent_folder_id}' in parents")

    drive = _drive_service()
    result = drive.files().list(
        q=" and ".join(q_parts),
        pageSize=min(max_results, 100),
        fields="files(id,name,mimeType,webViewLink,modifiedTime,size,parents)",
        orderBy="modifiedTime desc",
    ).execute()

    files = result.get("files", [])
    return json.dumps({"count": len(files), "files": files}, indent=2)


@mcp.tool()
def gdrive_read_file(
    file_id: str,
    export_format: Optional[str] = None,
) -> str:
    """
    Read/export the content of a Google Drive file.

    For Google Docs/Sheets/Slides, the file is exported to a readable format.
    For binary files (PDFs, images), base64-encoded content is returned.

    Args:
        file_id:       The Drive file ID (from gdrive_search results).
        export_format: Override export format. Options for Docs: 'txt', 'html',
                       'markdown', 'pdf'. For Sheets: 'csv', 'xlsx'. 
                       Defaults are: Docs→txt, Sheets→csv, Slides→txt, others→raw.

    Returns:
        JSON with file metadata and content string (or base64 for binaries).
    """
    import base64
    import io

    drive = _drive_service()

    # Fetch metadata first
    meta = drive.files().get(
        fileId=file_id,
        fields="id,name,mimeType,size,modifiedTime"
    ).execute()
    mime = meta.get("mimeType", "")

    # Google Workspace types require export
    export_mime_map = {
        "application/vnd.google-apps.document": {
            "txt": "text/plain",
            "html": "text/html",
            "markdown": "text/plain",   # Drive has no markdown export; use plain
            "pdf": "application/pdf",
        },
        "application/vnd.google-apps.spreadsheet": {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "html": "text/html",
        },
        "application/vnd.google-apps.presentation": {
            "txt": "text/plain",
            "pdf": "application/pdf",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        },
    }

    if mime in export_mime_map:
        formats = export_mime_map[mime]
        # Pick default or user-requested format
        chosen_key = export_format if export_format and export_format in formats else list(formats.keys())[0]
        chosen_mime = formats[chosen_key]
        data = drive.files().export_media(fileId=file_id, mimeType=chosen_mime).execute()
        if isinstance(data, bytes):
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                content = base64.b64encode(data).decode("ascii")
                return json.dumps({**meta, "format": chosen_key, "encoding": "base64", "content": content})
        else:
            content = str(data)
        return json.dumps({**meta, "format": chosen_key, "content": content[:50000]}, ensure_ascii=False)

    # Binary / raw file download
    request = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = None
    try:
        from googleapiclient.http import MediaIoBaseDownload
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    except Exception as e:
        return json.dumps({"error": str(e), **meta})

    raw = buf.getvalue()
    # Try UTF-8 text decode first
    try:
        content = raw.decode("utf-8")
        return json.dumps({**meta, "format": "text", "content": content[:50000]}, ensure_ascii=False)
    except UnicodeDecodeError:
        encoded = base64.b64encode(raw).decode("ascii")
        return json.dumps({**meta, "format": "binary", "encoding": "base64",
                           "size_bytes": len(raw), "content_preview": encoded[:2000]})


@mcp.tool()
def gdrive_list_folder(
    folder_id: str = "root",
    max_results: int = 50,
    include_subfolders: bool = False,
) -> str:
    """
    List files inside a Google Drive folder.

    Args:
        folder_id:          Drive folder ID, or 'root' for My Drive root.
        max_results:        Max files to list (default 50, max 200).
        include_subfolders: If True, recursively list subfolders too (shallow,
                            one level deep).

    Returns:
        JSON list of files in the folder.
    """
    drive = _drive_service()
    q = f"'{folder_id}' in parents and trashed=false"

    result = drive.files().list(
        q=q,
        pageSize=min(max_results, 200),
        fields="files(id,name,mimeType,webViewLink,modifiedTime,size)",
        orderBy="folder,name",
    ).execute()

    files = result.get("files", [])

    if include_subfolders:
        subfolders = [f for f in files if f["mimeType"] == "application/vnd.google-apps.folder"]
        for sf in subfolders[:10]:   # cap recursion at 10 subfolders
            sub_result = drive.files().list(
                q=f"'{sf['id']}' in parents and trashed=false",
                pageSize=50,
                fields="files(id,name,mimeType,webViewLink,modifiedTime,size)",
                orderBy="name",
            ).execute()
            sf["children"] = sub_result.get("files", [])

    return json.dumps({"folder_id": folder_id, "count": len(files), "files": files}, indent=2)


@mcp.tool()
def gdrive_get_file_metadata(file_id: str) -> str:
    """
    Get detailed metadata for a specific Drive file.

    Args:
        file_id: The Drive file ID.

    Returns:
        JSON object with full metadata: id, name, mimeType, size,
        webViewLink, owners, createdTime, modifiedTime, parents, shared.
    """
    drive = _drive_service()
    meta = drive.files().get(
        fileId=file_id,
        fields="id,name,mimeType,size,webViewLink,webContentLink,owners,createdTime,modifiedTime,parents,shared,description,starred"
    ).execute()
    return json.dumps(meta, indent=2)


@mcp.tool()
def gdrive_list_shared_drives() -> str:
    """
    List all Shared Drives (Team Drives) the authenticated user can access.

    Returns:
        JSON list of shared drives with id and name.
    """
    drive = _drive_service()
    result = drive.drives().list(pageSize=50).execute()
    drives = result.get("drives", [])
    return json.dumps({"count": len(drives), "drives": drives}, indent=2)


@mcp.tool()
def gdrive_auth_status() -> str:
    """
    Check whether the Google Drive authentication is working.
    Returns the authenticated user identity and drive quota info.

    Use this to verify credentials before using other tools.
    """
    try:
        drive = _drive_service()
        about = drive.about().get(fields="user,storageQuota").execute()
        return json.dumps({
            "status": "authenticated",
            "user": about.get("user", {}),
            "quota": about.get("storageQuota", {}),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ── Entry point ────────────────────────────────────────────────────────────────
# Expose the ASGI app so uvicorn / gunicorn can import it directly:
#   uvicorn apps.gdrive_mcp.server:mcp_app --port 8090
mcp_app = mcp.http_app(path="/mcp")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Google Drive MCP server on %s:%s", HOST, PORT)
    uvicorn.run(mcp_app, host=HOST, port=PORT, log_level="info")
