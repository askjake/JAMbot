# app/storage/__init__.py
"""
Unified storage abstraction layer.

Provides a single interface for file storage operations that can be backed by
either local filesystem or S3, controlled by settings.USE_LOCAL_STORAGE_ONLY.

Usage:
    from app.storage import (
        storage_upload_to_prefix,
        storage_upload_files_to_prefix,
        storage_download_from_prefix,
        storage_download_fileobj,
        storage_download_file_to_temp,
        storage_delete_by_prefix,
    )
"""

from .backend import (
    storage_upload_to_prefix,
    storage_upload_files_to_prefix,
    storage_download_from_prefix,
    storage_download_fileobj,
    storage_download_file_to_temp,
    storage_delete_by_prefix,
)

__all__ = [
    "storage_upload_to_prefix",
    "storage_upload_files_to_prefix",
    "storage_download_from_prefix",
    "storage_download_fileobj",
    "storage_download_file_to_temp",
    "storage_delete_by_prefix",
]
