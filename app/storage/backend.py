"""
app/storage/backend.py

Unified storage backend that dispatches to either local filesystem
or S3 based on settings.USE_LOCAL_STORAGE_ONLY.

All public functions have the SAME signature and return types as the
original aws/utils.py functions, so callers are drop-in replacements.
"""

import logging
import os
import os.path
import asyncio
import shutil
from io import BytesIO
from tempfile import SpooledTemporaryFile, NamedTemporaryFile
from typing import Optional, BinaryIO

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def _use_local() -> bool:
    """Check if local storage is enabled."""
    return settings.USE_LOCAL_STORAGE_ONLY.lower() == "true"


def _local_base_dir() -> str:
    """Return the base directory for local file storage."""
    return settings.LOCAL_UPLOADS_DIR


def _ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#                          LOCAL FILESYSTEM IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


async def _local_upload_to_prefix(
    bucket: str, prefix: str, file: BinaryIO, name: Optional[str] = None
) -> str:
    """
    Save a file locally under <LOCAL_UPLOADS_DIR>/<bucket>/<prefix>/<name>.
    The `name` parameter may contain subdirectories (e.g. "uuid/file.pdf").

    Returns the relative key (same format as S3 key): prefix/name
    """
    if not hasattr(file, "name") and not name:
        raise ValueError("File object must have a 'name' attribute, or a name must be supplied.")

    if prefix and not prefix.endswith("/"):
        prefix += "/"

    filename = name or os.path.basename(file.name)
    key = f"{prefix}{filename}"
    dest_path = os.path.join(_local_base_dir(), bucket, key)

    # Ensure the full parent directory tree exists (name may contain subdirs)
    _ensure_dir(os.path.dirname(dest_path))

    current_position = file.tell()
    try:
        file.seek(0)
        data = file.read()

        def _write():
            with open(dest_path, "wb") as out:
                out.write(data)

        await asyncio.to_thread(_write)
        logger.debug(f"Saved file locally: {dest_path}")
        return key
    except Exception as e:
        logger.exception(f"Error saving file {filename} locally to {dest_path}")
        raise
    finally:
        try:
            file.seek(current_position)
        except Exception:
            pass


async def _local_upload_files_to_prefix(
    bucket: str,
    prefix: str,
    files: list[BinaryIO],
    names: Optional[list[str]] = None,
    max_concurrency: int = 10,
) -> list[str | None]:
    """
    Save multiple files locally. Returns list of keys (or None on failure).
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    semaphore = asyncio.Semaphore(max_concurrency)

    async def upload_one(file: BinaryIO, name: Optional[str] = None) -> str | None:
        async with semaphore:
            if not hasattr(file, "name") and not name:
                logger.warning("File object missing name attribute, skipping")
                return None
            try:
                filename = name or os.path.basename(file.name)
                key = f"{prefix}{filename}"
                dest_path = os.path.join(_local_base_dir(), bucket, key)

                # Ensure full parent dir (name may contain subdirs like "uuid/file.pdf")
                _ensure_dir(os.path.dirname(dest_path))

                current_position = file.tell()
                try:
                    file.seek(0)
                    data = file.read()

                    def _write():
                        with open(dest_path, "wb") as out:
                            out.write(data)

                    await asyncio.to_thread(_write)
                    logger.debug(f"Saved file locally: {dest_path}")
                    return key
                finally:
                    try:
                        file.seek(current_position)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Error saving file locally: {e}", exc_info=True)
                return None

    if names:
        tasks = [upload_one(f, n) for f, n in zip(files, names)]
    else:
        tasks = [upload_one(f) for f in files]

    return list(await asyncio.gather(*tasks))


async def _local_download_from_prefix(
    bucket: str, prefix: str, max_size: int = -1, max_concurrency: int = 10
) -> dict[str, SpooledTemporaryFile]:
    """
    Load all files under a local prefix into SpooledTemporaryFiles.
    """
    if max_size < 0:
        max_size = settings.SPOOLED_MAX_SIZE

    base = os.path.join(_local_base_dir(), bucket, prefix)
    if not os.path.isdir(base):
        logger.warning(f"Local prefix directory does not exist: {base}")
        return {}

    results = {}
    for root, _dirs, filenames in os.walk(base):
        for fname in filenames:
            full_path = os.path.join(root, fname)
            temp = SpooledTemporaryFile(max_size=max_size)
            with open(full_path, "rb") as fh:
                shutil.copyfileobj(fh, temp)
            temp.seek(0)
            results[fname] = temp

    return results


async def _local_download_fileobj(bucket: str, key: str) -> BinaryIO:
    """
    Read a single local file into a BytesIO object.
    """
    file_path = os.path.join(_local_base_dir(), bucket, key)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Local file not found: {file_path}")

    buf = BytesIO()

    def _read():
        with open(file_path, "rb") as fh:
            shutil.copyfileobj(fh, buf)

    await asyncio.to_thread(_read)
    buf.seek(0)
    return buf


async def _local_download_file_to_temp(
    bucket: str, key: str, filename: str = None
) -> tuple[str, str]:
    """
    Copy a local file to a NamedTemporaryFile and return (path, filename).
    """
    if not filename:
        filename = os.path.basename(key)

    src_path = os.path.join(_local_base_dir(), bucket, key)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Local file not found: {src_path}")

    temp_file = NamedTemporaryFile(delete=False, suffix=f"-{filename}")
    try:
        def _copy():
            with open(src_path, "rb") as src:
                shutil.copyfileobj(src, temp_file)

        await asyncio.to_thread(_copy)
        temp_file.close()
        return temp_file.name, filename
    except Exception:
        temp_file.close()
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass
        raise


async def _local_delete_by_prefix(bucket: str, prefix: str) -> None:
    """
    Delete all files under a local prefix directory.
    """
    target = os.path.join(_local_base_dir(), bucket, prefix)
    if os.path.isdir(target):
        def _rm():
            shutil.rmtree(target, ignore_errors=True)
        await asyncio.to_thread(_rm)
        logger.debug(f"Deleted local directory: {target}")
    elif os.path.isfile(target):
        os.unlink(target)
        logger.debug(f"Deleted local file: {target}")
    else:
        logger.debug(f"Nothing to delete at: {target}")


# ═══════════════════════════════════════════════════════════════════════════════
#                          PUBLIC DISPATCH FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


async def storage_upload_to_prefix(
    bucket: str, prefix: str, file: BinaryIO, name: Optional[str] = None
) -> str:
    """Upload a file. Routes to local or S3 based on config."""
    if _use_local():
        return await _local_upload_to_prefix(bucket, prefix, file, name)
    else:
        from app.aws.utils import s3_upload_to_prefix
        return await s3_upload_to_prefix(bucket, prefix, file, name)


async def storage_upload_files_to_prefix(
    bucket: str,
    prefix: str,
    files: list[BinaryIO],
    names: Optional[list[str]] = None,
    max_concurrency: int = 10,
) -> list[str | None]:
    """Upload multiple files. Routes to local or S3 based on config."""
    if _use_local():
        return await _local_upload_files_to_prefix(bucket, prefix, files, names, max_concurrency)
    else:
        from app.aws.utils import s3_upload_files_to_prefix
        return await s3_upload_files_to_prefix(bucket, prefix, files, names, max_concurrency)


async def storage_download_from_prefix(
    bucket: str, prefix: str, max_size: int = -1, max_concurrency: int = 10
) -> dict[str, SpooledTemporaryFile]:
    """Download all files under a prefix. Routes to local or S3."""
    if _use_local():
        return await _local_download_from_prefix(bucket, prefix, max_size, max_concurrency)
    else:
        from app.aws.utils import s3_download_from_prefix
        return await s3_download_from_prefix(bucket, prefix, max_size, max_concurrency)


async def storage_download_fileobj(bucket: str, key: str) -> BinaryIO:
    """Download a single file to a BytesIO. Routes to local or S3."""
    if _use_local():
        return await _local_download_fileobj(bucket, key)
    else:
        from app.aws.utils import s3_download_fileobj
        return await s3_download_fileobj(bucket, key)


async def storage_download_file_to_temp(
    bucket: str, key: str, filename: str = None
) -> tuple[str, str]:
    """Download a file to a temp path. Routes to local or S3."""
    if _use_local():
        return await _local_download_file_to_temp(bucket, key, filename)
    else:
        from app.aws.utils import s3_download_file_to_temp
        return await s3_download_file_to_temp(bucket, key, filename)


async def storage_delete_by_prefix(bucket: str, prefix: str) -> None:
    """Delete all objects under a prefix. Routes to local or S3."""
    if _use_local():
        return await _local_delete_by_prefix(bucket, prefix)
    else:
        from app.aws.utils import s3_delete_by_prefix
        return await s3_delete_by_prefix(bucket, prefix)
