import logging
from tempfile import SpooledTemporaryFile, NamedTemporaryFile
from typing import Optional, BinaryIO
import os.path
import asyncio
from io import BytesIO

from botocore.exceptions import ClientError
from .clients import get_s3_client

from app.config import get_settings
from app.core.utils import copy_as_spooled_temporary_file

settings = get_settings()
logger = logging.getLogger(__name__)

async def s3_delete_by_prefix(bucket: str, prefix: str):
    """Delete all objects in a bucket with given prefix"""

    s3 = get_s3_client()
    if s3:
        try:
            res = await asyncio.to_thread(
                s3.list_objects_v2,
                Bucket=bucket,
                Prefix=prefix
            )
            if res.get("Contents"):
                objs = [{"Key": obj["Key"]} for obj in res["Contents"]]
                await asyncio.to_thread(
                    s3.delete_objects,
                    Bucket=bucket,
                    Delete={
                        "Objects": objs
                    }
                )
            while res.get("IsTruncated", False):
                res = await asyncio.to_thread(
                    s3.list_objects_v2,
                    Bucket=bucket,
                    Prefix=prefix,
                    ContinuationToken=res.get("NextContinuationToken")
                )
                if res.get("Contents"):
                    objs = [{"Key": obj["Key"]} for obj in res["Contents"]]
                    await asyncio.to_thread(
                        s3.delete_objects,
                        Bucket=bucket,
                        Delete={
                            "Objects": objs
                        }
                    )
        except ClientError as error:
            logger.error(
                f"Unexpected error batch deleting s3 objects in {bucket} with prefix {prefix}: {error}",
                exc_info=True
            )
            raise
    else:
        logger.error("S3 client not initialized")

async def s3_upload_to_prefix(bucket: str, prefix: str, file: BinaryIO, name: Optional[str] = None) -> str:
    """
    Upload the given file to S3 under a given prefix asynchronously.

    Args:
        bucket: S3 bucket name
        prefix: S3 key prefix (folder-like path)
        file: A file-like object with 'name' attribute in binary mode
        name: Optional file name. If `name` is set, `name` is used as file name in s3 rather than `file.name`

    Returns:
        The complete S3 key where the file was uploaded

    Raises:
        ConnectionError: If the S3 client is not initialized
        ValueError: If the file object doesn't have a name attribute
        ClientError: If there's an AWS S3-specific error during upload
        IOError: If there's an issue reading the file
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client not initialized")
        raise ConnectionError("boto3 S3 client not initialized.")

    # Ensure file has a name attribute
    if not hasattr(file, 'name') and not name:
        raise ValueError("File object must have a 'name' attribute, or a file name must be supplied.")

    # Ensure prefix ends with '/' if not empty
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    filename = name or os.path.basename(file.name)
    key = f"{prefix}{filename}"

    try:
        # Save current position to restore after upload
        current_position = file.tell()

        # Make a copy of the original since s3.upload_fileobj
        # automatically closes the input file-like object
        file_copy = copy_as_spooled_temporary_file(file)

        await asyncio.to_thread(
            s3.upload_fileobj,
            Fileobj=file_copy,
            Bucket=bucket,
            Key=key
        )

        logger.debug(f"Uploaded file {filename} to s3://{bucket}/{key}")
        return key

    except ClientError as e:
        logger.exception(f"AWS error uploading file {filename} to {bucket}/{key}: {str(e)}")
        raise
    except IOError as e:
        logger.exception(f"IO error reading file {filename}: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error uploading file {filename} to {bucket}/{key}")
        raise
    finally:
        # Ensure we restore file position even if there's an error
        try:
            file.seek(current_position)
        except:
            pass

async def s3_upload_files_to_prefix(
    bucket: str,
    prefix: str,
    files: list[BinaryIO],
    names: Optional[list[str]] = None,
    max_concurrency: int = 10
) -> list[str | None]:
    """
    Upload multiple files to S3 under a given prefix concurrently

    Args:
        bucket: S3 bucket name
        prefix: S3 key prefix (folder-like path)
        files: List of file-like objects with 'name' attribute
        name: Optional file names. If `name` is set, `name` is used as file name in s3 rather than `file.name`
        max_concurrency: Maximum number of concurrent uploads (default: 10)

    Returns:
        List[str | None]: List of s3 keys for successfully uploaded files, None for failed uploads. Ordered.
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client not initialized")
        return [None] * len(files)

    # Ensure prefix ends with '/' if not empty
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    # Helper function to upload a single file
    async def upload_single_file(file: BinaryIO, name: Optional[str] = None) -> str | None:
        if not hasattr(file, 'name') and not name:
            logger.warning(f"File object missing name attribute, skipping")
            return None

        try:
            filename = name or os.path.basename(file.name)
            key = f"{prefix}{filename}"

            # Save current position to restore after upload
            current_position = file.tell()

            # Make a copy of the original since s3.upload_fileobj
            # automatically closes the input file-like object
            file_copy = copy_as_spooled_temporary_file(file)

            await asyncio.to_thread(
                s3.upload_fileobj,
                Fileobj=file_copy,
                Bucket=bucket,
                Key=key
            )

            logger.debug(f"Uploaded file {filename} to s3://{bucket}/{key}")
            return key
        except Exception as e:
            logger.error(f"Error uploading file {filename}: {str(e)}", exc_info=True)
            return None
        finally:
            # Ensure we restore file position even if there's an error
            try:
                file.seek(current_position)
            except:
                pass

    semaphore = asyncio.Semaphore(max_concurrency)

    # Wrapper to apply the semaphore
    async def upload_with_limit(file: BinaryIO, name: Optional[str] = None) -> str | None:
        async with semaphore:
            return await upload_single_file(file, name)

    try:
        # Create upload tasks for all files
        if names:
            upload_tasks = [upload_with_limit(file, name) for file, name in zip(files, names)]
        else:
            upload_tasks = [upload_with_limit(file) for file in files]

        # Execute all uploads concurrently
        results = await asyncio.gather(*upload_tasks, return_exceptions=False)

        return results
    except Exception as e:
        logger.error(f"Unexpected error in s3_save_files_to_prefix: {str(e)}", exc_info=True)
        return [None] * len(files)

async def s3_download_from_prefix(
    bucket: str,
    prefix: str,
    max_size: int = -1,
    max_concurrency: int = 10
) -> dict[str, SpooledTemporaryFile]:
    """
    Download all files under a prefix from S3

    Args:
        bucket: S3 bucket name
        prefix: S3 key prefix to list and download objects from
        max_size: Maximum size in bytes before SpooledTemporaryFile rolls to disk (default: 10MB)

    Returns:
        list[SpooledTemporaryFile]: List of downloaded files as SpooledTemporaryFiles with metadata
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client not initialized")
        return {}

    if max_size < 0:
        max_size = settings.SPOOLED_MAX_SIZE

    # Helper function to download a single file asynchronously
    async def download_single_file(key: str) -> SpooledTemporaryFile:
        filename = os.path.basename(key)
        temp_file = SpooledTemporaryFile(max_size=max_size)

        try:
            await asyncio.to_thread(
                s3.download_fileobj,
                Bucket=bucket,
                Key=key,
                Fileobj=temp_file
            )
            temp_file.seek(0)
            logger.debug(f"Downloaded s3://{bucket}/{key} to temporary file")
            return filename, temp_file
        except Exception as e:
            logger.error(f"Error downloading s3://{bucket}/{key}: {str(e)}")
            temp_file.close()
            raise

    result_files = []
    semaphore = asyncio.Semaphore(max_concurrency)  # Semaphore for rate limit

    # Wrapper to apply the semaphore
    async def download_with_limit(key: str) -> SpooledTemporaryFile:
        async with semaphore:
            return await download_single_file(key)

    try:
        # List objects with the prefix
        res = await asyncio.to_thread(
            s3.list_objects_v2,
            Bucket=bucket,
            Prefix=prefix
        )

        all_keys = []

        # Gather all keys first
        if res.get("Contents"):
            all_keys.extend([obj["Key"] for obj in res["Contents"]])

        # Handle pagination
        while res.get("IsTruncated", False):
            res = await asyncio.to_thread(
                s3.list_objects_v2,
                Bucket=bucket,
                Prefix=prefix,
                ContinuationToken=res.get("NextContinuationToken")
            )

            if res.get("Contents"):
                all_keys.extend([obj["Key"] for obj in res["Contents"]])

        # Download all files concurrently with semaphore to limit concurrency
        if all_keys:
            download_tasks = [download_with_limit(key) for key in all_keys]
            result_files = await asyncio.gather(*download_tasks, return_exceptions=False)

        return {name: file for name, file in result_files}

    except ClientError as error:
        logger.error(
            f"Error downloading files from s3://{bucket}/{prefix}: {error}",
            exc_info=True
        )
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in s3_load_from_prefix: {str(e)}", exc_info=True)
        return {}

async def s3_download_fileobj(
    bucket: str,
    key: str,
) -> BinaryIO:
    """
    Download a single file from S3 to a file-like object.

    Args:
        bucket: S3 bucket name
        key: Exact S3 object key to download

    Returns:
        file-like object

    Raises:
        ConnectionError: If S3 client is not initialized
        ClientError: For S3-specific errors (e.g., file not found)
        IOError: For issues with writing to temporary file
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client not initialized")
        raise ConnectionError("boto3 S3 client not initialized")

    # Create a temporary file that will persist after closing
    file_obj = BytesIO()

    try:
        # Download directly to the temporary file
        await asyncio.to_thread(
            s3.download_fileobj,
            Bucket=bucket,
            Key=key,
            Fileobj=file_obj
        )

        file_obj.seek(0)

        logger.debug(f"Downloaded s3://{bucket}/{key} to file obj")
        return file_obj

    except ClientError as e:
        file_obj.close()

        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            logger.error(f"File not found in S3: s3://{bucket}/{key}")
        else:
            logger.exception(f"S3 error downloading s3://{bucket}/{key}: {str(e)}")
        raise

    except Exception as e:
        file_obj.close()

        logger.exception(f"Error downloading s3://{bucket}/{key}: {str(e)}")
        raise

async def s3_download_file_to_temp(
    bucket: str,
    key: str,
    filename: str = None,
) -> tuple[str, str]:
    """
    Download a single file from S3 to a temporary file on disk.

    Args:
        bucket: S3 bucket name
        key: Exact S3 object key to download
        filename: Optional filename to use (defaults to the basename of the key)
        delete_on_close: Whether to automatically delete the temp file when closed

    Returns:
        Tuple of (local_file_path, original_filename)

    Raises:
        ConnectionError: If S3 client is not initialized
        ClientError: For S3-specific errors (e.g., file not found)
        IOError: For issues with writing to temporary file
    """
    s3 = get_s3_client()
    if not s3:
        logger.error("S3 client not initialized")
        raise ConnectionError("boto3 S3 client not initialized")

    # Use key basename if filename not provided
    if not filename:
        filename = os.path.basename(key)

    # Create a temporary file that will persist after closing
    temp_file = NamedTemporaryFile(
        delete=False,
        suffix=f"-{filename}"
    )

    try:
        # Download directly to the temporary file
        await asyncio.to_thread(
            s3.download_fileobj,
            Bucket=bucket,
            Key=key,
            Fileobj=temp_file
        )

        # Close the file so it can be read by other processes
        temp_file.close()

        logger.debug(f"Downloaded s3://{bucket}/{key} to {temp_file.name}")
        return temp_file.name, filename

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')

        temp_file.close()
        try:
            os.unlink(temp_file.name)
        except:
            pass

        if error_code == 'NoSuchKey':
            logger.error(f"File not found in S3: s3://{bucket}/{key}")
        else:
            logger.exception(f"S3 error downloading s3://{bucket}/{key}: {str(e)}")
        raise

    except Exception as e:
        temp_file.close()
        try:
            os.unlink(temp_file.name)
        except:
            pass

        logger.exception(f"Error downloading s3://{bucket}/{key}: {str(e)}")
        raise