"""
app/attachment/utils.py  (REFACTORED)

Changes from original:
  - Replaced s3_upload_to_prefix with storage_upload_to_prefix
"""

from typing import Optional, BinaryIO
import io
import os
import asyncio
import logging

from PIL import Image, ImageOps

from app.background_mgr.service import TaskProgressTracker
from app.storage import storage_upload_to_prefix
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

async def downsize_image_if_oversized(
    file_obj: BinaryIO,
    filename: str,
    max_resolution: int,
    bg_tracker: Optional[TaskProgressTracker] = None,
    s3_prefix: Optional[str] = None
) -> BinaryIO:
    """  
    Check if image resolution is within limit, downsize if necessary.  

    Args:  
        file_obj: File-like object containing image data  
        filename: Original filename
        max_resolution: Resolution limit in pixels
        bg_tracker: Optional TaskProgressTracker object for progress tracking
        s3_prefix: Optional storage prefix to upload result to
    Returns:  
        File-like object with original or resized image  
    """
    if bg_tracker:
        await bg_tracker.start(3)

    try:
        # Open the image from file object 
        file_obj.seek(0)
        image = await asyncio.to_thread(Image.open, file_obj)
        image = ImageOps.exif_transpose(image)
        if bg_tracker:
            await bg_tracker.increment(1)

        # Calculate current resolution (total pixels)  
        current_resolution = image.width * image.height  

        # If within limit, return original file  
        if current_resolution <= max_resolution:  
            file_obj.seek(0)
            output_buffer = file_obj
        else:
            # Calculate new dimensions maintaining aspect ratio  
            scale_factor = (max_resolution / current_resolution) ** 0.5  

            new_width = int(image.width * scale_factor)  
            new_height = int(image.height * scale_factor)  

            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  

            # Create new file-like object with resized image  
            output_buffer = io.BytesIO()  

            # Preserve original format if possible, default to JPEG  
            format_to_save = image.format if image.format else 'JPEG'

            # Handle special cases for format compatibility  
            if format_to_save == 'JPEG' and resized_image.mode in ('RGBA', 'LA', 'P'):  
                resized_image = resized_image.convert('RGB')

            resized_image.save(output_buffer, format=format_to_save, quality=85)
            output_buffer.seek(0)
            
            if bg_tracker:
                await bg_tracker.increment(1)

        if s3_prefix:
            await storage_upload_to_prefix(
                bucket=settings.AWS_FILESTORE_BUCKET,
                prefix=s3_prefix,
                file=output_buffer,
                name=filename,
            )
        
        if bg_tracker:
            await bg_tracker.complete()

        return output_buffer

    except Exception as e:
        fail_msg = f"Error encountered when downsizing image: {str(e)}"
        logger.exception(fail_msg)
        if bg_tracker:
            await bg_tracker.fail(fail_msg)

        return None

def safe_unlink(file_path: str):
    if file_path:
        try:  
            os.unlink(file_path)  
        except FileNotFoundError:  
            pass  
        except Exception as e:  
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")

def parse_s3_url(s3_location: str) -> tuple[str, str]:
    """Parse bucket and key from s3:// URL (works for both local and S3 storage)"""
    if not s3_location.startswith("s3://"):
        logger.error(f"Invalid S3 location format: {s3_location}")
        raise ValueError("Invalid attachment storage location")

    s3_path = s3_location.replace("s3://", "")
    parts = s3_path.split("/", 1)

    if len(parts) != 2:
        logger.error(f"Invalid S3 path format: {s3_location}")
        raise ValueError("Invalid attachment storage location")

    return parts
