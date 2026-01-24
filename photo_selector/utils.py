"""
Utility functions for photo processing.
"""

import os
from datetime import datetime
from typing import Optional
from PIL import Image
from PIL.ExifTags import TAGS, IFD

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def get_photo_timestamp(image_path: str) -> Optional[datetime]:
    """
    Extract original creation date from photo EXIF data.

    Args:
        image_path: Path to the image file

    Returns:
        datetime object if timestamp found, None otherwise
    """
    datetime_result = None

    # First try EXIF data
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if exif:
                datetime_str = None

                # Check main EXIF for DateTimeOriginal
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        datetime_str = str(value)
                        break

                # Check IFD EXIF (nested EXIF data)
                if not datetime_str:
                    for ifd_key in IFD:
                        try:
                            ifd_data = exif.get_ifd(ifd_key)
                            if ifd_data:
                                for tag_id, value in ifd_data.items():
                                    tag = TAGS.get(tag_id, tag_id)
                                    if tag == 'DateTimeOriginal':
                                        datetime_str = str(value)
                                        break
                        except:
                            pass
                        if datetime_str:
                            break

                # Also try DateTime (fallback)
                if not datetime_str:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == 'DateTime':
                            datetime_str = str(value)
                            break

                if datetime_str:
                    # Format: "2024:01:15 14:30:45"
                    datetime_result = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")

    except Exception as e:
        pass

    # If EXIF didn't work, use file modification time as fallback
    if datetime_result is None:
        try:
            mtime = os.path.getmtime(image_path)
            datetime_result = datetime.fromtimestamp(mtime)
        except:
            pass

    return datetime_result


def get_thumbnail_name(filename: str) -> str:
    """
    Generate thumbnail filename from original filename.

    Args:
        filename: Original filename

    Returns:
        Thumbnail filename (thumb_<original>)
    """
    return f"thumb_{filename}"
