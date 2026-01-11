"""
Step 2: Temporal Segmentation
Split photos into time buckets (monthly or biweekly)
Prevents over-selecting one phase and ensures growth coverage
"""

import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from PIL import Image
from PIL.ExifTags import TAGS, IFD
import json

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class TemporalSegmenter:
    """Segment photos into time buckets based on creation date."""

    def __init__(self, bucket_type: str = "monthly"):
        """
        Initialize segmenter.

        Args:
            bucket_type: "monthly" (12 buckets) or "biweekly" (24 buckets)
        """
        self.bucket_type = bucket_type

    def get_photo_date(self, image_path: str) -> Optional[datetime]:
        """Extract original creation date from photo EXIF."""
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if not exif:
                    return None

                datetime_str = None

                # Check main EXIF
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'DateTimeOriginal':
                        datetime_str = str(value)
                        break

                # Check IFD EXIF
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

                if datetime_str:
                    # Format: "2024:01:15 14:30:45"
                    return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")

        except Exception as e:
            pass

        return None

    def get_bucket_key(self, dt: datetime) -> str:
        """Get bucket key for a datetime."""
        if self.bucket_type == "monthly":
            return f"{dt.year}-{dt.month:02d}"
        elif self.bucket_type == "biweekly":
            # First half (1-15) or second half (16-31)
            half = 1 if dt.day <= 15 else 2
            return f"{dt.year}-{dt.month:02d}-{half}"
        else:
            raise ValueError(f"Unknown bucket type: {self.bucket_type}")

    def segment_folder(self, folder_path: str,
                       image_extensions: set = None) -> Dict[str, List[Dict]]:
        """
        Segment all photos in a folder into time buckets.

        Args:
            folder_path: Path to folder containing images
            image_extensions: Valid image extensions

        Returns:
            Dictionary mapping bucket key to list of photo info dicts
        """
        if image_extensions is None:
            image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}

        folder = Path(folder_path)
        image_files = [f for f in folder.iterdir()
                       if f.suffix.lower() in image_extensions]

        print(f"Segmenting {len(image_files)} images into {self.bucket_type} buckets...")

        buckets = defaultdict(list)
        unknown_bucket = []

        for i, image_path in enumerate(image_files):
            if (i + 1) % 50 == 0:
                print(f"Processing [{i+1}/{len(image_files)}]")

            photo_info = {
                'filename': image_path.name,
                'filepath': str(image_path),
                'date': None,
                'bucket': None
            }

            dt = self.get_photo_date(str(image_path))

            if dt:
                photo_info['date'] = dt.isoformat()
                bucket_key = self.get_bucket_key(dt)
                photo_info['bucket'] = bucket_key
                buckets[bucket_key].append(photo_info)
            else:
                photo_info['bucket'] = 'unknown'
                unknown_bucket.append(photo_info)

        if unknown_bucket:
            buckets['unknown'] = unknown_bucket

        # Sort buckets by date
        sorted_buckets = dict(sorted(buckets.items()))

        # Print summary
        print(f"\n=== Temporal Segmentation Summary ===")
        print(f"Total buckets: {len(sorted_buckets)}")
        for bucket, photos in sorted_buckets.items():
            print(f"  {bucket}: {len(photos)} photos")

        return sorted_buckets

    def calculate_target_per_bucket(self, buckets: Dict[str, List],
                                     total_target: int = 350) -> Dict[str, int]:
        """
        Calculate how many photos to select from each bucket.

        Args:
            buckets: Dictionary of bucket -> photos
            total_target: Total number of photos to select

        Returns:
            Dictionary of bucket -> target count
        """
        # Exclude unknown bucket from proportional calculation
        known_buckets = {k: v for k, v in buckets.items() if k != 'unknown'}
        total_photos = sum(len(v) for v in known_buckets.values())

        if total_photos == 0:
            return {}

        targets = {}

        # Base allocation: proportional to number of photos
        # But with minimum and maximum constraints
        num_buckets = len(known_buckets)
        base_per_bucket = total_target / num_buckets if num_buckets > 0 else 0

        for bucket, photos in known_buckets.items():
            # Proportional allocation
            proportion = len(photos) / total_photos
            target = int(total_target * proportion)

            # Apply constraints
            min_target = max(1, int(base_per_bucket * 0.5))  # At least 50% of average
            max_target = min(len(photos), int(base_per_bucket * 2))  # At most 200% of average

            targets[bucket] = max(min_target, min(target, max_target))

        # Handle unknown bucket - allocate proportionally
        if 'unknown' in buckets and buckets['unknown']:
            unknown_proportion = len(buckets['unknown']) / (total_photos + len(buckets['unknown']))
            targets['unknown'] = max(1, int(total_target * unknown_proportion * 0.5))  # 50% weight

        # Adjust to hit target total
        current_total = sum(targets.values())
        if current_total < total_target:
            # Distribute remaining to buckets with more photos
            remaining = total_target - current_total
            sorted_by_size = sorted(known_buckets.keys(),
                                   key=lambda k: len(known_buckets[k]),
                                   reverse=True)
            for bucket in sorted_by_size:
                if remaining <= 0:
                    break
                add = min(remaining, len(buckets[bucket]) - targets[bucket])
                if add > 0:
                    targets[bucket] += add
                    remaining -= add

        return targets


def get_year_range(buckets: Dict[str, List]) -> Tuple[str, str]:
    """Get the date range covered by the photos."""
    dates = []
    for bucket, photos in buckets.items():
        if bucket != 'unknown':
            for photo in photos:
                if photo.get('date'):
                    dates.append(photo['date'])

    if not dates:
        return None, None

    dates.sort()
    return dates[0][:10], dates[-1][:10]  # Return just date part


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = r"C:\Users\tanis\Downloads\Ariya Millikin-20260106T125948Z-1-002\ariya"

    segmenter = TemporalSegmenter(bucket_type="monthly")
    buckets = segmenter.segment_folder(folder)

    start_date, end_date = get_year_range(buckets)
    print(f"\nDate range: {start_date} to {end_date}")

    targets = segmenter.calculate_target_per_bucket(buckets, total_target=50)
    print("\n=== Target Selection Per Bucket ===")
    for bucket, target in targets.items():
        print(f"  {bucket}: select {target} of {len(buckets[bucket])} photos")
