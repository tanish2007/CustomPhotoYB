import os
import json
from pathlib import Path
from collections import defaultdict

# Register HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from PIL import Image
from PIL.ExifTags import TAGS, IFD


def get_original_date(image):
    """Extract original creation date from image EXIF."""
    try:
        exif = image.getexif()
        if not exif:
            return None

        datetime_str = None

        # Check main EXIF for DateTimeOriginal first
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                datetime_str = str(value)
                break

        # Check IFD EXIF if not found
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
            # Format: "2024:01:15 14:30:45" -> "2024-01-15"
            date_part = datetime_str.split(' ')[0]
            return date_part.replace(':', '-')

    except:
        pass
    return None


def extract_metadata(file_path):
    """Extract original creation date from an image."""
    metadata = {
        'filename': os.path.basename(file_path),
        'original_creation_date': None
    }

    try:
        with Image.open(file_path) as img:
            metadata['original_creation_date'] = get_original_date(img)
    except Exception as e:
        metadata['error'] = str(e)

    return metadata


def process_folder(folder_path, output_file=None):
    """Process all images in a folder and bucket by date."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.heic', '.heif', '.webp'}

    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []

    results = []
    date_buckets = defaultdict(list)
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} image files")
    print("-" * 70)

    for i, file_path in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] {file_path.name}", end='\r')
        except:
            print(f"[{i}/{len(image_files)}] (filename with special chars)", end='\r')

        metadata = extract_metadata(str(file_path))
        results.append(metadata)

        # Add to date bucket
        date = metadata.get('original_creation_date') or 'Unknown'
        date_buckets[date].append(metadata['filename'])

    # Create output with buckets
    output_data = {
        'total_images': len(results),
        'images': results,
        'buckets_by_date': {date: {'count': len(files), 'files': files}
                           for date, files in sorted(date_buckets.items())}
    }

    # Save to JSON
    if output_file is None:
        output_file = folder / 'image_metadata.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "-" * 70)
    print(f"Saved to: {output_file}")
    print(f"Total: {len(results)} images")
    print(f"Date buckets: {len(date_buckets)}")

    # Print bucket summary
    print("\n=== DATE BUCKETS ===")
    for date in sorted(date_buckets.keys()):
        print(f"  {date}: {len(date_buckets[date])} images")

    return output_data


if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\tanis\Downloads\Ariya Millikin-20260106T125948Z-1-002\Ariya Millikin"
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_metadata.json')
    process_folder(folder, output)
