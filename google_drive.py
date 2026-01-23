"""
Google Drive integration for importing photos directly from Drive.
Uses Service Account authentication (no user OAuth required).
"""

import os
import re
import io
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Google API imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    print("Google Drive API not installed. Run: pip install google-api-python-client google-auth")


# Configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), 'credentials', 'service_account.json')

# Supported image types
IMAGE_MIMETYPES = [
    'image/jpeg',
    'image/png',
    'image/heic',
    'image/heif',
    'image/webp',
    'image/gif',
    'image/bmp'
]


def is_drive_available() -> bool:
    """Check if Google Drive integration is available."""
    if not GOOGLE_DRIVE_AVAILABLE:
        return False
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"Service account credentials not found at: {CREDENTIALS_PATH}")
        return False
    return True


def get_drive_service():
    """Create and return Google Drive service client."""
    if not is_drive_available():
        raise RuntimeError("Google Drive is not available. Check credentials.")

    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)


def extract_folder_id(url_or_id: str) -> str:
    """
    Extract folder ID from a Google Drive URL or return the ID if already an ID.

    Supports:
    - https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrS
    - https://drive.google.com/drive/u/0/folders/1AbCdEfGhIjKlMnOpQrS
    - 1AbCdEfGhIjKlMnOpQrS (just the ID)
    """
    # If it's already just an ID (no slashes), return it
    if '/' not in url_or_id and len(url_or_id) > 10:
        return url_or_id.strip()

    # Try to extract from URL
    patterns = [
        r'/folders/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract folder ID from: {url_or_id}")


def list_images_in_folder(folder_id: str, include_subfolders: bool = False) -> List[Dict]:
    """
    List all image files in a Google Drive folder.

    Args:
        folder_id: The Google Drive folder ID
        include_subfolders: Whether to recursively scan subfolders

    Returns:
        List of file dicts with 'id', 'name', 'mimeType', 'size'
    """
    service = get_drive_service()

    # Build query for images
    mime_query = " or ".join([f"mimeType='{m}'" for m in IMAGE_MIMETYPES])
    query = f"'{folder_id}' in parents and ({mime_query}) and trashed=false"

    all_files = []
    page_token = None

    # Pagination loop
    while True:
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=1000,  # Max allowed
            pageToken=page_token
        ).execute()

        files = results.get('files', [])
        all_files.extend(files)

        page_token = results.get('nextPageToken')
        if not page_token:
            break

    # Optionally scan subfolders
    if include_subfolders:
        # Find subfolders
        folder_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        folder_results = service.files().list(
            q=folder_query,
            fields="files(id, name)"
        ).execute()

        for subfolder in folder_results.get('files', []):
            subfolder_files = list_images_in_folder(subfolder['id'], include_subfolders=True)
            # Prefix with subfolder name for organization
            for f in subfolder_files:
                f['subfolder'] = subfolder['name']
            all_files.extend(subfolder_files)

    return all_files


def download_file(service, file_id: str, output_path: str) -> bool:
    """
    Download a single file from Google Drive.

    Args:
        service: Google Drive service client
        file_id: The file ID to download
        output_path: Local path to save the file

    Returns:
        True if successful, False otherwise
    """
    try:
        request = service.files().get_media(fileId=file_id)

        with io.FileIO(output_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

        return True
    except Exception as e:
        print(f"Error downloading file {file_id}: {e}")
        return False


def download_folder(
    folder_id: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    include_subfolders: bool = False,
    max_workers: int = 10  # Parallel downloads
) -> Dict:
    """
    Download all images from a Google Drive folder using parallel downloads.

    Args:
        folder_id: The Google Drive folder ID
        output_dir: Local directory to save files
        progress_callback: Optional callback(current, total, filename) for progress updates
        include_subfolders: Whether to include subfolders
        max_workers: Number of parallel download threads (default 10)

    Returns:
        Dict with 'success', 'downloaded', 'failed', 'total', 'files'
    """
    os.makedirs(output_dir, exist_ok=True)

    # List all images
    print(f"[Drive] Listing images in folder {folder_id}...")
    files = list_images_in_folder(folder_id, include_subfolders)
    total = len(files)

    if total == 0:
        return {
            'success': True,
            'downloaded': 0,
            'failed': 0,
            'total': 0,
            'files': [],
            'message': 'No images found in folder'
        }

    print(f"[Drive] Found {total} images. Starting parallel download ({max_workers} threads)...")

    # Thread-safe counters
    lock = threading.Lock()
    downloaded_count = [0]  # Use list for mutability in closure
    failed_count = [0]
    downloaded_files = []

    def download_single_file(file_info):
        """Download a single file (runs in thread)."""
        file_id = file_info['id']
        filename = file_info['name']

        # Handle subfolder organization
        if file_info.get('subfolder'):
            subfolder_path = os.path.join(output_dir, file_info['subfolder'])
            os.makedirs(subfolder_path, exist_ok=True)
            output_path = os.path.join(subfolder_path, filename)
        else:
            output_path = os.path.join(output_dir, filename)

        # Handle duplicate filenames (thread-safe)
        with lock:
            if os.path.exists(output_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(output_path):
                    new_filename = f"{base}_{counter}{ext}"
                    output_path = os.path.join(output_dir, new_filename)
                    counter += 1
                filename = os.path.basename(output_path)

        # Each thread gets its own service connection
        try:
            service = get_drive_service()
            success = download_file(service, file_id, output_path)
            return (filename, success)
        except Exception as e:
            print(f"[Drive] Error downloading {filename}: {e}")
            return (filename, False)

    # Parallel downloads using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_file = {executor.submit(download_single_file, f): f for f in files}

        completed = 0
        for future in as_completed(future_to_file):
            filename, success = future.result()
            completed += 1

            with lock:
                if success:
                    downloaded_count[0] += 1
                    downloaded_files.append(filename)
                else:
                    failed_count[0] += 1

            # Progress callback
            if progress_callback:
                progress_callback(completed, total, filename)

    downloaded = downloaded_count[0]
    failed = failed_count[0]

    print(f"[Drive] Download complete. Success: {downloaded}, Failed: {failed}")

    return {
        'success': failed == 0,
        'downloaded': downloaded,
        'failed': failed,
        'total': total,
        'files': downloaded_files,
        'message': f'Downloaded {downloaded}/{total} files'
    }


def get_folder_info(folder_id: str) -> Dict:
    """
    Get information about a Google Drive folder.

    Args:
        folder_id: The Google Drive folder ID

    Returns:
        Dict with folder info including name and image count
    """
    try:
        service = get_drive_service()

        # Get folder name
        folder = service.files().get(fileId=folder_id, fields='name').execute()
        folder_name = folder.get('name', 'Unknown')

        # Count images
        files = list_images_in_folder(folder_id)

        return {
            'success': True,
            'folder_id': folder_id,
            'folder_name': folder_name,
            'image_count': len(files),
            'images': [{'name': f['name'], 'size': f.get('size', 0)} for f in files[:10]]  # Preview first 10
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Test function
if __name__ == '__main__':
    print(f"Google Drive available: {is_drive_available()}")

    if is_drive_available():
        # Test with a folder ID
        test_folder = input("Enter folder ID or URL to test: ").strip()
        if test_folder:
            folder_id = extract_folder_id(test_folder)
            print(f"Extracted folder ID: {folder_id}")

            info = get_folder_info(folder_id)
            print(f"Folder info: {info}")
