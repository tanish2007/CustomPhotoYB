"""
Photo Selection Web App
Flask-based frontend for testing the photo selection pipeline
Now with AUTOMATIC selection - no target number needed!

Two-Stage Workflow with Review Step:
1. Upload reference photos of your child (2-3 photos)
2. Upload all event photos (e.g., 1000 photos)
3. System filters to find photos containing your child
4. USER REVIEWS filtered photos (can remove false positives)
5. Quality-based selection runs on confirmed photos
6. Final results shown
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np
from PIL import Image
import threading
import time

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'photo_selector_secret_key_2024'  # For session management

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
REFERENCE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'references')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'selected_photos')  # Auto-save location
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'heic', 'heif', 'webp'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5GB max (for large photo batches)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['MAX_FORM_MEMORY_SIZE'] = 5 * 1024 * 1024 * 1024  # 5GB for form data
app.config['MAX_FORM_PARTS'] = 10000  # Allow up to 10000 files in one upload

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_FOLDER, exist_ok=True)

# Store processing status
processing_jobs = {}

# Store face matchers for sessions (reuse to avoid reloading model)
face_matchers = {}

# Store chunked upload sessions
upload_sessions = {}


# Error handler for large uploads
@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(error):
    return jsonify({
        'error': 'Upload too large. Try uploading fewer files at once (max ~500 files per batch).'
    }), 413


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_thumbnail(image_path, thumb_path, size=(300, 300)):
    """Create a thumbnail for display with proper EXIF rotation."""
    from PIL import ExifTags
    try:
        with Image.open(image_path) as img:
            # Apply EXIF rotation before creating thumbnail
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass

            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumb_path, 'JPEG', quality=85)
        return True
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return False


def process_photos_face_filter_only(job_id, upload_dir, session_id=None):
    """
    Phase 1: Face filtering only.
    Scans all photos to find ones containing the target person.
    Returns filtered photos for user review before quality selection.
    """
    try:
        print(f"\n{'='*60}")
        print(f"[Job {job_id}] PHASE 1: Face Filtering Started")
        print(f"{'='*60}")

        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['progress'] = 5
        processing_jobs[job_id]['message'] = 'Loading face recognition AI...'

        print(f"[Job {job_id}] Loading InsightFace face recognition model...")

        from photo_selector.face_matcher import FaceMatcher

        # Get face matcher
        face_matcher = None
        if session_id and session_id in face_matchers:
            face_matcher = face_matchers[session_id]
            if face_matcher.get_reference_count() == 0:
                face_matcher = None

        if face_matcher is None:
            print(f"[Job {job_id}] ERROR: No reference photos loaded!")
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['message'] = 'No reference photos loaded'
            return

        ref_count = face_matcher.get_reference_count()
        print(f"[Job {job_id}] Reference photos loaded: {ref_count}")

        processing_jobs[job_id]['progress'] = 10
        processing_jobs[job_id]['message'] = 'Scanning photos for your child using InsightFace...'

        # Get all photo files
        photo_files = []
        for f in os.listdir(upload_dir):
            if allowed_file(f) and not f.startswith('thumb_'):
                photo_files.append(f)

        total_photos = len(photo_files)
        print(f"[Job {job_id}] Total photos to scan: {total_photos}")
        processing_jobs[job_id]['total_photos'] = total_photos
        processing_jobs[job_id]['message'] = f'Scanning {total_photos} photos for your child...'

        # Create thumbnails directory - always in uploads/<job_id>/thumbnails
        # This ensures thumbnails work for both browser upload and local folder mode
        is_local_folder = processing_jobs[job_id].get('is_local_folder', False)
        if is_local_folder:
            thumbs_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
        else:
            thumbs_dir = os.path.join(upload_dir, 'thumbnails')
        os.makedirs(thumbs_dir, exist_ok=True)

        # Get all photo paths
        photo_paths = [os.path.join(upload_dir, fn) for fn in photo_files]

        # Progress callback to update photos_checked
        def progress_callback(current, total, message):
            processing_jobs[job_id]['photos_checked'] = current
            processing_jobs[job_id]['message'] = f'Checked {current}/{total} photos...'
            # Update progress between 30-80%
            progress_pct = 30 + int((current / total) * 50) if total > 0 else 30
            processing_jobs[job_id]['progress'] = progress_pct

        # Run face filtering
        print(f"[Job {job_id}] Starting face detection and matching...")
        processing_jobs[job_id]['progress'] = 30
        filter_results = face_matcher.filter_photos(photo_paths, progress_callback=progress_callback)

        if 'error' in filter_results:
            print(f"[Job {job_id}] ERROR: Face matching failed - {filter_results['error']}")
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['message'] = f"Face matching error: {filter_results['error']}"
            return

        # Print statistics
        stats = filter_results.get('statistics', {})
        matched_count = len(filter_results.get('matched_photos', []))
        unmatched_count = len(filter_results.get('unmatched_photos', []))

        print(f"\n[Job {job_id}] Face Filtering Results:")
        print(f"  - Photos with your child: {matched_count}")
        print(f"  - Photos without match: {unmatched_count}")
        print(f"  - Photos with no faces: {stats.get('no_faces', 0)}")
        # Handle match_rate which may be a string or float
        match_rate = stats.get('match_rate', 0)
        if isinstance(match_rate, str):
            print(f"  - Match rate: {match_rate}")
        else:
            print(f"  - Match rate: {match_rate:.1%}")

        processing_jobs[job_id]['progress'] = 70
        processing_jobs[job_id]['message'] = f'Creating thumbnails: 0/{matched_count}'

        print(f"[Job {job_id}] Creating thumbnails for {matched_count} matched photos...")

        # Prepare filtered photo data
        filtered_photos = []
        for i, match in enumerate(filter_results['matched_photos']):
            filename = os.path.basename(match['path'])
            thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumb_path = os.path.join(thumbs_dir, thumb_name)

            create_thumbnail(match['path'], thumb_path)

            filtered_photos.append({
                'filename': filename,
                'thumbnail': thumb_name,
                'face_match_score': match['similarity'],
                'num_faces': match['num_faces'],
                'matched_face_idx': match.get('matched_face_idx', 0),
                'face_bboxes': match.get('face_bboxes', [])  # Cached face locations for scoring
            })

            # Progress update every 10 photos or on last photo
            if (i + 1) % 10 == 0 or (i + 1) == matched_count:
                progress = 70 + int((i / matched_count) * 25)
                processing_jobs[job_id]['progress'] = progress
                processing_jobs[job_id]['message'] = f'Creating thumbnails: {i + 1}/{matched_count}'
                print(f"[Job {job_id}] Thumbnails created: {i + 1}/{matched_count}")

        # Sort by face match score (highest first)
        filtered_photos.sort(key=lambda x: x['face_match_score'], reverse=True)

        # Store results for review
        review_data = {
            'total_uploaded': total_photos,
            'filtered_photos': filtered_photos,
            'statistics': filter_results['statistics'],
            'reference_count': face_matcher.get_reference_count()
        }

        # Save review data
        review_file = os.path.join(RESULTS_FOLDER, f"{job_id}_review.json")
        with open(review_file, 'w') as f:
            json.dump(review_data, f, indent=2, default=str)

        processing_jobs[job_id]['progress'] = 100
        processing_jobs[job_id]['status'] = 'review_pending'
        processing_jobs[job_id]['message'] = f'Found your child in {len(filtered_photos)} of {total_photos} photos!'
        processing_jobs[job_id]['review_data'] = review_data

        print(f"\n[Job {job_id}] PHASE 1 COMPLETE!")
        print(f"  - Found {len(filtered_photos)} photos of your child")
        print(f"  - Status: review_pending (waiting for user to confirm)")
        print(f"  - Review data saved to: {review_file}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"[Job {job_id}] EXCEPTION: {str(e)}")
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['message'] = str(e)
        import traceback
        traceback.print_exc()


def save_photos_by_month(job_id, upload_dir, selected_photos, rejected_photos, month_stats):
    """
    Automatically save both selected and not-selected photos organized by month.

    Creates folder structure:
    selected_photos/
    └── {job_id}_{timestamp}/
        ├── selected/
        │   ├── Jan/
        │   │   ├── photo1.jpg
        │   │   └── photo2.jpg
        │   ├── Feb/
        │   │   └── photo3.jpg
        │   └── ...
        ├── not_selected/
        │   ├── Jan/
        │   │   └── photo4.jpg
        │   ├── Feb/
        │   │   └── photo5.jpg
        │   └── ...
        └── summary.txt

    Args:
        job_id: The job identifier
        upload_dir: Source directory containing original photos
        selected_photos: List of selected photo dicts with 'filename' and 'month' keys
        rejected_photos: List of rejected photo dicts with 'filename' and 'month' keys
        month_stats: Statistics about each month's selection

    Returns:
        Path to the output folder
    """
    try:
        # Create output folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(OUTPUT_FOLDER, f"{job_id}_{timestamp}")
        os.makedirs(output_base, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  AUTO-SAVING PHOTOS BY MONTH (SELECTED & NOT SELECTED)")
        print(f"{'='*60}")
        print(f"  Output folder: {output_base}")

        # Create selected and not_selected folders
        selected_base = os.path.join(output_base, "selected")
        not_selected_base = os.path.join(output_base, "not_selected")
        os.makedirs(selected_base, exist_ok=True)
        os.makedirs(not_selected_base, exist_ok=True)

        # Group selected photos by month
        selected_by_month = {}
        for photo in selected_photos:
            month = photo.get('month', 'Unknown')
            if month not in selected_by_month:
                selected_by_month[month] = []
            selected_by_month[month].append(photo)

        # Group rejected photos by month
        rejected_by_month = {}
        for photo in rejected_photos:
            month = photo.get('month', 'Unknown')
            if month not in rejected_by_month:
                rejected_by_month[month] = []
            rejected_by_month[month].append(photo)

        # Copy SELECTED photos to month folders
        print(f"\n  --- SELECTED PHOTOS ---")
        total_selected_copied = 0
        for month, photos in selected_by_month.items():
            month_folder = os.path.join(selected_base, month)
            os.makedirs(month_folder, exist_ok=True)

            print(f"  [selected/{month}] Saving {len(photos)} photos...")

            for photo in photos:
                src_path = os.path.join(upload_dir, photo['filename'])
                dst_path = os.path.join(month_folder, photo['filename'])

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    total_selected_copied += 1

        # Copy NOT SELECTED photos to month folders
        print(f"\n  --- NOT SELECTED PHOTOS ---")
        total_rejected_copied = 0
        for month, photos in rejected_by_month.items():
            month_folder = os.path.join(not_selected_base, month)
            os.makedirs(month_folder, exist_ok=True)

            print(f"  [not_selected/{month}] Saving {len(photos)} photos...")

            for photo in photos:
                src_path = os.path.join(upload_dir, photo['filename'])
                dst_path = os.path.join(month_folder, photo['filename'])

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    total_rejected_copied += 1

        # Create summary file
        summary_path = os.path.join(output_base, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("  PHOTO SELECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Selected: {total_selected_copied} photos\n")
            f.write(f"Total Not Selected: {total_rejected_copied} photos\n")
            f.write(f"Grand Total: {total_selected_copied + total_rejected_copied} photos\n\n")

            f.write("-" * 40 + "\n")
            f.write("  BREAKDOWN BY MONTH\n")
            f.write("-" * 40 + "\n\n")
            f.write(f"{'Month':<12} {'Selected':>10} {'Not Selected':>14} {'Total':>8}\n")
            f.write(f"{'-'*12} {'-'*10} {'-'*14} {'-'*8}\n")

            for stat in month_stats:
                month = stat['month']
                selected = stat['selected']
                total = stat['total_photos']
                not_selected = total - selected
                f.write(f"{month:<12} {selected:>10} {not_selected:>14} {total:>8}\n")

            # Selected files by month
            f.write("\n" + "=" * 60 + "\n")
            f.write("  SELECTED FILES BY MONTH\n")
            f.write("=" * 60 + "\n")

            for month, photos in sorted(selected_by_month.items()):
                f.write(f"\n[{month}] - {len(photos)} selected photos:\n")
                for photo in sorted(photos, key=lambda x: x.get('score', 0), reverse=True):
                    score = photo.get('score', 0) * 100
                    cluster = photo.get('cluster_id', -1)
                    f.write(f"  + {photo['filename']} (Score: {score:.0f}%, Cluster: {cluster})\n")

            # Not selected files by month
            f.write("\n" + "=" * 60 + "\n")
            f.write("  NOT SELECTED FILES BY MONTH\n")
            f.write("=" * 60 + "\n")

            for month, photos in sorted(rejected_by_month.items()):
                f.write(f"\n[{month}] - {len(photos)} not selected photos:\n")
                for photo in sorted(photos, key=lambda x: x.get('score', 0), reverse=True):
                    score = photo.get('score', 0) * 100
                    cluster = photo.get('cluster_id', -1)
                    f.write(f"  - {photo['filename']} (Score: {score:.0f}%, Cluster: {cluster})\n")

        print(f"\n  SUMMARY:")
        print(f"  - Selected photos saved: {total_selected_copied}")
        print(f"  - Not selected photos saved: {total_rejected_copied}")
        print(f"  - Total photos saved: {total_selected_copied + total_rejected_copied}")
        print(f"  - Summary written to: {summary_path}")
        print(f"{'='*60}\n")

        return output_base

    except Exception as e:
        print(f"[ERROR] Failed to save photos by month: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_photos_quality_selection(job_id, upload_dir, quality_mode, similarity_threshold, confirmed_photos, face_data_cache=None):
    """
    Phase 2: Month-based category-aware photo selection.
    Selects ~40 best photos per month with category diversity.

    Args:
        face_data_cache: Dict of filename -> {'num_faces': int, 'face_bboxes': list}
                        Cached face data from Step 2 to avoid re-detection
    """
    face_data_cache = face_data_cache or {}
    try:
        print(f"\n{'='*60}")
        print(f"[Job {job_id}] PHASE 2: Monthly Category-Aware Selection Started")
        print(f"{'='*60}")
        print(f"[Job {job_id}] Confirmed photos: {len(confirmed_photos)}")
        print(f"[Job {job_id}] Quality mode: {quality_mode}")
        print(f"[Job {job_id}] Similarity threshold: {similarity_threshold}")

        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['progress'] = 5
        processing_jobs[job_id]['message'] = 'Loading AI models...'

        # Import the new monthly selector
        from photo_selector.embeddings import PhotoEmbedder
        from photo_selector.monthly_selector import MonthlyPhotoSelector

        # Determine target per month based on quality mode
        if quality_mode == 'keep_more':
            target_per_month = 60  # More photos per month
        elif quality_mode == 'strict':
            target_per_month = 25  # Fewer, higher quality
        else:  # balanced
            target_per_month = 40  # Default

        print(f"[Job {job_id}] Target per month: {target_per_month}")

        # Step 1: Generate embeddings for confirmed photos
        processing_jobs[job_id]['progress'] = 10
        processing_jobs[job_id]['message'] = 'Analyzing photos with CLIP AI...'

        print(f"[Job {job_id}] Generating CLIP embeddings for {len(confirmed_photos)} photos...")

        embedder = PhotoEmbedder()
        embeddings = {}

        for i, filename in enumerate(confirmed_photos):
            filepath = os.path.join(upload_dir, filename)
            if os.path.exists(filepath):
                img = embedder.load_image(filepath)
                if img is not None:
                    embedding = embedder.get_embedding(img)
                    if embedding is not None:
                        embeddings[filename] = embedding
                    img.close()

            # Update progress (10-30%)
            progress = 10 + int((i / len(confirmed_photos)) * 20)
            processing_jobs[job_id]['progress'] = progress

        print(f"[Job {job_id}] Embeddings generated: {len(embeddings)}")

        # Step 2: Initialize monthly selector
        processing_jobs[job_id]['progress'] = 35
        processing_jobs[job_id]['message'] = 'Grouping photos by month...'

        # Note: duplicate_threshold is for CLIP embedding similarity (0.85 catches exact near-dupes)
        # diversity_threshold ensures we don't select visually similar photos (different scenes)
        # This is separate from face similarity_threshold (0.4-0.5 for face matching)
        selector = MonthlyPhotoSelector(
            target_per_month=target_per_month,
            duplicate_threshold=0.85,    # Remove exact duplicates (same moment, slight angle change)
            diversity_threshold=0.75     # Ensure selected photos are visually diverse
        )

        # Step 3: Group photos by month (only confirmed photos)
        # We need to manually build the photos_by_month structure for confirmed photos
        from collections import defaultdict

        MONTH_NAMES = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
            5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
            9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }

        photos_by_month = defaultdict(list)

        for filename in confirmed_photos:
            filepath = os.path.join(upload_dir, filename)
            if not os.path.exists(filepath):
                continue

            dt = selector.get_photo_date(filepath)

            # Get cached face data if available
            cached_face = face_data_cache.get(filename, {})

            photo_info = {
                'filename': filename,
                'filepath': filepath,
                'date': dt.isoformat() if dt else None,
                'month': MONTH_NAMES.get(dt.month, "Unknown") if dt else "Unknown",
                'timestamp': dt.timestamp() if dt else None,
                # Cached face data from Step 2 (avoids re-detection)
                'num_faces': cached_face.get('num_faces'),
                'face_bboxes': cached_face.get('face_bboxes', [])
            }

            photos_by_month[photo_info['month']].append(photo_info)

        # Sort months in calendar order
        month_order = list(MONTH_NAMES.values()) + ['Unknown']
        photos_by_month = {m: photos_by_month[m] for m in month_order if m in photos_by_month}

        print(f"[Job {job_id}] Photos grouped into {len(photos_by_month)} months:")
        for month, photos in photos_by_month.items():
            print(f"  - {month}: {len(photos)} photos")

        # Step 4: Select best photos from each month (categories detected AFTER selection for speed)
        processing_jobs[job_id]['progress'] = 60
        processing_jobs[job_id]['message'] = 'Selecting best photos per month...'

        def progress_callback(msg):
            processing_jobs[job_id]['message'] = msg

        selection_results = selector.select_all_months(photos_by_month, embeddings, progress_callback)

        selected_photos = selection_results['selected']
        month_stats = selection_results['month_stats']
        summary = selection_results['summary']

        print(f"\n[Job {job_id}] Selection Results:")
        print(f"  - Total photos: {summary['total_photos']}")
        print(f"  - Selected: {summary['total_selected']}")
        print(f"  - Selection rate: {summary['selection_rate']*100:.1f}%")

        # Step 5: Detect categories ONLY for selected photos (much faster than all photos)
        processing_jobs[job_id]['progress'] = 75
        processing_jobs[job_id]['message'] = 'Detecting categories for selected photos...'

        print(f"[Job {job_id}] Detecting categories for {len(selected_photos)} selected photos...")
        selected_paths = [p['filepath'] for p in selected_photos]
        if selected_paths:
            selector._ensure_category_detector()
            categories = selector.category_detector.detect_categories_batch(selected_paths)
            for photo in selected_photos:
                # categories dict is keyed by filename, not filepath
                cat, conf = categories.get(photo['filename'], ('unknown', 0.0))
                photo['category'] = cat
                photo['category_confidence'] = conf

        # Update month_stats with category breakdown from selected photos only
        for stat in month_stats:
            month_name = stat['month']
            month_selected = [p for p in selected_photos if p.get('month') == month_name]
            cat_breakdown = {}
            for p in month_selected:
                cat = p.get('category', 'unknown')
                cat_breakdown[cat] = cat_breakdown.get(cat, 0) + 1
            stat['categories'] = cat_breakdown

        # Step 6: Build rejected list (photos not selected)
        selected_filenames = {p['filename'] for p in selected_photos}
        rejected_photos = []

        for month, photos in photos_by_month.items():
            for photo in photos:
                if photo['filename'] not in selected_filenames:
                    photo['rejection_reason'] = 'Not selected for month quota'
                    rejected_photos.append(photo)

        # Create thumbnails directory
        thumbs_dir = os.path.join(upload_dir, 'thumbnails')
        os.makedirs(thumbs_dir, exist_ok=True)

        # Calculate total thumbnails to create
        total_thumbnails = len(selected_photos) + len(rejected_photos)
        thumbnails_created = 0

        processing_jobs[job_id]['progress'] = 85
        processing_jobs[job_id]['message'] = f'Creating thumbnails: 0/{total_thumbnails}'

        # Build final results structure
        results = {
            'selected': [],
            'rejected': [],
            'summary': {
                'total_photos': summary['total_photos'],
                'selected_count': summary['total_selected'],
                'rejected_count': len(rejected_photos),
                'selection_rate': summary['selection_rate'],
                'face_filtering': {
                    'total_photos': processing_jobs[job_id].get('total_uploaded', len(confirmed_photos)),
                    'after_face_filter': len(confirmed_photos),
                    'user_confirmed': len(confirmed_photos)
                },
                'total_processed': len(confirmed_photos)
            },
            'month_stats': month_stats,
            'rejection_breakdown': {}
        }

        # Count rejection reasons
        rejection_counts = defaultdict(int)

        # Process selected photos
        for photo in selected_photos:
            filename = photo['filename']
            thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumb_path = os.path.join(thumbs_dir, thumb_name)

            create_thumbnail(os.path.join(upload_dir, filename), thumb_path)

            # Update thumbnail counter
            thumbnails_created += 1
            if thumbnails_created % 10 == 0 or thumbnails_created == total_thumbnails:
                processing_jobs[job_id]['message'] = f'Creating thumbnails: {thumbnails_created}/{total_thumbnails}'

            # Get embedding for this photo (convert to list for JSON serialization)
            photo_embedding = embeddings.get(filename)
            embedding_list = photo_embedding.tolist() if photo_embedding is not None else None

            results['selected'].append({
                'filename': filename,
                'thumbnail': thumb_name,
                'score': float(photo.get('total', 0)),
                'face_quality': float(photo.get('face_quality', 0)),
                'aesthetic_quality': float(photo.get('aesthetic_quality', 0)),
                'emotional_signal': float(photo.get('emotional_signal', 0)),
                'uniqueness': float(photo.get('uniqueness', 0)),
                'bucket': photo.get('month', 'unknown'),
                'month': photo.get('month', 'Unknown'),
                'category': photo.get('category', 'unknown'),
                'num_faces': int(photo.get('num_faces', 0)),
                'cluster_id': photo.get('cluster_id', -1),
                'max_similarity': float(photo.get('max_similarity', 0)),
                'embedding': embedding_list,
                'selection_reason': f"Best in {photo.get('category', 'category')} for {photo.get('month', 'month')}",
                'selection_detail': f"Selected from {photo.get('month', 'Unknown')} - Category: {photo.get('category', 'unknown')}"
            })

        # Process rejected photos
        for photo in rejected_photos:
            filename = photo['filename']
            thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumb_path = os.path.join(thumbs_dir, thumb_name)

            create_thumbnail(os.path.join(upload_dir, filename), thumb_path)

            # Update thumbnail counter
            thumbnails_created += 1
            if thumbnails_created % 10 == 0 or thumbnails_created == total_thumbnails:
                processing_jobs[job_id]['message'] = f'Creating thumbnails: {thumbnails_created}/{total_thumbnails}'

            # Simple rejection reason
            reason = "Better photos selected"
            rejection_counts[reason] += 1

            # Get embedding for this photo (convert to list for JSON serialization)
            photo_embedding = embeddings.get(filename)
            embedding_list = photo_embedding.tolist() if photo_embedding is not None else None

            results['rejected'].append({
                'filename': filename,
                'thumbnail': thumb_name,
                'score': float(photo.get('total', 0)),
                'face_quality': float(photo.get('face_quality', 0)),
                'aesthetic_quality': float(photo.get('aesthetic_quality', 0)),
                'bucket': photo.get('month', 'unknown'),
                'month': photo.get('month', 'Unknown'),
                'category': photo.get('category', 'unknown'),
                'cluster_id': photo.get('cluster_id', -1),
                'max_similarity': float(photo.get('max_similarity', 0)),
                'embedding': embedding_list,
                'reason': reason,
                'reason_detail': f"Category: {photo.get('category', 'unknown')}"
            })

        results['rejection_breakdown'] = dict(rejection_counts)

        # Sort by score
        results['selected'].sort(key=lambda x: x['score'], reverse=True)
        results['rejected'].sort(key=lambda x: x['score'], reverse=True)

        # Save results
        results_file = os.path.join(RESULTS_FOLDER, f"{job_id}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        processing_jobs[job_id]['status'] = 'complete'
        processing_jobs[job_id]['progress'] = 100
        processing_jobs[job_id]['message'] = 'Selection complete!'
        processing_jobs[job_id]['results'] = results

        print(f"\n[Job {job_id}] PHASE 2 COMPLETE!")
        print(f"  - Final selection: {len(results['selected'])} photos")
        print(f"  - Filtered out: {len(results['rejected'])} photos")
        print(f"  - Results saved to: {results_file}")
        print(f"\n=== Month Distribution ===")
        for stat in month_stats:
            print(f"  {stat['month']}: {stat['selected']}/{stat['total_photos']} ({stat['category_summary']})")
        print(f"{'='*60}\n")

        # Auto-save both selected and not-selected photos organized by month
        output_folder = save_photos_by_month(job_id, upload_dir, selected_photos, rejected_photos, month_stats)
        if output_folder:
            processing_jobs[job_id]['output_folder'] = output_folder
            print(f"[Job {job_id}] Photos auto-saved to: {output_folder}")

    except Exception as e:
        print(f"[Job {job_id}] EXCEPTION: {str(e)}")
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['message'] = str(e)
        import traceback
        traceback.print_exc()


def process_photos_automatic(job_id, upload_dir, quality_mode, similarity_threshold, session_id=None):
    """
    Full automatic processing (no review step) - used when no reference photos loaded.
    Processes all photos with quality-based selection.
    """
    try:
        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['progress'] = 5
        processing_jobs[job_id]['message'] = 'Loading AI models...'

        # Import pipeline components
        from photo_selector.embeddings import PhotoEmbedder
        from photo_selector.temporal import TemporalSegmenter
        from photo_selector.clustering import PhotoClusterer, BucketClusterManager
        from photo_selector.scoring import PhotoScorer, ClusterScorer
        from photo_selector.auto_selector import SmartPhotoSelector, SelectionReason

        # Step 1: Embeddings
        processing_jobs[job_id]['progress'] = 20
        processing_jobs[job_id]['message'] = 'Analyzing photos with CLIP AI...'

        embedder = PhotoEmbedder()
        embeddings = embedder.process_folder(upload_dir)

        processing_jobs[job_id]['progress'] = 40
        processing_jobs[job_id]['message'] = 'Organizing by date...'

        # Step 2: Temporal segmentation
        segmenter = TemporalSegmenter(bucket_type="monthly")
        buckets = segmenter.segment_folder(upload_dir)

        # For clustering, use a reasonable estimate (will be refined by auto-selector)
        estimated_target = max(10, len(embeddings) // 3)
        targets = segmenter.calculate_target_per_bucket(buckets, estimated_target)

        processing_jobs[job_id]['progress'] = 50
        processing_jobs[job_id]['message'] = 'Grouping similar photos (adaptive clustering)...'

        # Step 3: Clustering (HDBSCAN automatically finds natural groupings)
        clusterer = BucketClusterManager(PhotoClusterer(method="hdbscan", min_cluster_size=3, temporal_gap_hours=6.0))
        cluster_results = clusterer.cluster_all_buckets(buckets, embeddings, targets, use_adaptive=False)

        processing_jobs[job_id]['progress'] = 60
        processing_jobs[job_id]['message'] = 'Scoring photo quality...'

        # Step 4: Score ALL photos
        scorer = ClusterScorer(PhotoScorer())
        all_scores = {}

        for bucket_key, bucket_data in cluster_results.items():
            filenames = bucket_data['filenames']
            labels = np.array(bucket_data['labels'])
            bucket_embeddings = np.array([embeddings[fn] for fn in filenames])

            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                cluster_filenames = [filenames[i] for i in cluster_indices]
                cluster_embs = bucket_embeddings[cluster_mask]
                cluster_paths = [os.path.join(upload_dir, fn) for fn in cluster_filenames]

                scores = scorer.score_cluster(cluster_paths, cluster_embs)

                for score in scores:
                    score['bucket'] = bucket_key
                    score['cluster'] = int(cluster_id)
                    score['cluster_key'] = f"{bucket_key}_cluster_{cluster_id}"
                    all_scores[score['filename']] = score

        processing_jobs[job_id]['progress'] = 75
        processing_jobs[job_id]['message'] = 'AI deciding which photos to keep...'

        # Step 5: AUTOMATIC SELECTION
        auto_selector = SmartPhotoSelector(
            quality_mode=quality_mode,
            similarity_threshold=similarity_threshold
        )

        selection_results = auto_selector.process_all_photos(
            all_scores, embeddings, cluster_results
        )

        processing_jobs[job_id]['progress'] = 90
        processing_jobs[job_id]['message'] = 'Preparing results...'

        # Create thumbnails directory
        thumbs_dir = os.path.join(upload_dir, 'thumbnails')
        os.makedirs(thumbs_dir, exist_ok=True)

        # Prepare results
        results = {
            'selected': [],
            'rejected': [],
            'summary': selection_results['summary'],
            'rejection_breakdown': selection_results['rejection_breakdown'],
            'bucket_stats': selection_results['bucket_stats']
        }

        # Process selected photos
        for photo in selection_results['selected']:
            filename = photo['filename']
            thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumb_path = os.path.join(thumbs_dir, thumb_name)

            create_thumbnail(os.path.join(upload_dir, filename), thumb_path)

            reason = photo.get('selection_reason', None)
            if isinstance(reason, SelectionReason):
                reason_text = reason.value
            else:
                reason_text = str(reason) if reason else 'High quality photo'

            results['selected'].append({
                'filename': filename,
                'thumbnail': thumb_name,
                'score': float(photo.get('total', 0)),
                'face_quality': float(photo.get('face_quality', 0)),
                'aesthetic_quality': float(photo.get('aesthetic_quality', 0)),
                'emotional_signal': float(photo.get('emotional_signal', 0)),
                'uniqueness': float(photo.get('uniqueness', 0)),
                'bucket': photo.get('bucket', 'unknown'),
                'num_faces': int(photo.get('num_faces', 0)),
                'selection_reason': reason_text,
                'selection_detail': photo.get('selection_detail', reason_text)
            })

        # Process rejected photos
        for photo in selection_results['rejected']:
            filename = photo['filename']
            thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumb_path = os.path.join(thumbs_dir, thumb_name)

            create_thumbnail(os.path.join(upload_dir, filename), thumb_path)

            reason = photo.get('rejection_reason', None)
            if isinstance(reason, SelectionReason):
                reason_text = reason.value
            else:
                reason_text = str(reason) if reason else 'Did not meet quality threshold'

            results['rejected'].append({
                'filename': filename,
                'thumbnail': thumb_name,
                'score': float(photo.get('total', 0)),
                'face_quality': float(photo.get('face_quality', 0)),
                'aesthetic_quality': float(photo.get('aesthetic_quality', 0)),
                'bucket': photo.get('bucket', 'unknown'),
                'reason': reason_text,
                'reason_detail': photo.get('rejection_detail', '')
            })

        # Sort by score
        results['selected'].sort(key=lambda x: x['score'], reverse=True)
        results['rejected'].sort(key=lambda x: x['score'], reverse=True)

        # Save results
        results_file = os.path.join(RESULTS_FOLDER, f"{job_id}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        processing_jobs[job_id]['status'] = 'complete'
        processing_jobs[job_id]['progress'] = 100
        processing_jobs[job_id]['message'] = 'Selection complete!'
        processing_jobs[job_id]['results'] = results

    except Exception as e:
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['message'] = str(e)
        import traceback
        traceback.print_exc()


@app.route('/')
def index():
    """Main page - redirects to step 1 (reference upload)."""
    return render_template('index.html')


@app.route('/step1')
def step1_reference():
    """Step 1: Upload reference photos of target person."""
    # Create a new session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())[:8]
    return render_template('step1_reference.html', session_id=session['session_id'])


@app.route('/step2')
def step2_upload():
    """Step 2: Upload all event photos."""
    session_id = session.get('session_id')
    if not session_id:
        return render_template('index.html')

    # Check if we have reference photos loaded
    ref_count = 0
    if session_id in face_matchers:
        ref_count = face_matchers[session_id].get_reference_count()

    return render_template('step2_upload.html',
                          session_id=session_id,
                          reference_count=ref_count)


@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    """Handle reference photo uploads (2-3 photos of target person)."""
    from photo_selector.face_matcher import FaceMatcher

    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    # Get or create session ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
        session['session_id'] = session_id

    # Create reference directory for this session
    ref_dir = os.path.join(REFERENCE_FOLDER, session_id)
    os.makedirs(ref_dir, exist_ok=True)

    # Initialize face matcher for this session if not exists
    if session_id not in face_matchers:
        face_matchers[session_id] = FaceMatcher(similarity_threshold=0.5)

    matcher = face_matchers[session_id]

    # Process each reference photo
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(ref_dir, filename)
            file.save(filepath)

            # Add to face matcher
            result = matcher.add_reference_photo(filepath)
            result['filename'] = filename

            # Create thumbnail for preview
            thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumb_path = os.path.join(ref_dir, thumb_name)
            create_thumbnail(filepath, thumb_path, size=(150, 150))
            result['thumbnail'] = thumb_name

            results.append(result)

    return jsonify({
        'session_id': session_id,
        'results': results,
        'total_references': matcher.get_reference_count(),
        'message': f'Loaded {matcher.get_reference_count()} reference face(s)'
    })


@app.route('/reference_status')
def reference_status():
    """Get current reference photo status."""
    session_id = session.get('session_id')
    if not session_id or session_id not in face_matchers:
        return jsonify({
            'session_id': session_id,
            'reference_count': 0,
            'ready': False
        })

    matcher = face_matchers[session_id]
    return jsonify({
        'session_id': session_id,
        'reference_count': matcher.get_reference_count(),
        'ready': matcher.get_reference_count() >= 1
    })


@app.route('/clear_references', methods=['POST'])
def clear_references():
    """Clear all reference photos for current session."""
    session_id = session.get('session_id')

    if session_id and session_id in face_matchers:
        face_matchers[session_id].clear_references()

        # Delete reference files
        ref_dir = os.path.join(REFERENCE_FOLDER, session_id)
        if os.path.exists(ref_dir):
            shutil.rmtree(ref_dir)

    return jsonify({'message': 'References cleared', 'reference_count': 0})


@app.route('/reference_thumbnail/<filename>')
def get_reference_thumbnail(filename):
    """Serve reference photo thumbnails."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session'}), 404
    ref_dir = os.path.join(REFERENCE_FOLDER, session_id)
    return send_from_directory(ref_dir, filename)


# ============== CHUNKED UPLOAD ENDPOINTS ==============
# These endpoints allow uploading large batches of photos in smaller chunks
# to avoid 413 (Request Entity Too Large) errors on Hugging Face Spaces

@app.route('/upload_init', methods=['POST'])
def upload_init():
    """Initialize a chunked upload session."""
    data = request.json
    total_files = data.get('total_files', 0)
    quality_mode = data.get('quality_mode', 'balanced')
    similarity_threshold = data.get('similarity_threshold', 0.92)

    # Create a unique session ID for this upload
    upload_session_id = str(uuid.uuid4())[:8]
    upload_dir = os.path.join(UPLOAD_FOLDER, upload_session_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Get face matcher session
    face_session_id = session.get('session_id')

    # Store session info
    upload_sessions[upload_session_id] = {
        'upload_dir': upload_dir,
        'total_files': total_files,
        'uploaded_files': [],
        'quality_mode': quality_mode,
        'similarity_threshold': similarity_threshold,
        'face_session_id': face_session_id,
        'created_at': time.time()
    }

    print(f"\n[Upload Session {upload_session_id}] Initialized for {total_files} files")

    return jsonify({
        'session_id': upload_session_id,
        'message': 'Upload session initialized'
    })


@app.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    """Handle a chunk of files in a chunked upload."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    session_id = request.form.get('session_id')
    if not session_id or session_id not in upload_sessions:
        return jsonify({'error': 'Invalid upload session'}), 400

    upload_info = upload_sessions[session_id]
    upload_dir = upload_info['upload_dir']

    files = request.files.getlist('files')
    saved_count = 0

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Handle duplicate filenames
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(upload_dir, filename)):
                filename = f"{base}_{counter}{ext}"
                counter += 1

            file.save(os.path.join(upload_dir, filename))
            upload_info['uploaded_files'].append(filename)
            saved_count += 1

    chunk_index = request.form.get('chunk_index', '?')
    print(f"[Upload Session {session_id}] Chunk {chunk_index}: saved {saved_count} files (total: {len(upload_info['uploaded_files'])})")

    return jsonify({
        'success': True,
        'saved': saved_count,
        'total_uploaded': len(upload_info['uploaded_files'])
    })


@app.route('/upload_complete', methods=['POST'])
def upload_complete():
    """Complete a chunked upload and start processing."""
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in upload_sessions:
        return jsonify({'error': 'Invalid upload session'}), 400

    upload_info = upload_sessions[session_id]
    upload_dir = upload_info['upload_dir']
    saved_files = upload_info['uploaded_files']
    quality_mode = upload_info['quality_mode']
    similarity_threshold = upload_info['similarity_threshold']
    face_session_id = upload_info['face_session_id']

    if not saved_files:
        shutil.rmtree(upload_dir)
        del upload_sessions[session_id]
        return jsonify({'error': 'No valid image files uploaded'}), 400

    # Check if we have reference photos loaded
    has_references = False
    ref_count = 0
    if face_session_id and face_session_id in face_matchers:
        ref_count = face_matchers[face_session_id].get_reference_count()
        has_references = ref_count > 0

    # Create job (use same session_id as job_id for simplicity)
    job_id = session_id

    # Initialize job
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 30,  # Start at 30% since upload is done
        'message': 'Starting AI processing...',
        'total_files': len(saved_files),
        'total_uploaded': len(saved_files),
        'upload_dir': upload_dir,
        'session_id': face_session_id,
        'has_reference_photos': has_references,
        'reference_count': ref_count,
        'quality_mode': quality_mode,
        'similarity_threshold': similarity_threshold,
        'results': None
    }

    # Clean up upload session
    del upload_sessions[session_id]

    # Decide which processing mode to use
    if has_references:
        print(f"\n[Job {job_id}] NEW JOB (Chunked Upload) - Face Filtering Mode")
        print(f"  - Files uploaded: {len(saved_files)}")
        print(f"  - Reference photos: {ref_count}")
        thread = threading.Thread(
            target=process_photos_face_filter_only,
            args=(job_id, upload_dir, face_session_id)
        )
        message = f'Scanning {len(saved_files)} photos to find your child using {ref_count} reference(s)...'
    else:
        print(f"\n[Job {job_id}] NEW JOB (Chunked Upload) - No Face Filtering")
        print(f"  - Files uploaded: {len(saved_files)}")
        thread = threading.Thread(
            target=process_photos_quality_selection,
            args=(job_id, upload_dir, quality_mode, similarity_threshold)
        )
        message = f'Selecting best photos from {len(saved_files)} images...'

    thread.daemon = True
    thread.start()

    processing_jobs[job_id]['message'] = message

    return jsonify({
        'job_id': job_id,
        'message': message,
        'total_files': len(saved_files)
    })


# ============== END CHUNKED UPLOAD ENDPOINTS ==============


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and start processing."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    # Get parameters - now using quality_mode instead of target
    quality_mode = request.form.get('quality_mode', 'balanced')
    similarity_threshold = float(request.form.get('similarity', 0.92))

    # Get session ID for face matching
    session_id = session.get('session_id')

    # Create job
    job_id = str(uuid.uuid4())[:8]
    upload_dir = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Save files
    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Handle duplicate filenames
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(upload_dir, filename)):
                filename = f"{base}_{counter}{ext}"
                counter += 1

            file.save(os.path.join(upload_dir, filename))
            saved_files.append(filename)

    if not saved_files:
        shutil.rmtree(upload_dir)
        return jsonify({'error': 'No valid image files'}), 400

    # Check if we have reference photos loaded
    has_references = False
    ref_count = 0
    if session_id and session_id in face_matchers:
        ref_count = face_matchers[session_id].get_reference_count()
        has_references = ref_count > 0

    # Initialize job
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Uploading files...',
        'total_files': len(saved_files),
        'total_uploaded': len(saved_files),
        'upload_dir': upload_dir,
        'session_id': session_id,
        'has_reference_photos': has_references,
        'reference_count': ref_count,
        'quality_mode': quality_mode,
        'similarity_threshold': similarity_threshold,
        'results': None
    }

    # Decide which processing mode to use
    if has_references:
        # With reference photos: Phase 1 = face filtering only, then review step
        print(f"\n[Job {job_id}] NEW JOB - Face Filtering Mode")
        print(f"  - Files uploaded: {len(saved_files)}")
        print(f"  - Reference photos: {ref_count}")
        print(f"  - Session ID: {session_id}")
        thread = threading.Thread(
            target=process_photos_face_filter_only,
            args=(job_id, upload_dir, session_id)
        )
        message = f'Scanning {len(saved_files)} photos to find your child using {ref_count} reference(s)...'
    else:
        # Without reference photos: Full automatic processing (no review step)
        print(f"\n[Job {job_id}] NEW JOB - Full Automatic Mode")
        print(f"  - Files uploaded: {len(saved_files)}")
        print(f"  - Quality mode: {quality_mode}")
        print(f"  - Similarity threshold: {similarity_threshold}")
        thread = threading.Thread(
            target=process_photos_automatic,
            args=(job_id, upload_dir, quality_mode, similarity_threshold, session_id)
        )
        message = 'Processing started - AI will automatically select the best photos!'

    thread.start()

    return jsonify({
        'job_id': job_id,
        'files_uploaded': len(saved_files),
        'has_reference_photos': has_references,
        'reference_count': ref_count,
        'message': message,
        'needs_review': has_references  # Client should redirect to review page
    })


@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    """Process photos from a local folder path (for large batches)."""
    data = request.get_json()
    folder_path = data.get('folder_path', '').strip()
    quality_mode = data.get('quality_mode', 'balanced')
    similarity_threshold = float(data.get('similarity_threshold', 0.92))

    if not folder_path:
        return jsonify({'error': 'No folder path provided'}), 400

    # Validate folder exists
    if not os.path.isdir(folder_path):
        return jsonify({'error': f'Folder not found: {folder_path}'}), 400

    # Get session ID for face matching
    session_id = session.get('session_id')

    # Create job with reference to original folder
    job_id = str(uuid.uuid4())[:8]

    # Count valid image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
    image_files = [f for f in os.listdir(folder_path)
                   if os.path.splitext(f.lower())[1] in image_extensions]

    if not image_files:
        return jsonify({'error': 'No valid image files found in folder'}), 400

    print(f"\n[Job {job_id}] LOCAL FOLDER MODE")
    print(f"  - Folder: {folder_path}")
    print(f"  - Images found: {len(image_files)}")

    # Check if we have reference photos loaded
    has_references = False
    ref_count = 0
    if session_id and session_id in face_matchers:
        ref_count = face_matchers[session_id].get_reference_count()
        has_references = ref_count > 0

    # Create thumbnails directory
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
    os.makedirs(thumb_dir, exist_ok=True)

    # Initialize job - use original folder path as upload_dir
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Preparing to process photos...',
        'total_files': len(image_files),
        'total_uploaded': len(image_files),
        'upload_dir': folder_path,  # Point to original folder
        'thumb_dir': thumb_dir,
        'session_id': session_id,
        'has_reference_photos': has_references,
        'reference_count': ref_count,
        'quality_mode': quality_mode,
        'similarity_threshold': similarity_threshold,
        'is_local_folder': True,  # Flag for local folder mode
        'results': None
    }

    # Decide which processing mode to use
    if has_references:
        print(f"  - Reference photos: {ref_count}")
        print(f"  - Mode: Face Filtering")
        thread = threading.Thread(
            target=process_photos_face_filter_only,
            args=(job_id, folder_path, session_id)
        )
        message = f'Scanning {len(image_files)} photos to find your child...'
    else:
        print(f"  - Mode: Full Automatic")
        thread = threading.Thread(
            target=process_photos_automatic,
            args=(job_id, folder_path, quality_mode, similarity_threshold, session_id)
        )
        message = 'Processing started - AI will automatically select the best photos!'

    thread.start()

    return jsonify({
        'job_id': job_id,
        'files_found': len(image_files),
        'has_reference_photos': has_references,
        'reference_count': ref_count,
        'message': message,
        'needs_review': has_references
    })


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]
    response = {
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'total_photos': job.get('total_photos', 0),
        'photos_checked': job.get('photos_checked', 0)
    }

    if job['status'] == 'complete' and job['results']:
        response['summary'] = job['results']['summary']

    return jsonify(response)


@app.route('/results/<job_id>')
def get_results(job_id):
    """Get processing results."""
    try:
        if job_id not in processing_jobs:
            # Try loading from file
            results_file = os.path.join(RESULTS_FOLDER, f"{job_id}.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    return jsonify(json.load(f))
            return jsonify({'error': 'Job not found'}), 404

        job = processing_jobs[job_id]
        if job['status'] != 'complete':
            return jsonify({'error': 'Processing not complete', 'status': job['status'], 'message': job.get('message', '')}), 400

        # Try from memory first, then file
        if 'results' in job and job['results']:
            return jsonify(job['results'])

        # Fallback to file
        results_file = os.path.join(RESULTS_FOLDER, f"{job_id}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return jsonify(json.load(f))

        return jsonify({'error': 'Results not found'}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/thumbnail/<job_id>/<filename>')
def get_thumbnail(job_id, filename):
    """Serve thumbnail images."""
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
    return send_from_directory(thumb_dir, filename)


@app.route('/photo/<job_id>/<filename>')
def get_photo(job_id, filename):
    """Serve full-size photos with proper EXIF rotation handling."""
    from io import BytesIO
    from PIL import ExifTags

    photo_dir = os.path.join(UPLOAD_FOLDER, job_id)
    filepath = os.path.join(photo_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    ext = os.path.splitext(filename)[1].lower()

    # Handle HEIC/HEIF - convert to JPEG
    if ext in ['.heic', '.heif']:
        try:
            img = Image.open(filepath)
            img = img.convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            return send_file(buffer, mimetype='image/jpeg')
        except Exception as e:
            print(f"Error converting HEIC: {e}")
            return send_from_directory(photo_dir, filename)

    # Handle JPG/JPEG - apply EXIF rotation
    if ext in ['.jpg', '.jpeg']:
        try:
            img = Image.open(filepath)

            # Get EXIF orientation and rotate if needed
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass

            # Convert to RGB if needed (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            return send_file(buffer, mimetype='image/jpeg')
        except Exception as e:
            print(f"Error processing JPEG: {e}")
            return send_from_directory(photo_dir, filename)

    # Other formats - serve directly
    return send_from_directory(photo_dir, filename)


@app.route('/download/<job_id>')
def download_selected(job_id):
    """Download selected photos as zip."""
    import zipfile
    from io import BytesIO

    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Processing not complete'}), 400

    results = job.get('results', {})
    selected = results.get('selected', [])
    upload_dir = job.get('upload_dir', '')

    if not selected:
        return jsonify({'error': 'No selected photos found'}), 404

    if not upload_dir:
        return jsonify({'error': 'Upload directory not found'}), 404

    # Create zip file
    memory_file = BytesIO()
    files_added = 0
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for photo in selected:
            filename = photo.get('filename', '')
            photo_path = os.path.join(upload_dir, filename)
            if os.path.exists(photo_path):
                zf.write(photo_path, filename)
                files_added += 1
            else:
                print(f"[Download] File not found: {photo_path}")

    if files_added == 0:
        return jsonify({'error': f'No files found in {upload_dir}. Files may have been cleaned up.'}), 404

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'selected_photos_{job_id}.zip'
    )


@app.route('/download_filtered/<job_id>')
def download_filtered(job_id):
    """Download all filtered photos (after face matching, before quality selection)."""
    import zipfile
    from io import BytesIO

    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]

    # Get filtered photos from review data
    filtered_photos = []
    if 'review_data' in job:
        filtered_photos = [p['filename'] for p in job['review_data'].get('filtered_photos', [])]
    else:
        # Try to load from file
        review_file = os.path.join(RESULTS_FOLDER, f"{job_id}_review.json")
        if os.path.exists(review_file):
            with open(review_file, 'r') as f:
                review_data = json.load(f)
            filtered_photos = [p['filename'] for p in review_data.get('filtered_photos', [])]

    if not filtered_photos:
        return jsonify({'error': 'No filtered photos found'}), 404

    # Create zip file
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in filtered_photos:
            photo_path = os.path.join(job['upload_dir'], filename)
            if os.path.exists(photo_path):
                zf.write(photo_path, filename)

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'filtered_photos_{job_id}.zip'
    )


@app.route('/cleanup/<job_id>', methods=['POST'])
def cleanup_job(job_id):
    """Clean up job files."""
    if job_id in processing_jobs:
        upload_dir = processing_jobs[job_id].get('upload_dir')
        if upload_dir and os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        del processing_jobs[job_id]

    results_file = os.path.join(RESULTS_FOLDER, f"{job_id}.json")
    if os.path.exists(results_file):
        os.remove(results_file)

    # Also clean up review file
    review_file = os.path.join(RESULTS_FOLDER, f"{job_id}_review.json")
    if os.path.exists(review_file):
        os.remove(review_file)

    return jsonify({'message': 'Cleaned up'})


# ==================== REVIEW WORKFLOW ROUTES ====================

@app.route('/step3_review/<job_id>')
def step3_review(job_id):
    """Step 3: Review filtered photos before quality selection."""
    if job_id not in processing_jobs:
        return render_template('index.html')

    job = processing_jobs[job_id]

    # Check if face filtering is complete
    if job['status'] not in ['review_pending', 'complete']:
        # Still processing or error - redirect back to step2
        return render_template('step2_upload.html',
                              session_id=session.get('session_id'),
                              reference_count=job.get('reference_count', 0))

    return render_template('step3_review.html', job_id=job_id)


@app.route('/review_data/<job_id>')
def get_review_data(job_id):
    """Get the filtered photos data for review."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]

    # Check if we have review data
    if 'review_data' in job:
        return jsonify(job['review_data'])

    # Try to load from file
    review_file = os.path.join(RESULTS_FOLDER, f"{job_id}_review.json")
    if os.path.exists(review_file):
        with open(review_file, 'r') as f:
            review_data = json.load(f)
        return jsonify(review_data)

    return jsonify({'error': 'Review data not found'}), 404


@app.route('/review_thumbnail/<job_id>/<filename>')
def get_review_thumbnail(job_id, filename):
    """Serve thumbnail for review page."""
    # Thumbnails are always stored in uploads/<job_id>/thumbnails
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
    if os.path.exists(os.path.join(thumb_dir, filename)):
        return send_from_directory(thumb_dir, filename)

    # Fallback: check if thumbnails are in the upload_dir (for older jobs)
    if job_id in processing_jobs:
        job = processing_jobs[job_id]
        upload_dir = job.get('upload_dir', '')
        fallback_dir = os.path.join(upload_dir, 'thumbnails')
        if os.path.exists(os.path.join(fallback_dir, filename)):
            return send_from_directory(fallback_dir, filename)

    return send_from_directory(thumb_dir, filename)


@app.route('/review_photo/<job_id>/<filename>')
def get_review_photo(job_id, filename):
    """Serve full-size photo for review modal with EXIF rotation handling."""
    from io import BytesIO
    from PIL import ExifTags

    photo_dir = os.path.join(UPLOAD_FOLDER, job_id)
    filepath = os.path.join(photo_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    ext = os.path.splitext(filename)[1].lower()

    # Handle HEIC/HEIF - convert to JPEG
    if ext in ['.heic', '.heif']:
        try:
            img = Image.open(filepath)
            img = img.convert('RGB')
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            return send_file(buffer, mimetype='image/jpeg')
        except Exception as e:
            print(f"Error converting HEIC: {e}")
            return send_from_directory(photo_dir, filename)

    # Handle JPG/JPEG - apply EXIF rotation
    if ext in ['.jpg', '.jpeg']:
        try:
            img = Image.open(filepath)

            # Get EXIF orientation and rotate if needed
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = img._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation_value == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation_value == 8:
                        img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                pass

            if img.mode != 'RGB':
                img = img.convert('RGB')

            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            return send_file(buffer, mimetype='image/jpeg')
        except Exception as e:
            print(f"Error processing JPEG: {e}")
            return send_from_directory(photo_dir, filename)

    return send_from_directory(photo_dir, filename)


@app.route('/confirm_selection/<job_id>', methods=['POST'])
def confirm_selection(job_id):
    """User confirms their selection - proceed to quality-based selection."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]

    # Get confirmed photos from request
    data = request.get_json()
    if not data or 'selected_photos' not in data:
        return jsonify({'error': 'No photos selected'}), 400

    confirmed_photos = data['selected_photos']
    if len(confirmed_photos) == 0:
        return jsonify({'error': 'At least one photo must be selected'}), 400

    # Get processing parameters from job
    quality_mode = job.get('quality_mode', 'balanced')
    similarity_threshold = job.get('similarity_threshold', 0.92)
    upload_dir = job.get('upload_dir')

    # Load cached face data from review_data (to avoid re-detection in scoring)
    face_data_cache = {}
    if 'review_data' in job:
        for photo in job['review_data'].get('filtered_photos', []):
            filename = photo.get('filename')
            if filename:
                face_data_cache[filename] = {
                    'num_faces': photo.get('num_faces', 0),
                    'face_bboxes': photo.get('face_bboxes', [])
                }
    else:
        # Try loading from review file
        review_file = os.path.join(RESULTS_FOLDER, f"{job_id}_review.json")
        if os.path.exists(review_file):
            with open(review_file, 'r') as f:
                review_data = json.load(f)
            for photo in review_data.get('filtered_photos', []):
                filename = photo.get('filename')
                if filename:
                    face_data_cache[filename] = {
                        'num_faces': photo.get('num_faces', 0),
                        'face_bboxes': photo.get('face_bboxes', [])
                    }

    print(f"[Job {job_id}] Loaded face data cache for {len(face_data_cache)} photos")

    # Update job status
    job['status'] = 'processing'
    job['progress'] = 0
    job['message'] = 'Starting quality-based selection...'
    job['confirmed_photos'] = confirmed_photos

    # Start phase 2 processing
    thread = threading.Thread(
        target=process_photos_quality_selection,
        args=(job_id, upload_dir, quality_mode, similarity_threshold, confirmed_photos, face_data_cache)
    )
    thread.start()

    return jsonify({
        'message': f'Processing {len(confirmed_photos)} confirmed photos...',
        'confirmed_count': len(confirmed_photos)
    })


@app.route('/step4_results/<job_id>')
def step4_results(job_id):
    """Step 4: Final results page."""
    if job_id not in processing_jobs:
        return render_template('index.html')

    job = processing_jobs[job_id]

    # Check reference count from session
    session_id = session.get('session_id')
    ref_count = 0
    if session_id and session_id in face_matchers:
        ref_count = face_matchers[session_id].get_reference_count()

    return render_template('step4_results.html',
                          job_id=job_id,
                          reference_count=ref_count)


# ==================== TEST SINGLE MONTH ROUTES ====================

@app.route('/test-month')
def test_month_page():
    """Test page for single month photo selection."""
    return render_template('test_month.html')


@app.route('/test-month/start', methods=['POST'])
def test_month_start():
    """Start processing a single month folder."""
    data = request.get_json()
    folder_path = data.get('folder_path', '').strip()
    target = int(data.get('target', 40))
    organize_by_month = data.get('organize_by_month', False)

    if not folder_path:
        return jsonify({'error': 'No folder path provided'}), 400

    if not os.path.isdir(folder_path):
        return jsonify({'error': f'Folder not found: {folder_path}'}), 400

    # Count valid image files
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
    image_files = [f for f in os.listdir(folder_path)
                   if os.path.splitext(f.lower())[1] in extensions]

    if not image_files:
        return jsonify({'error': 'No valid image files found in folder'}), 400

    # Create job
    job_id = str(uuid.uuid4())[:8]

    # Create thumbnails directory
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
    os.makedirs(thumb_dir, exist_ok=True)

    processing_jobs[job_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'Starting test...',
        'folder_path': folder_path,
        'thumb_dir': thumb_dir,
        'target': target,
        'total_files': len(image_files),
        'results': None,
        'organize_by_month': organize_by_month
    }

    # Start processing in background
    thread = threading.Thread(
        target=process_test_month,
        args=(job_id, folder_path, target, thumb_dir, organize_by_month)
    )
    thread.start()

    return jsonify({
        'job_id': job_id,
        'total_photos': len(image_files),
        'target': target,
        'organize_by_month': organize_by_month,
        'message': f'Processing {len(image_files)} photos...'
    })


@app.route('/test-month/upload', methods=['POST'])
def test_month_upload():
    """Handle uploaded photos for test-month (for HuggingFace deployment)."""
    if 'photos' not in request.files:
        return jsonify({'error': 'No photos uploaded'}), 400

    files = request.files.getlist('photos')
    target = int(request.form.get('target', 40))
    organize_by_month = request.form.get('organize_by_month', 'false').lower() == 'true'

    if not files or len(files) == 0:
        return jsonify({'error': 'No photos uploaded'}), 400

    # Filter valid image files
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
    valid_files = [f for f in files if f.filename and
                   os.path.splitext(f.filename.lower())[1] in extensions]

    if not valid_files:
        return jsonify({'error': 'No valid image files uploaded'}), 400

    # Create job and upload directory
    job_id = str(uuid.uuid4())[:8]
    upload_dir = os.path.join(UPLOAD_FOLDER, job_id, 'photos')
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(thumb_dir, exist_ok=True)

    # Save uploaded files
    saved_files = []
    for f in valid_files:
        filename = secure_filename(f.filename)
        # Handle duplicate filenames
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(os.path.join(upload_dir, filename)):
            filename = f"{base}_{counter}{ext}"
            counter += 1

        filepath = os.path.join(upload_dir, filename)
        f.save(filepath)
        saved_files.append(filename)

    processing_jobs[job_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'Starting test...',
        'folder_path': upload_dir,  # Use upload dir as folder path
        'thumb_dir': thumb_dir,
        'target': target,
        'total_files': len(saved_files),
        'results': None,
        'is_upload': True,
        'organize_by_month': organize_by_month
    }

    # Start processing in background
    thread = threading.Thread(
        target=process_test_month,
        args=(job_id, upload_dir, target, thumb_dir, organize_by_month)
    )
    thread.start()

    return jsonify({
        'job_id': job_id,
        'total_photos': len(saved_files),
        'target': target,
        'organize_by_month': organize_by_month,
        'message': f'Processing {len(saved_files)} uploaded photos...'
    })


@app.route('/test-month/upload-init', methods=['POST'])
def test_month_upload_init():
    """Initialize chunked upload for test-month."""
    data = request.json
    total_files = data.get('total_files', 0)
    target = data.get('target', 40)
    organize_by_month = data.get('organize_by_month', False)

    job_id = str(uuid.uuid4())[:8]
    upload_dir = os.path.join(UPLOAD_FOLDER, job_id, 'photos')
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(thumb_dir, exist_ok=True)

    # Store upload session
    session_id = f"test_{job_id}"
    upload_sessions[session_id] = {
        'job_id': job_id,
        'upload_dir': upload_dir,
        'thumb_dir': thumb_dir,
        'target': target,
        'organize_by_month': organize_by_month,
        'total_files': total_files,
        'uploaded_files': []
    }

    print(f"[Test-Month Upload {job_id}] Initialized for {total_files} files")

    return jsonify({
        'session_id': session_id,
        'job_id': job_id
    })


@app.route('/test-month/upload-chunk', methods=['POST'])
def test_month_upload_chunk():
    """Handle a chunk of files for test-month."""
    session_id = request.form.get('session_id')
    if not session_id or session_id not in upload_sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session_data = upload_sessions[session_id]
    upload_dir = session_data['upload_dir']
    files = request.files.getlist('files')

    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
    saved_count = 0

    for f in files:
        if f and f.filename:
            ext = os.path.splitext(f.filename.lower())[1]
            if ext in extensions:
                filename = secure_filename(f.filename)
                # Handle duplicate filenames
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(os.path.join(upload_dir, filename)):
                    filename = f"{base}_{counter}{ext}"
                    counter += 1

                f.save(os.path.join(upload_dir, filename))
                session_data['uploaded_files'].append(filename)
                saved_count += 1

    chunk_index = request.form.get('chunk_index', '?')
    print(f"[Test-Month Upload {session_data['job_id']}] Chunk {chunk_index}: saved {saved_count} files (total: {len(session_data['uploaded_files'])})")

    return jsonify({
        'uploaded': len(session_data['uploaded_files']),
        'total': session_data['total_files']
    })


@app.route('/test-month/upload-complete', methods=['POST'])
def test_month_upload_complete():
    """Complete chunked upload and start processing for test-month."""
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in upload_sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session_data = upload_sessions[session_id]
    job_id = session_data['job_id']
    upload_dir = session_data['upload_dir']
    thumb_dir = session_data['thumb_dir']
    target = session_data['target']
    organize_by_month = session_data['organize_by_month']
    saved_files = session_data['uploaded_files']

    # Clean up session
    del upload_sessions[session_id]

    if not saved_files:
        return jsonify({'error': 'No valid image files uploaded'}), 400

    print(f"[Test-Month Upload {job_id}] Complete: {len(saved_files)} files, starting processing...")

    # Create processing job
    processing_jobs[job_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'Starting test...',
        'folder_path': upload_dir,
        'thumb_dir': thumb_dir,
        'target': target,
        'total_files': len(saved_files),
        'results': None,
        'is_upload': True,
        'organize_by_month': organize_by_month
    }

    # Start processing in background
    thread = threading.Thread(
        target=process_test_month,
        args=(job_id, upload_dir, target, thumb_dir, organize_by_month)
    )
    thread.start()

    return jsonify({
        'job_id': job_id,
        'total_photos': len(saved_files),
        'target': target,
        'organize_by_month': organize_by_month,
        'message': f'Processing {len(saved_files)} uploaded photos...'
    })


def process_test_month(job_id, folder_path, target, thumb_dir, organize_by_month=False):
    """Process photos for testing with category-aware selection.

    If organize_by_month is True, groups photos by EXIF date and runs
    selection per month (same as main app Step 4).
    """
    try:
        from photo_selector.monthly_selector import MonthlyPhotoSelector, CategoryDetector
        from photo_selector.embeddings import PhotoEmbedder
        from photo_selector.scoring import PhotoScorer
        from datetime import datetime

        job = processing_jobs[job_id]

        # Get all photos
        extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
        photo_files = [f for f in os.listdir(folder_path)
                       if os.path.splitext(f.lower())[1] in extensions]
        photo_paths = [os.path.join(folder_path, f) for f in photo_files]

        job['message'] = 'Loading CLIP model...'
        job['progress'] = 5

        # Initialize embedder and selector
        embedder = PhotoEmbedder()
        selector = MonthlyPhotoSelector()

        # Step 1: Generate embeddings
        job['message'] = f'Generating embeddings for {len(photo_paths)} photos...'
        job['progress'] = 10
        embeddings = embedder.process_folder(folder_path)
        job['progress'] = 30

        # Step 2: Detect categories for all photos
        job['message'] = 'Detecting photo categories...'
        job['progress'] = 35
        selector._ensure_category_detector()
        categories = selector.category_detector.detect_categories_batch(photo_paths)
        job['progress'] = 45

        # Step 3: Score photos and add category + timestamp
        job['message'] = 'Scoring photos...'
        scorer = PhotoScorer()
        scored_photos = []

        for i, photo_path in enumerate(photo_paths):
            filename = os.path.basename(photo_path)
            scores = scorer.score_photo(photo_path)

            # Get category
            cat, conf = categories.get(filename, ('unknown', 0.0))

            # Get timestamp from EXIF
            dt = selector.get_photo_date(photo_path)

            scored_photos.append({
                'filename': filename,
                'filepath': photo_path,
                'total': scores.get('total', 0),
                'face_quality': scores.get('face_quality', 0),
                'aesthetic_quality': scores.get('aesthetic_quality', 0),
                'emotional_signal': scores.get('emotional_signal', 0),
                'uniqueness': scores.get('uniqueness', 0.5),
                'num_faces': scores.get('num_faces', 0),
                'category': cat,
                'category_confidence': conf,
                'timestamp': dt.timestamp() if dt else None
            })

            if (i + 1) % 10 == 0:
                job['progress'] = 45 + int((i / len(photo_paths)) * 20)
                job['message'] = f'Scoring photos... {i + 1}/{len(photo_paths)}'

        job['progress'] = 70

        # Step 4: Run category-aware HDBSCAN selection
        if organize_by_month:
            # Group photos by month using EXIF dates
            job['message'] = 'Grouping photos by month...'

            # Month names for mapping
            MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']

            photos_by_month = {}
            for photo in scored_photos:
                ts = photo.get('timestamp')
                if ts:
                    dt = datetime.fromtimestamp(ts)
                    month_name = MONTH_NAMES[dt.month - 1]
                else:
                    month_name = 'Unknown'

                photo['month'] = month_name
                if month_name not in photos_by_month:
                    photos_by_month[month_name] = []
                photos_by_month[month_name].append(photo)

            # Calculate target per month (proportional allocation)
            total_photos = len(scored_photos)
            selected = []
            month_stats = []

            for month_name, month_photos in photos_by_month.items():
                # Proportional target for this month
                month_proportion = len(month_photos) / total_photos
                month_target = max(1, int(target * month_proportion))

                job['message'] = f'Processing {month_name} ({len(month_photos)} photos)...'

                # Get embeddings for this month's photos
                month_embeddings = {p['filename']: embeddings.get(p['filename']) for p in month_photos}

                # Run selection for this month
                month_selected = selector.select_hybrid_hdbscan(month_photos, month_embeddings, target=month_target)

                # Add month info to each selected photo
                for photo in month_selected:
                    photo['month'] = month_name

                selected.extend(month_selected)

                month_stats.append({
                    'month': month_name,
                    'total_photos': len(month_photos),
                    'selected': len(month_selected),
                    'target': month_target
                })

            print(f"[Test Month {job_id}] Organized by month: {len(photos_by_month)} months, {len(selected)} total selected")
        else:
            # Single batch selection (original behavior)
            job['message'] = 'Running category-aware clustering and selection...'
            selected = selector.select_hybrid_hdbscan(scored_photos, embeddings, target=target)
            # Add 'Unknown' month to all photos when not organized
            for photo in selected:
                photo['month'] = 'Unknown'
            for photo in scored_photos:
                photo['month'] = 'Unknown'
            month_stats = []

        job['progress'] = 85
        job['message'] = 'Creating thumbnails...'

        # Create thumbnails and build results
        selected_results = []
        for photo in selected:
            filename = photo['filename']
            filepath = photo['filepath']
            thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumb_path = os.path.join(thumb_dir, thumb_name)

            create_thumbnail(filepath, thumb_path)

            # Get embedding for this photo
            photo_emb = embeddings.get(filename)
            embedding_list = photo_emb.tolist() if photo_emb is not None else None

            # Format timestamp for display
            ts = photo.get('timestamp')
            datetime_str = ''
            if ts:
                dt = datetime.fromtimestamp(ts)
                datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')

            selected_results.append({
                'filename': filename,
                'thumbnail': thumb_name,
                'score': float(photo.get('total', 0)),
                'face_quality': float(photo.get('face_quality', 0)),
                'aesthetic_quality': float(photo.get('aesthetic_quality', 0)),
                'emotional_signal': float(photo.get('emotional_signal', 0)),
                'uniqueness': float(photo.get('uniqueness', 0)),
                'num_faces': int(photo.get('num_faces', 0)),
                'multi_face_bonus': float(photo.get('multi_face_bonus', 0)),
                'cluster_id': photo.get('cluster_id', -1),
                'max_similarity': float(photo.get('max_similarity', 0)),
                'category': photo.get('category', 'unknown'),
                'category_confidence': float(photo.get('category_confidence', 0)),
                'event_id': photo.get('event_id', -1),
                'selection_reason': photo.get('selection_reason', ''),
                'datetime': datetime_str,
                'embedding': embedding_list,
                'month': photo.get('month', 'Unknown')
            })

        # Build rejected list
        selected_filenames = {p['filename'] for p in selected}
        rejected_results = []

        for photo in scored_photos:
            if photo['filename'] not in selected_filenames:
                filename = photo['filename']
                filepath = photo['filepath']
                thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
                thumb_path = os.path.join(thumb_dir, thumb_name)

                create_thumbnail(filepath, thumb_path)

                photo_emb = embeddings.get(filename)
                embedding_list = photo_emb.tolist() if photo_emb is not None else None

                # Format timestamp for display
                ts = photo.get('timestamp')
                datetime_str = ''
                if ts:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(ts)
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')

                rejected_results.append({
                    'filename': filename,
                    'thumbnail': thumb_name,
                    'score': float(photo.get('total', 0)),
                    'face_quality': float(photo.get('face_quality', 0)),
                    'aesthetic_quality': float(photo.get('aesthetic_quality', 0)),
                    'num_faces': int(photo.get('num_faces', 0)),
                    'cluster_id': photo.get('cluster_id', -1),
                    'category': photo.get('category', 'unknown'),
                    'event_id': photo.get('event_id', -1),
                    'embedding': embedding_list,
                    'max_similarity': float(photo.get('max_similarity', 0)),
                    'selection_reason': photo.get('rejection_reason', 'Not selected'),
                    'datetime': datetime_str,
                    'month': photo.get('month', 'Unknown')
                })

        # Sort results
        selected_results.sort(key=lambda x: x['score'], reverse=True)
        rejected_results.sort(key=lambda x: x['score'], reverse=True)

        # Cluster distribution
        cluster_counts = {}
        for photo in selected_results:
            cid = photo.get('cluster_id', -1)
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        # Category distribution
        category_counts = {}
        for photo in selected_results:
            cat = photo.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Build results
        job['results'] = {
            'selected': selected_results,
            'rejected': rejected_results,
            'summary': {
                'total_photos': len(photo_paths),
                'selected_count': len(selected_results),
                'rejected_count': len(rejected_results),
                'target': target
            },
            'cluster_distribution': cluster_counts,
            'category_distribution': category_counts,
            'organized_by_month': organize_by_month,
            'month_stats': month_stats
        }

        job['status'] = 'complete'
        job['progress'] = 100
        job['message'] = f'Done! Selected {len(selected_results)} of {len(photo_paths)} photos'

        print(f"\n[Test Month {job_id}] Complete!")
        print(f"  - Total: {len(photo_paths)}")
        print(f"  - Selected: {len(selected_results)}")
        print(f"  - Organized by month: {organize_by_month}")
        if month_stats:
            print(f"  - Month stats: {month_stats}")
        print(f"  - Clusters: {cluster_counts}")
        print(f"  - Categories: {category_counts}")

    except Exception as e:
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['message'] = str(e)
        import traceback
        traceback.print_exc()


@app.route('/test-month/status/<job_id>')
def test_month_status(job_id):
    """Get test month job status."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message']
    })


@app.route('/test-month/results/<job_id>')
def test_month_results(job_id):
    """Get test month results."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Not complete', 'status': job['status']}), 400

    return jsonify(job['results'])


@app.route('/test-month/thumbnail/<job_id>/<filename>')
def test_month_thumbnail(job_id, filename):
    """Serve test month thumbnails."""
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
    return send_from_directory(thumb_dir, filename)


@app.route('/test-month/download/<job_id>')
def test_month_download(job_id):
    """Download selected photos from test-month as ZIP."""
    import zipfile
    from io import BytesIO

    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Processing not complete'}), 400

    results = job.get('results', {})
    selected = results.get('selected', [])
    folder_path = job.get('folder_path', '')

    if not selected:
        return jsonify({'error': 'No selected photos'}), 404

    if not folder_path:
        return jsonify({'error': 'Folder path not found'}), 404

    # Create zip file
    memory_file = BytesIO()
    files_added = 0
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for photo in selected:
            filename = photo.get('filename', '')
            # Build full path from folder_path + filename
            photo_path = os.path.join(folder_path, filename)
            if os.path.exists(photo_path):
                zf.write(photo_path, filename)
                files_added += 1

    if files_added == 0:
        return jsonify({'error': 'No files could be added to ZIP'}), 404

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'test_selected_{job_id}.zip'
    )


if __name__ == '__main__':
    print("""
    ============================================
        PHOTO SELECTION WEB APP
        Open http://localhost:5000 in your browser

        NEW: Automatic selection mode!
        The AI decides which photos to keep.

        TEST: /test-month for single folder testing
    ============================================
    """)
    # Use port 7860 for Hugging Face Spaces, 5000 for local
    import os
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
