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
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'heic', 'heif', 'webp'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5GB max (for large photo batches)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['MAX_FORM_MEMORY_SIZE'] = 5 * 1024 * 1024 * 1024  # 5GB for form data

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_FOLDER, exist_ok=True)

# Store processing status
processing_jobs = {}

# Store face matchers for sessions (reuse to avoid reloading model)
face_matchers = {}


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
        processing_jobs[job_id]['message'] = f'Scanning {total_photos} photos for your child...'

        # Create thumbnails directory
        thumbs_dir = os.path.join(upload_dir, 'thumbnails')
        os.makedirs(thumbs_dir, exist_ok=True)

        # Get all photo paths
        photo_paths = [os.path.join(upload_dir, fn) for fn in photo_files]

        # Run face filtering
        print(f"[Job {job_id}] Starting face detection and matching...")
        processing_jobs[job_id]['progress'] = 30
        filter_results = face_matcher.filter_photos(photo_paths)

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
        processing_jobs[job_id]['message'] = 'Creating thumbnails...'

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
                'matched_face_idx': match.get('matched_face_idx', 0)
            })

            # Progress update every 10 photos
            if (i + 1) % 10 == 0:
                progress = 70 + int((i / matched_count) * 25)
                processing_jobs[job_id]['progress'] = progress
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


def process_photos_quality_selection(job_id, upload_dir, quality_mode, similarity_threshold, confirmed_photos):
    """
    Phase 2: Quality-based selection on confirmed photos.
    Runs after user has reviewed and confirmed the face-filtered photos.
    """
    try:
        print(f"\n{'='*60}")
        print(f"[Job {job_id}] PHASE 2: Quality Selection Started")
        print(f"{'='*60}")
        print(f"[Job {job_id}] Confirmed photos: {len(confirmed_photos)}")
        print(f"[Job {job_id}] Quality mode: {quality_mode}")
        print(f"[Job {job_id}] Similarity threshold: {similarity_threshold}")

        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['progress'] = 5
        processing_jobs[job_id]['message'] = 'Loading AI models for quality analysis...'

        print(f"[Job {job_id}] Loading CLIP model and pipeline components...")

        # Import pipeline components
        from photo_selector.embeddings import PhotoEmbedder
        from photo_selector.temporal import TemporalSegmenter
        from photo_selector.clustering import PhotoClusterer, BucketClusterManager
        from photo_selector.scoring import PhotoScorer, ClusterScorer
        from photo_selector.auto_selector import SmartPhotoSelector, SelectionReason

        # Step 1: Generate embeddings for confirmed photos only
        processing_jobs[job_id]['progress'] = 20
        processing_jobs[job_id]['message'] = 'Analyzing confirmed photos with CLIP AI...'

        print(f"[Job {job_id}] Generating CLIP embeddings for {len(confirmed_photos)} photos...")

        embedder = PhotoEmbedder()

        # Generate embeddings only for confirmed photos
        embeddings = {}
        for i, filename in enumerate(confirmed_photos):
            filepath = os.path.join(upload_dir, filename)
            if os.path.exists(filepath):
                # Load image first, then get embedding
                img = embedder.load_image(filepath)
                if img is not None:
                    embedding = embedder.get_embedding(img)
                    if embedding is not None:
                        embeddings[filename] = embedding
                    img.close()

            # Update progress
            progress = 20 + int((i / len(confirmed_photos)) * 20)
            processing_jobs[job_id]['progress'] = progress

        processing_jobs[job_id]['progress'] = 40
        processing_jobs[job_id]['message'] = 'Organizing by date...'

        print(f"[Job {job_id}] Embeddings generated: {len(embeddings)}")
        print(f"[Job {job_id}] Organizing photos by date (temporal segmentation)...")

        # Step 2: Temporal segmentation (only for confirmed photos)
        segmenter = TemporalSegmenter(bucket_type="monthly")

        # Create a filtered folder view for confirmed photos
        buckets = {}
        for filename in confirmed_photos:
            filepath = os.path.join(upload_dir, filename)
            if os.path.exists(filepath):
                photo_date = segmenter.get_photo_date(filepath)
                if photo_date:
                    bucket_key = segmenter.get_bucket_key(photo_date)
                else:
                    bucket_key = 'unknown'
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(filename)

        # Calculate targets
        estimated_target = max(10, len(embeddings) // 3)
        targets = {}
        total_photos = sum(len(files) for files in buckets.values())
        for bucket_key, files in buckets.items():
            bucket_ratio = len(files) / total_photos if total_photos > 0 else 0
            targets[bucket_key] = max(1, int(estimated_target * bucket_ratio))

        processing_jobs[job_id]['progress'] = 50
        processing_jobs[job_id]['message'] = 'Grouping similar photos...'

        print(f"[Job {job_id}] Time buckets created: {len(buckets)}")
        for bucket_key, files in buckets.items():
            print(f"  - {bucket_key}: {len(files)} photos")
        print(f"[Job {job_id}] Running HDBSCAN clustering...")

        # Step 3: Clustering
        clusterer = BucketClusterManager(PhotoClusterer(method="hdbscan", min_cluster_size=3, temporal_gap_hours=6.0))

        # Build cluster_results structure
        cluster_results = {}
        for bucket_key, filenames in buckets.items():
            if len(filenames) == 0:
                continue

            bucket_embeddings = [embeddings[fn] for fn in filenames if fn in embeddings]
            valid_filenames = [fn for fn in filenames if fn in embeddings]

            if len(valid_filenames) < 2:
                # Single photo in bucket - assign to cluster 0
                cluster_results[bucket_key] = {
                    'filenames': valid_filenames,
                    'labels': [0] * len(valid_filenames),
                    'target': targets.get(bucket_key, 1)
                }
            else:
                # Use clusterer for this bucket
                # cluster_all_buckets expects List[Dict] with 'filename' key
                bucket_data = {bucket_key: [{'filename': fn} for fn in valid_filenames]}
                bucket_targets = {bucket_key: targets.get(bucket_key, 1)}
                result = clusterer.cluster_all_buckets(bucket_data, embeddings, bucket_targets, use_adaptive=False)
                cluster_results.update(result)

        processing_jobs[job_id]['progress'] = 60
        processing_jobs[job_id]['message'] = 'Scoring photo quality...'

        total_clusters = sum(len(set(bucket_data['labels'])) for bucket_data in cluster_results.values())
        print(f"[Job {job_id}] Clustering complete: {total_clusters} clusters across {len(cluster_results)} buckets")
        print(f"[Job {job_id}] Scoring photo quality (face, aesthetic, emotional, uniqueness)...")

        # Step 4: Score photos
        scorer = ClusterScorer(PhotoScorer())
        all_scores = {}

        for bucket_key, bucket_data in cluster_results.items():
            filenames = bucket_data['filenames']
            labels = np.array(bucket_data['labels'])
            bucket_embeddings = np.array([embeddings[fn] for fn in filenames if fn in embeddings])

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
        processing_jobs[job_id]['message'] = 'AI selecting best photos...'

        print(f"[Job {job_id}] Photos scored: {len(all_scores)}")
        print(f"[Job {job_id}] Running automatic selection (mode: {quality_mode})...")

        # Step 5: Automatic selection
        auto_selector = SmartPhotoSelector(
            quality_mode=quality_mode,
            similarity_threshold=similarity_threshold
        )

        selection_results = auto_selector.process_all_photos(
            all_scores, embeddings, cluster_results
        )

        print(f"\n[Job {job_id}] Selection Results:")
        print(f"  - Selected: {len(selection_results['selected'])} photos")
        print(f"  - Rejected: {len(selection_results['rejected'])} photos")

        # Add review stats to summary
        selection_results['summary']['face_filtering'] = {
            'total_photos': processing_jobs[job_id].get('total_uploaded', len(confirmed_photos)),
            'after_face_filter': len(confirmed_photos),
            'user_confirmed': len(confirmed_photos)
        }
        # Add total_processed for template compatibility
        selection_results['summary']['total_processed'] = len(confirmed_photos)

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

        print(f"\n[Job {job_id}] PHASE 2 COMPLETE!")
        print(f"  - Final selection: {len(results['selected'])} photos")
        print(f"  - Filtered out: {len(results['rejected'])} photos")
        print(f"  - Results saved to: {results_file}")
        print(f"{'='*60}\n")

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


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]
    response = {
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message']
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

    # Create zip file
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for photo in job['results']['selected']:
            photo_path = os.path.join(job['upload_dir'], photo['filename'])
            if os.path.exists(photo_path):
                zf.write(photo_path, photo['filename'])

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'selected_photos_{job_id}.zip'
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
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
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

    # Update job status
    job['status'] = 'processing'
    job['progress'] = 0
    job['message'] = 'Starting quality-based selection...'
    job['confirmed_photos'] = confirmed_photos

    # Start phase 2 processing
    thread = threading.Thread(
        target=process_photos_quality_selection,
        args=(job_id, upload_dir, quality_mode, similarity_threshold, confirmed_photos)
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


if __name__ == '__main__':
    print("""
    ============================================
        PHOTO SELECTION WEB APP
        Open http://localhost:5000 in your browser

        NEW: Automatic selection mode!
        The AI decides which photos to keep.
    ============================================
    """)
    app.run(debug=True, host='0.0.0.0', port=5000)
