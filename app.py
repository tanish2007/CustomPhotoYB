"""
Photo Selection Web App
Flask-based frontend for testing the photo selection pipeline
Now with AUTOMATIC selection - no target number needed!
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
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

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'heic', 'heif', 'webp'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Store processing status
processing_jobs = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_thumbnail(image_path, thumb_path, size=(300, 300)):
    """Create a thumbnail for display."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumb_path, 'JPEG', quality=85)
        return True
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        return False


def process_photos_automatic(job_id, upload_dir, quality_mode, similarity_threshold, reference_photo_path=None):
    """
    Background task to process photos with AUTOMATIC selection.
    No target number - the algorithm decides what to keep based on quality.
    Optionally filters to only photos containing a specific person (from reference photo).
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
        from photo_selector.face_matcher import FaceMatcher

        # Optional: Face matching if reference photo provided
        face_matcher = None
        if reference_photo_path and os.path.exists(reference_photo_path):
            processing_jobs[job_id]['progress'] = 10
            processing_jobs[job_id]['message'] = 'Learning target person\'s face...'

            face_matcher = FaceMatcher(reference_photo_path)
            if face_matcher.reference_encoding is None:
                processing_jobs[job_id]['message'] = 'Warning: Could not detect face in reference photo. Processing all photos...'
                face_matcher = None

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
        # Temporal gap splitting: Photos >6 hours apart in same cluster = different events
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

        # Optional: Filter by face matching BEFORE automatic selection
        filtered_scores = all_scores
        filtered_embeddings = embeddings
        face_filter_stats = {'total_photos': len(all_scores), 'after_face_filter': len(all_scores), 'filtered_out': 0}

        if face_matcher is not None:
            processing_jobs[job_id]['progress'] = 70
            processing_jobs[job_id]['message'] = 'Identifying photos with target person...'

            photos_with_target = []
            photos_without_target = []

            for filename, score_data in all_scores.items():
                photo_path = os.path.join(upload_dir, filename)
                person_found, match_score, num_faces = face_matcher.find_target_person_in_photo(photo_path, threshold=0.55)

                # Add face match info to score data
                score_data['face_match_score'] = match_score
                score_data['face_match_found'] = person_found

                if person_found:
                    photos_with_target.append(filename)
                else:
                    photos_without_target.append(filename)

            # Filter scores and embeddings to only include photos with target person
            filtered_scores = {fn: all_scores[fn] for fn in photos_with_target}
            filtered_embeddings = {fn: embeddings[fn] for fn in photos_with_target if fn in embeddings}

            # Update cluster_results to only include filtered photos
            for bucket_key, bucket_data in cluster_results.items():
                original_filenames = bucket_data['filenames']
                filtered_filenames = [fn for fn in original_filenames if fn in photos_with_target]

                # Update filenames and labels
                if len(filtered_filenames) > 0:
                    # Find indices of kept photos
                    kept_indices = [i for i, fn in enumerate(original_filenames) if fn in photos_with_target]
                    bucket_data['filenames'] = filtered_filenames
                    bucket_data['labels'] = [bucket_data['labels'][i] for i in kept_indices]

            face_filter_stats['after_face_filter'] = len(filtered_scores)
            face_filter_stats['filtered_out'] = face_filter_stats['total_photos'] - face_filter_stats['after_face_filter']

            processing_jobs[job_id]['message'] = f'Found target person in {len(filtered_scores)} photos...'

        processing_jobs[job_id]['progress'] = 75
        processing_jobs[job_id]['message'] = 'AI deciding which photos to keep...'

        # Step 5: AUTOMATIC SELECTION (on filtered photos if face matching was used)
        auto_selector = SmartPhotoSelector(
            quality_mode=quality_mode,
            similarity_threshold=similarity_threshold
        )

        selection_results = auto_selector.process_all_photos(
            filtered_scores, filtered_embeddings, cluster_results
        )

        # Add face filter stats to summary
        selection_results['summary']['face_filtering'] = face_filter_stats

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

            # Get selection reason
            reason = photo.get('selection_reason', None)
            if isinstance(reason, SelectionReason):
                reason_text = reason.value
            else:
                reason_text = str(reason) if reason else 'High quality photo'

            results['selected'].append({
                'filename': filename,
                'thumbnail': thumb_name,
                'score': photo.get('total', 0),
                'face_quality': photo.get('face_quality', 0),
                'aesthetic_quality': photo.get('aesthetic_quality', 0),
                'emotional_signal': photo.get('emotional_signal', 0),
                'uniqueness': photo.get('uniqueness', 0),
                'bucket': photo.get('bucket', 'unknown'),
                'num_faces': photo.get('num_faces', 0),
                'selection_reason': reason_text,
                'selection_detail': photo.get('selection_detail', reason_text)
            })

        # Process rejected photos
        for photo in selection_results['rejected']:
            filename = photo['filename']
            thumb_name = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumb_path = os.path.join(thumbs_dir, thumb_name)

            create_thumbnail(os.path.join(upload_dir, filename), thumb_path)

            # Get rejection reason
            reason = photo.get('rejection_reason', None)
            if isinstance(reason, SelectionReason):
                reason_text = reason.value
            else:
                reason_text = str(reason) if reason else 'Did not meet quality threshold'

            results['rejected'].append({
                'filename': filename,
                'thumbnail': thumb_name,
                'score': photo.get('total', 0),
                'face_quality': photo.get('face_quality', 0),
                'aesthetic_quality': photo.get('aesthetic_quality', 0),
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
    return render_template('index.html')


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

    # Create job
    job_id = str(uuid.uuid4())[:8]
    upload_dir = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Check for reference photo (optional - for face matching)
    reference_photo_path = None
    if 'reference_photo' in request.files:
        ref_file = request.files['reference_photo']
        if ref_file and allowed_file(ref_file.filename):
            ref_filename = secure_filename('reference_' + ref_file.filename)
            reference_photo_path = os.path.join(upload_dir, ref_filename)
            ref_file.save(reference_photo_path)

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

    # Initialize job
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Uploading files...',
        'total_files': len(saved_files),
        'upload_dir': upload_dir,
        'has_reference_photo': reference_photo_path is not None,
        'results': None
    }

    # Start processing in background with AUTOMATIC selection
    thread = threading.Thread(
        target=process_photos_automatic,
        args=(job_id, upload_dir, quality_mode, similarity_threshold, reference_photo_path)
    )
    thread.start()

    return jsonify({
        'job_id': job_id,
        'files_uploaded': len(saved_files),
        'message': 'Processing started - AI will automatically select the best photos!'
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
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Processing not complete'}), 400

    return jsonify(job['results'])


@app.route('/thumbnail/<job_id>/<filename>')
def get_thumbnail(job_id, filename):
    """Serve thumbnail images."""
    thumb_dir = os.path.join(UPLOAD_FOLDER, job_id, 'thumbnails')
    return send_from_directory(thumb_dir, filename)


@app.route('/photo/<job_id>/<filename>')
def get_photo(job_id, filename):
    """Serve full-size photos."""
    photo_dir = os.path.join(UPLOAD_FOLDER, job_id)
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

    return jsonify({'message': 'Cleaned up'})


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
