"""
Face Matcher Module
Uses InsightFace for accurate face detection and recognition.
Filters photos to find those containing a specific person (e.g., your child).
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import cv2
from tqdm import tqdm
from pathlib import Path

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# Try to import InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not installed. Install with: pip install insightface onnxruntime")


class FaceMatcher:
    """
    Matches faces across photos using InsightFace embeddings (ArcFace).

    Workflow:
    1. Extract face embeddings from reference photos (2-3 photos of target person)
    2. Create average embedding for the target person
    3. Scan all event photos and find matches
    """

    def __init__(self,
                 similarity_threshold: float = 0.5,
                 detection_threshold: float = 0.5,
                 model_name: str = 'buffalo_s'):
        """
        Initialize FaceMatcher.

        Args:
            similarity_threshold: Minimum cosine similarity for a face match (0.4-0.6 recommended)
            detection_threshold: Minimum confidence for face detection
            model_name: InsightFace model to use ('buffalo_l' is most accurate, 'buffalo_s' is faster)
        """
        self.similarity_threshold = similarity_threshold
        self.detection_threshold = detection_threshold
        self.model_name = model_name
        self.face_analyzer = None
        self.reference_embeddings = []
        self.average_embedding = None
        self.is_initialized = False

        if INSIGHTFACE_AVAILABLE:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the InsightFace model."""
        try:
            # Try GPU first
            self.face_analyzer = FaceAnalysis(
                name=self.model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self.is_initialized = True
            print(f"InsightFace model '{self.model_name}' loaded successfully (GPU)")
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU-only execution...")
            try:
                self.face_analyzer = FaceAnalysis(
                    name=self.model_name,
                    providers=['CPUExecutionProvider']
                )
                self.face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                self.is_initialized = True
                print(f"InsightFace model '{self.model_name}' loaded successfully (CPU)")
            except Exception as e2:
                print(f"Failed to load InsightFace: {e2}")
                self.face_analyzer = None
                self.is_initialized = False

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image and convert to BGR format for InsightFace.
        Supports HEIC/HEIF formats.
        """
        try:
            # Load with PIL (handles more formats including HEIC)
            pil_image = Image.open(image_path)

            # Handle rotation from EXIF
            try:
                from PIL import ExifTags
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = pil_image._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        pil_image = pil_image.rotate(180, expand=True)
                    elif orientation_value == 6:
                        pil_image = pil_image.rotate(270, expand=True)
                    elif orientation_value == 8:
                        pil_image = pil_image.rotate(90, expand=True)
            except:
                pass

            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert to numpy array (RGB)
            rgb_image = np.array(pil_image)

            # Convert RGB to BGR for InsightFace/OpenCV
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            return bgr_image

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def extract_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Extract all faces from an image.

        Returns:
            List of face dictionaries containing:
            - 'embedding': 512-dim face embedding
            - 'bbox': bounding box [x1, y1, x2, y2]
            - 'det_score': detection confidence
        """
        if self.face_analyzer is None:
            return []

        try:
            faces = self.face_analyzer.get(image)
            return [
                {
                    'embedding': face.embedding,
                    'bbox': face.bbox.tolist(),
                    'det_score': float(face.det_score)
                }
                for face in faces
                if face.det_score >= self.detection_threshold
            ]
        except Exception as e:
            print(f"Error extracting faces: {e}")
            return []

    def add_reference_photo(self, image_path: str) -> Dict:
        """
        Add a reference photo of the target person.
        Extracts the largest/most prominent face as the reference.

        Args:
            image_path: Path to reference photo

        Returns:
            Dict with status and face info
        """
        if not self.is_initialized:
            return {'success': False, 'error': 'InsightFace not initialized'}

        image = self.load_image(image_path)
        if image is None:
            return {'success': False, 'error': 'Failed to load image'}

        faces = self.extract_faces(image)

        if not faces:
            return {'success': False, 'error': 'No face detected in reference photo'}

        # Select the largest face (by bounding box area)
        largest_face = max(faces, key=lambda f:
            (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])
        )

        self.reference_embeddings.append(largest_face['embedding'])
        self._update_average_embedding()

        return {
            'success': True,
            'faces_detected': len(faces),
            'face_used': {
                'bbox': largest_face['bbox'],
                'confidence': largest_face['det_score']
            },
            'total_references': len(self.reference_embeddings)
        }

    def _update_average_embedding(self):
        """Update the average embedding from all reference photos."""
        if self.reference_embeddings:
            self.average_embedding = np.mean(self.reference_embeddings, axis=0)
            # Normalize the average embedding
            norm = np.linalg.norm(self.average_embedding)
            if norm > 0:
                self.average_embedding = self.average_embedding / norm

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings.

        Returns:
            Similarity score between -1 and 1 (higher = more similar)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        emb1_norm = embedding1 / norm1
        emb2_norm = embedding2 / norm2

        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)

        return float(similarity)

    def check_photo_for_target(self, image_path: str) -> Dict:
        """
        Check if the target person appears in a photo.

        Args:
            image_path: Path to photo to check

        Returns:
            Dict with:
            - 'contains_target': bool
            - 'best_match_similarity': float (highest similarity to target)
            - 'all_face_similarities': list of all face similarities
            - 'num_faces': number of faces in photo
        """
        if self.average_embedding is None:
            return {
                'contains_target': False,
                'error': 'No reference photos loaded'
            }

        image = self.load_image(image_path)
        if image is None:
            return {
                'contains_target': False,
                'error': 'Failed to load image'
            }

        faces = self.extract_faces(image)

        if not faces:
            return {
                'contains_target': False,
                'best_match_similarity': 0.0,
                'all_face_similarities': [],
                'num_faces': 0
            }

        # Compare each face to the target
        similarities = []
        for face in faces:
            sim = self.compute_similarity(face['embedding'], self.average_embedding)
            similarities.append({
                'similarity': sim,
                'bbox': face['bbox'],
                'det_score': face['det_score']
            })

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        best_match = similarities[0]['similarity']
        contains_target = best_match >= self.similarity_threshold

        # Extract face bboxes for caching (to avoid re-detection in scoring)
        face_bboxes = [face['bbox'] for face in faces]

        return {
            'contains_target': contains_target,
            'best_match_similarity': best_match,
            'all_face_similarities': similarities,
            'num_faces': len(faces),
            'face_bboxes': face_bboxes  # Cached face locations for scoring
        }

    def filter_photos(self,
                      photo_paths: List[str],
                      progress_callback=None,
                      max_workers: int = 8) -> Dict:
        """
        Filter a list of photos to find those containing the target person.
        Uses parallel processing for ~4x speedup.

        Args:
            photo_paths: List of paths to photos to filter
            progress_callback: Optional callback function(current, total, message)
            max_workers: Number of parallel threads (default 8)

        Returns:
            Dict with:
            - 'matched_photos': List of photos containing target
            - 'unmatched_photos': List of photos without target
            - 'no_faces_photos': List of photos with no detected faces
            - 'error_photos': List of photos that couldn't be processed
            - 'statistics': Summary statistics
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        if self.average_embedding is None:
            return {'error': 'No reference photos loaded. Add reference photos first.'}

        matched = []
        unmatched = []
        no_faces = []
        errors = []

        total = len(photo_paths)
        processed_count = [0]  # Use list for mutable counter in closure
        lock = threading.Lock()

        def process_single_photo(photo_path):
            """Process a single photo and return result."""
            result = self.check_photo_for_target(photo_path)
            return photo_path, result

        print(f"[Face Filter] Processing {total} photos with {max_workers} workers...")

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(process_single_photo, path): path
                for path in photo_paths
            }

            # Process results as they complete
            for future in as_completed(future_to_path):
                photo_path, result = future.result()

                with lock:
                    processed_count[0] += 1
                    # Get filename from path
                    filename = Path(photo_path).name
                    # Print progress every 10 photos
                    if processed_count[0] % 10 == 0 or processed_count[0] == total:
                        print(f"[Face Filter] [{processed_count[0]}/{total}] Checked: {filename}")
                    if progress_callback and processed_count[0] % 50 == 0:
                        progress_callback(processed_count[0], total, f"Scanned {processed_count[0]}/{total} photos")

                if 'error' in result:
                    if result['error'] == 'Failed to load image':
                        errors.append({
                            'path': photo_path,
                            'error': result['error']
                        })
                    continue

                if result['num_faces'] == 0:
                    no_faces.append({
                        'path': photo_path,
                        'num_faces': 0
                    })
                elif result['contains_target']:
                    matched.append({
                        'path': photo_path,
                        'similarity': result['best_match_similarity'],
                        'num_faces': result['num_faces'],
                        'all_similarities': result['all_face_similarities'],
                        'face_bboxes': result.get('face_bboxes', [])  # Cached for scoring
                    })
                else:
                    unmatched.append({
                        'path': photo_path,
                        'best_similarity': result['best_match_similarity'],
                        'num_faces': result['num_faces']
                    })

        # Sort matched photos by similarity (highest first)
        matched.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'matched_photos': matched,
            'unmatched_photos': unmatched,
            'no_faces_photos': no_faces,
            'error_photos': errors,
            'statistics': {
                'total_scanned': total,
                'matched': len(matched),
                'unmatched': len(unmatched),
                'no_faces': len(no_faces),
                'errors': len(errors),
                'match_rate': f"{(len(matched) / total * 100):.1f}%" if total > 0 else "0%"
            }
        }

    def clear_references(self):
        """Clear all reference embeddings."""
        self.reference_embeddings = []
        self.average_embedding = None

    def get_reference_count(self) -> int:
        """Get the number of reference photos loaded."""
        return len(self.reference_embeddings)

    def set_similarity_threshold(self, threshold: float):
        """
        Update the similarity threshold.

        Args:
            threshold: New threshold (0.3-0.7 recommended)
                      Lower = more matches (more false positives)
                      Higher = fewer matches (might miss some)
        """
        self.similarity_threshold = max(0.1, min(0.9, threshold))

    def get_borderline_matches(self,
                               photo_paths: List[str],
                               margin: float = 0.1) -> List[Dict]:
        """
        Get photos that are close to the threshold (for manual review).

        Args:
            photo_paths: List of photo paths to check
            margin: How far below threshold to consider (default 0.1)

        Returns:
            List of borderline photos with their similarity scores
        """
        borderline = []
        lower_bound = self.similarity_threshold - margin

        for photo_path in tqdm(photo_paths, desc="Finding borderline matches"):
            result = self.check_photo_for_target(photo_path)

            if result.get('num_faces', 0) > 0:
                sim = result['best_match_similarity']
                if lower_bound <= sim < self.similarity_threshold:
                    borderline.append({
                        'path': photo_path,
                        'similarity': sim,
                        'num_faces': result['num_faces']
                    })

        borderline.sort(key=lambda x: x['similarity'], reverse=True)
        return borderline


# Convenience function for quick filtering
def filter_photos_by_person(reference_photos: List[str],
                            event_photos: List[str],
                            similarity_threshold: float = 0.5) -> Dict:
    """
    Quick function to filter photos containing a specific person.

    Args:
        reference_photos: 2-3 photos of the target person
        event_photos: All photos to filter
        similarity_threshold: Matching threshold (0.4-0.6 recommended)

    Returns:
        Dict with matched and unmatched photos
    """
    matcher = FaceMatcher(similarity_threshold=similarity_threshold)

    # Load reference photos
    print(f"Loading {len(reference_photos)} reference photos...")
    for ref_photo in reference_photos:
        result = matcher.add_reference_photo(ref_photo)
        if not result['success']:
            print(f"Warning: Failed to add reference {ref_photo}: {result.get('error')}")
        else:
            print(f"  Added: {Path(ref_photo).name} (confidence: {result['face_used']['confidence']:.2f})")

    if matcher.get_reference_count() == 0:
        return {'error': 'No valid reference photos loaded'}

    print(f"\nScanning {len(event_photos)} photos for target person...")
    # Filter event photos
    return matcher.filter_photos(event_photos)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python face_matcher.py <reference_photo> <photo_to_check_or_directory>")
        print("\nExample:")
        print("  python face_matcher.py child_ref.jpg event_photo.jpg")
        print("  python face_matcher.py child_ref.jpg ./event_photos/")
        sys.exit(1)

    reference_photo = sys.argv[1]
    test_path = sys.argv[2]

    # Initialize matcher
    matcher = FaceMatcher(similarity_threshold=0.5)

    # Add reference
    result = matcher.add_reference_photo(reference_photo)
    if not result['success']:
        print(f"Error: {result.get('error')}")
        sys.exit(1)

    print(f"Reference loaded: {Path(reference_photo).name}")
    print(f"Faces in reference: {result['faces_detected']}")

    # Check if test_path is a directory or file
    test_path = Path(test_path)

    if test_path.is_dir():
        # Get all image files
        extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
        photos = [str(p) for p in test_path.iterdir()
                  if p.suffix.lower() in extensions]

        print(f"\nScanning {len(photos)} photos...")
        results = matcher.filter_photos(photos)

        print(f"\n=== Results ===")
        print(f"Total scanned: {results['statistics']['total_scanned']}")
        print(f"Matched: {results['statistics']['matched']}")
        print(f"No faces: {results['statistics']['no_faces']}")
        print(f"Match rate: {results['statistics']['match_rate']}")

        if results['matched_photos']:
            print(f"\nMatched photos:")
            for match in results['matched_photos'][:10]:  # Show top 10
                print(f"  {Path(match['path']).name}: {match['similarity']:.2%}")
    else:
        # Single file
        result = matcher.check_photo_for_target(str(test_path))

        print(f"\nChecking: {test_path.name}")
        print(f"Faces detected: {result.get('num_faces', 0)}")
        print(f"Target found: {result.get('contains_target', False)}")
        print(f"Best match: {result.get('best_match_similarity', 0):.2%}")
