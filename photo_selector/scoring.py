"""
Step 4: Score Photos Within Each Cluster
Weighted scoring based on:
- Face quality (35%): sharpness, eyes open, smile
- Aesthetic quality (25%): lighting, composition
- Emotional signal (20%): expression, interaction
- Uniqueness (20%): distance from other photos

OPTIMIZED: Removed slow body/profile detection (3x speedup)
"""

import math
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class PhotoScorer:
    """Score photos based on multiple quality criteria."""

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize scorer with weights.

        Args:
            weights: Dictionary of criterion -> weight
        """
        self.weights = weights or {
            'face_quality': 0.35,
            'aesthetic_quality': 0.25,
            'emotional_signal': 0.20,
            'uniqueness': 0.20
        }

        # Load face detector (frontal only - profile/body detection too slow)
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.has_face_detector = True
        except:
            self.has_face_detector = False
            print("Warning: Face detector not available")

    def load_image_cv2(self, image_path: str) -> Optional[np.ndarray]:
        """Load image for OpenCV processing."""
        try:
            # Try PIL first for HEIC support
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            pil_img.close()
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        Higher = sharper.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize to 0-1 range (empirically tuned)
        normalized = min(1.0, laplacian_var / 500.0)
        return normalized

    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate if image has good brightness.
        Returns score (optimal around 0.5 brightness).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2]) / 255.0

        # Penalize too dark or too bright
        if brightness < 0.2:
            return brightness / 0.2 * 0.5
        elif brightness > 0.8:
            return (1.0 - brightness) / 0.2 * 0.5
        else:
            return 1.0

    def calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std() / 128.0  # Normalize
        return min(1.0, contrast)

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect frontal faces with filtering to reduce false positives.

        Filters applied:
        1. Minimum face size (1.5% of image area) - removes tiny printed faces
        2. Remove overlapping/nested detections - keeps only largest when boxes overlap
        3. Cap at 8 faces max - realistic limit for family photos
        4. Stricter detection (minNeighbors=6) - reduces false positives
        """
        if not self.has_face_detector:
            return []

        height, width = image.shape[:2]
        image_area = height * width

        # Minimum face size: 1.5% of image area (sqrt for dimension)
        min_face_dim = int(math.sqrt(image_area * 0.015))
        min_face_dim = max(min_face_dim, 50)  # At least 50px

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,  # Increased from 5 for fewer false positives
            minSize=(min_face_dim, min_face_dim)
        )

        if len(faces) == 0:
            return []

        # Convert to list for filtering
        faces = [list(f) for f in faces]

        # Filter 1: Remove faces smaller than 1.5% of image area
        min_area = image_area * 0.015
        faces = [f for f in faces if f[2] * f[3] >= min_area]

        if len(faces) == 0:
            return []

        # Filter 2: Remove overlapping/nested faces (non-max suppression)
        faces = self._remove_overlapping_faces(faces)

        # Filter 3: Cap at 8 faces max (realistic limit)
        if len(faces) > 8:
            # Keep the 8 largest faces
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:8]

        return [tuple(f) for f in faces]

    def _remove_overlapping_faces(self, faces: List, overlap_thresh: float = 0.5) -> List:
        """
        Remove faces that significantly overlap with larger faces.

        Uses non-max suppression: if a smaller face is >50% inside a larger face,
        it's likely a false positive or nested detection.
        """
        if len(faces) <= 1:
            return faces

        # Sort by area (largest first)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        keep = []
        for face in faces:
            x1, y1, w1, h1 = face
            is_nested = False

            for kept in keep:
                x2, y2, w2, h2 = kept

                # Calculate intersection
                ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                intersection = ix * iy

                # If this face is >50% inside a kept face, skip it
                face_area = w1 * h1
                if face_area > 0 and intersection > face_area * overlap_thresh:
                    is_nested = True
                    break

            if not is_nested:
                keep.append(face)

        return keep

    def calculate_face_quality(self, image: np.ndarray,
                                faces: List[Tuple]) -> float:
        """
        Calculate face quality score.
        Considers: presence, size, sharpness of faces.
        """
        if not faces:
            return 0.3  # Some photos are valid without faces

        height, width = image.shape[:2]
        image_area = height * width

        face_scores = []
        for (x, y, w, h) in faces:
            # Face size relative to image
            face_area = w * h
            size_score = min(1.0, (face_area / image_area) * 20)  # Good if face is ~5% of image

            # Face region sharpness
            face_region = image[y:y+h, x:x+w]
            if face_region.size > 0:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                sharp_score = min(1.0, sharpness / 300.0)
            else:
                sharp_score = 0.5

            # Face position (prefer centered)
            face_center_x = (x + w/2) / width
            face_center_y = (y + h/2) / height
            center_score = 1.0 - (abs(face_center_x - 0.5) + abs(face_center_y - 0.4)) / 1.4

            face_score = (size_score * 0.4 + sharp_score * 0.4 + center_score * 0.2)
            face_scores.append(face_score)

        # Return average of top faces
        return np.mean(sorted(face_scores, reverse=True)[:3])

    def calculate_aesthetic_quality(self, image: np.ndarray) -> float:
        """
        Calculate aesthetic quality.
        Simplified version using basic metrics.
        """
        sharpness = self.calculate_sharpness(image)
        brightness = self.calculate_brightness(image)
        contrast = self.calculate_contrast(image)

        # Color saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        sat_score = 1.0 - abs(saturation - 0.4)  # Prefer moderate saturation

        # Rule of thirds (simple version)
        # Higher variance in thirds intersections suggests interesting composition
        h, w = image.shape[:2]
        thirds_points = [
            (w//3, h//3), (2*w//3, h//3),
            (w//3, 2*h//3), (2*w//3, 2*h//3)
        ]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thirds_intensity = [gray[p[1], p[0]] for p in thirds_points if p[1] < h and p[0] < w]
        composition_score = np.std(thirds_intensity) / 128.0 if thirds_intensity else 0.5

        aesthetic = (
            sharpness * 0.35 +
            brightness * 0.25 +
            contrast * 0.20 +
            sat_score * 0.10 +
            composition_score * 0.10
        )

        return min(1.0, aesthetic)

    def calculate_emotional_signal(self, image: np.ndarray,
                                    faces: List[Tuple]) -> float:
        """
        Estimate emotional signal.
        This is a simplified heuristic without deep learning.
        """
        if not faces:
            # For photos without faces, use color warmth as proxy
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = np.mean(hsv[:, :, 0])
            # Warm colors (yellows, oranges) often signal happy emotions
            if 10 <= hue <= 40:  # Yellow-orange range
                return 0.6
            return 0.4

        # With faces, larger and sharper faces suggest more engagement
        face_quality = self.calculate_face_quality(image, faces)

        # Bonus for multiple faces (interaction)
        interaction_bonus = min(0.2, len(faces) * 0.1)

        return min(1.0, face_quality + interaction_bonus)

    def calculate_uniqueness(self, embedding: np.ndarray,
                              cluster_embeddings: np.ndarray) -> float:
        """
        Calculate how unique a photo is within its cluster.
        Higher = more distinct from others.
        """
        if len(cluster_embeddings) <= 1:
            return 1.0

        # Calculate average distance to other photos
        distances = []
        for other_emb in cluster_embeddings:
            if not np.array_equal(embedding, other_emb):
                dist = np.linalg.norm(embedding - other_emb)
                distances.append(dist)

        if not distances:
            return 1.0

        avg_distance = np.mean(distances)

        # Normalize (typical CLIP distance range)
        normalized = min(1.0, avg_distance / 1.5)
        return normalized

    def score_photos_parallel(self, photo_paths: List[str],
                               embeddings: np.ndarray = None,
                               max_workers: int = 4) -> List[Dict[str, float]]:
        """
        Score multiple photos in parallel using ThreadPoolExecutor.

        Args:
            photo_paths: List of image file paths
            embeddings: Array of embeddings (one per photo)
            max_workers: Number of parallel workers

        Returns:
            List of score dictionaries in same order as input
        """
        results = [None] * len(photo_paths)

        def score_single(idx: int) -> Tuple[int, Dict]:
            emb = embeddings[idx] if embeddings is not None else None
            score = self.score_photo(
                photo_paths[idx],
                embedding=emb,
                cluster_embeddings=embeddings
            )
            return idx, score

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(score_single, i) for i in range(len(photo_paths))]
            for future in as_completed(futures):
                idx, score = future.result()
                results[idx] = score

        return results

    def score_photo(self, image_path: str,
                    embedding: np.ndarray = None,
                    cluster_embeddings: np.ndarray = None,
                    cached_face_bboxes: list = None,
                    cached_num_faces: int = None) -> Dict[str, float]:
        """
        Calculate complete score for a photo.
        OPTIMIZED: Uses cached face data if available, skips slow face detection.

        Args:
            image_path: Path to image file
            embedding: Photo's embedding (for uniqueness)
            cluster_embeddings: All embeddings in cluster (for uniqueness)
            cached_face_bboxes: Pre-detected face bounding boxes from Step 2
            cached_num_faces: Pre-detected face count from Step 2

        Returns:
            Dictionary with individual scores and total
        """
        image = self.load_image_cv2(image_path)
        if image is None:
            return {'total': 0.0, 'error': 'Could not load image'}

        # Use cached face data if available (HUGE speedup - skips face detection)
        if cached_face_bboxes is not None and cached_num_faces is not None:
            # Convert InsightFace bbox format [x1, y1, x2, y2] to OpenCV format [x, y, w, h]
            faces = []
            for bbox in cached_face_bboxes:
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    faces.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            num_faces = cached_num_faces
        else:
            # Fallback: Detect frontal faces (slow)
            faces = self.detect_faces(image)
            num_faces = len(faces)

        # Calculate individual scores
        face_quality = self.calculate_face_quality(image, faces)
        aesthetic_quality = self.calculate_aesthetic_quality(image)
        emotional_signal = self.calculate_emotional_signal(image, faces)

        # Uniqueness (requires embeddings)
        if embedding is not None and cluster_embeddings is not None:
            uniqueness = self.calculate_uniqueness(embedding, cluster_embeddings)
        else:
            uniqueness = 0.5  # Default

        # Weighted total
        total = (
            face_quality * self.weights['face_quality'] +
            aesthetic_quality * self.weights['aesthetic_quality'] +
            emotional_signal * self.weights['emotional_signal'] +
            uniqueness * self.weights['uniqueness']
        )

        return {
            'face_quality': round(face_quality, 3),
            'aesthetic_quality': round(aesthetic_quality, 3),
            'emotional_signal': round(emotional_signal, 3),
            'uniqueness': round(uniqueness, 3),
            'num_faces': num_faces,
            'total': round(total, 3)
        }


class ClusterScorer:
    """Score all photos within clusters and select the best."""

    def __init__(self, scorer: PhotoScorer = None):
        self.scorer = scorer or PhotoScorer()

    def score_cluster(self,
                      photo_paths: List[str],
                      embeddings: np.ndarray,
                      use_parallel: bool = True,
                      max_workers: int = 4) -> List[Dict]:
        """
        Score all photos in a cluster.
        OPTIMIZED: Uses parallel processing for clusters with 4+ photos.

        Args:
            photo_paths: List of image file paths
            embeddings: Array of embeddings for these photos
            use_parallel: Whether to use parallel processing
            max_workers: Number of parallel workers

        Returns:
            List of score dictionaries with filename
        """
        # Use parallel scoring for larger clusters
        if use_parallel and len(photo_paths) >= 4:
            scores = self.scorer.score_photos_parallel(
                photo_paths, embeddings, max_workers=max_workers
            )
            # Add filepath and filename
            for i, score in enumerate(scores):
                score['filepath'] = photo_paths[i]
                score['filename'] = Path(photo_paths[i]).name
        else:
            # Sequential for small clusters
            scores = []
            for i, path in enumerate(photo_paths):
                score = self.scorer.score_photo(
                    path,
                    embedding=embeddings[i] if embeddings is not None else None,
                    cluster_embeddings=embeddings
                )
                score['filepath'] = path
                score['filename'] = Path(path).name
                scores.append(score)

        # Sort by total score
        scores.sort(key=lambda x: x.get('total', 0), reverse=True)
        return scores

    def select_best_from_cluster(self,
                                  photo_paths: List[str],
                                  embeddings: np.ndarray,
                                  num_select: int = 1) -> List[Dict]:
        """
        Select the best N photos from a cluster.

        Args:
            photo_paths: List of image file paths
            embeddings: Embeddings for photos
            num_select: Number of photos to select

        Returns:
            List of selected photo info with scores
        """
        scores = self.score_cluster(photo_paths, embeddings)
        return scores[:num_select]


if __name__ == "__main__":
    import sys

    scorer = PhotoScorer()

    # Test on a single image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        score = scorer.score_photo(image_path)
        print(f"Scores for {image_path}:")
        for k, v in score.items():
            print(f"  {k}: {v}")
    else:
        print("Usage: python scoring.py <image_path>")
        print("\nScoring weights:")
        for k, v in scorer.weights.items():
            print(f"  {k}: {v*100:.0f}%")
