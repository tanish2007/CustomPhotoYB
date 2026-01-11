"""
Step 4: Score Photos Within Each Cluster
Weighted scoring based on:
- Face quality (35%): sharpness, eyes open, smile
- Aesthetic quality (25%): lighting, composition
- Emotional signal (20%): expression, interaction
- Uniqueness (20%): distance from other photos
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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

        # Load face detectors (frontal and profile)
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            # Profile face detector for side views
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            # Full body detector to identify people even when not facing camera
            self.body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_fullbody.xml'
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
        """Detect frontal faces in image."""
        if not self.has_face_detector:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return [tuple(f) for f in faces]

    def detect_profile_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect profile (side view) faces in image."""
        if not self.has_face_detector:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect profiles looking left
        profiles_left = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Detect profiles looking right (flip image)
        gray_flipped = cv2.flip(gray, 1)
        profiles_right = self.profile_cascade.detectMultiScale(
            gray_flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Combine results (flip right profiles back to original coordinates)
        all_profiles = list(profiles_left)
        for (x, y, w, h) in profiles_right:
            flipped_x = gray.shape[1] - x - w
            all_profiles.append((flipped_x, y, w, h))

        return [tuple(f) for f in all_profiles]

    def detect_bodies(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect full bodies in image (people not facing camera)."""
        if not self.has_face_detector:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use smaller scale image for body detection (faster and more reliable)
        scale = 0.5
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale)

        bodies = self.body_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 60)
        )

        # Scale back to original size
        return [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in bodies]

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

    def score_photo(self, image_path: str,
                    embedding: np.ndarray = None,
                    cluster_embeddings: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate complete score for a photo.

        Args:
            image_path: Path to image file
            embedding: Photo's embedding (for uniqueness)
            cluster_embeddings: All embeddings in cluster (for uniqueness)

        Returns:
            Dictionary with individual scores and total
        """
        image = self.load_image_cv2(image_path)
        if image is None:
            return {'total': 0.0, 'error': 'Could not load image'}

        # Detect frontal faces (people looking at camera)
        frontal_faces = self.detect_faces(image)

        # Detect profile faces (people looking sideways)
        profile_faces = self.detect_profile_faces(image)

        # Detect bodies (people in photo, regardless of face orientation)
        bodies = self.detect_bodies(image)

        # Calculate pose penalty:
        # If we detect bodies/profiles but NO frontal faces, people are not facing camera
        pose_penalty = 0.0
        has_people = len(frontal_faces) > 0 or len(profile_faces) > 0 or len(bodies) > 0

        if has_people and len(frontal_faces) == 0:
            # People detected but not facing camera - significant penalty
            if len(profile_faces) > 0:
                # Side profile - moderate penalty (at least we see some face)
                pose_penalty = 0.15
            elif len(bodies) > 0:
                # Only bodies, no faces at all - major penalty (backs turned, etc.)
                pose_penalty = 0.25

        # Calculate individual scores using frontal faces for quality
        face_quality = self.calculate_face_quality(image, frontal_faces)

        # Apply pose penalty to face quality
        if pose_penalty > 0 and len(frontal_faces) == 0:
            # If no frontal faces but people detected, reduce face quality significantly
            face_quality = max(0.1, face_quality - pose_penalty)

        aesthetic_quality = self.calculate_aesthetic_quality(image)
        emotional_signal = self.calculate_emotional_signal(image, frontal_faces)

        # Also penalize emotional signal if people aren't facing camera
        if pose_penalty > 0:
            emotional_signal = max(0.1, emotional_signal - pose_penalty)

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
            'num_faces': len(frontal_faces),
            'num_profiles': len(profile_faces),
            'num_bodies': len(bodies),
            'pose_penalty': round(pose_penalty, 3),
            'total': round(total, 3)
        }


class ClusterScorer:
    """Score all photos within clusters and select the best."""

    def __init__(self, scorer: PhotoScorer = None):
        self.scorer = scorer or PhotoScorer()

    def score_cluster(self,
                      photo_paths: List[str],
                      embeddings: np.ndarray) -> List[Dict]:
        """
        Score all photos in a cluster.

        Args:
            photo_paths: List of image file paths
            embeddings: Array of embeddings for these photos

        Returns:
            List of score dictionaries with filename
        """
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
