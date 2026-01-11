"""
Face Matching Module
Identifies photos containing a specific person based on a reference photo.
Uses OpenCV's face recognition with feature matching.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class FaceMatcher:
    """Match faces across photos to identify a specific person."""

    def __init__(self, reference_image_path: str = None):
        """
        Initialize face matcher with optional reference image.

        Args:
            reference_image_path: Path to photo of the person to identify
        """
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Try to load face recognizer (LBP is available in OpenCV)
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.has_recognizer = True
        except:
            self.has_recognizer = False
            print("Warning: Face recognizer not available. Using simpler matching.")

        self.reference_face = None
        self.reference_encoding = None

        if reference_image_path:
            self.load_reference_face(reference_image_path)

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image using PIL (for HEIC support) and convert to OpenCV format."""
        try:
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            pil_img.close()
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image. Returns list of (x, y, w, h) tuples."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return [tuple(f) for f in faces]

    def extract_face_features(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract features from a face region.
        Uses histogram of oriented gradients (HOG) as features.
        """
        x, y, w, h = face_rect

        # Extract face region
        face_roi = image[y:y+h, x:x+w]

        # Convert to grayscale
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Resize to standard size
        standard_size = (128, 128)
        resized = cv2.resize(gray_face, standard_size)

        # Compute histogram features
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Compute edge features using Sobel
        sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_hist = cv2.calcHist([edge_mag.astype(np.uint8)], [0], None, [256], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()

        # Combine features
        features = np.concatenate([hist, edge_hist])

        return features

    def load_reference_face(self, image_path: str) -> bool:
        """
        Load reference photo and extract face features.

        Returns:
            True if face was successfully extracted, False otherwise
        """
        image = self.load_image(image_path)
        if image is None:
            return False

        faces = self.detect_faces(image)

        if len(faces) == 0:
            print(f"No face detected in reference image: {image_path}")
            return False

        # Use the largest face (assumed to be the main subject)
        largest_face = max(faces, key=lambda f: f[2] * f[3])

        # Store reference face image
        x, y, w, h = largest_face
        self.reference_face = image[y:y+h, x:x+w].copy()

        # Extract features
        self.reference_encoding = self.extract_face_features(image, largest_face)

        print(f"Reference face loaded from {Path(image_path).name}")
        return True

    def compare_faces(self, face_encoding: np.ndarray, threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Compare a face encoding with the reference face.

        Args:
            face_encoding: Feature vector to compare
            threshold: Similarity threshold (0-1, lower = more similar)

        Returns:
            Tuple of (is_match, similarity_score)
        """
        if self.reference_encoding is None:
            return False, 0.0

        # Compute cosine similarity
        dot_product = np.dot(face_encoding, self.reference_encoding)
        norm1 = np.linalg.norm(face_encoding)
        norm2 = np.linalg.norm(self.reference_encoding)

        if norm1 == 0 or norm2 == 0:
            return False, 0.0

        similarity = dot_product / (norm1 * norm2)

        # Normalize to 0-1 range
        similarity = (similarity + 1) / 2  # Cosine is -1 to 1

        is_match = similarity >= threshold

        return is_match, similarity

    def find_target_person_in_photo(self, image_path: str, threshold: float = 0.6) -> Tuple[bool, float, int]:
        """
        Check if the target person appears in a photo.

        Args:
            image_path: Path to photo to check
            threshold: Match threshold

        Returns:
            Tuple of (person_found, best_match_score, num_faces)
        """
        if self.reference_encoding is None:
            return False, 0.0, 0

        image = self.load_image(image_path)
        if image is None:
            return False, 0.0, 0

        faces = self.detect_faces(image)

        if len(faces) == 0:
            return False, 0.0, 0

        # Compare each face with reference
        best_match = False
        best_score = 0.0

        for face_rect in faces:
            face_encoding = self.extract_face_features(image, face_rect)
            is_match, score = self.compare_faces(face_encoding, threshold)

            if score > best_score:
                best_score = score
                best_match = is_match

        return best_match, best_score, len(faces)

    def filter_photos_by_person(self,
                                 photo_paths: List[str],
                                 threshold: float = 0.6,
                                 min_confidence: float = 0.5) -> List[dict]:
        """
        Filter a list of photos to only include ones with the target person.

        Args:
            photo_paths: List of photo file paths
            threshold: Match threshold
            min_confidence: Minimum confidence to include photo

        Returns:
            List of dicts with photo info and match scores
        """
        if self.reference_encoding is None:
            print("No reference face loaded. Cannot filter photos.")
            return []

        results = []

        for photo_path in photo_paths:
            person_found, match_score, num_faces = self.find_target_person_in_photo(
                photo_path, threshold
            )

            if person_found and match_score >= min_confidence:
                results.append({
                    'filepath': photo_path,
                    'filename': Path(photo_path).name,
                    'match_score': match_score,
                    'num_faces': num_faces,
                    'contains_target': True
                })

        # Sort by match score
        results.sort(key=lambda x: x['match_score'], reverse=True)

        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python face_matcher.py <reference_photo> <photo_to_check>")
        sys.exit(1)

    reference_photo = sys.argv[1]
    test_photo = sys.argv[2]

    # Test face matching
    matcher = FaceMatcher(reference_photo)

    if matcher.reference_encoding is not None:
        found, score, num_faces = matcher.find_target_person_in_photo(test_photo)

        print(f"\nChecking: {Path(test_photo).name}")
        print(f"Faces detected: {num_faces}")
        print(f"Target person found: {found}")
        print(f"Match confidence: {score:.1%}")
