"""
Month-Based Category-Aware Photo Selection

Selects ~40 best photos per month ensuring:
1. Temporal distribution (by month)
2. Category diversity (auto-detected via CLIP)
3. Quality scoring with tiebreakers
4. No duplicates
"""

import os
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from PIL import Image
from PIL.ExifTags import TAGS, IFD
import torch
import hdbscan

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# CLIP for category detection
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available for category detection")


# Simplified category prompts for CLIP classification
# Fewer, broader categories = higher accuracy
CATEGORY_PROMPTS = [
    ("outdoor", "children playing outside in nature, at a park, playground, beach, or garden"),
    ("indoor", "children inside a house, room, or building with furniture and walls visible"),
    ("portrait", "a close-up photo focusing on a child's face, head and shoulders"),
    ("group", "multiple people together in a group photo, family gathering, or party"),
    ("activity", "children actively doing something: sports, arts, games, dancing, or playing"),
    ("event", "a special occasion with decorations: birthday party, holiday, graduation, performance"),
]

# Minimum confidence threshold for category assignment
MIN_CATEGORY_CONFIDENCE = 0.20  # Lower threshold for broader categories

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}


class CategoryDetector:
    """Detect photo categories using CLIP."""

    def __init__(self, device: str = None):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP required. Install: pip install git+https://github.com/openai/CLIP.git")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP for category detection on {self.device}...")

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # Pre-encode category text prompts
        self.categories = [cat for cat, _ in CATEGORY_PROMPTS]
        prompts = [prompt for _, prompt in CATEGORY_PROMPTS]

        with torch.no_grad():
            text_tokens = clip.tokenize(prompts).to(self.device)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        print(f"Category detector ready with {len(self.categories)} categories")

    def detect_category(self, image_path: str) -> Tuple[str, float]:
        """
        Detect the best matching category for an image.

        Returns:
            Tuple of (category_name, confidence_score)
        """
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            with torch.no_grad():
                image_input = self.preprocess(img).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Calculate similarity with all categories
                similarities = (image_features @ self.text_features.T).squeeze(0)

                # Get best match
                best_idx = similarities.argmax().item()
                confidence = similarities[best_idx].item()

            img.close()
            return self.categories[best_idx], confidence

        except Exception as e:
            print(f"Error detecting category for {image_path}: {e}")
            return "unknown", 0.0

    def detect_categories_batch(self, image_paths: List[str],
                                 batch_size: int = 32) -> Dict[str, Tuple[str, float]]:
        """
        Detect categories for multiple images efficiently.

        Returns:
            Dict mapping filename to (category, confidence)
        """
        results = {}

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []

            for path in batch_paths:
                try:
                    img = Image.open(path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    batch_images.append(self.preprocess(img))
                    valid_paths.append(path)
                    img.close()
                except Exception as e:
                    results[Path(path).name] = ("unknown", 0.0)

            if batch_images:
                with torch.no_grad():
                    image_input = torch.stack(batch_images).to(self.device)
                    image_features = self.model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    similarities = image_features @ self.text_features.T

                    for j, path in enumerate(valid_paths):
                        best_idx = similarities[j].argmax().item()
                        confidence = similarities[j][best_idx].item()
                        # Only assign category if confidence is above threshold
                        if confidence >= MIN_CATEGORY_CONFIDENCE:
                            results[Path(path).name] = (self.categories[best_idx], confidence)
                        else:
                            results[Path(path).name] = ("candid", confidence)

            if (i + batch_size) % 100 == 0:
                print(f"  Categorized {min(i + batch_size, len(image_paths))}/{len(image_paths)} photos")

        return results


class MonthlyPhotoSelector:
    """Select best photos organized by month with visual diversity."""

    def __init__(self,
                 target_per_month: int = 40,
                 duplicate_threshold: float = 0.85,
                 diversity_threshold: float = 0.75,
                 device: str = None):
        """
        Initialize selector.

        Args:
            target_per_month: Target number of photos per month
            duplicate_threshold: Similarity threshold for duplicates (0.85 catches near-dupes from same moment)
            diversity_threshold: Skip photos with similarity > this to already-selected (ensures diversity)
            device: 'cuda' or 'cpu'
        """
        self.target_per_month = target_per_month
        self.duplicate_threshold = duplicate_threshold
        self.diversity_threshold = diversity_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.category_detector = None  # Lazy loaded
        self.scorer = None  # Lazy loaded

    def _ensure_category_detector(self):
        """Lazy load category detector."""
        if self.category_detector is None:
            self.category_detector = CategoryDetector(device=self.device)

    def _ensure_scorer(self):
        """Lazy load photo scorer."""
        if self.scorer is None:
            from .scoring import PhotoScorer
            self.scorer = PhotoScorer()

    def get_photo_date(self, image_path: str) -> Optional[datetime]:
        """Extract creation date from photo EXIF."""
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
                    return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")

        except Exception as e:
            pass

        return None

    def get_month_key(self, dt: datetime) -> str:
        """Get month name from datetime (ignoring year)."""
        return MONTH_NAMES.get(dt.month, "Unknown")

    def group_photos_by_month(self, folder_path: str,
                               image_extensions: set = None) -> Dict[str, List[Dict]]:
        """
        Group all photos by month.

        Returns:
            Dict mapping month name to list of photo info dicts
        """
        if image_extensions is None:
            image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}

        folder = Path(folder_path)
        image_files = [f for f in folder.iterdir()
                       if f.suffix.lower() in image_extensions]

        print("\n")
        print("=" * 70)
        print("  TEMPORAL GROUPING - GROUPING PHOTOS BY MONTH")
        print("=" * 70)
        print(f"  Folder: {folder_path}")
        print(f"  Total images found: {len(image_files)}")
        print(f"  Supported extensions: {', '.join(image_extensions)}")
        print(f"\n  Extracting EXIF dates...")

        months = defaultdict(list)
        no_date_count = 0

        for i, image_path in enumerate(image_files):
            if (i + 1) % 100 == 0:
                print(f"    [{i+1}/{len(image_files)}] processed...")

            photo_info = {
                'filename': image_path.name,
                'filepath': str(image_path),
                'date': None,
                'month': None,
                'timestamp': None
            }

            dt = self.get_photo_date(str(image_path))

            if dt:
                photo_info['date'] = dt.isoformat()
                photo_info['month'] = self.get_month_key(dt)
                photo_info['timestamp'] = dt.timestamp()
                months[photo_info['month']].append(photo_info)
            else:
                photo_info['month'] = 'Unknown'
                months['Unknown'].append(photo_info)
                no_date_count += 1

        # Sort months in calendar order
        month_order = list(MONTH_NAMES.values()) + ['Unknown']
        sorted_months = {m: months[m] for m in month_order if m in months}

        print(f"\n  TEMPORAL DISTRIBUTION:")
        print(f"  {'-'*50}")
        print(f"  Photos with valid date: {len(image_files) - no_date_count}")
        print(f"  Photos without date (Unknown): {no_date_count}")
        print(f"\n  {'Month':<12} {'Count':>8} {'Distribution':<40}")
        print(f"  {'-'*12} {'-'*8} {'-'*40}")

        max_count = max(len(p) for p in sorted_months.values()) if sorted_months else 1
        for month, photos in sorted_months.items():
            count = len(photos)
            bar_len = int((count / max_count) * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            print(f"  {month:<12} {count:>8} {bar}")

        print(f"  {'-'*12} {'-'*8} {'-'*40}")
        print(f"  {'TOTAL':<12} {len(image_files):>8}")
        print("=" * 70)

        return sorted_months

    def detect_categories(self, photos_by_month: Dict[str, List[Dict]],
                          progress_callback=None) -> Dict[str, List[Dict]]:
        """
        Detect categories for all photos.

        Returns:
            Updated photos_by_month with category info added
        """
        self._ensure_category_detector()

        print("\n")
        print("=" * 70)
        print("  CATEGORY DETECTION - STARTING")
        print("=" * 70)

        # Collect all photo paths
        all_paths = []
        for month, photos in photos_by_month.items():
            for photo in photos:
                all_paths.append(photo['filepath'])

        print(f"  Total photos to categorize: {len(all_paths)}")
        print(f"  Available categories: {', '.join(self.category_detector.categories)}")
        print(f"\n  Processing...")

        # Batch detect categories
        categories = self.category_detector.detect_categories_batch(all_paths)

        # Add category info back to photos
        for month, photos in photos_by_month.items():
            for photo in photos:
                cat, conf = categories.get(photo['filename'], ('unknown', 0.0))
                photo['category'] = cat
                photo['category_confidence'] = conf

        # Print overall category distribution
        print("\n  OVERALL CATEGORY DISTRIBUTION:")
        print(f"  {'-'*50}")
        cat_counts = defaultdict(int)
        cat_confidence = defaultdict(list)
        for month, photos in photos_by_month.items():
            for photo in photos:
                cat_counts[photo['category']] += 1
                cat_confidence[photo['category']].append(photo['category_confidence'])

        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            avg_conf = sum(cat_confidence[cat]) / len(cat_confidence[cat]) if cat_confidence[cat] else 0
            pct = count / len(all_paths) * 100
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f"  {cat:15}: {count:5} ({pct:5.1f}%) | Avg conf: {avg_conf:.2f}")

        # Print category distribution per month
        print(f"\n  CATEGORY DISTRIBUTION BY MONTH:")
        print(f"  {'-'*50}")
        for month, photos in photos_by_month.items():
            month_cats = defaultdict(int)
            for photo in photos:
                month_cats[photo['category']] += 1
            top_cats = sorted(month_cats.items(), key=lambda x: -x[1])[:4]
            cat_str = ", ".join([f"{c}:{n}" for c, n in top_cats])
            print(f"  {month:12}: {len(photos):4} photos | {cat_str}")

        print("=" * 70)
        return photos_by_month

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1_norm, emb2_norm))

    def remove_duplicates(self, photos: List[Dict],
                          embeddings: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Remove near-duplicate photos.

        Returns:
            List of unique photos
        """
        if len(photos) <= 1:
            return photos

        unique = []

        for photo in photos:
            emb = embeddings.get(photo['filename'])
            if emb is None:
                unique.append(photo)
                continue

            is_duplicate = False
            for existing in unique:
                exist_emb = embeddings.get(existing['filename'])
                if exist_emb is not None:
                    sim = self.compute_similarity(emb, exist_emb)
                    if sim > self.duplicate_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(photo)

        return unique

    def score_photos(self, photos: List[Dict],
                     embeddings: Dict[str, np.ndarray],
                     all_month_embeddings: np.ndarray = None) -> List[Dict]:
        """
        Score all photos with quality heuristics.
        OPTIMIZED: Uses cached face data from photos dict if available.

        Returns:
            Photos with scores added, sorted by total score
        """
        self._ensure_scorer()

        cached_count = 0
        for photo in photos:
            emb = embeddings.get(photo['filename'])

            # Check for cached face data (from Step 2 face filtering)
            cached_bboxes = photo.get('face_bboxes')
            cached_num_faces = photo.get('num_faces')

            if cached_bboxes is not None and cached_num_faces is not None:
                cached_count += 1

            scores = self.scorer.score_photo(
                photo['filepath'],
                embedding=emb,
                cluster_embeddings=all_month_embeddings,
                cached_face_bboxes=cached_bboxes,
                cached_num_faces=cached_num_faces
            )

            photo.update(scores)

        if cached_count > 0:
            print(f"  [OPTIMIZED] Used cached face data for {cached_count}/{len(photos)} photos (skipped face detection)")

        return photos

    def select_greedy_diverse(self, photos: List[Dict],
                               embeddings: Dict[str, np.ndarray],
                               target: int,
                               month_name: str = "Unknown") -> List[Dict]:
        """
        Select photos using greedy diverse selection.

        Algorithm:
        1. Sort all photos by quality score (highest first)
        2. Pick the top photo
        3. Skip any photos too similar to already-selected ones
        4. Repeat until target is reached

        This ensures we get the BEST photos while guaranteeing DIVERSITY.

        Args:
            photos: Scored photos to select from
            embeddings: Photo embeddings for similarity
            target: Number of photos to select
            month_name: For logging

        Returns:
            List of selected photos
        """
        if len(photos) == 0:
            return []

        if len(photos) <= target:
            return photos

        # Sort by score (highest first)
        sorted_photos = sorted(photos, key=lambda x: x.get('total', 0), reverse=True)

        selected = []
        skipped_similar = 0

        print(f"\n  [Greedy Selection] Selecting {target} diverse photos...")
        print(f"  Diversity threshold: {self.diversity_threshold} (skip if similarity > this)")

        for photo in sorted_photos:
            if len(selected) >= target:
                break

            emb = embeddings.get(photo['filename'])
            if emb is None:
                # No embedding, just add it
                selected.append(photo)
                continue

            # Check similarity to all already-selected photos
            too_similar = False
            for sel_photo in selected:
                sel_emb = embeddings.get(sel_photo['filename'])
                if sel_emb is not None:
                    sim = self.compute_similarity(emb, sel_emb)
                    if sim > self.diversity_threshold:
                        too_similar = True
                        skipped_similar += 1
                        break

            if not too_similar:
                selected.append(photo)

        print(f"  Selected: {len(selected)}, Skipped (too similar): {skipped_similar}")

        # If we didn't reach target (all remaining were too similar),
        # relax threshold and try again with remaining photos
        if len(selected) < target:
            remaining_target = target - len(selected)
            remaining_photos = [p for p in sorted_photos if p not in selected]

            if remaining_photos:
                print(f"  [Relaxed pass] Need {remaining_target} more, relaxing threshold to 0.90...")
                relaxed_threshold = 0.90

                for photo in remaining_photos:
                    if len(selected) >= target:
                        break

                    emb = embeddings.get(photo['filename'])
                    if emb is None:
                        selected.append(photo)
                        continue

                    too_similar = False
                    for sel_photo in selected:
                        sel_emb = embeddings.get(sel_photo['filename'])
                        if sel_emb is not None:
                            sim = self.compute_similarity(emb, sel_emb)
                            if sim > relaxed_threshold:
                                too_similar = True
                                break

                    if not too_similar:
                        selected.append(photo)

                print(f"  After relaxed pass: {len(selected)} selected")

        return selected

    def calculate_cluster_score(self, cluster_photos: List[Dict]) -> float:
        """
        Calculate priority score for a cluster.

        Formula: 0.50 × best_photo_score + 0.30 × log(cluster_size) + 0.20 × mean_photo_score

        This favors:
        - High-quality moments (best photo)
        - Real events (larger clusters = more photos taken)
        - Consistent quality (mean score)
        """
        if not cluster_photos:
            return 0.0

        scores = [p.get('total', 0) for p in cluster_photos]
        best_score = max(scores)
        mean_score = sum(scores) / len(scores)
        size_score = math.log(len(cluster_photos) + 1)  # +1 to avoid log(1)=0

        # Normalize size_score (log(100) ≈ 4.6, so divide by 5 to get 0-1 range)
        size_score_normalized = min(size_score / 5.0, 1.0)

        cluster_score = (
            0.50 * best_score +
            0.30 * size_score_normalized +
            0.20 * mean_score
        )

        return cluster_score

    def is_same_event(self, photo1: Dict, photo2: Dict, time_window_minutes: int = 30) -> bool:
        """
        Check if two photos are from the same event based on timestamp proximity.

        Photos taken within `time_window_minutes` of each other are considered
        from the same event.

        Args:
            photo1, photo2: Photo dicts with optional 'timestamp' key (unix timestamp)
            time_window_minutes: Minutes threshold for same-event detection (default 30)

        Returns:
            True if photos are from the same event, False otherwise
        """
        ts1 = photo1.get('timestamp')
        ts2 = photo2.get('timestamp')

        # If either photo has no timestamp, can't determine
        if ts1 is None or ts2 is None:
            return False

        time_diff_seconds = abs(ts1 - ts2)
        time_diff_minutes = time_diff_seconds / 60.0

        return time_diff_minutes <= time_window_minutes

    def count_same_event_photos(self, candidate: Dict, selected: List[Dict],
                                 time_window_minutes: int = 30) -> int:
        """
        Count how many already-selected photos are from the same event as candidate.

        Args:
            candidate: Photo to check
            selected: List of already selected photos
            time_window_minutes: Minutes threshold for same-event detection

        Returns:
            Count of selected photos from the same event
        """
        count = 0
        for sel in selected:
            if self.is_same_event(candidate, sel, time_window_minutes):
                count += 1
        return count

    def assign_event_ids(self, photos: List[Dict], time_window_minutes: int = 30) -> List[Dict]:
        """
        Assign event_id to photos based on timestamp proximity.

        Photos taken within `time_window_minutes` of each other are grouped
        as the same event. This helps identify burst shots and same-moment photos.

        Args:
            photos: List of photo dicts with optional 'timestamp' key
            time_window_minutes: Minutes threshold for same-event grouping (default 30)

        Returns:
            Photos with 'event_id' added
        """
        if not photos:
            return photos

        # Separate photos with and without timestamps
        photos_with_ts = [(i, p) for i, p in enumerate(photos) if p.get('timestamp') is not None]
        photos_without_ts = [(i, p) for i, p in enumerate(photos) if p.get('timestamp') is None]

        # Sort by timestamp
        photos_with_ts.sort(key=lambda x: x[1]['timestamp'])

        current_event_id = 0
        last_timestamp = None

        # Assign event_ids to photos with timestamps
        for idx, photo in photos_with_ts:
            ts = photo['timestamp']

            if last_timestamp is None:
                # First photo starts event 0
                photos[idx]['event_id'] = current_event_id
            else:
                time_diff_minutes = (ts - last_timestamp) / 60.0
                if time_diff_minutes > time_window_minutes:
                    # New event
                    current_event_id += 1
                photos[idx]['event_id'] = current_event_id

            last_timestamp = ts

        # Assign unique event_ids to photos without timestamps (each is its own event)
        for idx, photo in photos_without_ts:
            current_event_id += 1
            photos[idx]['event_id'] = current_event_id

        return photos

    def cluster_photos_hdbscan(self, photos: List[Dict],
                               embeddings: Dict[str, np.ndarray]) -> Dict[int, List[Dict]]:
        """
        Cluster photos using HDBSCAN to find natural groupings (events/moments).

        Args:
            photos: List of photo dicts (must have 'filename' key)
            embeddings: Dict mapping filename to CLIP embedding

        Returns:
            Dict mapping cluster_id to list of photos in that cluster
        """
        # Build embedding matrix
        photo_embeddings = []
        valid_photos = []

        for photo in photos:
            emb = embeddings.get(photo['filename'])
            if emb is not None:
                photo_embeddings.append(emb)
                valid_photos.append(photo)

        if len(valid_photos) < 3:
            # Too few photos, treat each as its own cluster
            return {i: [p] for i, p in enumerate(valid_photos)}

        embedding_matrix = np.array(photo_embeddings)

        # Normalize embeddings for cosine-like behavior with euclidean metric
        # L2 normalization: euclidean distance on normalized vectors ≈ cosine distance
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embedding_matrix = embedding_matrix / norms

        # Adaptive min_cluster_size: 2% of photos, minimum 2, maximum 8
        # Lower values = more clusters, less fallback
        n = len(valid_photos)
        min_cluster_size = max(2, min(int(0.02 * n), 8))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,  # Relaxed: allows sparser clusters
            metric='euclidean',  # Euclidean on normalized vectors ≈ cosine distance
            cluster_selection_method='eom'  # Excess of Mass for stable clusters
        )

        labels = clusterer.fit_predict(embedding_matrix)

        # Group photos by cluster
        clusters = {}
        noise_cluster_id = max(labels) + 1 if len(labels) > 0 else 0

        for i, (photo, label) in enumerate(zip(valid_photos, labels)):
            if label == -1:
                # Noise points become individual clusters
                clusters[noise_cluster_id] = [photo]
                noise_cluster_id += 1
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(photo)

        print(f"  HDBSCAN found {len(clusters)} clusters from {len(valid_photos)} photos")

        return clusters

    def get_max_picks_for_cluster(self, cluster_size: int) -> int:
        """
        Determine maximum photos to pick from a cluster based on its size.

        Very generous limits - diversity check (0.92) prevents only true duplicates.
        This allows different scenes with same people to be selected.

        Cluster Size → Max Photos
        1-2         → 2
        3-5         → 4
        6-10        → 8
        11-20       → 12
        21-40       → 16
        >40         → 20
        """
        if cluster_size <= 2:
            return 2
        elif cluster_size <= 5:
            return 4
        elif cluster_size <= 10:
            return 8
        elif cluster_size <= 20:
            return 12
        elif cluster_size <= 40:
            return 16
        else:
            return 20

    def select_hybrid_hdbscan(self, photos: List[Dict],
                              embeddings: Dict[str, np.ndarray],
                              target: int,
                              month_name: str = "Unknown") -> List[Dict]:
        """
        Select photos using quality-first approach with diversity constraints.

        Algorithm:
        1. Assign event_ids based on timestamps (30-min window)
        2. Cluster with HDBSCAN for visual similarity info
        3. Combine ALL photos and sort by score
        4. Select by score with:
           - Event diversity: one photo per event_id
           - Visual similarity check: skip if >80% similar to selected

        This ensures high-quality photos are selected regardless of cluster status
        (a 74% "noise" photo beats a 73% clustered photo).

        Args:
            photos: Scored photos to select from (should have 'category' key)
            embeddings: Photo embeddings for clustering
            target: Number of photos to select
            month_name: For logging

        Returns:
            List of selected photos
        """
        if len(photos) == 0:
            return []

        if len(photos) <= target:
            return photos

        print(f"\n  [Quality-First Selection] Target: {target} photos")

        # Step 0: Assign event_ids based on timestamps
        photos = self.assign_event_ids(photos, time_window_minutes=30)

        # Step 1: Cluster photos with HDBSCAN (for cluster_id info in results)
        clusters = self.cluster_photos_hdbscan(photos, embeddings)
        print(f"  Found {len(clusters)} clusters")

        # Step 2: Flatten all photos with cluster_id and sort by score
        all_photos = []
        for cid, cluster_photos in clusters.items():
            for photo in cluster_photos:
                photo['cluster_id'] = int(cid)
                all_photos.append(photo)

        all_photos.sort(key=lambda x: x.get('total', 0), reverse=True)

        # Step 3: Select by quality with diversity constraints
        selected = []
        selected_files = set()
        used_event_ids = set()
        used_cluster_ids = set()

        cross_cluster_threshold = 0.80  # Skip if >80% similar (only within same cluster)

        def get_max_similarity(candidate, selected_list, same_cluster_only=True):
            """
            Check if candidate is too similar to already selected.

            Args:
                candidate: Photo to check
                selected_list: Already selected photos
                same_cluster_only: If True, only check similarity against photos
                                   in the same cluster (default). This prevents
                                   rejecting different moments with same people.
            """
            candidate_emb = embeddings.get(candidate['filename'])
            candidate_cluster = candidate.get('cluster_id', -1)
            max_sim = 0.0
            max_sim_photo = None

            if candidate_emb is not None:
                for sel in selected_list:
                    # If same_cluster_only, skip photos from different clusters
                    if same_cluster_only:
                        sel_cluster = sel.get('cluster_id', -1)
                        if candidate_cluster != sel_cluster:
                            continue  # Different cluster = different moment, don't compare

                    sel_emb = embeddings.get(sel['filename'])
                    if sel_emb is not None:
                        sim = self.compute_similarity(candidate_emb, sel_emb)
                        if sim > max_sim:
                            max_sim = sim
                            max_sim_photo = sel['filename']
                        if sim > cross_cluster_threshold:
                            return True, max_sim, max_sim_photo

            return False, max_sim, max_sim_photo

        print(f"\n  Selecting by quality score with event + cluster diversity...")
        skipped_event = 0
        skipped_cluster = 0
        skipped_similar = 0

        for photo in all_photos:
            if len(selected) >= target:
                break

            event_id = photo.get('event_id', -1)
            cluster_id = photo.get('cluster_id', -1)

            # Diversity constraint 1: Skip if already picked from this event
            if event_id != -1 and event_id in used_event_ids:
                skipped_event += 1
                continue

            # Diversity constraint 2: Skip if already picked from this cluster
            if cluster_id != -1 and cluster_id in used_cluster_ids:
                skipped_cluster += 1
                continue

            # Diversity constraint 3: Skip if too similar to already selected (same cluster only)
            should_skip, max_sim, _ = get_max_similarity(photo, selected, same_cluster_only=True)
            if should_skip:
                skipped_similar += 1
                continue

            # Select this photo
            photo['max_similarity'] = round(max_sim, 4)
            photo['selection_reason'] = "Quality"
            selected.append(photo)
            selected_files.add(photo['filename'])
            if event_id != -1:
                used_event_ids.add(event_id)
            if cluster_id != -1:
                used_cluster_ids.add(cluster_id)

        print(f"  Selected: {len(selected)}")
        print(f"  Skipped (same event): {skipped_event}")
        print(f"  Skipped (same cluster): {skipped_cluster}")
        print(f"  Skipped (too similar): {skipped_similar}")

        # If we still need more photos, relax event constraint in two sub-passes
        if len(selected) < target:
            remaining = target - len(selected)
            print(f"\n  [Relaxed pass] Need {remaining} more...")

            # Sub-pass 1: Pick from new event OR new cluster
            print(f"  Sub-pass 1: Prioritizing new events OR new clusters...")
            for photo in all_photos:
                if len(selected) >= target:
                    break

                if photo['filename'] in selected_files:
                    continue

                event_id = photo.get('event_id', -1)
                cluster_id = photo.get('cluster_id', -1)

                # Check if this is a new event or new cluster
                is_new_event = event_id == -1 or event_id not in used_event_ids
                is_new_cluster = cluster_id == -1 or cluster_id not in used_cluster_ids

                # Accept if: new event OR new cluster
                if not is_new_event and not is_new_cluster:
                    continue

                should_skip, max_sim, _ = get_max_similarity(photo, selected, same_cluster_only=True)
                if should_skip:
                    continue

                photo['max_similarity'] = round(max_sim, 4)
                photo['selection_reason'] = "Quality (new event/cluster)"
                selected.append(photo)
                selected_files.add(photo['filename'])
                if event_id != -1:
                    used_event_ids.add(event_id)
                if cluster_id != -1:
                    used_cluster_ids.add(cluster_id)

            print(f"  After sub-pass 1: {len(selected)} photos")

            # Sub-pass 2: Allow 2nd photo per event AND per cluster
            if len(selected) < target:
                print(f"  Sub-pass 2: Allowing 2nd photo per event/cluster...")

                # Count photos per event AND per cluster
                event_counts = {}
                cluster_counts = {}
                for sel in selected:
                    eid = sel.get('event_id', -1)
                    cid = sel.get('cluster_id', -1)
                    if eid != -1:
                        event_counts[eid] = event_counts.get(eid, 0) + 1
                    if cid != -1:
                        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

                max_per_event = 2
                max_per_cluster = 2

                for photo in all_photos:
                    if len(selected) >= target:
                        break

                    if photo['filename'] in selected_files:
                        continue

                    event_id = photo.get('event_id', -1)
                    cluster_id = photo.get('cluster_id', -1)

                    # Check both limits
                    if event_id != -1 and event_counts.get(event_id, 0) >= max_per_event:
                        continue
                    if cluster_id != -1 and cluster_counts.get(cluster_id, 0) >= max_per_cluster:
                        continue

                    should_skip, max_sim, _ = get_max_similarity(photo, selected, same_cluster_only=True)
                    if should_skip:
                        continue

                    photo['max_similarity'] = round(max_sim, 4)
                    photo['selection_reason'] = "Quality (2nd from event/cluster)"
                    selected.append(photo)
                    selected_files.add(photo['filename'])
                    if event_id != -1:
                        event_counts[event_id] = event_counts.get(event_id, 0) + 1
                    if cluster_id != -1:
                        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

                print(f"  After sub-pass 2: {len(selected)} photos")

            # Sub-pass 3: Final fallback - just pick by quality (no event/cluster limits)
            if len(selected) < target:
                print(f"  Sub-pass 3: Final quality fallback...")
                for photo in all_photos:
                    if len(selected) >= target:
                        break

                    if photo['filename'] in selected_files:
                        continue

                    # In fallback, still check similarity within same cluster
                    should_skip, max_sim, _ = get_max_similarity(photo, selected, same_cluster_only=True)
                    if should_skip:
                        continue

                    photo['max_similarity'] = round(max_sim, 4)
                    photo['selection_reason'] = "Quality (fallback)"
                    selected.append(photo)
                    selected_files.add(photo['filename'])

                print(f"  After sub-pass 3: {len(selected)} photos")

        # Log category distribution
        final_cats = defaultdict(int)
        for p in selected:
            final_cats[p.get('category', 'unknown')] += 1

        print(f"\n  Final selection: {len(selected)} photos")
        print(f"  Category distribution: {dict(final_cats)}")

        # Build lookup maps for debugging: which photo used which event/cluster
        event_to_photo = {}
        cluster_to_photo = {}
        for sel in selected:
            eid = sel.get('event_id', -1)
            cid = sel.get('cluster_id', -1)
            if eid != -1 and eid not in event_to_photo:
                event_to_photo[eid] = sel['filename']
            if cid != -1 and cid not in cluster_to_photo:
                cluster_to_photo[cid] = sel['filename']

        # Mark rejection reasons for unselected photos (for debugging)
        print(f"\n  Marking rejection reasons for unselected photos...")
        for photo in all_photos:
            if photo['filename'] in selected_files:
                continue  # Already selected

            event_id = photo.get('event_id', -1)
            cluster_id = photo.get('cluster_id', -1)

            # Determine why this photo wasn't selected
            if event_id != -1 and event_id in used_event_ids:
                blocking_photo = event_to_photo.get(event_id, 'unknown')
                photo['rejection_reason'] = f"Event {event_id} used by {blocking_photo}"
            elif cluster_id != -1 and cluster_id in used_cluster_ids:
                blocking_photo = cluster_to_photo.get(cluster_id, 'unknown')
                photo['rejection_reason'] = f"Cluster {cluster_id} used by {blocking_photo}"
            else:
                # Check similarity to selected photos (same cluster only)
                should_skip, max_sim, most_similar_photo = get_max_similarity(photo, selected, same_cluster_only=True)
                photo['max_similarity'] = round(max_sim, 4)
                if should_skip:
                    photo['rejection_reason'] = f"Too similar ({max_sim*100:.1f}%) to {most_similar_photo or 'unknown'}"
                else:
                    # Would have been selected but target was already reached
                    photo['rejection_reason'] = "Target reached"

        return selected

    def select_from_month(self, photos: List[Dict],
                          embeddings: Dict[str, np.ndarray],
                          target: int = None,
                          month_name: str = "Unknown") -> Tuple[List[Dict], Dict]:
        """
        Select best photos from a single month using hybrid HDBSCAN + quality selection.
        Categories are detected for display purposes but selection uses visual diversity.

        Returns:
            Tuple of (selected_photos, category_breakdown)
        """
        target = target or self.target_per_month

        print(f"\n{'='*60}")
        print(f"  PROCESSING MONTH: {month_name}")
        print(f"{'='*60}")
        print(f"  Input photos: {len(photos)}")
        print(f"  Target selection: {target}")

        if len(photos) == 0:
            print(f"  [!] No photos in this month, skipping...")
            return [], {}

        # Step 1: Remove exact/near duplicates first
        print(f"\n  [Step 1] Removing duplicates (threshold: {self.duplicate_threshold})...")
        unique_photos = self.remove_duplicates(photos, embeddings)
        duplicates_removed = len(photos) - len(unique_photos)
        print(f"  Duplicates removed: {duplicates_removed}")
        print(f"  Unique photos: {len(unique_photos)}")

        if len(unique_photos) <= target:
            print(f"  [!] Fewer photos than target, taking all {len(unique_photos)}")
            return unique_photos, self._get_category_breakdown(unique_photos)

        # Step 2: Get embeddings array for uniqueness scoring
        print(f"\n  [Step 2] Preparing embeddings for scoring...")
        month_embeddings = []
        for photo in unique_photos:
            emb = embeddings.get(photo['filename'])
            if emb is not None:
                month_embeddings.append(emb)
        month_embeddings = np.array(month_embeddings) if month_embeddings else None
        print(f"  Embeddings ready: {len(month_embeddings) if month_embeddings is not None else 0}")

        # Step 3: Score all photos
        print(f"\n  [Step 3] Scoring photos (face quality, aesthetic, emotional, uniqueness)...")
        scored_photos = self.score_photos(unique_photos, embeddings, month_embeddings)

        # Show top 10 scores
        sorted_by_score = sorted(scored_photos, key=lambda x: x.get('total', 0), reverse=True)
        print(f"  Top 10 by score:")
        for i, p in enumerate(sorted_by_score[:10]):
            print(f"    {i+1:2}. {p['filename'][:30]:30} | Score: {p.get('total', 0):.3f} | Face: {p.get('face_quality', 0):.2f} | Cat: {p.get('category', '?')}")

        # Step 4: Show category distribution (for info only, not used in selection)
        print(f"\n  [Step 4] Category distribution (display only):")
        by_category = defaultdict(int)
        for photo in scored_photos:
            by_category[photo.get('category', 'unknown')] += 1
        for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
            print(f"    - {cat:15}: {count:4} photos")

        # Step 5: Hybrid HDBSCAN + quality selection (replaces greedy-diverse)
        print(f"\n  [Step 5] Hybrid HDBSCAN + quality selection...")
        selected = self.select_hybrid_hdbscan(scored_photos, embeddings, target, month_name)

        # Final summary for this month
        final_breakdown = self._get_category_breakdown(selected)
        print(f"\n  [RESULT] {month_name}: Selected {len(selected)} photos")
        print(f"  Final category distribution:")
        for cat, count in sorted(final_breakdown.items(), key=lambda x: -x[1]):
            print(f"    - {cat:15}: {count:3} photos")

        # Show selected photo summary
        print(f"\n  Selected photos (by score):")
        for i, p in enumerate(sorted(selected, key=lambda x: x.get('total', 0), reverse=True)[:5]):
            print(f"    {i+1}. {p['filename'][:35]:35} | Score: {p.get('total', 0):.3f} | Cat: {p.get('category', '?')}")
        if len(selected) > 5:
            print(f"    ... and {len(selected) - 5} more")

        return selected, final_breakdown

    def _sort_with_tiebreakers(self, photos: List[Dict],
                                embeddings: Dict[str, np.ndarray]) -> List[Dict]:
        """Sort photos by score with tiebreakers."""

        def sort_key(photo):
            # Primary: total score (higher better)
            total = photo.get('total', 0)

            # Tiebreaker 1: uniqueness
            uniqueness = photo.get('uniqueness', 0.5)

            # Tiebreaker 2: timestamp for spread (use hour of day)
            timestamp = photo.get('timestamp', 0)
            time_spread = (timestamp % 86400) / 86400 if timestamp else 0.5

            # Combine with decreasing weights
            return (total, uniqueness, time_spread)

        return sorted(photos, key=sort_key, reverse=True)

    def _is_duplicate_of_selected(self, photo: Dict,
                                   selected: List[Dict],
                                   embeddings: Dict[str, np.ndarray]) -> bool:
        """Check if photo is duplicate of any selected photo."""
        emb = embeddings.get(photo['filename'])
        if emb is None:
            return False

        for sel in selected:
            sel_emb = embeddings.get(sel['filename'])
            if sel_emb is not None:
                sim = self.compute_similarity(emb, sel_emb)
                if sim > self.duplicate_threshold:
                    return True

        return False

    def _get_category_breakdown(self, photos: List[Dict]) -> Dict[str, int]:
        """Get count of photos per category."""
        breakdown = defaultdict(int)
        for photo in photos:
            breakdown[photo.get('category', 'unknown')] += 1
        return dict(breakdown)

    def select_all_months(self, photos_by_month: Dict[str, List[Dict]],
                          embeddings: Dict[str, np.ndarray],
                          progress_callback=None) -> Dict:
        """
        Select best photos from all months.

        Returns:
            Dict with selected photos and summary statistics
        """
        print("\n")
        print("=" * 70)
        print("  MONTHLY PHOTO SELECTION - STARTING")
        print("=" * 70)
        print(f"  Target per month: {self.target_per_month}")
        print(f"  Duplicate threshold: {self.duplicate_threshold}")
        print(f"  Total months to process: {len(photos_by_month)}")

        all_selected = []
        month_stats = []

        month_order = list(MONTH_NAMES.values()) + ['Unknown']

        for month in month_order:
            if month not in photos_by_month:
                continue

            photos = photos_by_month[month]

            if progress_callback:
                progress_callback(f"Selecting from {month}...")

            selected, category_breakdown = self.select_from_month(
                photos, embeddings, self.target_per_month, month_name=month
            )

            all_selected.extend(selected)

            # Build category string for display
            cat_str = ", ".join([f"{cat}:{cnt}" for cat, cnt in
                                sorted(category_breakdown.items(), key=lambda x: -x[1])[:3]])

            stat = {
                'month': month,
                'total_photos': len(photos),
                'selected': len(selected),
                'categories': category_breakdown,
                'category_summary': cat_str
            }
            month_stats.append(stat)

        # Summary
        total_input = sum(s['total_photos'] for s in month_stats)
        total_selected = len(all_selected)

        print("\n")
        print("=" * 70)
        print("  FINAL SUMMARY")
        print("=" * 70)
        print(f"\n  {'Month':<12} {'Input':>8} {'Selected':>10} {'Categories':<40}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*40}")
        for stat in month_stats:
            print(f"  {stat['month']:<12} {stat['total_photos']:>8} {stat['selected']:>10} {stat['category_summary']:<40}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*40}")
        print(f"  {'TOTAL':<12} {total_input:>8} {total_selected:>10}")
        print(f"\n  Selection rate: {total_selected/total_input*100:.1f}%" if total_input > 0 else "")
        print("=" * 70)

        return {
            'selected': all_selected,
            'month_stats': month_stats,
            'summary': {
                'total_photos': total_input,
                'total_selected': total_selected,
                'selection_rate': round(total_selected / total_input, 3) if total_input > 0 else 0
            }
        }


def run_monthly_selection(folder_path: str,
                          embeddings: Dict[str, np.ndarray],
                          target_per_month: int = 40,
                          duplicate_threshold: float = 0.85,
                          diversity_threshold: float = 0.75,
                          progress_callback=None) -> Dict:
    """
    Main entry point for monthly photo selection.

    Uses greedy diverse selection: picks highest-scoring photos while
    ensuring visual diversity (skips photos too similar to already-selected).

    Args:
        folder_path: Path to folder with photos
        embeddings: Pre-computed CLIP embeddings
        target_per_month: Target photos per month
        duplicate_threshold: Similarity threshold for exact duplicates (default 0.85)
        diversity_threshold: Skip photos with similarity > this to selected (default 0.80)
        progress_callback: Optional callback for progress updates

    Returns:
        Dict with selected photos
    """
    selector = MonthlyPhotoSelector(
        target_per_month=target_per_month,
        duplicate_threshold=duplicate_threshold,
        diversity_threshold=diversity_threshold
    )

    # Step 1: Group by month
    if progress_callback:
        progress_callback("Grouping photos by month...")
    photos_by_month = selector.group_photos_by_month(folder_path)

    # Step 2: Detect categories
    if progress_callback:
        progress_callback("Detecting photo categories...")
    photos_by_month = selector.detect_categories(photos_by_month)

    # Step 3: Select from each month
    if progress_callback:
        progress_callback("Selecting best photos...")
    results = selector.select_all_months(photos_by_month, embeddings, progress_callback)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = r"C:\Users\tanis\Downloads\test_photos"

    # Load embeddings if available
    embeddings = {}
    emb_file = os.path.join(os.path.dirname(__file__), "photo_embeddings.npz")
    if os.path.exists(emb_file):
        data = np.load(emb_file)
        embeddings = {fn: emb for fn, emb in zip(data['filenames'], data['embeddings'])}

    results = run_monthly_selection(folder, embeddings, target_per_month=40)

    print("\n=== Month Distribution Table ===")
    print(f"{'Month':<10} {'Total':>8} {'Selected':>10} {'Categories':<40}")
    print("-" * 70)
    for stat in results['month_stats']:
        print(f"{stat['month']:<10} {stat['total_photos']:>8} {stat['selected']:>10} {stat['category_summary']:<40}")
    print("-" * 70)
    print(f"{'TOTAL':<10} {results['summary']['total_photos']:>8} {results['summary']['total_selected']:>10}")
