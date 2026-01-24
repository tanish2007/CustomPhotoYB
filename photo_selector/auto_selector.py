"""
Automatic Photo Selection
Decides which photos to include based on quality, not a fixed target number.

Selection Criteria:
1. Quality Threshold - Only keep photos above a minimum quality score
2. Diversity Requirement - Ensure variety across time and content
3. Duplicate Removal - Remove near-identical photos
4. Per-Cluster Selection - Keep best photos from each activity/scene cluster
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SelectionReason(Enum):
    """Reasons for selecting or rejecting a photo."""
    SELECTED_HIGH_QUALITY = "High quality photo"
    SELECTED_BEST_IN_CLUSTER = "Best photo in its group"
    SELECTED_UNIQUE_MOMENT = "Unique moment"
    SELECTED_FOR_VARIETY = "Selected for variety"
    SELECTED_TIME_COVERAGE = "Represents this time period"
    REJECTED_LOW_QUALITY = "Quality below threshold"
    REJECTED_BLURRY = "Too blurry"
    REJECTED_BAD_FACES = "Poor face quality"
    REJECTED_DUPLICATE = "Too similar to another photo"
    REJECTED_REDUNDANT = "Better version exists in same group"
    REJECTED_POOR_LIGHTING = "Poor lighting/exposure"


@dataclass
class PhotoDecision:
    """Decision about a single photo."""
    filename: str
    selected: bool
    reason: SelectionReason
    score: float
    details: Dict


class AutomaticSelector:
    """
    Automatically decide which photos to keep based on quality metrics.

    No fixed target - keeps all photos that meet quality criteria.
    Ensures variety across time periods and content types.
    """

    def __init__(self,
                 quality_threshold: float = 0.5,
                 blur_threshold: float = 0.3,
                 face_quality_threshold: float = 0.25,
                 similarity_threshold: float = 0.92,
                 min_photos_per_bucket: int = 1,
                 min_photos_per_cluster: int = 1,
                 max_similar_in_cluster: int = 3,
                 diversity_bonus: float = 0.1):
        """
        Initialize automatic selector.

        Args:
            quality_threshold: Minimum overall score (0-1) to keep a photo
            blur_threshold: Minimum sharpness score (0-1)
            face_quality_threshold: Minimum face quality if faces detected
            similarity_threshold: Max similarity before considered duplicate
            min_photos_per_bucket: Ensure at least N photos per time bucket
            min_photos_per_cluster: Ensure at least N photos per cluster (for variety)
            max_similar_in_cluster: Max photos to keep from very similar group
            diversity_bonus: Score bonus for unique content (0-1)
        """
        self.quality_threshold = quality_threshold
        self.blur_threshold = blur_threshold
        self.face_quality_threshold = face_quality_threshold
        self.similarity_threshold = similarity_threshold
        self.min_photos_per_bucket = min_photos_per_bucket
        self.min_photos_per_cluster = min_photos_per_cluster
        self.max_similar_in_cluster = max_similar_in_cluster
        self.diversity_bonus = diversity_bonus

    def generate_selection_description(self, score_data: Dict, cluster_rank: int = 1,
                                       cluster_size: int = 1) -> str:
        """
        Generate a human-readable description of why this photo was selected.

        Args:
            score_data: Photo scoring data
            cluster_rank: Rank within cluster (1 = best)
            cluster_size: Total photos in cluster

        Returns:
            Descriptive explanation
        """
        total_score = score_data.get('total', 0)
        face_quality = score_data.get('face_quality', 0)
        aesthetic = score_data.get('aesthetic_quality', 0)
        emotional = score_data.get('emotional_signal', 0)
        uniqueness = score_data.get('uniqueness', 0)
        num_faces = score_data.get('num_faces', 0)

        # Find the strongest quality
        qualities = {
            'face_quality': face_quality,
            'aesthetic_quality': aesthetic,
            'emotional_signal': emotional,
            'uniqueness': uniqueness
        }
        best_quality = max(qualities.items(), key=lambda x: x[1])

        # Build description
        parts = []

        # Overall ranking context
        if cluster_size > 1:
            if cluster_rank == 1:
                parts.append("Best photo from this moment")
            elif cluster_rank == 2:
                parts.append("Second-best from this moment")
            else:
                parts.append(f"Top {cluster_rank} from this moment")
        else:
            parts.append("Unique moment captured")

        # Add specific strengths
        strengths = []
        if face_quality >= 0.7:
            if num_faces > 1:
                strengths.append("everyone looks great")
            else:
                strengths.append("clear, well-composed faces")
        elif face_quality >= 0.5:
            strengths.append("good facial expressions")

        if aesthetic >= 0.7:
            strengths.append("excellent photo quality")
        elif aesthetic >= 0.5:
            strengths.append("well-lit and sharp")

        if emotional >= 0.7:
            if num_faces > 1:
                strengths.append("great interaction captured")
            else:
                strengths.append("genuine emotion")
        elif emotional >= 0.5:
            strengths.append("nice moment")

        if uniqueness >= 0.7:
            strengths.append("distinct from other photos")

        # Combine strengths
        if strengths:
            if len(strengths) == 1:
                parts.append(f"with {strengths[0]}")
            elif len(strengths) == 2:
                parts.append(f"with {strengths[0]} and {strengths[1]}")
            else:
                parts.append(f"with {', '.join(strengths[:-1])}, and {strengths[-1]}")

        # Add score as context
        score_percent = int(total_score * 100)
        description = " ".join(parts) + f" (quality: {score_percent}%)"

        return description

    def analyze_photo(self, score_data: Dict) -> Tuple[bool, SelectionReason, str]:
        """
        Analyze a single photo and decide if it should be selected.

        Returns:
            Tuple of (should_select, reason, explanation)
        """
        total_score = score_data.get('total', 0)
        face_quality = score_data.get('face_quality', 0)
        aesthetic = score_data.get('aesthetic_quality', 0)
        num_faces = score_data.get('num_faces', 0)

        # Check for blur (aesthetic includes sharpness)
        if aesthetic < self.blur_threshold:
            return False, SelectionReason.REJECTED_BLURRY, f"Sharpness too low ({aesthetic:.2f})"

        # Check face quality if faces are present
        if num_faces > 0 and face_quality < self.face_quality_threshold:
            return False, SelectionReason.REJECTED_BAD_FACES, f"Face quality too low ({face_quality:.2f})"

        # Check overall quality
        if total_score < self.quality_threshold:
            return False, SelectionReason.REJECTED_LOW_QUALITY, f"Score below threshold ({total_score:.2f} < {self.quality_threshold})"

        # Passed all checks
        return True, SelectionReason.SELECTED_HIGH_QUALITY, f"Good quality ({total_score:.2f})"

    def remove_duplicates(self,
                          candidates: List[Dict],
                          embeddings: Dict[str, np.ndarray]) -> Tuple[List[Dict], List[Dict]]:
        """
        Remove near-duplicate photos, keeping the best one.

        Returns:
            Tuple of (selected, rejected)
        """
        if not candidates:
            return [], []

        # Sort by score (best first)
        sorted_candidates = sorted(candidates, key=lambda x: x.get('total', 0), reverse=True)

        selected = []
        rejected = []
        selected_embeddings = []

        for photo in sorted_candidates:
            filename = photo['filename']
            if filename not in embeddings:
                continue

            emb = embeddings[filename]
            emb_normalized = emb / (np.linalg.norm(emb) + 1e-8)

            # Check similarity with already selected photos
            is_duplicate = False
            for sel_emb in selected_embeddings:
                similarity = float(np.dot(emb_normalized, sel_emb))
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                photo['rejection_reason'] = SelectionReason.REJECTED_DUPLICATE
                photo['rejection_detail'] = f"Similar to already selected photo (>{self.similarity_threshold:.0%})"
                rejected.append(photo)
            else:
                selected.append(photo)
                selected_embeddings.append(emb_normalized)

        return selected, rejected

    def select_from_cluster(self,
                            cluster_photos: List[Dict],
                            embeddings: Dict[str, np.ndarray],
                            ensure_variety: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Select best photos from a cluster, limiting similar photos.
        GUARANTEES at least 1 photo per cluster for variety (with lowered quality bar).
        STRICTLY checks for duplicates within cluster using embeddings.

        Returns:
            Tuple of (selected, rejected)
        """
        if not cluster_photos:
            return [], []

        # Sort by score
        sorted_photos = sorted(cluster_photos, key=lambda x: x.get('total', 0), reverse=True)

        selected = []
        rejected = []
        variety_picks = []  # Photos picked for variety even if below threshold
        selected_embeddings = []  # Track embeddings for duplicate detection
        cluster_size = len(cluster_photos)

        for rank, photo in enumerate(sorted_photos, start=1):
            filename = photo.get('filename', '')

            # First, check if photo passes quality threshold
            passes, reason, detail = self.analyze_photo(photo)

            if not passes:
                # If we need variety and don't have enough, consider lower quality photos
                if (ensure_variety and
                    len(selected) + len(variety_picks) < self.min_photos_per_cluster and
                    photo.get('total', 0) >= self.quality_threshold * 0.5):  # 50% of threshold for variety
                    photo['selection_reason'] = SelectionReason.SELECTED_UNIQUE_MOMENT
                    # Generate descriptive reason for variety pick
                    base_desc = self.generate_selection_description(photo, cluster_rank=rank, cluster_size=cluster_size)
                    photo['selection_detail'] = f"Chosen for variety - {base_desc.lower()}"
                    variety_picks.append(photo)
                else:
                    photo['rejection_reason'] = reason
                    photo['rejection_detail'] = detail
                    rejected.append(photo)
                continue

            # Check if we already have enough photos from this cluster
            if len(selected) >= self.max_similar_in_cluster:
                photo['rejection_reason'] = SelectionReason.REJECTED_REDUNDANT
                photo['rejection_detail'] = f"Already have {self.max_similar_in_cluster} photos from this group"
                rejected.append(photo)
                continue

            # STRICT duplicate check within cluster using embeddings
            if filename in embeddings and len(selected_embeddings) > 0:
                emb = embeddings[filename]
                emb_normalized = emb / (np.linalg.norm(emb) + 1e-8)

                is_too_similar = False
                for sel_emb in selected_embeddings:
                    similarity = float(np.dot(emb_normalized, sel_emb))
                    # Use a STRICTER threshold within clusters (0.85 instead of 0.92)
                    # Photos in same cluster are already similar, so we need stricter check
                    strict_threshold = min(self.similarity_threshold, 0.85)
                    if similarity > strict_threshold:
                        is_too_similar = True
                        photo['rejection_reason'] = SelectionReason.REJECTED_DUPLICATE
                        photo['rejection_detail'] = f"Too similar to another photo in this group ({similarity:.0%} match)"
                        rejected.append(photo)
                        break

                if is_too_similar:
                    continue

                # Add embedding for future comparisons
                selected_embeddings.append(emb_normalized)
            elif filename in embeddings:
                # First photo - add its embedding
                emb = embeddings[filename]
                selected_embeddings.append(emb / (np.linalg.norm(emb) + 1e-8))

            # Mark with selection reason if not already set
            if 'selection_reason' not in photo:
                photo['selection_reason'] = reason
                # Generate descriptive explanation
                photo['selection_detail'] = self.generate_selection_description(
                    photo, cluster_rank=rank, cluster_size=cluster_size
                )

            selected.append(photo)

        # Add variety picks to selected
        selected.extend(variety_picks)

        # GUARANTEE: If cluster still has no selections and ensure_variety is True,
        # pick the absolute best photo even if it's below our lowered threshold
        if ensure_variety and len(selected) == 0 and len(sorted_photos) > 0:
            best_photo = sorted_photos[0]  # Already sorted by score
            # Only skip if the photo is EXTREMELY poor (below 30% of threshold)
            if best_photo.get('total', 0) >= self.quality_threshold * 0.3:
                best_photo['selection_reason'] = SelectionReason.SELECTED_FOR_VARIETY
                # Generate descriptive reason
                base_desc = self.generate_selection_description(best_photo, cluster_rank=1, cluster_size=cluster_size)
                best_photo['selection_detail'] = f"Only photo representing this unique activity - {base_desc.lower()}"
                selected.append(best_photo)
                # Remove from rejected if it was there
                rejected = [p for p in rejected if p.get('filename') != best_photo.get('filename')]

        return selected, rejected

    def ensure_time_coverage(self,
                             bucket_selections: Dict[str, List[Dict]],
                             bucket_all_photos: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Ensure each time bucket has minimum representation.

        If a bucket has no selections, pick the best available photo.
        """
        for bucket_key, selections in bucket_selections.items():
            if len(selections) < self.min_photos_per_bucket:
                # Get all photos from this bucket
                all_photos = bucket_all_photos.get(bucket_key, [])

                # Sort by score and try to add more
                sorted_photos = sorted(all_photos, key=lambda x: x.get('total', 0), reverse=True)

                for photo in sorted_photos:
                    if photo not in selections:
                        # Lower our standards slightly for time coverage
                        if photo.get('total', 0) >= self.quality_threshold * 0.7:
                            photo['selection_reason'] = SelectionReason.SELECTED_UNIQUE_MOMENT
                            photo['selection_detail'] = f"Best available for {bucket_key}"
                            selections.append(photo)

                        if len(selections) >= self.min_photos_per_bucket:
                            break

        return bucket_selections


class SmartPhotoSelector:
    """
    Complete automatic selection pipeline.
    Combines all criteria to make intelligent decisions.
    Ensures variety across time AND content (different activities/scenes).
    """

    def __init__(self,
                 quality_mode: str = "balanced",
                 similarity_threshold: float = 0.92):
        """
        Initialize smart selector.

        Args:
            quality_mode: "strict" (fewer, higher quality),
                         "balanced" (moderate),
                         "lenient" (keep more photos)
            similarity_threshold: How different photos must be (0.8-0.99)
        """
        # Set thresholds based on mode
        # Each mode balances quality vs variety differently
        # Within-cluster similarity is checked STRICTLY at 0.85 threshold
        if quality_mode == "strict":
            self.selector = AutomaticSelector(
                quality_threshold=0.65,
                blur_threshold=0.4,
                face_quality_threshold=0.35,
                similarity_threshold=similarity_threshold,
                min_photos_per_bucket=1,      # At least 1 per month
                min_photos_per_cluster=1,     # At least 1 per activity/scene
                max_similar_in_cluster=1,     # Only 1 photo per similar group (strictest)
                diversity_bonus=0.05          # Small bonus for unique content
            )
        elif quality_mode == "lenient":
            self.selector = AutomaticSelector(
                quality_threshold=0.35,
                blur_threshold=0.2,
                face_quality_threshold=0.15,
                similarity_threshold=similarity_threshold,
                min_photos_per_bucket=2,      # At least 2 per month
                min_photos_per_cluster=1,     # At least 1 per activity/scene
                max_similar_in_cluster=3,     # Up to 3 similar photos allowed
                diversity_bonus=0.15          # Larger bonus for variety
            )
        else:  # balanced
            self.selector = AutomaticSelector(
                quality_threshold=0.50,
                blur_threshold=0.3,
                face_quality_threshold=0.25,
                similarity_threshold=similarity_threshold,
                min_photos_per_bucket=1,      # At least 1 per month
                min_photos_per_cluster=1,     # At least 1 per activity/scene
                max_similar_in_cluster=2,     # Up to 2 similar photos (reduced from 3)
                diversity_bonus=0.10          # Moderate bonus for variety
            )

        self.quality_mode = quality_mode

    def process_all_photos(self,
                           all_scores: Dict[str, Dict],
                           embeddings: Dict[str, np.ndarray],
                           cluster_results: Dict[str, Dict]) -> Dict:
        """
        Process all photos and make selection decisions.
        Ensures variety across both time (buckets) and content (clusters).

        Returns:
            Dictionary with selected, rejected, and statistics
        """
        all_selected = []
        all_rejected = []
        bucket_stats = {}
        variety_stats = {'clusters_represented': 0, 'total_clusters': 0}

        # Track selections per bucket for time coverage
        bucket_selections = {}
        bucket_all_photos = {}

        for bucket_key, bucket_data in cluster_results.items():
            filenames = bucket_data.get('filenames', [])
            labels = bucket_data.get('labels', [])

            bucket_selected = []
            bucket_rejected = []

            # Group photos by cluster (each cluster = different activity/scene)
            cluster_photos = {}
            for filename, label in zip(filenames, labels):
                if label not in cluster_photos:
                    cluster_photos[label] = []

                score_data = all_scores.get(filename, {})
                score_data['filename'] = filename
                score_data['bucket'] = bucket_key
                score_data['cluster'] = label
                cluster_photos[label].append(score_data)

            variety_stats['total_clusters'] += len(cluster_photos)

            # Track cluster selections for variety enforcement
            cluster_selections = {}

            # Process each cluster (activity/scene)
            for cluster_id, photos in cluster_photos.items():
                selected, rejected = self.selector.select_from_cluster(
                    photos, embeddings, ensure_variety=True
                )
                cluster_selections[cluster_id] = selected
                bucket_selected.extend(selected)
                bucket_rejected.extend(rejected)

                if len(selected) > 0:
                    variety_stats['clusters_represented'] += 1

            # Store for time coverage check
            bucket_selections[bucket_key] = bucket_selected
            bucket_all_photos[bucket_key] = [p for photos in cluster_photos.values() for p in photos]

            # Remove duplicates across clusters within bucket
            bucket_selected, dup_rejected = self.selector.remove_duplicates(
                bucket_selected, embeddings
            )
            bucket_rejected.extend(dup_rejected)

            all_selected.extend(bucket_selected)
            all_rejected.extend(bucket_rejected)

            bucket_stats[bucket_key] = {
                'total': len(filenames),
                'selected': len(bucket_selected),
                'rejected': len(bucket_rejected),
                'clusters': len(cluster_photos),
                'clusters_with_selection': len([s for s in cluster_selections.values() if len(s) > 0])
            }

        # Ensure time coverage - at least min_photos_per_bucket from each time period
        bucket_selections = self.selector.ensure_time_coverage(
            bucket_selections, bucket_all_photos
        )

        # Check if any new photos were added for time coverage
        for bucket_key, selections in bucket_selections.items():
            for photo in selections:
                if photo not in all_selected:
                    # Remove from rejected if it was there
                    all_rejected = [p for p in all_rejected if p.get('filename') != photo.get('filename')]
                    all_selected.append(photo)

        # Global deduplication across buckets
        all_selected, global_dups = self.selector.remove_duplicates(
            all_selected, embeddings
        )
        all_rejected.extend(global_dups)

        # Calculate statistics
        total_photos = len(all_scores)
        selected_count = len(all_selected)

        # Categorize rejections
        rejection_stats = {}
        for photo in all_rejected:
            reason = photo.get('rejection_reason', SelectionReason.REJECTED_LOW_QUALITY)
            reason_name = reason.value if isinstance(reason, SelectionReason) else str(reason)
            rejection_stats[reason_name] = rejection_stats.get(reason_name, 0) + 1

        return {
            'selected': all_selected,
            'rejected': all_rejected,
            'summary': {
                'total_photos': total_photos,
                'selected_count': selected_count,
                'rejected_count': len(all_rejected),
                'selection_rate': f"{(selected_count/total_photos*100):.1f}%" if total_photos > 0 else "0%",
                'quality_mode': self.quality_mode,
                'variety': {
                    'time_periods': len(bucket_stats),
                    'activity_clusters': variety_stats['total_clusters'],
                    'clusters_represented': variety_stats['clusters_represented']
                },
                'thresholds': {
                    'quality': self.selector.quality_threshold,
                    'blur': self.selector.blur_threshold,
                    'face_quality': self.selector.face_quality_threshold,
                    'similarity': self.selector.similarity_threshold
                }
            },
            'rejection_breakdown': rejection_stats,
            'bucket_stats': bucket_stats
        }


if __name__ == "__main__":
    # Test with dummy data
    selector = SmartPhotoSelector(quality_mode="balanced")

    # Simulate scores
    test_scores = {
        'photo1.jpg': {'total': 0.85, 'face_quality': 0.9, 'aesthetic_quality': 0.8, 'num_faces': 1},
        'photo2.jpg': {'total': 0.45, 'face_quality': 0.3, 'aesthetic_quality': 0.5, 'num_faces': 1},
        'photo3.jpg': {'total': 0.72, 'face_quality': 0.7, 'aesthetic_quality': 0.75, 'num_faces': 2},
        'photo4.jpg': {'total': 0.25, 'face_quality': 0.1, 'aesthetic_quality': 0.2, 'num_faces': 1},
        'photo5.jpg': {'total': 0.65, 'face_quality': 0.6, 'aesthetic_quality': 0.7, 'num_faces': 0},
    }

    for filename, scores in test_scores.items():
        passes, reason, detail = selector.selector.analyze_photo(scores)
        status = "✅ KEEP" if passes else "❌ REJECT"
        print(f"{filename}: {status} - {reason.value} ({detail})")
