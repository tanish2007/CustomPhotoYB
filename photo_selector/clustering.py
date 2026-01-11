"""
Step 3: Cluster Photos Within Each Time Bucket
Uses K-Means or Hierarchical Clustering for diversity
Ensures different activities, settings, and moods are represented
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import hdbscan
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class PhotoClusterer:
    """Cluster photos within time buckets for diversity."""

    def __init__(self, method: str = "hdbscan", min_cluster_size: int = 3,
                 temporal_gap_hours: float = 6.0):
        """
        Initialize clusterer.

        Args:
            method: "kmeans", "hierarchical", or "hdbscan"
            min_cluster_size: Minimum cluster size for HDBSCAN (default: 3)
            temporal_gap_hours: Hours gap to split clusters into separate events (default: 6.0)
                              Set to None to disable temporal splitting
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.temporal_gap_hours = temporal_gap_hours

    def determine_num_clusters(self, num_photos: int,
                                target_selection: int) -> int:
        """
        Determine optimal number of clusters.

        Args:
            num_photos: Total photos in bucket
            target_selection: How many photos we want to select

        Returns:
            Number of clusters to use
        """
        if num_photos <= target_selection:
            # If we have fewer photos than target, each photo is its own cluster
            return num_photos

        # Aim for slightly more clusters than target to have choices
        # But not too many that clusters become meaningless
        num_clusters = min(
            max(target_selection, 2),  # At least 2 clusters
            num_photos // 2  # At most half the photos
        )

        return num_clusters

    def determine_optimal_clusters_adaptive(self, embeddings: np.ndarray,
                                             num_photos: int) -> int:
        """
        Determine optimal number of clusters using silhouette analysis.
        Tries different K values and picks the one with best silhouette score.
        This finds NATURAL groupings based on actual photo similarity.

        Args:
            embeddings: Photo embeddings array
            num_photos: Total photos in bucket

        Returns:
            Optimal number of clusters
        """
        if num_photos < 4:
            # Too few photos for meaningful clustering
            return num_photos

        # Define range of K values to try
        # Min: 2 clusters
        # Max: photos//3 (ensures at least 3 photos per cluster on average)
        min_k = 2
        max_k = min(num_photos // 3, 20)  # Cap at 20 to avoid too many clusters

        if max_k < min_k:
            return min_k

        # For efficiency, sample strategic K values instead of trying all
        # This reduces computation while still finding good K
        if max_k - min_k <= 5:
            # Small range, try all
            candidate_ks = list(range(min_k, max_k + 1))
        else:
            # Large range, sample strategically
            candidate_ks = [
                min_k,                                      # Minimum
                min_k + (max_k - min_k) // 4,              # 25% point
                min_k + (max_k - min_k) // 2,              # 50% point (middle)
                min_k + 3 * (max_k - min_k) // 4,          # 75% point
                max_k                                       # Maximum
            ]
            # Remove duplicates and sort
            candidate_ks = sorted(list(set(candidate_ks)))

        print(f"  Testing K values: {candidate_ks} (from {num_photos} photos)")

        best_k = min_k
        best_score = -1

        for k in candidate_ks:
            if k >= num_photos or k < 2:
                continue

            try:
                # Cluster with this K
                if self.method == "kmeans":
                    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                else:
                    clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')

                labels = clusterer.fit_predict(embeddings)

                # Calculate silhouette score (measures cluster quality)
                # Score ranges from -1 to 1:
                #   1 = perfect clusters (very distinct)
                #   0 = overlapping clusters
                #  -1 = wrong clustering
                score = silhouette_score(embeddings, labels)

                print(f"    K={k}: silhouette={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                print(f"    K={k}: failed ({e})")
                continue

        print(f"  -> Optimal K={best_k} (silhouette={best_score:.3f})")
        return best_k

    def split_cluster_by_temporal_gaps(self, photo_indices: List[int],
                                      timestamps: List[datetime]) -> List[List[int]]:
        """
        Split a cluster into sub-clusters based on temporal gaps.

        Args:
            photo_indices: List of photo indices in the cluster
            timestamps: List of timestamps corresponding to photo_indices

        Returns:
            List of sub-clusters, where each sub-cluster is a list of photo indices
        """
        if self.temporal_gap_hours is None or len(photo_indices) <= 1:
            return [photo_indices]

        # Sort by timestamp
        sorted_pairs = sorted(zip(photo_indices, timestamps), key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in sorted_pairs]
        sorted_timestamps = [ts for _, ts in sorted_pairs]

        sub_clusters = []
        current_cluster = [sorted_indices[0]]

        for i in range(1, len(sorted_indices)):
            # Calculate time gap in hours
            time_gap = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds() / 3600

            if time_gap > self.temporal_gap_hours:
                # Large gap detected - start new sub-cluster
                sub_clusters.append(current_cluster)
                current_cluster = [sorted_indices[i]]
            else:
                # Same event - add to current sub-cluster
                current_cluster.append(sorted_indices[i])

        # Don't forget the last sub-cluster
        sub_clusters.append(current_cluster)

        return sub_clusters

    def _apply_temporal_splitting(self, labels: np.ndarray,
                                  timestamps: List[datetime]) -> np.ndarray:
        """
        Apply temporal gap splitting to existing cluster labels.

        Args:
            labels: Current cluster labels
            timestamps: Timestamps for each photo

        Returns:
            Updated labels with temporal splitting applied
        """
        new_labels = labels.copy()
        next_cluster_id = labels.max() + 1
        num_splits = 0

        # Process each cluster
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0].tolist()

            # Get timestamps for this cluster
            cluster_timestamps = [timestamps[i] for i in cluster_indices]

            # Split by temporal gaps
            sub_clusters = self.split_cluster_by_temporal_gaps(cluster_indices, cluster_timestamps)

            # If split occurred, reassign labels
            if len(sub_clusters) > 1:
                num_splits += len(sub_clusters) - 1
                # Keep first sub-cluster with original label
                # Assign new labels to remaining sub-clusters
                for i, sub_cluster in enumerate(sub_clusters[1:], start=1):
                    for idx in sub_cluster:
                        new_labels[idx] = next_cluster_id
                    next_cluster_id += 1

        if num_splits > 0:
            print(f"  -> Temporal splitting: {num_splits} gaps detected ({self.temporal_gap_hours}h threshold)")

        return new_labels

    def cluster_photos(self, embeddings: np.ndarray,
                       num_clusters: int = None,
                       timestamps: Optional[List[datetime]] = None) -> np.ndarray:
        """
        Cluster photos based on their embeddings.

        Args:
            embeddings: Array of shape (num_photos, embedding_dim)
            num_clusters: Number of clusters (ignored for HDBSCAN, which auto-determines)
            timestamps: Optional list of timestamps for temporal gap splitting

        Returns:
            Array of cluster labels
        """
        if self.method == "hdbscan":
            num_photos = len(embeddings)

            # HDBSCAN requires at least 3 photos to work properly
            # For very small buckets, treat each photo as its own cluster
            if num_photos < 3:
                return np.arange(num_photos)

            # Adjust parameters based on number of photos
            # min_cluster_size must be > 1 and <= num_photos
            actual_min_cluster_size = max(2, min(self.min_cluster_size, num_photos))
            # min_samples must be >= 1 and <= num_photos
            actual_min_samples = max(1, min(1, num_photos))

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=actual_min_cluster_size,
                min_samples=actual_min_samples,
                metric='euclidean',
                cluster_selection_method='eom'  # Excess of Mass for more stable clusters
            )
            labels = clusterer.fit_predict(embeddings)

            # HDBSCAN marks noise points as -1, we'll treat them as individual clusters
            # This ensures every photo gets assigned to a cluster
            if np.any(labels == -1):
                max_label = labels.max()
                noise_mask = labels == -1
                noise_indices = np.where(noise_mask)[0]
                for i, idx in enumerate(noise_indices):
                    labels[idx] = max_label + 1 + i

            # Apply temporal gap splitting if timestamps are provided
            if timestamps is not None and self.temporal_gap_hours is not None:
                labels = self._apply_temporal_splitting(labels, timestamps)

            return labels

        # For K-Means and Hierarchical, we need num_clusters
        if len(embeddings) <= num_clusters:
            # Each photo is its own cluster
            return np.arange(len(embeddings))

        if num_clusters <= 1:
            return np.zeros(len(embeddings), dtype=int)

        if self.method == "kmeans":
            clusterer = KMeans(
                n_clusters=num_clusters,
                random_state=42,
                n_init=10
            )
        elif self.method == "hierarchical":
            clusterer = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage='ward'
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        labels = clusterer.fit_predict(embeddings)
        return labels

    def get_cluster_representatives(self, embeddings: np.ndarray,
                                     labels: np.ndarray) -> Dict[int, int]:
        """
        Find the most representative photo (centroid) in each cluster.

        Args:
            embeddings: Photo embeddings
            labels: Cluster labels

        Returns:
            Dictionary mapping cluster_id to photo index (closest to centroid)
        """
        representatives = {}
        unique_labels = np.unique(labels)

        for cluster_id in unique_labels:
            # Get indices of photos in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = embeddings[cluster_mask]

            if len(cluster_embeddings) == 1:
                representatives[cluster_id] = cluster_indices[0]
                continue

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Find photo closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = np.argmin(distances)
            representatives[cluster_id] = cluster_indices[closest_idx]

        return representatives

    def get_cluster_diversity_scores(self, embeddings: np.ndarray,
                                      labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate diversity score for each cluster.
        Higher score = more diverse cluster.

        Args:
            embeddings: Photo embeddings
            labels: Cluster labels

        Returns:
            Dictionary mapping cluster_id to diversity score
        """
        diversity_scores = {}
        unique_labels = np.unique(labels)

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]

            if len(cluster_embeddings) <= 1:
                diversity_scores[cluster_id] = 0.0
                continue

            # Calculate average pairwise distance
            distances = []
            for i in range(len(cluster_embeddings)):
                for j in range(i + 1, len(cluster_embeddings)):
                    dist = np.linalg.norm(cluster_embeddings[i] - cluster_embeddings[j])
                    distances.append(dist)

            diversity_scores[cluster_id] = np.mean(distances) if distances else 0.0

        return diversity_scores


class BucketClusterManager:
    """Manage clustering across all time buckets."""

    def __init__(self, clusterer: PhotoClusterer = None):
        """
        Initialize manager.

        Args:
            clusterer: PhotoClusterer instance
        """
        self.clusterer = clusterer or PhotoClusterer(method="hdbscan")

    def cluster_all_buckets(self,
                            buckets: Dict[str, List[Dict]],
                            embeddings: Dict[str, np.ndarray],
                            targets: Dict[str, int],
                            use_adaptive: bool = True) -> Dict[str, Dict]:
        """
        Cluster photos in all buckets.

        Args:
            buckets: Dict of bucket_key -> list of photo info
            embeddings: Dict of filename -> embedding
            targets: Dict of bucket_key -> target selection count
            use_adaptive: If True, use adaptive K selection with silhouette scores

        Returns:
            Dict with clustering results per bucket
        """
        results = {}

        for bucket_key, photos in buckets.items():
            if not photos:
                continue

            target = targets.get(bucket_key, 1)

            # Get embeddings for photos in this bucket
            bucket_embeddings = []
            bucket_filenames = []
            bucket_timestamps = []
            for photo in photos:
                filename = photo['filename']
                if filename in embeddings:
                    bucket_embeddings.append(embeddings[filename])
                    bucket_filenames.append(filename)
                    # Extract timestamp from photo info
                    if photo.get('date'):
                        try:
                            ts = datetime.fromisoformat(photo['date'])
                            bucket_timestamps.append(ts)
                        except:
                            bucket_timestamps.append(None)
                    else:
                        bucket_timestamps.append(None)

            if not bucket_embeddings:
                print(f"Warning: No embeddings found for bucket {bucket_key}")
                continue

            bucket_embeddings = np.array(bucket_embeddings)

            # Only use timestamps if all photos have them
            use_timestamps = all(ts is not None for ts in bucket_timestamps)
            timestamps_param = bucket_timestamps if use_timestamps else None

            # Determine number of clusters
            if self.clusterer.method == "hdbscan":
                # HDBSCAN automatically determines clusters
                print(f"\nBucket {bucket_key}: HDBSCAN clustering (auto K)...")
                labels = self.clusterer.cluster_photos(bucket_embeddings, timestamps=timestamps_param)
                num_clusters = len(np.unique(labels))
                print(f"  -> HDBSCAN found {num_clusters} natural clusters")
            elif use_adaptive:
                print(f"\nBucket {bucket_key}: Adaptive clustering...")
                num_clusters = self.clusterer.determine_optimal_clusters_adaptive(
                    bucket_embeddings, len(bucket_embeddings)
                )
                # Cluster
                labels = self.clusterer.cluster_photos(bucket_embeddings, num_clusters, timestamps=timestamps_param)
            else:
                num_clusters = self.clusterer.determine_num_clusters(
                    len(bucket_embeddings), target
                )
                # Cluster
                labels = self.clusterer.cluster_photos(bucket_embeddings, num_clusters, timestamps=timestamps_param)

            # Get representatives
            representatives = self.clusterer.get_cluster_representatives(
                bucket_embeddings, labels
            )

            # Map indices back to filenames
            results[bucket_key] = {
                'num_photos': len(bucket_filenames),
                'num_clusters': num_clusters,
                'target_selection': target,
                'filenames': bucket_filenames,
                'labels': labels.tolist(),
                'cluster_representatives': {
                    str(k): bucket_filenames[v]
                    for k, v in representatives.items()
                },
                'cluster_sizes': {
                    str(i): int(np.sum(labels == i))
                    for i in np.unique(labels)
                }
            }

            print(f"Bucket {bucket_key}: {len(bucket_filenames)} photos -> "
                  f"{num_clusters} clusters (target: {target})")

        return results


def analyze_cluster_quality(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Analyze the quality of clustering.

    Returns:
        Dictionary with clustering metrics
    """
    if len(np.unique(labels)) < 2 or len(embeddings) < 3:
        return {'silhouette_score': None, 'num_clusters': len(np.unique(labels))}

    try:
        sil_score = silhouette_score(embeddings, labels)
    except:
        sil_score = None

    return {
        'silhouette_score': sil_score,
        'num_clusters': len(np.unique(labels)),
        'cluster_sizes': {int(i): int(np.sum(labels == i)) for i in np.unique(labels)}
    }


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)

    # Simulate embeddings
    num_photos = 50
    embedding_dim = 512
    embeddings = np.random.randn(num_photos, embedding_dim)

    # Create clusterer
    clusterer = PhotoClusterer(method="kmeans")

    # Cluster
    num_clusters = 10
    labels = clusterer.cluster_photos(embeddings, num_clusters)

    # Analyze
    quality = analyze_cluster_quality(embeddings, labels)
    print(f"Clustering quality: {quality}")

    # Get representatives
    reps = clusterer.get_cluster_representatives(embeddings, labels)
    print(f"Representatives: {reps}")
