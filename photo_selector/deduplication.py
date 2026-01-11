"""
Step 5: Global Deduplication Pass
Remove near-duplicate photos after selection
Replace with next-best from same cluster
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from pathlib import Path


class PhotoDeduplicator:
    """Remove near-duplicate photos from selection."""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplicator.

        Args:
            similarity_threshold: Photos with similarity > this are considered duplicates
        """
        self.similarity_threshold = similarity_threshold

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Embeddings should already be normalized, but normalize just in case
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1_norm, emb2_norm))

    def find_duplicates(self,
                        selected_photos: List[Dict],
                        embeddings: Dict[str, np.ndarray]) -> List[Tuple[int, int, float]]:
        """
        Find duplicate pairs in selected photos.

        Args:
            selected_photos: List of selected photo info dicts
            embeddings: Dict of filename -> embedding

        Returns:
            List of (index1, index2, similarity) tuples for duplicates
        """
        duplicates = []

        for i in range(len(selected_photos)):
            for j in range(i + 1, len(selected_photos)):
                fname1 = selected_photos[i]['filename']
                fname2 = selected_photos[j]['filename']

                if fname1 not in embeddings or fname2 not in embeddings:
                    continue

                sim = self.compute_similarity(embeddings[fname1], embeddings[fname2])

                if sim > self.similarity_threshold:
                    duplicates.append((i, j, sim))

        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates

    def find_all_similar_pairs(self,
                                selected_photos: List[Dict],
                                embeddings: Dict[str, np.ndarray],
                                threshold: float = 0.85) -> List[Dict]:
        """
        Find all similar pairs for analysis.

        Args:
            selected_photos: List of selected photos
            embeddings: Embeddings dict
            threshold: Minimum similarity to report

        Returns:
            List of similarity info dicts
        """
        pairs = []

        for i in range(len(selected_photos)):
            for j in range(i + 1, len(selected_photos)):
                fname1 = selected_photos[i]['filename']
                fname2 = selected_photos[j]['filename']

                if fname1 not in embeddings or fname2 not in embeddings:
                    continue

                sim = self.compute_similarity(embeddings[fname1], embeddings[fname2])

                if sim > threshold:
                    pairs.append({
                        'photo1': fname1,
                        'photo2': fname2,
                        'similarity': round(sim, 4),
                        'bucket1': selected_photos[i].get('bucket'),
                        'bucket2': selected_photos[j].get('bucket')
                    })

        pairs.sort(key=lambda x: x['similarity'], reverse=True)
        return pairs


class SelectionDeduplicator:
    """Manage deduplication with replacement from clusters."""

    def __init__(self,
                 deduplicator: PhotoDeduplicator = None,
                 similarity_threshold: float = 0.95):
        self.deduplicator = deduplicator or PhotoDeduplicator(similarity_threshold)
        self.similarity_threshold = similarity_threshold

    def deduplicate_with_replacement(self,
                                      selected_photos: List[Dict],
                                      cluster_alternatives: Dict[str, List[Dict]],
                                      embeddings: Dict[str, np.ndarray],
                                      max_iterations: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """
        Deduplicate selected photos, replacing with alternatives.

        Args:
            selected_photos: List of selected photo info dicts
            cluster_alternatives: Dict mapping cluster_key to sorted alternatives
            embeddings: Dict of filename -> embedding
            max_iterations: Max replacement iterations

        Returns:
            Tuple of (final selection, removed photos)
        """
        current_selection = list(selected_photos)
        removed = []

        for iteration in range(max_iterations):
            duplicates = self.deduplicator.find_duplicates(current_selection, embeddings)

            if not duplicates:
                break

            print(f"Dedup iteration {iteration + 1}: found {len(duplicates)} duplicate pairs")

            # Process each duplicate pair
            for idx1, idx2, similarity in duplicates:
                if idx1 >= len(current_selection) or idx2 >= len(current_selection):
                    continue

                photo1 = current_selection[idx1]
                photo2 = current_selection[idx2]

                # Decide which to remove (lower score)
                score1 = photo1.get('total', 0)
                score2 = photo2.get('total', 0)

                if score1 >= score2:
                    to_remove_idx = idx2
                    to_remove = photo2
                else:
                    to_remove_idx = idx1
                    to_remove = photo1

                # Find replacement
                cluster_key = to_remove.get('cluster_key', '')
                alternatives = cluster_alternatives.get(cluster_key, [])

                replacement = None
                selected_filenames = {p['filename'] for p in current_selection}

                for alt in alternatives:
                    if alt['filename'] not in selected_filenames:
                        # Check this alternative isn't too similar to existing selections
                        alt_emb = embeddings.get(alt['filename'])
                        if alt_emb is not None:
                            too_similar = False
                            for existing in current_selection:
                                if existing['filename'] == to_remove['filename']:
                                    continue
                                exist_emb = embeddings.get(existing['filename'])
                                if exist_emb is not None:
                                    sim = self.deduplicator.compute_similarity(alt_emb, exist_emb)
                                    if sim > self.similarity_threshold:
                                        too_similar = True
                                        break
                            if not too_similar:
                                replacement = alt
                                break

                # Remove duplicate
                removed.append(to_remove)
                current_selection = [p for p in current_selection
                                    if p['filename'] != to_remove['filename']]

                # Add replacement if found
                if replacement:
                    replacement['replaced'] = to_remove['filename']
                    current_selection.append(replacement)
                    print(f"  Replaced {to_remove['filename']} with {replacement['filename']}")
                else:
                    print(f"  Removed {to_remove['filename']} (no replacement available)")

        return current_selection, removed

    def final_similarity_check(self,
                                selected_photos: List[Dict],
                                embeddings: Dict[str, np.ndarray],
                                threshold: float = 0.90) -> Dict:
        """
        Final check for remaining similar photos.

        Returns:
            Summary dict with similarity statistics
        """
        pairs = self.deduplicator.find_all_similar_pairs(
            selected_photos, embeddings, threshold
        )

        return {
            'num_similar_pairs': len(pairs),
            'most_similar': pairs[:5] if pairs else [],
            'high_similarity_count': sum(1 for p in pairs if p['similarity'] > 0.95)
        }


def build_cluster_alternatives(cluster_results: Dict,
                                all_scores: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """
    Build a mapping of cluster -> sorted alternatives.

    Args:
        cluster_results: Results from clustering step
        all_scores: All photo scores

    Returns:
        Dict mapping cluster_key to sorted list of alternatives
    """
    alternatives = {}

    for bucket_key, bucket_data in cluster_results.items():
        filenames = bucket_data.get('filenames', [])
        labels = bucket_data.get('labels', [])

        # Group by cluster
        cluster_photos = {}
        for fname, label in zip(filenames, labels):
            cluster_key = f"{bucket_key}_cluster_{label}"
            if cluster_key not in cluster_photos:
                cluster_photos[cluster_key] = []

            photo_info = all_scores.get(fname, {'filename': fname, 'total': 0})
            photo_info['filename'] = fname
            photo_info['bucket'] = bucket_key
            photo_info['cluster_key'] = cluster_key
            cluster_photos[cluster_key].append(photo_info)

        # Sort each cluster by score
        for cluster_key, photos in cluster_photos.items():
            alternatives[cluster_key] = sorted(
                photos,
                key=lambda x: x.get('total', 0),
                reverse=True
            )

    return alternatives


if __name__ == "__main__":
    # Example usage
    dedup = PhotoDeduplicator(similarity_threshold=0.95)

    # Test with dummy data
    np.random.seed(42)

    # Create some embeddings with duplicates
    base_emb = np.random.randn(512)
    base_emb = base_emb / np.linalg.norm(base_emb)

    embeddings = {
        'photo1.jpg': base_emb,
        'photo2.jpg': base_emb + np.random.randn(512) * 0.01,  # Very similar
        'photo3.jpg': np.random.randn(512),  # Different
        'photo4.jpg': np.random.randn(512),  # Different
    }

    # Normalize all
    for k in embeddings:
        embeddings[k] = embeddings[k] / np.linalg.norm(embeddings[k])

    selected = [
        {'filename': 'photo1.jpg', 'total': 0.8},
        {'filename': 'photo2.jpg', 'total': 0.75},
        {'filename': 'photo3.jpg', 'total': 0.9},
        {'filename': 'photo4.jpg', 'total': 0.85},
    ]

    duplicates = dedup.find_duplicates(selected, embeddings)
    print(f"Found {len(duplicates)} duplicate pairs:")
    for i, j, sim in duplicates:
        print(f"  {selected[i]['filename']} <-> {selected[j]['filename']}: {sim:.4f}")
