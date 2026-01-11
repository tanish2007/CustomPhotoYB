"""
Main Photo Selection Pipeline
Combines all steps to select the best representative photos of your child's year
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class PhotoSelectionPipeline:
    """
    Complete pipeline for selecting best representative photos.

    Steps:
    1. Generate CLIP embeddings for all photos
    2. Segment photos by time (monthly/biweekly)
    3. Cluster within each time bucket
    4. Score photos and select best from each cluster
    5. Global deduplication pass
    """

    def __init__(self,
                 target_photos: int = 350,
                 bucket_type: str = "monthly",
                 clustering_method: str = "kmeans",
                 similarity_threshold: float = 0.95,
                 output_dir: str = None):
        """
        Initialize pipeline.

        Args:
            target_photos: Total number of photos to select
            bucket_type: "monthly" or "biweekly"
            clustering_method: "kmeans" or "hierarchical"
            similarity_threshold: Threshold for duplicate detection
            output_dir: Directory to save results
        """
        self.target_photos = target_photos
        self.bucket_type = bucket_type
        self.clustering_method = clustering_method
        self.similarity_threshold = similarity_threshold
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))

        # Will be initialized when needed
        self.embedder = None
        self.segmenter = None
        self.clusterer = None
        self.scorer = None
        self.deduplicator = None

        # Data storage
        self.embeddings = {}
        self.buckets = {}
        self.cluster_results = {}
        self.all_scores = {}
        self.selected_photos = []

    def _init_components(self):
        """Initialize all pipeline components."""
        from .embeddings import PhotoEmbedder
        from .temporal import TemporalSegmenter
        from .clustering import PhotoClusterer, BucketClusterManager
        from .scoring import PhotoScorer, ClusterScorer
        from .deduplication import PhotoDeduplicator, SelectionDeduplicator

        print("Initializing pipeline components...")

        self.embedder = PhotoEmbedder()
        self.segmenter = TemporalSegmenter(bucket_type=self.bucket_type)
        self.clusterer = BucketClusterManager(
            PhotoClusterer(method=self.clustering_method)
        )
        self.scorer = ClusterScorer(PhotoScorer())
        self.deduplicator = SelectionDeduplicator(
            PhotoDeduplicator(self.similarity_threshold),
            self.similarity_threshold
        )

    def run(self, folder_path: str,
            skip_embeddings: bool = False,
            embeddings_file: str = None) -> Dict:
        """
        Run the complete pipeline.

        Args:
            folder_path: Path to folder containing photos
            skip_embeddings: If True, load from file instead of computing
            embeddings_file: Path to saved embeddings file

        Returns:
            Dictionary with results and selected photos
        """
        self._init_components()
        folder = Path(folder_path)

        print(f"\n{'='*60}")
        print(f"PHOTO SELECTION PIPELINE")
        print(f"{'='*60}")
        print(f"Folder: {folder_path}")
        print(f"Target: {self.target_photos} photos")
        print(f"Bucket type: {self.bucket_type}")
        print(f"{'='*60}\n")

        # Step 1: Embeddings
        print("\n[STEP 1/5] Generating embeddings...")
        if skip_embeddings and embeddings_file and os.path.exists(embeddings_file):
            from .embeddings import PhotoEmbedder
            self.embeddings = PhotoEmbedder.load_embeddings(embeddings_file)
            print(f"Loaded {len(self.embeddings)} embeddings from file")
        else:
            self.embeddings = self.embedder.process_folder(folder_path)
            # Save embeddings
            emb_file = os.path.join(self.output_dir, "photo_embeddings.npz")
            self.embedder.save_embeddings(self.embeddings, emb_file)

        # Step 2: Temporal segmentation
        print("\n[STEP 2/5] Segmenting by time...")
        self.buckets = self.segmenter.segment_folder(folder_path)
        targets = self.segmenter.calculate_target_per_bucket(
            self.buckets, self.target_photos
        )

        print("\nTarget selection per bucket:")
        for bucket, target in sorted(targets.items()):
            available = len(self.buckets.get(bucket, []))
            print(f"  {bucket}: {target} of {available}")

        # Step 3: Clustering
        print("\n[STEP 3/5] Clustering within time buckets...")
        self.cluster_results = self.clusterer.cluster_all_buckets(
            self.buckets, self.embeddings, targets
        )

        # Step 4: Scoring and selection
        print("\n[STEP 4/5] Scoring and selecting best photos...")
        self.selected_photos = []
        self.all_scores = {}

        for bucket_key, bucket_data in self.cluster_results.items():
            filenames = bucket_data['filenames']
            labels = np.array(bucket_data['labels'])
            target = bucket_data['target_selection']

            # Get embeddings for this bucket
            bucket_embeddings = np.array([
                self.embeddings[fn] for fn in filenames
            ])

            # Group by cluster
            unique_labels = np.unique(labels)
            photos_per_cluster = max(1, target // len(unique_labels))

            bucket_selected = []

            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                cluster_filenames = [filenames[i] for i in cluster_indices]
                cluster_embs = bucket_embeddings[cluster_mask]

                # Get file paths
                cluster_paths = [
                    str(folder / fn) for fn in cluster_filenames
                ]

                # Score and select
                scores = self.scorer.score_cluster(cluster_paths, cluster_embs)

                # Store all scores
                for score in scores:
                    score['bucket'] = bucket_key
                    score['cluster'] = int(cluster_id)
                    score['cluster_key'] = f"{bucket_key}_cluster_{cluster_id}"
                    self.all_scores[score['filename']] = score

                # Select best from this cluster
                num_select = min(photos_per_cluster, len(scores))
                bucket_selected.extend(scores[:num_select])

            # Trim to target if we selected too many
            bucket_selected.sort(key=lambda x: x.get('total', 0), reverse=True)
            self.selected_photos.extend(bucket_selected[:target])

            print(f"  {bucket_key}: selected {min(len(bucket_selected), target)} photos")

        print(f"\nTotal selected before dedup: {len(self.selected_photos)}")

        # Step 5: Deduplication
        print("\n[STEP 5/5] Global deduplication...")
        from .deduplication import build_cluster_alternatives

        cluster_alternatives = build_cluster_alternatives(
            self.cluster_results, self.all_scores
        )

        final_selection, removed = self.deduplicator.deduplicate_with_replacement(
            self.selected_photos,
            cluster_alternatives,
            self.embeddings
        )

        self.selected_photos = final_selection

        # Final check
        similarity_stats = self.deduplicator.final_similarity_check(
            self.selected_photos, self.embeddings
        )

        print(f"\n{'='*60}")
        print("SELECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Final selection: {len(self.selected_photos)} photos")
        print(f"Removed duplicates: {len(removed)}")
        print(f"Remaining similar pairs (>0.90): {similarity_stats['num_similar_pairs']}")
        print(f"High similarity pairs (>0.95): {similarity_stats['high_similarity_count']}")

        # Save results
        results = self._save_results(folder_path, removed, similarity_stats)

        return results

    def _save_results(self, folder_path: str,
                      removed: List[Dict],
                      similarity_stats: Dict) -> Dict:
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare results
        results = {
            'metadata': {
                'source_folder': folder_path,
                'timestamp': timestamp,
                'target_photos': self.target_photos,
                'bucket_type': self.bucket_type,
                'similarity_threshold': self.similarity_threshold
            },
            'summary': {
                'total_photos': len(self.embeddings),
                'selected_photos': len(self.selected_photos),
                'removed_duplicates': len(removed),
                'buckets': len(self.buckets),
                'similarity_stats': similarity_stats
            },
            'selected_photos': [
                {
                    'filename': p['filename'],
                    'filepath': p.get('filepath', ''),
                    'bucket': p.get('bucket', ''),
                    'score': p.get('total', 0),
                    'face_quality': p.get('face_quality', 0),
                    'aesthetic_quality': p.get('aesthetic_quality', 0),
                    'emotional_signal': p.get('emotional_signal', 0),
                    'uniqueness': p.get('uniqueness', 0)
                }
                for p in self.selected_photos
            ],
            'removed_duplicates': [
                {'filename': p['filename'], 'reason': 'duplicate'}
                for p in removed
            ],
            'buckets_summary': {
                bucket: {
                    'total_photos': len(photos),
                    'selected': sum(1 for p in self.selected_photos
                                   if p.get('bucket') == bucket)
                }
                for bucket, photos in self.buckets.items()
            }
        }

        # Save JSON
        results_file = os.path.join(self.output_dir, f"selection_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

        # Save simple list of selected files
        selected_list_file = os.path.join(self.output_dir, f"selected_photos_{timestamp}.txt")
        with open(selected_list_file, 'w', encoding='utf-8') as f:
            for p in sorted(self.selected_photos, key=lambda x: x.get('bucket', '')):
                f.write(f"{p['filename']}\n")
        print(f"Selected list saved to: {selected_list_file}")

        return results

    def copy_selected_photos(self, source_folder: str,
                              destination_folder: str):
        """Copy selected photos to a new folder."""
        dest = Path(destination_folder)
        dest.mkdir(parents=True, exist_ok=True)

        source = Path(source_folder)

        copied = 0
        for photo in self.selected_photos:
            src_path = source / photo['filename']
            if src_path.exists():
                dst_path = dest / photo['filename']
                shutil.copy2(src_path, dst_path)
                copied += 1

        print(f"Copied {copied} photos to {destination_folder}")
        return copied


def run_pipeline(folder_path: str,
                 target_photos: int = 350,
                 bucket_type: str = "monthly",
                 output_dir: str = None) -> Dict:
    """
    Convenience function to run the pipeline.

    Args:
        folder_path: Path to folder with photos
        target_photos: Number of photos to select
        bucket_type: "monthly" or "biweekly"
        output_dir: Where to save results

    Returns:
        Results dictionary
    """
    pipeline = PhotoSelectionPipeline(
        target_photos=target_photos,
        bucket_type=bucket_type,
        output_dir=output_dir
    )

    return pipeline.run(folder_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <folder_path> [target_photos]")
        print("\nExample:")
        print("  python pipeline.py /path/to/photos 350")
        sys.exit(1)

    folder = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 350

    results = run_pipeline(folder, target_photos=target)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
