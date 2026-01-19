"""
Test script: Select best 40 photos from a single month folder.
Usage: python test_single_month.py <folder_path> [target_count]

Example:
    python test_single_month.py "C:/Photos/2024/January" 40
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from photo_selector.monthly_selector import MonthlyPhotoSelector


def test_single_month(folder_path: str, target: int = 40):
    """
    Test photo selection on a single folder.

    Args:
        folder_path: Path to folder containing photos
        target: Number of photos to select (default 40)
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return

    # Count photos
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
    photos = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    print(f"\n{'='*60}")
    print(f"SINGLE MONTH TEST")
    print(f"{'='*60}")
    print(f"Folder: {folder}")
    print(f"Photos found: {len(photos)}")
    print(f"Target selection: {target}")
    print(f"{'='*60}\n")

    if len(photos) == 0:
        print("No photos found in folder!")
        return

    # Initialize selector
    print("Initializing selector (loading CLIP model)...")
    selector = MonthlyPhotoSelector()

    # Step 1: Generate embeddings
    print(f"\n[Step 1] Generating CLIP embeddings for {len(photos)} photos...")
    photo_paths = [str(p) for p in photos]
    embeddings = selector.generate_embeddings(photo_paths)
    print(f"Generated embeddings for {len(embeddings)} photos")

    # Step 2: Score photos
    print(f"\n[Step 2] Scoring photos...")
    from photo_selector.scoring import PhotoScorer
    scorer = PhotoScorer()

    scored_photos = []
    for i, photo_path in enumerate(photo_paths):
        if (i + 1) % 10 == 0:
            print(f"  Scoring {i + 1}/{len(photo_paths)}...")

        filename = Path(photo_path).name
        emb = embeddings.get(filename)

        # Get scores
        scores = scorer.score_photo(photo_path)

        scored_photos.append({
            'filename': filename,
            'filepath': photo_path,
            'total': scores.get('total', 0),
            'face_quality': scores.get('face_quality', 0),
            'aesthetic_quality': scores.get('aesthetic_quality', 0),
            'emotional_signal': scores.get('emotional_signal', 0),
            'uniqueness': scores.get('uniqueness', 0.5),
            'num_faces': scores.get('num_faces', 0)
        })

    print(f"Scored {len(scored_photos)} photos")

    # Step 3: Cluster and select using HDBSCAN
    print(f"\n[Step 3] Running HDBSCAN clustering and selection...")
    selected = selector.select_hybrid_hdbscan(
        scored_photos,
        embeddings,
        target=target
    )

    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total photos: {len(photos)}")
    print(f"Selected: {len(selected)}")
    print(f"{'='*60}\n")

    # Show selected photos
    print("Selected photos (ranked by score):\n")
    print(f"{'#':<4} {'Score':>6} {'Faces':>6} {'Cluster':>8} {'Similarity':>10} {'Filename':<40}")
    print("-" * 80)

    for i, photo in enumerate(selected, 1):
        score = photo.get('total', 0) * 100
        faces = photo.get('num_faces', 0)
        cluster = photo.get('cluster_id', -1)
        cluster_label = f"C{cluster}" if cluster >= 0 else "Fallback"
        similarity = photo.get('max_similarity', 0) * 100
        filename = photo.get('filename', '?')[:38]

        print(f"{i:<4} {score:>5.1f}% {faces:>6} {cluster_label:>8} {similarity:>9.1f}% {filename:<40}")

    # Cluster distribution
    print(f"\n{'='*60}")
    print("CLUSTER DISTRIBUTION")
    print(f"{'='*60}")

    cluster_counts = {}
    for photo in selected:
        cid = photo.get('cluster_id', -1)
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    for cid in sorted(cluster_counts.keys()):
        label = f"Cluster {cid}" if cid >= 0 else "Fallback"
        count = cluster_counts[cid]
        bar = "█" * count
        print(f"  {label:<12}: {count:>3} {bar}")

    print(f"\n{'='*60}")

    return selected


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo folder provided. Please specify a folder path.")
        sys.exit(1)

    folder_path = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    test_single_month(folder_path, target)
