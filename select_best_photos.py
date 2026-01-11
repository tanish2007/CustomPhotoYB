#!/usr/bin/env python3
"""
Best Photo Selection for Child's Yearly Album
==============================================

This script selects the best representative photos from a year of your child's photos.

Algorithm:
1. Generate CLIP embeddings - captures faces, activities, backgrounds, emotions
2. Temporal segmentation - splits by month to ensure year-round coverage
3. Cluster within time buckets - ensures diversity (different activities/settings)
4. Score photos - weighted scoring based on face quality, aesthetics, emotion, uniqueness
5. Global deduplication - removes near-duplicates

Usage:
    python select_best_photos.py <folder_path> [options]

Examples:
    python select_best_photos.py "C:/Photos/2024" --target 350
    python select_best_photos.py "C:/Photos/2024" --target 100 --bucket biweekly
"""

import argparse
import os
import sys

# Add the photo_selector package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Select best representative photos for a yearly album",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scoring Weights:
  - Face quality (35%%): sharpness, eyes open, face size/position
  - Aesthetic quality (25%%): lighting, composition, contrast
  - Emotional signal (20%%): expression, interaction, warmth
  - Uniqueness (20%%): distinctiveness within cluster

Tips:
  - Start with a smaller target (50-100) to test
  - Use 'biweekly' bucket type for more granular time coverage
  - Lower similarity threshold (0.90) for stricter deduplication
        """
    )

    parser.add_argument(
        "folder",
        help="Path to folder containing photos"
    )

    parser.add_argument(
        "--target", "-t",
        type=int,
        default=350,
        help="Target number of photos to select (default: 350)"
    )

    parser.add_argument(
        "--bucket", "-b",
        choices=["monthly", "biweekly"],
        default="monthly",
        help="Time bucket type (default: monthly)"
    )

    parser.add_argument(
        "--similarity", "-s",
        type=float,
        default=0.95,
        help="Similarity threshold for duplicates 0.0-1.0 (default: 0.95)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output directory for results (default: same as script)"
    )

    parser.add_argument(
        "--copy-to",
        help="Copy selected photos to this folder"
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation, load from file"
    )

    parser.add_argument(
        "--embeddings-file",
        help="Path to saved embeddings file (for --skip-embeddings)"
    )

    args = parser.parse_args()

    # Validate folder
    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    # Set output directory
    output_dir = args.output or os.path.dirname(os.path.abspath(__file__))

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     BEST PHOTO SELECTION FOR YEARLY ALBUM                 ║
    ║     Powered by CLIP + ML Clustering + Quality Scoring     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    try:
        from photo_selector.pipeline import PhotoSelectionPipeline

        pipeline = PhotoSelectionPipeline(
            target_photos=args.target,
            bucket_type=args.bucket,
            similarity_threshold=args.similarity,
            output_dir=output_dir
        )

        results = pipeline.run(
            args.folder,
            skip_embeddings=args.skip_embeddings,
            embeddings_file=args.embeddings_file
        )

        # Copy photos if requested
        if args.copy_to:
            pipeline.copy_selected_photos(args.folder, args.copy_to)

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Source folder: {args.folder}")
        print(f"Total photos analyzed: {results['summary']['total_photos']}")
        print(f"Photos selected: {results['summary']['selected_photos']}")
        print(f"Duplicates removed: {results['summary']['removed_duplicates']}")
        print(f"Time buckets: {results['summary']['buckets']}")
        print("\nResults saved to:", output_dir)

        if args.copy_to:
            print(f"Selected photos copied to: {args.copy_to}")

        return 0

    except ImportError as e:
        print(f"\nError: Missing required package: {e}")
        print("\nPlease install requirements:")
        print("  pip install torch torchvision pillow pillow-heif opencv-python scikit-learn")
        print("  pip install git+https://github.com/openai/CLIP.git")
        return 1

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
