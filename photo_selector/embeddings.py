"""
Step 1: Generate CLIP embeddings for photos
Captures: faces, activities, backgrounds, emotions, context
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional
import json

# Try to import CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class PhotoEmbedder:
    """Generate CLIP embeddings for photos."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the CLIP model.

        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-L/14, etc.)
            device: 'cuda' or 'cpu', auto-detected if None
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is required. Install with: pip install git+https://github.com/openai/CLIP.git")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on {self.device}...")

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.embedding_dim = self.model.visual.output_dim

        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess an image."""
        try:
            img = Image.open(image_path)
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for a single image."""
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_input)
            # Normalize the embedding
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()

    def process_folder(self, folder_path: str,
                       image_extensions: set = None,
                       batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Process all images in a folder and generate embeddings.

        Args:
            folder_path: Path to folder containing images
            image_extensions: Set of valid extensions
            batch_size: Number of images to process at once

        Returns:
            Dictionary mapping filename to embedding
        """
        if image_extensions is None:
            image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}

        folder = Path(folder_path)
        image_files = [f for f in folder.iterdir()
                       if f.suffix.lower() in image_extensions]

        print(f"Found {len(image_files)} images in {folder_path}")

        embeddings = {}
        errors = []

        for i, image_path in enumerate(image_files):
            if (i + 1) % 10 == 0:
                print(f"Processing [{i+1}/{len(image_files)}] {image_path.name}")

            try:
                img = self.load_image(str(image_path))
                if img is not None:
                    embedding = self.get_embedding(img)
                    embeddings[image_path.name] = embedding
                    img.close()
            except Exception as e:
                errors.append((image_path.name, str(e)))

        print(f"\nProcessed {len(embeddings)} images successfully")
        if errors:
            print(f"Errors on {len(errors)} images")

        return embeddings

    def save_embeddings(self, embeddings: Dict[str, np.ndarray],
                        output_path: str):
        """Save embeddings to a numpy file."""
        # Convert to serializable format
        data = {
            'filenames': list(embeddings.keys()),
            'embeddings': np.array(list(embeddings.values()))
        }
        np.savez(output_path, **data)
        print(f"Saved embeddings to {output_path}")

    @staticmethod
    def load_embeddings(input_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from a numpy file."""
        data = np.load(input_path)
        filenames = data['filenames']
        embeddings_array = data['embeddings']
        return {fn: emb for fn, emb in zip(filenames, embeddings_array)}


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(emb1, emb2))


def find_similar_photos(embeddings: Dict[str, np.ndarray],
                        query_filename: str,
                        top_k: int = 10) -> List[Tuple[str, float]]:
    """Find most similar photos to a query photo."""
    query_emb = embeddings[query_filename]

    similarities = []
    for filename, emb in embeddings.items():
        if filename != query_filename:
            sim = compute_similarity(query_emb, emb)
            similarities.append((filename, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = r"C:\Users\tanis\Downloads\Ariya Millikin-20260106T125948Z-1-002\ariya"

    embedder = PhotoEmbedder()
    embeddings = embedder.process_folder(folder)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    embedder.save_embeddings(embeddings, os.path.join(output_dir, "photo_embeddings.npz"))
