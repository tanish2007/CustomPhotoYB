"""
CLIP embeddings for photo clustering.
CLIP (Contrastive Language-Image Pre-training) by OpenAI.

Uses ViT-B/32 by default (512-dim embeddings)
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional

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


class CLIPEmbedder:
    """Generate CLIP embeddings for photos."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the CLIP model.

        Args:
            model_name: CLIP model variant. Options:
                - "ViT-B/32" (512-dim, fastest)
                - "ViT-B/16" (512-dim, better quality)
                - "ViT-L/14" (768-dim, best quality)
                - "ViT-L/14@336px" (768-dim, highest resolution)
            device: 'cuda' or 'cpu', auto-detected if None
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is required. Install with: pip install git+https://github.com/openai/CLIP.git")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on {self.device}...")

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.embedding_dim = self.model.visual.output_dim
        self.model_name = model_name

        print(f"CLIP loaded. Embedding dimension: {self.embedding_dim}")

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

    def get_embeddings_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Get CLIP embeddings for a batch of images."""
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            with torch.no_grad():
                # Preprocess all images in batch
                image_inputs = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)
                embeddings = self.model.encode_image(image_inputs)

                # Normalize
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def process_folder(self, folder_path: str,
                       image_extensions: set = None,
                       batch_size: int = 32,
                       use_batching: bool = True) -> Dict[str, np.ndarray]:
        """
        Process all images in a folder and generate embeddings.

        Args:
            folder_path: Path to folder containing images
            image_extensions: Set of valid extensions
            batch_size: Number of images to process at once
            use_batching: Whether to use batch processing (faster but more memory)

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

        if use_batching and len(image_files) > batch_size:
            # Batch processing for efficiency
            print(f"Using batch processing (batch_size={batch_size})...")

            for batch_start in range(0, len(image_files), batch_size):
                batch_end = min(batch_start + batch_size, len(image_files))
                batch_files = image_files[batch_start:batch_end]

                print(f"Processing batch [{batch_start+1}-{batch_end}/{len(image_files)}]")

                batch_images = []
                batch_names = []

                for image_path in batch_files:
                    try:
                        img = self.load_image(str(image_path))
                        if img is not None:
                            batch_images.append(img)
                            batch_names.append(image_path.name)
                    except Exception as e:
                        errors.append((image_path.name, str(e)))

                if batch_images:
                    try:
                        batch_embeddings = self.get_embeddings_batch(batch_images)
                        for name, emb in zip(batch_names, batch_embeddings):
                            embeddings[name] = emb
                    except Exception as e:
                        print(f"Batch processing failed, falling back to individual: {e}")
                        for img, name in zip(batch_images, batch_names):
                            try:
                                embeddings[name] = self.get_embedding(img)
                            except Exception as e2:
                                errors.append((name, str(e2)))

                    # Close images
                    for img in batch_images:
                        img.close()
        else:
            # Individual processing
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
        data = {
            'filenames': list(embeddings.keys()),
            'embeddings': np.array(list(embeddings.values())),
            'model': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        np.savez(output_path, **data)
        print(f"Saved CLIP embeddings to {output_path}")

    @staticmethod
    def load_embeddings(input_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from a numpy file."""
        data = np.load(input_path, allow_pickle=True)
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
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        print("Usage: python clip_embeddings.py <folder_path>")
        print("\nThis will generate CLIP embeddings for all images in the folder.")
        sys.exit(0)

    embedder = CLIPEmbedder()
    embeddings = embedder.process_folder(folder)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    embedder.save_embeddings(embeddings, os.path.join(output_dir, "clip_embeddings.npz"))
