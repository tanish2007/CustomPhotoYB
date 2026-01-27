"""
Supabase Storage Integration for Photo Selection App
Handles persistent storage of dataset metadata (not photos) in Supabase.
Also provides global embedding cache for CLIP/SigLIP embeddings.
"""

import os
import json
import base64
import hashlib
import numpy as np
from typing import Optional, List, Dict, Any

# Supabase credentials
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://cqnyibiopjcwuxmyqbgy.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
BUCKET_NAME = 'datasets'

# Initialize Supabase client (lazy loading)
_supabase_client = None

def get_supabase_client():
    """Get or create Supabase client."""
    global _supabase_client

    if not SUPABASE_KEY:
        print("[Supabase] No SUPABASE_KEY found in environment")
        return None

    if _supabase_client is None:
        try:
            from supabase import create_client
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            print(f"[Supabase] Connected to {SUPABASE_URL}")
        except ImportError:
            print("[Supabase] supabase-py not installed. Run: pip install supabase")
            return None
        except Exception as e:
            print(f"[Supabase] Connection error: {e}")
            return None

    return _supabase_client


def is_supabase_available() -> bool:
    """Check if Supabase is configured and available."""
    return get_supabase_client() is not None


def _get_dataset_registry(client) -> List[str]:
    """
    Get the list of dataset names from the registry file.
    Returns None if there's an error reading (to prevent accidental overwrite).
    Returns [] only if file doesn't exist yet.
    """
    try:
        storage = client.storage.from_(BUCKET_NAME)
        response = storage.download("_registry.json")
        registry = json.loads(response.decode('utf-8'))
        return registry.get('datasets', [])
    except Exception as e:
        error_str = str(e).lower()
        # Only return empty if file doesn't exist (not for other errors)
        if 'not found' in error_str or '404' in error_str or 'does not exist' in error_str:
            print("[Supabase] Registry file doesn't exist yet, starting fresh")
            return []
        else:
            # For other errors, return None to prevent accidental overwrite
            print(f"[Supabase] ERROR reading registry: {e}")
            return None


def _update_dataset_registry(client, dataset_name: str, action: str = 'add'):
    """Update the registry file with dataset names."""
    try:
        storage = client.storage.from_(BUCKET_NAME)

        # Get current registry
        datasets = _get_dataset_registry(client)

        # If we couldn't read the registry (error, not "not found"), don't overwrite
        if datasets is None:
            print(f"[Supabase] Skipping registry update - couldn't read existing registry safely")
            return

        if action == 'add' and dataset_name not in datasets:
            datasets.append(dataset_name)
        elif action == 'remove' and dataset_name in datasets:
            datasets.remove(dataset_name)
        else:
            return  # No changes needed

        # Save updated registry
        registry_data = json.dumps({'datasets': datasets}, indent=2).encode('utf-8')

        # Try to update (upsert)
        try:
            storage.update(
                path="_registry.json",
                file=registry_data,
                file_options={"content-type": "application/json"}
            )
        except Exception:
            # File doesn't exist, create it
            storage.upload(
                path="_registry.json",
                file=registry_data,
                file_options={"content-type": "application/json"}
            )

        print(f"[Supabase] Registry updated: {action} '{dataset_name}'")
    except Exception as e:
        print(f"[Supabase] Error updating registry: {e}")


def save_dataset_to_supabase(
    dataset_name: str,
    embeddings_data: bytes,
    face_results: dict,
    metadata: dict
) -> bool:
    """
    Save dataset files to Supabase Storage.

    Args:
        dataset_name: Unique name for the dataset (folder name)
        embeddings_data: Binary data of reference_embeddings.npz
        face_results: Dictionary of face detection results
        metadata: Dataset metadata dictionary

    Returns:
        True if successful, False otherwise
    """
    client = get_supabase_client()
    if not client:
        print("[Supabase] Client not available, skipping cloud save")
        return False

    try:
        # 1. Upload reference embeddings (.npz file)
        embeddings_path = f"{dataset_name}/reference_embeddings.npz"
        result = client.storage.from_(BUCKET_NAME).upload(
            path=embeddings_path,
            file=embeddings_data,
            file_options={"content-type": "application/octet-stream"}
        )
        print(f"[Supabase] Uploaded {embeddings_path}: {result}")

        # 2. Upload face results (JSON)
        face_results_path = f"{dataset_name}/face_results.json"
        face_results_bytes = json.dumps(face_results, indent=2).encode('utf-8')
        result = client.storage.from_(BUCKET_NAME).upload(
            path=face_results_path,
            file=face_results_bytes,
            file_options={"content-type": "application/json"}
        )
        print(f"[Supabase] Uploaded {face_results_path}: {result}")

        # 3. Upload metadata (JSON)
        metadata_path = f"{dataset_name}/metadata.json"
        metadata_bytes = json.dumps(metadata, indent=2).encode('utf-8')
        result = client.storage.from_(BUCKET_NAME).upload(
            path=metadata_path,
            file=metadata_bytes,
            file_options={"content-type": "application/json"}
        )
        print(f"[Supabase] Uploaded {metadata_path}")

        # 4. Update the registry file (list of all dataset names)
        _update_dataset_registry(client, dataset_name, action='add')

        print(f"[Supabase] Dataset '{dataset_name}' saved successfully")
        return True

    except Exception as e:
        print(f"[Supabase] Error saving dataset: {e}")
        return False


def load_dataset_from_supabase(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Load dataset files from Supabase Storage.

    Args:
        dataset_name: Name of the dataset to load

    Returns:
        Dictionary with 'embeddings_data', 'face_results', 'metadata' or None if failed
    """
    client = get_supabase_client()
    if not client:
        print("[Supabase] Client not available")
        return None

    try:
        result = {}

        # 1. Download reference embeddings
        embeddings_path = f"{dataset_name}/reference_embeddings.npz"
        response = client.storage.from_(BUCKET_NAME).download(embeddings_path)
        result['embeddings_data'] = response
        print(f"[Supabase] Downloaded {embeddings_path}")

        # 2. Download face results
        face_results_path = f"{dataset_name}/face_results.json"
        response = client.storage.from_(BUCKET_NAME).download(face_results_path)
        result['face_results'] = json.loads(response.decode('utf-8'))
        print(f"[Supabase] Downloaded {face_results_path}")

        # 3. Download metadata
        metadata_path = f"{dataset_name}/metadata.json"
        response = client.storage.from_(BUCKET_NAME).download(metadata_path)
        result['metadata'] = json.loads(response.decode('utf-8'))
        print(f"[Supabase] Downloaded {metadata_path}")

        print(f"[Supabase] Dataset '{dataset_name}' loaded successfully")
        return result

    except Exception as e:
        print(f"[Supabase] Error loading dataset: {e}")
        return None


def list_datasets_from_supabase() -> List[Dict[str, Any]]:
    """
    List all datasets stored in Supabase.

    Returns:
        List of dataset metadata dictionaries
    """
    client = get_supabase_client()
    if not client:
        print("[Supabase] Client not available")
        return []

    try:
        storage = client.storage.from_(BUCKET_NAME)

        # Get dataset names from registry
        dataset_names = _get_dataset_registry(client)
        print(f"[Supabase] Registry contains: {dataset_names}")

        # If registry read failed (None), return empty to be safe
        if dataset_names is None:
            print("[Supabase] Could not read registry, returning empty list")
            return []

        # If registry is empty, try to find existing datasets by checking known names
        # This handles the case where datasets were saved before registry was implemented
        if not dataset_names:
            print("[Supabase] Registry empty, checking for existing datasets...")
            # Try some known/common dataset names
            potential_names = ['testing']
            for name in potential_names:
                try:
                    storage.download(f"{name}/metadata.json")
                    dataset_names.append(name)
                    print(f"[Supabase] Found existing dataset: {name}")
                except Exception:
                    pass

        datasets = []
        for folder_name in dataset_names:
            try:
                metadata_path = f"{folder_name}/metadata.json"
                metadata_response = storage.download(metadata_path)
                metadata = json.loads(metadata_response.decode('utf-8'))
                metadata['folder_name'] = folder_name
                metadata['source'] = 'supabase'
                datasets.append(metadata)
                print(f"[Supabase] Loaded metadata for {folder_name}")
            except Exception as e:
                print(f"[Supabase] Could not load metadata for {folder_name}: {e}")
                # Add basic info without full metadata
                datasets.append({
                    'name': folder_name,
                    'folder_name': folder_name,
                    'source': 'supabase',
                    'total_photos': 0,
                    'created_at': None
                })

        print(f"[Supabase] Found {len(datasets)} datasets")
        return datasets

    except Exception as e:
        print(f"[Supabase] Error listing datasets: {e}")
        import traceback
        traceback.print_exc()
        return []


def delete_dataset_from_supabase(dataset_name: str) -> bool:
    """
    Delete a dataset from Supabase Storage.

    Args:
        dataset_name: Name of the dataset to delete

    Returns:
        True if successful, False otherwise
    """
    client = get_supabase_client()
    if not client:
        print("[Supabase] Client not available")
        return False

    try:
        # List all files in the dataset folder
        files = client.storage.from_(BUCKET_NAME).list(dataset_name)

        # Delete each file
        file_paths = [f"{dataset_name}/{f['name']}" for f in files if f.get('name')]

        if file_paths:
            client.storage.from_(BUCKET_NAME).remove(file_paths)
            print(f"[Supabase] Deleted {len(file_paths)} files from '{dataset_name}'")

        # Remove from registry
        _update_dataset_registry(client, dataset_name, action='remove')

        print(f"[Supabase] Dataset '{dataset_name}' deleted successfully")
        return True

    except Exception as e:
        print(f"[Supabase] Error deleting dataset: {e}")
        return False


def check_dataset_exists_in_supabase(dataset_name: str) -> bool:
    """
    Check if a dataset exists in Supabase.

    Args:
        dataset_name: Name of the dataset to check

    Returns:
        True if exists, False otherwise
    """
    client = get_supabase_client()
    if not client:
        return False

    try:
        # Try to list files in the dataset folder
        files = client.storage.from_(BUCKET_NAME).list(dataset_name)
        return len(files) > 0
    except:
        return False


# =============================================================================
# GLOBAL EMBEDDING CACHE
# =============================================================================
# Stores CLIP/SigLIP embeddings in Supabase database table for reuse.
# Table schema (create in Supabase Dashboard):
#
# CREATE TABLE image_embeddings (
#     id BIGSERIAL PRIMARY KEY,
#     image_hash TEXT NOT NULL,
#     embedding_model TEXT NOT NULL,
#     embedding TEXT NOT NULL,
#     embedding_dim INTEGER NOT NULL,
#     created_at TIMESTAMPTZ DEFAULT NOW(),
#     UNIQUE(image_hash, embedding_model)
# );
# CREATE INDEX idx_image_embeddings_hash_model ON image_embeddings(image_hash, embedding_model);
# =============================================================================

EMBEDDING_TABLE = 'image_embeddings'


def compute_file_hash(filepath: str) -> Optional[str]:
    """
    Compute MD5 hash of a file.

    Args:
        filepath: Path to the image file

    Returns:
        MD5 hash string or None if error
    """
    try:
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Read in chunks for memory efficiency
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        print(f"[EmbeddingCache] Error hashing {filepath}: {e}")
        return None


def _embedding_to_base64(embedding: np.ndarray) -> str:
    """Convert numpy embedding to base64 string for storage."""
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode('utf-8')


def _base64_to_embedding(b64_str: str, dim: int) -> np.ndarray:
    """Convert base64 string back to numpy embedding."""
    bytes_data = base64.b64decode(b64_str)
    return np.frombuffer(bytes_data, dtype=np.float32).reshape(dim)


def get_cached_embeddings_batch(
    image_hashes: List[str],
    embedding_model: str
) -> Dict[str, np.ndarray]:
    """
    Get cached embeddings for multiple images in one query.

    Args:
        image_hashes: List of MD5 hashes to look up
        embedding_model: Model name ('siglip' or 'clip')

    Returns:
        Dict mapping hash -> embedding for found entries
    """
    client = get_supabase_client()
    if not client or not image_hashes:
        return {}

    try:
        # Query all hashes at once
        response = client.table(EMBEDDING_TABLE).select(
            'image_hash, embedding, embedding_dim'
        ).in_('image_hash', image_hashes).eq('embedding_model', embedding_model).execute()

        result = {}
        for row in response.data:
            embedding = _base64_to_embedding(row['embedding'], row['embedding_dim'])
            result[row['image_hash']] = embedding

        print(f"[EmbeddingCache] Found {len(result)}/{len(image_hashes)} cached embeddings for {embedding_model}")
        return result

    except Exception as e:
        print(f"[EmbeddingCache] Error fetching batch: {e}")
        return {}


def save_embeddings_batch(
    embeddings: Dict[str, np.ndarray],
    image_hashes: Dict[str, str],
    embedding_model: str
) -> int:
    """
    Save multiple embeddings to cache.

    Args:
        embeddings: Dict mapping filename -> embedding
        image_hashes: Dict mapping filename -> hash
        embedding_model: Model name ('siglip' or 'clip')

    Returns:
        Number of embeddings saved
    """
    client = get_supabase_client()
    if not client or not embeddings:
        return 0

    try:
        # Prepare batch insert data
        rows = []
        for filename, embedding in embeddings.items():
            img_hash = image_hashes.get(filename)
            if img_hash and embedding is not None:
                rows.append({
                    'image_hash': img_hash,
                    'embedding_model': embedding_model,
                    'embedding': _embedding_to_base64(embedding),
                    'embedding_dim': len(embedding)
                })

        if not rows:
            return 0

        # Insert with upsert (ignore conflicts)
        # Batch in chunks of 100 to avoid request size limits
        saved = 0
        chunk_size = 100
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            try:
                client.table(EMBEDDING_TABLE).upsert(
                    chunk,
                    on_conflict='image_hash,embedding_model'
                ).execute()
                saved += len(chunk)
            except Exception as e:
                print(f"[EmbeddingCache] Error saving chunk {i//chunk_size}: {e}")

        print(f"[EmbeddingCache] Saved {saved} new embeddings for {embedding_model}")
        return saved

    except Exception as e:
        print(f"[EmbeddingCache] Error saving batch: {e}")
        return 0


def get_embedding_cache_stats() -> Dict[str, Any]:
    """Get statistics about the embedding cache."""
    client = get_supabase_client()
    if not client:
        return {'available': False}

    try:
        # Count by model
        response = client.table(EMBEDDING_TABLE).select(
            'embedding_model',
            count='exact'
        ).execute()

        # Get counts per model
        siglip_count = client.table(EMBEDDING_TABLE).select(
            'id', count='exact'
        ).eq('embedding_model', 'siglip').execute()

        clip_count = client.table(EMBEDDING_TABLE).select(
            'id', count='exact'
        ).eq('embedding_model', 'clip').execute()

        return {
            'available': True,
            'siglip_count': siglip_count.count or 0,
            'clip_count': clip_count.count or 0,
            'total': (siglip_count.count or 0) + (clip_count.count or 0)
        }

    except Exception as e:
        print(f"[EmbeddingCache] Error getting stats: {e}")
        return {'available': False, 'error': str(e)}
