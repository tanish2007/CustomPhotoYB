"""
Supabase Storage Integration for Photo Selection App
Handles persistent storage of dataset metadata (not photos) in Supabase.
"""

import os
import json
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
    """Get the list of dataset names from the registry file."""
    try:
        storage = client.storage.from_(BUCKET_NAME)
        response = storage.download("_registry.json")
        registry = json.loads(response.decode('utf-8'))
        return registry.get('datasets', [])
    except Exception:
        # Registry doesn't exist yet
        return []


def _update_dataset_registry(client, dataset_name: str, action: str = 'add'):
    """Update the registry file with dataset names."""
    try:
        storage = client.storage.from_(BUCKET_NAME)

        # Get current registry
        datasets = _get_dataset_registry(client)

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
