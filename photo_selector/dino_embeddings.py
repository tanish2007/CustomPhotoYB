"""
EXPERIMENTAL: DINOv2 embeddings for scene-based clustering

DINOv2 is a self-supervised vision model that focuses on visual structure,
textures, and spatial layout rather than semantic concepts like CLIP.

Hypothesis: DINOv2 might cluster photos by "visual scene" (background,
environment) rather than by "person identity" like CLIP does.

Usage:
    python dino_embeddings.py <folder_path>

This will:
1. Generate DINOv2 embeddings for all photos
2. Generate CLIP embeddings for comparison
3. Cluster both with HDBSCAN
4. Generate an HTML visualization and open it in browser
"""

import os
import sys
import base64
import webbrowser
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import List, Dict, Optional, Tuple
from io import BytesIO
import json

try:
    import hdbscan
except ImportError:
    print("hdbscan not installed. Run: pip install hdbscan")
    sys.exit(1)

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class DinoEmbedder:
    """Generate DINOv2 embeddings for photos."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None):
        """
        Initialize the DINOv2 model.

        Args:
            model_name: HuggingFace model name (dinov2-small, dinov2-base, dinov2-large)
            device: 'cuda' or 'cpu', auto-detected if None
        """
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError("transformers is required. Install with: pip install transformers")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv2 model '{model_name}' on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # DINOv2-base has 768-dim embeddings
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess an image."""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Get DINOv2 embedding for a single image."""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :]
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()

    def process_folder(self, folder_path: str,
                       image_extensions: set = None) -> Dict[str, np.ndarray]:
        """
        Process all images in a folder and generate embeddings.

        Args:
            folder_path: Path to folder containing images
            image_extensions: Set of valid extensions

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

    def save_embeddings(self, embeddings: Dict[str, np.ndarray], output_path: str):
        """Save embeddings to a numpy file."""
        data = {
            'filenames': list(embeddings.keys()),
            'embeddings': np.array(list(embeddings.values()))
        }
        np.savez(output_path, **data)
        print(f"Saved DINOv2 embeddings to {output_path}")

    @staticmethod
    def load_embeddings(input_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from a numpy file."""
        data = np.load(input_path)
        filenames = data['filenames']
        embeddings_array = data['embeddings']
        return {fn: emb for fn, emb in zip(filenames, embeddings_array)}


def cluster_embeddings(embeddings: Dict[str, np.ndarray],
                       min_cluster_size: int = 3,
                       epsilon: float = 0.3) -> Dict[str, int]:
    """
    Cluster embeddings using HDBSCAN.

    Returns:
        Dictionary mapping filename to cluster_id (-1 = noise)
    """
    filenames = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[fn] for fn in filenames])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=epsilon
    )

    labels = clusterer.fit_predict(embedding_matrix)

    return {fn: int(label) for fn, label in zip(filenames, labels)}


def create_thumbnail_base64(image_path: str, max_size: int = 150) -> str:
    """Create a base64-encoded thumbnail for embedding in HTML."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Create thumbnail
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=80)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        img.close()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")
        return ""


def generate_html_comparison(folder_path: str,
                             clip_clusters: Dict[str, int],
                             dino_clusters: Dict[str, int],
                             output_path: str) -> str:
    """
    Generate an HTML file with visual comparison of CLIP vs DINOv2 clustering.

    Returns:
        Path to the generated HTML file
    """
    folder = Path(folder_path)

    # Get all filenames
    all_files = list(set(clip_clusters.keys()) | set(dino_clusters.keys()))

    # Generate thumbnails
    print("\nGenerating thumbnails for HTML visualization...")
    thumbnails = {}
    for i, fn in enumerate(all_files):
        if (i + 1) % 20 == 0:
            print(f"  Thumbnails [{i+1}/{len(all_files)}]")
        img_path = folder / fn
        if img_path.exists():
            thumbnails[fn] = create_thumbnail_base64(str(img_path))

    # Group by clusters
    def group_by_cluster(clusters: Dict[str, int]) -> Dict[int, List[str]]:
        groups = {}
        for fn, cid in clusters.items():
            if cid not in groups:
                groups[cid] = []
            groups[cid].append(fn)
        # Sort files within each cluster
        for cid in groups:
            groups[cid].sort()
        return groups

    clip_groups = group_by_cluster(clip_clusters)
    dino_groups = group_by_cluster(dino_clusters)

    # Prepare data for JavaScript
    photo_data = {}
    for fn in all_files:
        photo_data[fn] = {
            'thumbnail': thumbnails.get(fn, ''),
            'clip_cluster': clip_clusters.get(fn, -1),
            'dino_cluster': dino_clusters.get(fn, -1)
        }

    # Stats
    clip_cluster_count = len([c for c in clip_groups.keys() if c != -1])
    dino_cluster_count = len([c for c in dino_groups.keys() if c != -1])
    clip_noise = len(clip_groups.get(-1, []))
    dino_noise = len(dino_groups.get(-1, []))

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CLIP vs DINOv2 Clustering Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
            color: #333;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px 20px;
            text-align: center;
            color: white;
        }}

        .header h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .header p {{
            opacity: 0.9;
            font-size: 14px;
        }}

        .stats-row {{
            display: flex;
            justify-content: center;
            gap: 20px;
            padding: 20px;
            background: white;
            border-bottom: 1px solid #e5e7eb;
            flex-wrap: wrap;
        }}

        .stat-card {{
            text-align: center;
            padding: 15px 25px;
            background: #f9fafb;
            border-radius: 10px;
        }}

        .stat-card.clip {{
            border: 2px solid #3b82f6;
        }}

        .stat-card.dino {{
            border: 2px solid #10b981;
        }}

        .stat-value {{
            font-size: 28px;
            font-weight: 700;
        }}

        .stat-card.clip .stat-value {{
            color: #3b82f6;
        }}

        .stat-card.dino .stat-value {{
            color: #10b981;
        }}

        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }}

        .tabs {{
            display: flex;
            justify-content: center;
            gap: 10px;
            padding: 15px;
            background: white;
            border-bottom: 1px solid #e5e7eb;
        }}

        .tab {{
            padding: 10px 25px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            background: #f0f0f0;
            color: #666;
        }}

        .tab:hover {{
            background: #e0e0e0;
        }}

        .tab.active {{
            color: white;
        }}

        .tab.active.clip-tab {{
            background: #3b82f6;
        }}

        .tab.active.dino-tab {{
            background: #10b981;
        }}

        .tab.active.side-tab {{
            background: #8b5cf6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .view {{
            display: none;
        }}

        .view.active {{
            display: block;
        }}

        .cluster-card {{
            background: white;
            border-radius: 12px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        .cluster-header {{
            padding: 12px 16px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .cluster-header.clip {{
            background: #3b82f6;
        }}

        .cluster-header.dino {{
            background: #10b981;
        }}

        .cluster-header.noise {{
            background: #6b7280;
        }}

        .cluster-title {{
            font-weight: 600;
            font-size: 14px;
        }}

        .cluster-count {{
            font-size: 13px;
            opacity: 0.9;
        }}

        .cluster-photos {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            padding: 15px;
        }}

        .photo-card {{
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid transparent;
        }}

        .photo-card:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}

        .photo-card img {{
            width: 100%;
            height: 100px;
            object-fit: cover;
            display: block;
        }}

        .photo-card .other-cluster {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.7);
            color: white;
            font-size: 10px;
            padding: 3px 6px;
            text-align: center;
        }}

        .photo-card.highlighted {{
            border-color: #fbbf24;
            box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.5);
        }}

        /* Side by Side View */
        .side-by-side {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .side-column {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
        }}

        .side-column-header {{
            padding: 15px;
            color: white;
            text-align: center;
            font-weight: 600;
        }}

        .side-column-header.clip {{
            background: #3b82f6;
        }}

        .side-column-header.dino {{
            background: #10b981;
        }}

        .side-column-content {{
            padding: 15px;
            max-height: 70vh;
            overflow-y: auto;
        }}

        /* Modal */
        .modal {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }}

        .modal.active {{
            display: flex;
        }}

        .modal-content {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            max-width: 500px;
            text-align: center;
        }}

        .modal-content img {{
            max-width: 300px;
            max-height: 300px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}

        .modal-content h3 {{
            font-size: 14px;
            margin-bottom: 15px;
            word-break: break-all;
        }}

        .modal-clusters {{
            display: flex;
            justify-content: center;
            gap: 20px;
        }}

        .modal-cluster {{
            padding: 10px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
        }}

        .modal-cluster.clip {{
            background: #3b82f6;
        }}

        .modal-cluster.dino {{
            background: #10b981;
        }}

        .modal-close {{
            margin-top: 20px;
            padding: 10px 30px;
            background: #e0e0e0;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
        }}

        @media (max-width: 768px) {{
            .side-by-side {{
                grid-template-columns: 1fr;
            }}

            .cluster-photos {{
                grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
            }}

            .photo-card img {{
                height: 70px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CLIP vs DINOv2 Clustering Comparison</h1>
        <p>Experiment: Does DINOv2 cluster by scene instead of person identity?</p>
    </div>

    <div class="stats-row">
        <div class="stat-card clip">
            <div class="stat-value">{clip_cluster_count}</div>
            <div class="stat-label">CLIP Clusters</div>
        </div>
        <div class="stat-card clip">
            <div class="stat-value">{clip_noise}</div>
            <div class="stat-label">CLIP Noise</div>
        </div>
        <div class="stat-card dino">
            <div class="stat-value">{dino_cluster_count}</div>
            <div class="stat-label">DINOv2 Clusters</div>
        </div>
        <div class="stat-card dino">
            <div class="stat-value">{dino_noise}</div>
            <div class="stat-label">DINOv2 Noise</div>
        </div>
    </div>

    <div class="tabs">
        <button class="tab clip-tab active" onclick="showView('clip')">CLIP Clusters</button>
        <button class="tab dino-tab" onclick="showView('dino')">DINOv2 Clusters</button>
        <button class="tab side-tab" onclick="showView('side')">Side by Side</button>
    </div>

    <div class="container">
        <div id="clip-view" class="view active"></div>
        <div id="dino-view" class="view"></div>
        <div id="side-view" class="view"></div>
    </div>

    <!-- Modal -->
    <div id="photo-modal" class="modal" onclick="closeModal()">
        <div class="modal-content" onclick="event.stopPropagation()">
            <img id="modal-img" src="" alt="">
            <h3 id="modal-filename"></h3>
            <div class="modal-clusters">
                <div class="modal-cluster clip">
                    CLIP: <span id="modal-clip-cluster"></span>
                </div>
                <div class="modal-cluster dino">
                    DINOv2: <span id="modal-dino-cluster"></span>
                </div>
            </div>
            <button class="modal-close" onclick="closeModal()">Close</button>
        </div>
    </div>

    <script>
        const photoData = {json.dumps(photo_data)};
        const clipGroups = {json.dumps({str(k): v for k, v in clip_groups.items()})};
        const dinoGroups = {json.dumps({str(k): v for k, v in dino_groups.items()})};

        function showView(view) {{
            // Update tabs
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab.${{view}}-tab`).classList.add('active');

            // Update views
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.getElementById(`${{view}}-view`).classList.add('active');
        }}

        function renderClusters(groups, type, containerId) {{
            const container = document.getElementById(containerId);
            container.innerHTML = '';

            // Sort clusters by ID (noise last)
            const sortedIds = Object.keys(groups).map(Number).sort((a, b) => {{
                if (a === -1) return 1;
                if (b === -1) return -1;
                return a - b;
            }});

            for (const clusterId of sortedIds) {{
                const files = groups[clusterId];
                const isNoise = clusterId === -1;

                const card = document.createElement('div');
                card.className = 'cluster-card';

                const headerClass = isNoise ? 'noise' : type;
                const title = isNoise ? 'Noise (Unclustered)' : `Cluster #${{clusterId}}`;

                card.innerHTML = `
                    <div class="cluster-header ${{headerClass}}">
                        <span class="cluster-title">${{title}}</span>
                        <span class="cluster-count">${{files.length}} photos</span>
                    </div>
                    <div class="cluster-photos" id="${{type}}-cluster-${{clusterId}}"></div>
                `;

                container.appendChild(card);

                const photosContainer = card.querySelector('.cluster-photos');
                for (const fn of files) {{
                    const data = photoData[fn];
                    if (!data || !data.thumbnail) continue;

                    const otherCluster = type === 'clip' ? data.dino_cluster : data.clip_cluster;
                    const otherType = type === 'clip' ? 'DINO' : 'CLIP';

                    const photoCard = document.createElement('div');
                    photoCard.className = 'photo-card';
                    photoCard.onclick = () => openModal(fn);
                    photoCard.innerHTML = `
                        <img src="${{data.thumbnail}}" alt="${{fn}}">
                        <div class="other-cluster">${{otherType}}: #${{otherCluster}}</div>
                    `;
                    photosContainer.appendChild(photoCard);
                }}
            }}
        }}

        function renderSideBySide() {{
            const container = document.getElementById('side-view');
            container.innerHTML = `
                <div class="side-by-side">
                    <div class="side-column">
                        <div class="side-column-header clip">CLIP Clusters (Semantic)</div>
                        <div class="side-column-content" id="side-clip"></div>
                    </div>
                    <div class="side-column">
                        <div class="side-column-header dino">DINOv2 Clusters (Visual)</div>
                        <div class="side-column-content" id="side-dino"></div>
                    </div>
                </div>
            `;

            renderClusters(clipGroups, 'clip', 'side-clip');
            renderClusters(dinoGroups, 'dino', 'side-dino');
        }}

        function openModal(filename) {{
            const data = photoData[filename];
            document.getElementById('modal-img').src = data.thumbnail;
            document.getElementById('modal-filename').textContent = filename;
            document.getElementById('modal-clip-cluster').textContent =
                data.clip_cluster === -1 ? 'Noise' : `#${{data.clip_cluster}}`;
            document.getElementById('modal-dino-cluster').textContent =
                data.dino_cluster === -1 ? 'Noise' : `#${{data.dino_cluster}}`;
            document.getElementById('photo-modal').classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('photo-modal').classList.remove('active');
        }}

        // Initialize
        renderClusters(clipGroups, 'clip', 'clip-view');
        renderClusters(dinoGroups, 'dino', 'dino-view');
        renderSideBySide();

        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
        }});
    </script>
</body>
</html>
'''

    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nGenerated HTML visualization: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python dino_embeddings.py <folder_path>")
        print("\nThis will compare CLIP vs DINOv2 clustering on your photos.")
        print("Requirements: pip install transformers hdbscan")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory")
        sys.exit(1)

    # Generate DINOv2 embeddings
    print("\n" + "="*50)
    print("STEP 1: Generating DINOv2 embeddings...")
    print("="*50)
    dino_embedder = DinoEmbedder()
    dino_embeddings = dino_embedder.process_folder(folder)

    if len(dino_embeddings) == 0:
        print("No images processed. Exiting.")
        sys.exit(1)

    # Generate CLIP embeddings for comparison
    print("\n" + "="*50)
    print("STEP 2: Generating CLIP embeddings for comparison...")
    print("="*50)
    clip_embeddings = None
    try:
        from embeddings import PhotoEmbedder
        clip_embedder = PhotoEmbedder()
        clip_embeddings = clip_embedder.process_folder(folder)
    except Exception as e:
        print(f"CLIP embedder not available: {e}")
        print("Will only show DINOv2 clusters.")

    # Cluster both
    print("\n" + "="*50)
    print("STEP 3: Clustering embeddings...")
    print("="*50)

    n_photos = len(dino_embeddings)
    min_cluster_size = max(2, min(int(0.02 * n_photos), 8))
    print(f"Using min_cluster_size={min_cluster_size} for {n_photos} photos")

    dino_clusters = cluster_embeddings(dino_embeddings, min_cluster_size=min_cluster_size)

    if clip_embeddings:
        clip_clusters = cluster_embeddings(clip_embeddings, min_cluster_size=min_cluster_size)
    else:
        # Use empty clusters if CLIP not available
        clip_clusters = {fn: -1 for fn in dino_embeddings.keys()}

    # Generate HTML visualization
    print("\n" + "="*50)
    print("STEP 4: Generating HTML visualization...")
    print("="*50)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(output_dir, "cluster_comparison.html")

    generate_html_comparison(folder, clip_clusters, dino_clusters, html_path)

    # Save embeddings
    dino_embedder.save_embeddings(dino_embeddings, os.path.join(output_dir, "dino_embeddings.npz"))

    # Open in browser
    print("\nOpening visualization in browser...")
    webbrowser.open(f"file://{html_path}")

    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    print(f"\nVisualization: {html_path}")
    print("\nWhat to look for:")
    print("  - If DINOv2 has MORE clusters: It's finding more distinct visual scenes")
    print("  - If CLIP cluster gets SPLIT by DINOv2: Same person in different places")
    print("  - If DINOv2 clusters same-background photos: It's working as expected")


if __name__ == "__main__":
    main()
