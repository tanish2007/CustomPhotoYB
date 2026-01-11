# CustomYB - Smart Photo Selection System

An intelligent photo selection application that uses AI to automatically select the best photos from your collection based on quality, diversity, and temporal coverage.

## Features

- **Automatic Photo Selection**: AI-powered selection using CLIP embeddings and quality scoring
- **Smart Clustering**: Groups similar photos using HDBSCAN with temporal gap detection
- **Quality Scoring**: Multi-factor scoring system evaluating:
  - Face Quality (35%): Face detection, sharpness, size, positioning
  - Aesthetic Quality (25%): Sharpness, brightness, contrast, composition
  - Emotional Signal (20%): Facial expressions, interactions
  - Uniqueness (20%): Diversity within clusters
- **Temporal Segmentation**: Organizes photos by time to ensure coverage across different periods
- **Web Interface**: Easy-to-use Flask-based web UI
- **Flexible Selection Modes**:
  - Strict: 1 photo per cluster
  - Balanced: 2 photos per cluster (default)
  - Lenient: 3 photos per cluster

## Technology Stack

- **Backend**: Python, Flask
- **AI/ML**:
  - CLIP (sentence-transformers) for image embeddings
  - HDBSCAN for density-based clustering
  - OpenCV for face detection and image processing
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: Pillow, pillow-heif (HEIC support)

## Architecture

```
Photo Upload
    ↓
Temporal Segmentation (monthly/biweekly buckets)
    ↓
CLIP Embeddings (512-dim vectors)
    ↓
HDBSCAN Clustering (with temporal gap splitting)
    ↓
Quality Scoring (face, aesthetic, emotional, uniqueness)
    ↓
Selection (based on quality mode + similarity thresholds)
    ↓
Results (selected + rejected photos)
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tanish2007/CustomYB.git
cd CustomYB
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload photos and let the AI select the best ones!

### Configuration Options

You can customize the selection behavior by modifying parameters in `app.py`:

```python
# Clustering method
clusterer = PhotoClusterer(
    method="hdbscan",           # Clustering algorithm
    min_cluster_size=3,         # Minimum photos per cluster
    temporal_gap_hours=6.0      # Hours between events
)

# Selection mode
selector = SmartPhotoSelector(
    quality_mode="balanced",    # strict/balanced/lenient
    similarity_threshold=0.92   # Global similarity threshold
)
```

## Project Structure

```
CustomYB/
├── app.py                          # Main Flask application
├── photo_selector/
│   ├── __init__.py
│   ├── auto_selector.py           # Photo selection logic
│   ├── clustering.py              # HDBSCAN clustering + temporal splitting
│   ├── embeddings.py              # CLIP embedding generation
│   ├── scoring.py                 # Quality scoring system
│   └── temporal.py                # Temporal segmentation
├── static/
│   ├── css/
│   │   └── style.css              # Web UI styles
│   └── js/
│       └── upload.js              # Client-side upload logic
├── templates/
│   ├── index.html                 # Main upload page
│   └── results.html               # Results display page
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## How It Works

### 1. Temporal Segmentation
Photos are first organized into time buckets (monthly or biweekly) based on EXIF timestamps to ensure coverage across different time periods.

### 2. Embedding Generation
Each photo is converted into a 512-dimensional vector using CLIP that captures visual and semantic features.

### 3. Clustering
HDBSCAN groups visually similar photos together. Temporal gap splitting ensures photos taken hours apart (e.g., different events in the same location) are separated.

### 4. Quality Scoring
Each photo receives scores in four categories:
- **Face Quality**: Detects faces using OpenCV Haar Cascades and scores based on size, sharpness, and positioning
- **Aesthetic Quality**: Evaluates overall image quality (sharpness, lighting, composition)
- **Emotional Signal**: Assesses engagement and emotion in faces
- **Uniqueness**: Measures how different the photo is from others in its cluster

### 5. Selection
The system selects the best photos from each cluster while:
- Enforcing maximum photos per cluster (based on quality mode)
- Removing duplicates using 85% within-cluster similarity threshold
- Ensuring global diversity with 92% cross-cluster similarity threshold

## API Endpoints

- `GET /` - Main upload interface
- `POST /upload` - Upload photos for processing
- `GET /status/<session_id>` - Check processing status
- `GET /results/<session_id>` - Get selection results
- `GET /photo/<session_id>/<filename>` - Retrieve full-size photo
- `GET /thumbnail/<session_id>/<filename>` - Retrieve thumbnail
- `POST /cleanup/<session_id>` - Clean up session data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- CLIP model by OpenAI
- HDBSCAN algorithm by Leland McInnes
- OpenCV for computer vision capabilities

## Contact

For questions or issues, please open an issue on GitHub.
