// Photo Selection App JavaScript
// Automatic Selection Mode - AI decides which photos to keep

let selectedFiles = [];
let currentJobId = null;
let pollInterval = null;
let resultsData = null;
let currentQualityMode = 'balanced';

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const previewGrid = document.getElementById('preview-grid');
const fileCount = document.getElementById('file-count');
const similaritySlider = document.getElementById('similarity');
const similarityValue = document.getElementById('similarity-value');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupDropZone();
    setupFileInput();
    setupSimilaritySlider();
    setupQualityButtons();
});

// Drop Zone Setup
function setupDropZone() {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
}

// File Input Setup
function setupFileInput() {
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

// Similarity Slider
function setupSimilaritySlider() {
    similaritySlider.addEventListener('input', (e) => {
        const value = Math.round(parseFloat(e.target.value) * 100);
        similarityValue.textContent = value + '%';
    });
}

// Quality Mode Buttons Setup
function setupQualityButtons() {
    // Set initial state
    document.querySelectorAll('.quality-btn').forEach(btn => {
        if (btn.dataset.mode === 'balanced') {
            btn.classList.add('active');
        }
    });
}

// Set Quality Mode
function setQualityMode(mode) {
    currentQualityMode = mode;

    // Update button states
    document.querySelectorAll('.quality-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
}

// Handle Selected Files
function handleFiles(files) {
    const validFiles = Array.from(files).filter(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        return ['jpg', 'jpeg', 'png', 'heic', 'heif', 'webp'].includes(ext);
    });

    if (validFiles.length === 0) {
        alert('No valid image files selected.');
        return;
    }

    selectedFiles = [...selectedFiles, ...validFiles];
    updateFilePreview();
}

// Update File Preview
function updateFilePreview() {
    if (selectedFiles.length === 0) {
        filePreview.classList.add('hidden');
        return;
    }

    filePreview.classList.remove('hidden');
    fileCount.textContent = selectedFiles.length;
    previewGrid.innerHTML = '';

    selectedFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'preview-item';

        // Create thumbnail
        if (file.type.startsWith('image/') && !file.name.toLowerCase().endsWith('.heic')) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            item.appendChild(img);
        } else {
            // For HEIC or unknown types, show placeholder
            item.innerHTML = `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#334155;font-size:24px;">photo</div>`;
        }

        // Remove button
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '&times;';
        removeBtn.onclick = (e) => {
            e.stopPropagation();
            removeFile(index);
        };
        item.appendChild(removeBtn);

        previewGrid.appendChild(item);
    });
}

// Remove File
function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFilePreview();
}

// Start Processing
async function startProcessing() {
    if (selectedFiles.length === 0) {
        alert('Please select some photos first.');
        return;
    }

    const similarity = parseFloat(document.getElementById('similarity').value);

    // Show processing section
    document.getElementById('upload-section').classList.add('hidden');
    document.getElementById('processing-section').classList.remove('hidden');

    // Create form data
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    formData.append('quality_mode', currentQualityMode);
    formData.append('similarity', similarity);

    try {
        // Upload files
        updateProgress(5, 'Uploading files...');

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const data = await response.json();
        currentJobId = data.job_id;

        // Start polling for status
        pollInterval = setInterval(checkStatus, 1000);

    } catch (error) {
        console.error('Error:', error);
        alert('Error uploading files: ' + error.message);
        resetToUpload();
    }
}

// Check Processing Status
async function checkStatus() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`/status/${currentJobId}`);
        const data = await response.json();

        updateProgress(data.progress, data.message);
        updateProcessingSteps(data.progress);

        if (data.status === 'complete') {
            clearInterval(pollInterval);
            await loadResults();
        } else if (data.status === 'review_pending') {
            // Face filtering complete - redirect to review page
            clearInterval(pollInterval);
            window.location.href = `/step3_review/${currentJobId}`;
        } else if (data.status === 'error') {
            clearInterval(pollInterval);
            alert('Processing error: ' + data.message);
            resetToUpload();
        }

    } catch (error) {
        console.error('Status check error:', error);
    }
}

// Update Progress
function updateProgress(percent, message) {
    document.getElementById('progress-fill').style.width = `${percent}%`;
    document.getElementById('progress-percent').textContent = Math.round(percent);
    document.getElementById('processing-message').textContent = message || 'Processing...';
}

// Update Processing Steps
function updateProcessingSteps(progress) {
    // Check if we're in face filtering mode (has step-0) or full processing mode
    const hasFaceFiltering = document.getElementById('step-0') !== null;

    let steps;
    if (hasFaceFiltering) {
        // Face filtering mode: 3 steps (step-0, step-1, step-2)
        steps = [
            { id: 'step-0', threshold: 10 },
            { id: 'step-1', threshold: 70 },
            { id: 'step-2', threshold: 90 }
        ];
    } else {
        // Full processing mode: 5 steps (step-1 through step-5)
        steps = [
            { id: 'step-1', threshold: 20 },
            { id: 'step-2', threshold: 40 },
            { id: 'step-3', threshold: 50 },
            { id: 'step-4', threshold: 60 },
            { id: 'step-5', threshold: 80 }
        ];
    }

    steps.forEach((step, index) => {
        const el = document.getElementById(step.id);
        if (!el) return; // Skip if element doesn't exist

        if (progress >= step.threshold) {
            el.classList.add('complete');
            el.classList.remove('active');
        } else if (index === 0 || progress >= steps[index - 1].threshold) {
            el.classList.add('active');
            el.classList.remove('complete');
        } else {
            el.classList.remove('active', 'complete');
        }
    });
}

// Load Results
async function loadResults() {
    try {
        const response = await fetch(`/results/${currentJobId}`);
        resultsData = await response.json();

        displayResults(resultsData);

        document.getElementById('processing-section').classList.add('hidden');
        document.getElementById('results-section').classList.remove('hidden');

    } catch (error) {
        console.error('Error loading results:', error);
        alert('Error loading results');
        resetToUpload();
    }
}

// Display Results
function displayResults(data) {
    // Update stats
    const totalPhotos = data.summary.total_photos;
    const selectedCount = data.summary.selected_count;
    const rejectedCount = data.summary.rejected_count;
    const selectionRate = data.summary.selection_rate;

    document.getElementById('stat-total').textContent = totalPhotos;
    document.getElementById('stat-selected').textContent = selectedCount;
    document.getElementById('stat-rejected').textContent = rejectedCount;

    // Handle optional elements
    const statRate = document.getElementById('stat-rate');
    if (statRate) statRate.textContent = selectionRate;

    // Handle face filtering stats if available
    const faceFiltering = data.summary.face_filtering;
    const statFound = document.getElementById('stat-found');
    if (statFound && faceFiltering) {
        statFound.textContent = faceFiltering.after_face_filter || faceFiltering.user_confirmed || 0;
    }

    // Display rejection breakdown
    displayRejectionBreakdown(data.rejection_breakdown);

    // Display selected photos
    const selectedGrid = document.getElementById('selected-grid');
    selectedGrid.innerHTML = '';

    if (data.selected.length === 0) {
        selectedGrid.innerHTML = '<div class="no-photos">No photos selected. Try using "Keep More" mode.</div>';
    } else {
        data.selected.forEach(photo => {
            selectedGrid.appendChild(createPhotoCard(photo, false));
        });
    }

    // Display rejected photos
    const rejectedGrid = document.getElementById('rejected-grid');
    rejectedGrid.innerHTML = '';

    if (data.rejected.length === 0) {
        rejectedGrid.innerHTML = '<div class="no-photos">No photos filtered out.</div>';
    } else {
        data.rejected.forEach(photo => {
            rejectedGrid.appendChild(createPhotoCard(photo, true));
        });
    }
}

// Display Rejection Breakdown
function displayRejectionBreakdown(breakdown) {
    const container = document.getElementById('rejection-bars');
    const breakdownSection = document.getElementById('rejection-breakdown');

    if (!breakdown || Object.keys(breakdown).length === 0) {
        breakdownSection.classList.add('hidden');
        return;
    }

    breakdownSection.classList.remove('hidden');
    container.innerHTML = '';

    // Calculate total for percentages
    const total = Object.values(breakdown).reduce((sum, count) => sum + count, 0);

    // Sort by count descending
    const sortedReasons = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);

    // Color mapping for reasons
    const reasonColors = {
        'Quality below threshold': '#ef4444',
        'Too blurry': '#f59e0b',
        'Poor face quality': '#8b5cf6',
        'Too similar to another photo': '#3b82f6',
        'Better version exists in same group': '#6366f1'
    };

    sortedReasons.forEach(([reason, count]) => {
        const percent = Math.round((count / total) * 100);
        const color = reasonColors[reason] || '#64748b';

        const bar = document.createElement('div');
        bar.className = 'rejection-bar-item';
        bar.innerHTML = `
            <div class="rejection-bar-label">
                <span class="rejection-reason">${reason}</span>
                <span class="rejection-count">${count} photos (${percent}%)</span>
            </div>
            <div class="rejection-bar-track">
                <div class="rejection-bar-fill" style="width: ${percent}%; background: ${color}"></div>
            </div>
        `;
        container.appendChild(bar);
    });
}

// Create Photo Card
function createPhotoCard(photo, isRejected) {
    const card = document.createElement('div');
    card.className = `photo-card ${isRejected ? 'rejected' : ''}`;
    card.onclick = () => openModal(photo, isRejected);

    const score = photo.score || 0;
    const scorePercent = Math.round(score * 100);

    let reasonHtml = '';
    if (isRejected && photo.reason) {
        reasonHtml = `<div class="photo-reason rejected-reason">${photo.reason}</div>`;
    } else if (!isRejected && photo.selection_detail) {
        // Show selection reason for selected photos
        reasonHtml = `<div class="photo-reason selected-reason">${photo.selection_detail}</div>`;
    }

    card.innerHTML = `
        <img src="/thumbnail/${currentJobId}/${photo.thumbnail}" alt="${photo.filename}" loading="lazy">
        <div class="photo-info">
            <div class="photo-filename" title="${photo.filename}">${photo.filename}</div>
            <div class="photo-score">
                <div class="score-bar">
                    <div class="score-fill" style="width: ${scorePercent}%"></div>
                </div>
                <span class="score-value">${scorePercent}%</span>
            </div>
            ${reasonHtml}
        </div>
    `;

    return card;
}

// Open Modal
function openModal(photo, isRejected = false) {
    const modal = document.getElementById('photo-modal');
    const modalImage = document.getElementById('modal-image');
    const modalFilename = document.getElementById('modal-filename');
    const modalReason = document.getElementById('modal-reason');
    const modalScores = document.getElementById('modal-scores');

    modalImage.src = `/photo/${currentJobId}/${photo.filename}`;
    modalFilename.textContent = photo.filename;

    // Show selection/rejection reason
    if (isRejected && photo.reason) {
        modalReason.innerHTML = `<div class="reason-badge rejected">${photo.reason}</div>`;
        if (photo.reason_detail) {
            modalReason.innerHTML += `<div class="reason-detail">${photo.reason_detail}</div>`;
        }
        modalReason.classList.remove('hidden');
    } else if (!isRejected && photo.selection_detail) {
        // Show selection reason with detail
        modalReason.innerHTML = `<div class="reason-badge selected">✓ Selected</div>`;
        modalReason.innerHTML += `<div class="reason-detail">${photo.selection_detail}</div>`;
        modalReason.classList.remove('hidden');
    } else {
        modalReason.innerHTML = '<div class="reason-badge selected">✓ Selected by AI</div>';
        modalReason.classList.remove('hidden');
    }

    // Build score breakdown
    const scores = [
        { label: 'Overall Score', value: photo.score, color: '#6366f1' },
        { label: 'Face Quality', value: photo.face_quality, color: '#10b981' },
        { label: 'Aesthetic', value: photo.aesthetic_quality, color: '#f59e0b' },
        { label: 'Emotional', value: photo.emotional_signal, color: '#ef4444' },
        { label: 'Uniqueness', value: photo.uniqueness, color: '#8b5cf6' }
    ];

    modalScores.innerHTML = scores.map(s => {
        const percent = Math.round((s.value || 0) * 100);
        return `
            <div class="score-item">
                <div class="score-item-label">${s.label}</div>
                <div class="score-item-value" style="color: ${s.color}">${percent}%</div>
                <div class="score-item-bar">
                    <div class="score-item-fill" style="width: ${percent}%; background: ${s.color}"></div>
                </div>
            </div>
        `;
    }).join('');

    modal.classList.remove('hidden');
}

// Close Modal
function closeModal() {
    document.getElementById('photo-modal').classList.add('hidden');
}

// Close modal on background click
document.getElementById('photo-modal').addEventListener('click', (e) => {
    if (e.target.id === 'photo-modal') {
        closeModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
    }
});

// Switch Tab
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });
}

// Download Selected Photos
async function downloadSelected() {
    if (!currentJobId) return;

    try {
        window.location.href = `/download/${currentJobId}`;
    } catch (error) {
        alert('Error downloading files');
    }
}

// Reset App
async function resetApp() {
    if (currentJobId) {
        try {
            await fetch(`/cleanup/${currentJobId}`, { method: 'POST' });
        } catch (error) {
            console.error('Cleanup error:', error);
        }
    }

    resetToUpload();
}

// Reset to Upload State
function resetToUpload() {
    selectedFiles = [];
    currentJobId = null;
    resultsData = null;
    currentQualityMode = 'balanced';

    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }

    // Reset UI
    document.getElementById('upload-section').classList.remove('hidden');
    document.getElementById('processing-section').classList.add('hidden');
    document.getElementById('results-section').classList.add('hidden');

    // Reset file preview
    filePreview.classList.add('hidden');
    previewGrid.innerHTML = '';
    fileCount.textContent = '0';
    fileInput.value = '';

    // Reset quality mode buttons
    document.querySelectorAll('.quality-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === 'balanced');
    });

    // Reset progress
    updateProgress(0, '');

    // Reset processing steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'complete');
    });
}
