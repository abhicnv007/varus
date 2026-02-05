/**
 * calTAD Landmark Annotation Tool
 */

// Landmark definitions
const LANDMARKS = {
    ap: [
        { id: 'screw_tip', name: 'Screw Tip', description: 'End of lag screw inside femoral head' },
        { id: 'calcar_point_1', name: 'Calcar Point 1', description: 'First point on medial femoral neck cortex' },
        { id: 'calcar_point_2', name: 'Calcar Point 2', description: 'Second point on medial cortex' },
        { id: 'screw_edge_1', name: 'Screw Edge 1', description: 'Point on one edge of screw shaft' },
        { id: 'screw_edge_2', name: 'Screw Edge 2', description: 'Point on opposite edge of screw shaft' },
    ],
    lateral: [
        { id: 'screw_tip', name: 'Screw Tip', description: 'End of lag screw inside femoral head' },
        { id: 'femoral_head_apex', name: 'Femoral Head Apex', description: 'Superior-most point of femoral head' },
        { id: 'screw_edge_1', name: 'Screw Edge 1', description: 'Point on one edge of screw shaft' },
        { id: 'screw_edge_2', name: 'Screw Edge 2', description: 'Point on opposite edge of screw shaft' },
    ]
};

// Colors for landmarks
const COLORS = [
    '#e94560', // red
    '#4ade80', // green
    '#60a5fa', // blue
    '#fbbf24', // yellow
    '#c084fc', // purple
];

// State
let canvas, ctx;
let image = null;
let imageName = null;
let viewType = 'ap';
let annotations = {};
let currentLandmarkIndex = 0;
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('annotation-canvas');
    ctx = canvas.getContext('2d');

    // Event listeners
    document.getElementById('file-upload').addEventListener('change', handleFileUpload);
    document.getElementById('image-select').addEventListener('change', handleImageSelect);
    document.querySelectorAll('input[name="view-type"]').forEach(radio => {
        radio.addEventListener('change', handleViewTypeChange);
    });

    // Canvas events
    canvas.addEventListener('click', handleCanvasClick);
    canvas.addEventListener('wheel', handleWheel);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseUp);

    // Load existing images
    loadImageList();
    updateUI();
    resizeCanvas();

    window.addEventListener('resize', resizeCanvas);
});

function resizeCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    render();
}

// Image handling
async function loadImageList() {
    try {
        const response = await fetch('/images');
        const data = await response.json();
        const select = document.getElementById('image-select');

        // Clear existing options except first
        while (select.options.length > 1) {
            select.remove(1);
        }

        data.images.forEach(img => {
            const option = document.createElement('option');
            option.value = img;
            option.textContent = img;
            select.appendChild(option);
        });
    } catch (error) {
        showStatus('Failed to load images', 'error');
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        showStatus('Uploading...', 'info');
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) throw new Error('Upload failed');

        await loadImageList();
        document.getElementById('image-select').value = file.name;
        await loadImage(file.name);
        showStatus('Image uploaded', 'success');
    } catch (error) {
        showStatus('Upload failed', 'error');
    }

    e.target.value = '';
}

async function handleImageSelect(e) {
    const name = e.target.value;
    if (name) {
        await loadImage(name);
    }
}

async function loadImage(name) {
    imageName = name;
    image = new Image();
    image.onload = async () => {
        resetView();
        await loadExistingAnnotation();
        render();
    };
    image.src = `/images/${name}`;
}

async function loadExistingAnnotation() {
    annotations = {};
    currentLandmarkIndex = 0;

    try {
        const response = await fetch(`/annotations/${imageName}?view_type=${viewType}`);
        if (response.ok) {
            const data = await response.json();
            annotations = data.landmarks || {};

            // Find first missing landmark
            const landmarks = LANDMARKS[viewType];
            for (let i = 0; i < landmarks.length; i++) {
                if (!annotations[landmarks[i].id]) {
                    currentLandmarkIndex = i;
                    break;
                }
                if (i === landmarks.length - 1) {
                    currentLandmarkIndex = landmarks.length;
                }
            }
        }
    } catch (error) {
        // No existing annotation
    }

    updateUI();
}

function handleViewTypeChange(e) {
    viewType = e.target.value;
    if (imageName) {
        loadExistingAnnotation();
    } else {
        updateUI();
    }
}

// Canvas interaction
function handleCanvasClick(e) {
    if (!image || isDragging) return;

    const landmarks = LANDMARKS[viewType];
    if (currentLandmarkIndex >= landmarks.length) return;

    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    // Convert to image coordinates
    const imgX = (canvasX - offsetX) / scale;
    const imgY = (canvasY - offsetY) / scale;

    // Check bounds
    if (imgX < 0 || imgX > image.width || imgY < 0 || imgY > image.height) {
        return;
    }

    const landmark = landmarks[currentLandmarkIndex];
    annotations[landmark.id] = { x: Math.round(imgX), y: Math.round(imgY) };
    currentLandmarkIndex++;

    updateUI();
    render();
}

function handleWheel(e) {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.1, Math.min(10, scale * zoomFactor));

    // Zoom toward mouse position
    offsetX = mouseX - (mouseX - offsetX) * (newScale / scale);
    offsetY = mouseY - (mouseY - offsetY) * (newScale / scale);
    scale = newScale;

    render();
}

function handleMouseDown(e) {
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
        isDragging = true;
        dragStartX = e.clientX - offsetX;
        dragStartY = e.clientY - offsetY;
        canvas.style.cursor = 'grabbing';
    }
}

function handleMouseMove(e) {
    if (isDragging) {
        offsetX = e.clientX - dragStartX;
        offsetY = e.clientY - dragStartY;
        render();
    }
}

function handleMouseUp() {
    isDragging = false;
    canvas.style.cursor = 'crosshair';
}

// Zoom controls
function zoomIn() {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const newScale = Math.min(10, scale * 1.2);

    offsetX = centerX - (centerX - offsetX) * (newScale / scale);
    offsetY = centerY - (centerY - offsetY) * (newScale / scale);
    scale = newScale;

    render();
}

function zoomOut() {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const newScale = Math.max(0.1, scale * 0.8);

    offsetX = centerX - (centerX - offsetX) * (newScale / scale);
    offsetY = centerY - (centerY - offsetY) * (newScale / scale);
    scale = newScale;

    render();
}

function resetView() {
    if (!image) {
        scale = 1;
        offsetX = 0;
        offsetY = 0;
        return;
    }

    // Fit image to canvas with padding
    const padding = 40;
    const scaleX = (canvas.width - padding * 2) / image.width;
    const scaleY = (canvas.height - padding * 2) / image.height;
    scale = Math.min(scaleX, scaleY, 1);

    offsetX = (canvas.width - image.width * scale) / 2;
    offsetY = (canvas.height - image.height * scale) / 2;

    render();
}

// Rendering
function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!image) {
        ctx.fillStyle = '#888';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Upload or select an image to begin', canvas.width / 2, canvas.height / 2);
        return;
    }

    // Draw image
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.drawImage(image, 0, 0);
    ctx.restore();

    // Draw landmarks
    const landmarks = LANDMARKS[viewType];
    landmarks.forEach((landmark, i) => {
        const point = annotations[landmark.id];
        if (point) {
            const x = point.x * scale + offsetX;
            const y = point.y * scale + offsetY;
            const color = COLORS[i % COLORS.length];

            // Draw marker
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Draw number
            ctx.fillStyle = 'white';
            ctx.font = 'bold 10px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(i + 1, x, y);
        }
    });
}

// Actions
function undoLast() {
    if (currentLandmarkIndex === 0) return;

    currentLandmarkIndex--;
    const landmark = LANDMARKS[viewType][currentLandmarkIndex];
    delete annotations[landmark.id];

    updateUI();
    render();
}

async function saveAnnotation() {
    if (!imageName) return;

    const data = {
        image_name: imageName,
        view_type: viewType,
        landmarks: annotations,
    };

    try {
        showStatus('Saving...', 'info');
        const response = await fetch('/annotations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        if (!response.ok) throw new Error('Save failed');

        showStatus('Annotation saved!', 'success');
    } catch (error) {
        showStatus('Failed to save', 'error');
    }
}

function logout() {
    // Clear auth by making request with wrong credentials
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '/', true, 'logout', 'logout');
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4) {
            window.location.reload();
        }
    };
    xhr.send();
}

// UI updates
function updateUI() {
    const landmarks = LANDMARKS[viewType];
    const landmarksList = document.getElementById('landmarks-ul');
    const currentDiv = document.getElementById('current-landmark');
    const descriptionP = document.getElementById('landmark-description');

    // Update landmarks list
    landmarksList.innerHTML = '';
    landmarks.forEach((landmark, i) => {
        const li = document.createElement('li');
        const point = annotations[landmark.id];

        if (point) {
            li.className = 'completed';
            li.innerHTML = `
                <span>${i + 1}. ${landmark.name}</span>
                <span class="coords">(${point.x}, ${point.y})</span>
            `;
        } else if (i === currentLandmarkIndex) {
            li.className = 'current';
            li.innerHTML = `<span>${i + 1}. ${landmark.name}</span>`;
        } else {
            li.className = 'pending';
            li.innerHTML = `<span>${i + 1}. ${landmark.name}</span>`;
        }

        landmarksList.appendChild(li);
    });

    // Update current landmark info
    if (currentLandmarkIndex < landmarks.length) {
        const current = landmarks[currentLandmarkIndex];
        currentDiv.innerHTML = `
            <span class="landmark-name">${current.name}</span>
            <span class="landmark-progress">(${currentLandmarkIndex + 1}/${landmarks.length})</span>
        `;
        descriptionP.textContent = current.description;
    } else {
        currentDiv.innerHTML = `
            <span class="landmark-name">Complete!</span>
            <span class="landmark-progress">(${landmarks.length}/${landmarks.length})</span>
        `;
        descriptionP.textContent = 'All landmarks annotated. Click Save to store.';
    }

    // Update buttons
    document.getElementById('undo-btn').disabled = currentLandmarkIndex === 0;
    document.getElementById('save-btn').disabled = Object.keys(annotations).length === 0;
}

function showStatus(message, type) {
    const statusDiv = document.getElementById('status-message');
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;

    if (type !== 'info') {
        setTimeout(() => {
            statusDiv.textContent = '';
            statusDiv.className = 'status';
        }, 3000);
    }
}
