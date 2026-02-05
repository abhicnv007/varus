/**
 * calTAD Landmark Annotation Tool - Patient-centric workflow
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
let currentPatient = null;
let currentPatientData = null;
let viewType = 'ap';
let annotations = {};
let currentLandmarkIndex = 0;
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let statusData = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('annotation-canvas');
    ctx = canvas.getContext('2d');

    // Event listeners
    document.getElementById('file-upload').addEventListener('change', handleFileUpload);
    document.getElementById('patient-select').addEventListener('change', handlePatientSelect);

    // Canvas events
    canvas.addEventListener('click', handleCanvasClick);
    canvas.addEventListener('wheel', handleWheel);
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseUp);

    // Load data
    loadStatus();
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

// Data loading
async function loadStatus() {
    try {
        const response = await fetch('/status');
        statusData = await response.json();
        updatePatientSelect();
        updateProgressBar();
    } catch (error) {
        showStatus('Failed to load data', 'error');
    }
}

function updatePatientSelect() {
    const select = document.getElementById('patient-select');
    const currentValue = select.value;

    while (select.options.length > 1) {
        select.remove(1);
    }

    if (!statusData) return;

    statusData.patients.forEach(p => {
        const option = document.createElement('option');
        option.value = p.id;

        let status = '';
        if (p.complete) {
            status = ' [Complete]';
        } else if (p.ap_annotated || p.lateral_annotated) {
            const parts = [];
            if (p.ap_annotated) parts.push('AP');
            if (p.lateral_annotated) parts.push('LAT');
            status = ` [${parts.join('+')}]`;
        } else if (p.has_ap || p.has_lateral) {
            status = ' [Images uploaded]';
        }

        option.textContent = p.name + status;
        select.appendChild(option);
    });

    if (currentValue) {
        select.value = currentValue;
    }
}

function updateProgressBar() {
    if (!statusData) return;

    const { stats } = statusData;
    const pct = stats.total_patients > 0
        ? Math.round((stats.complete / stats.total_patients) * 100)
        : 0;

    document.getElementById('progress-complete').textContent =
        `${stats.complete}/${stats.total_patients} complete`;
    document.getElementById('progress-ap').textContent =
        `AP: ${stats.ap_annotated}`;
    document.getElementById('progress-lateral').textContent =
        `Lateral: ${stats.lateral_annotated}`;
    document.getElementById('progress-fill').style.width = `${pct}%`;
}

// Patient handling
async function handlePatientSelect(e) {
    const patientId = e.target.value;
    if (patientId) {
        await loadPatient(parseInt(patientId));
    } else {
        currentPatient = null;
        currentPatientData = null;
        image = null;
        document.getElementById('view-controls').style.display = 'none';
        document.getElementById('patient-info').style.display = 'none';
        updateUI();
        render();
    }
}

async function loadPatient(patientId) {
    try {
        const response = await fetch(`/patients/${patientId}`);
        if (!response.ok) throw new Error('Failed to load patient');

        currentPatientData = await response.json();
        currentPatient = currentPatientData.patient;

        document.getElementById('view-controls').style.display = 'flex';
        document.getElementById('patient-info').style.display = 'block';
        document.getElementById('patient-name').textContent = currentPatient.name;

        updatePatientStatus();
        await loadCurrentView();
    } catch (error) {
        showStatus('Failed to load patient', 'error');
    }
}

function updatePatientStatus() {
    if (!currentPatientData) return;

    const images = currentPatientData.images;
    const apImage = images.find(i => i.view_type === 'ap');
    const lateralImage = images.find(i => i.view_type === 'lateral');

    // Update status badges
    const apBadge = document.getElementById('status-ap');
    const lateralBadge = document.getElementById('status-lateral');

    if (apImage?.has_annotation) {
        apBadge.textContent = 'AP: Done';
        apBadge.className = 'status-badge annotated';
    } else if (apImage) {
        apBadge.textContent = 'AP: Uploaded';
        apBadge.className = 'status-badge uploaded';
    } else {
        apBadge.textContent = 'AP: -';
        apBadge.className = 'status-badge';
    }

    if (lateralImage?.has_annotation) {
        lateralBadge.textContent = 'Lateral: Done';
        lateralBadge.className = 'status-badge annotated';
    } else if (lateralImage) {
        lateralBadge.textContent = 'Lateral: Uploaded';
        lateralBadge.className = 'status-badge uploaded';
    } else {
        lateralBadge.textContent = 'Lateral: -';
        lateralBadge.className = 'status-badge';
    }

    // Update view buttons
    const btnAp = document.getElementById('btn-ap');
    const btnLateral = document.getElementById('btn-lateral');

    btnAp.classList.toggle('has-image', !!apImage);
    btnAp.classList.toggle('annotated', !!apImage?.has_annotation);
    btnLateral.classList.toggle('has-image', !!lateralImage);
    btnLateral.classList.toggle('annotated', !!lateralImage?.has_annotation);
}

async function selectView(view) {
    viewType = view;

    document.getElementById('btn-ap').classList.toggle('active', view === 'ap');
    document.getElementById('btn-lateral').classList.toggle('active', view === 'lateral');

    await loadCurrentView();
}

async function loadCurrentView() {
    if (!currentPatientData) return;

    const images = currentPatientData.images;
    const currentImage = images.find(i => i.view_type === viewType);

    if (currentImage) {
        // Load image
        image = new Image();
        image.onload = async () => {
            resetView();
            await loadExistingAnnotation();
            render();
        };
        image.src = `/images/${currentImage.filename}`;
    } else {
        image = null;
        annotations = {};
        currentLandmarkIndex = 0;
        updateUI();
        render();
    }
}

async function loadExistingAnnotation() {
    annotations = {};
    currentLandmarkIndex = 0;

    if (!currentPatient) return;

    try {
        const response = await fetch(`/annotations/${currentPatient.id}/${viewType}`);
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

// File upload
async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file || !currentPatient) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_id', currentPatient.id);
    formData.append('view_type', viewType);

    try {
        showStatus('Uploading...', 'info');
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }

        // Reload patient data
        await loadPatient(currentPatient.id);
        await loadStatus();
        showStatus('Image uploaded', 'success');
    } catch (error) {
        showStatus(error.message || 'Upload failed', 'error');
    }

    e.target.value = '';
}

// Patient modal
function showAddPatientModal() {
    document.getElementById('add-patient-modal').classList.add('show');
    document.getElementById('patient-name-input').focus();
}

function hideAddPatientModal() {
    document.getElementById('add-patient-modal').classList.remove('show');
    document.getElementById('add-patient-form').reset();
}

async function addPatient(e) {
    e.preventDefault();

    const name = document.getElementById('patient-name-input').value.trim();
    const notes = document.getElementById('patient-notes-input').value.trim();

    if (!name) return;

    try {
        showStatus('Adding patient...', 'info');
        const response = await fetch('/patients', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, notes: notes || null }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to add patient');
        }

        const data = await response.json();
        hideAddPatientModal();
        await loadStatus();

        // Select the new patient
        document.getElementById('patient-select').value = data.id;
        await loadPatient(data.id);

        showStatus('Patient added', 'success');
    } catch (error) {
        showStatus(error.message || 'Failed to add patient', 'error');
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

    const imgX = (canvasX - offsetX) / scale;
    const imgY = (canvasY - offsetY) / scale;

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

    if (!currentPatient) {
        ctx.fillStyle = '#888';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Select a patient to begin', canvas.width / 2, canvas.height / 2);
        return;
    }

    if (!image) {
        ctx.fillStyle = '#888';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`Upload ${viewType.toUpperCase()} image for this patient`, canvas.width / 2, canvas.height / 2);
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

            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.stroke();

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
    if (!currentPatient || !image) return;

    const data = {
        patient_id: currentPatient.id,
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

        // Reload to update status
        await loadPatient(currentPatient.id);
        await loadStatus();
        showStatus('Annotation saved!', 'success');
    } catch (error) {
        showStatus('Failed to save', 'error');
    }
}

function logout() {
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

    if (!currentPatient) {
        currentDiv.innerHTML = `
            <span class="landmark-name">-</span>
            <span class="landmark-progress">(0/0)</span>
        `;
        descriptionP.textContent = 'Select a patient to begin';
    } else if (!image) {
        currentDiv.innerHTML = `
            <span class="landmark-name">-</span>
            <span class="landmark-progress">(0/${landmarks.length})</span>
        `;
        descriptionP.textContent = `Upload ${viewType.toUpperCase()} image first`;
    } else if (currentLandmarkIndex < landmarks.length) {
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

    document.getElementById('undo-btn').disabled = currentLandmarkIndex === 0;
    document.getElementById('save-btn').disabled = !image || Object.keys(annotations).length === 0;
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
