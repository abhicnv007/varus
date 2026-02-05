"""
FastAPI backend for calTAD landmark annotation tool.
"""
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database import Database

# Configuration from environment variables
USERNAME = os.environ.get("CALTAD_USERNAME", "caltad")
PASSWORD = os.environ.get("CALTAD_PASSWORD", "annotate123")

# Paths - use DATA_DIR env var for Railway volume mount
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR)))
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "caltad.db"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize database
db = Database(DB_PATH)

app = FastAPI(title="calTAD Annotation Tool")
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Verify HTTP Basic Auth credentials."""
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class Landmark(BaseModel):
    x: float
    y: float


class PatientCreate(BaseModel):
    name: str
    notes: Optional[str] = None


class AnnotationData(BaseModel):
    patient_id: int
    view_type: str
    landmarks: dict[str, Landmark]


# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(username: str = Depends(verify_credentials)):
    """Serve the main annotation interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


# Patient endpoints
@app.post("/patients")
async def create_patient(
    data: PatientCreate,
    username: str = Depends(verify_credentials),
):
    """Create a new patient."""
    existing = db.get_patient_by_name(data.name)
    if existing:
        raise HTTPException(status_code=400, detail="Patient with this name already exists")

    patient_id = db.create_patient(data.name, data.notes)
    return {"id": patient_id, "name": data.name}


@app.get("/patients")
async def list_patients(username: str = Depends(verify_credentials)):
    """List all patients with annotation status."""
    patients = db.list_patients()
    return {"patients": patients}


@app.get("/patients/{patient_id}")
async def get_patient(patient_id: int, username: str = Depends(verify_credentials)):
    """Get patient details with images."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    images = db.get_patient_images(patient_id)
    return {"patient": patient, "images": images}


@app.delete("/patients/{patient_id}")
async def delete_patient(patient_id: int, username: str = Depends(verify_credentials)):
    """Delete a patient and all associated data."""
    # Get images to delete files
    images = db.get_patient_images(patient_id)
    for img in images:
        file_path = UPLOADS_DIR / img['filename']
        if file_path.exists():
            file_path.unlink()

    if not db.delete_patient(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    return {"message": "Patient deleted"}


# Image endpoints
@app.post("/upload")
async def upload_image(
    patient_id: int = Form(...),
    view_type: str = Form(...),
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials),
):
    """Upload an X-ray image for a patient."""
    # Validate patient exists
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Validate view type
    if view_type not in ('ap', 'lateral'):
        raise HTTPException(status_code=400, detail="view_type must be 'ap' or 'lateral'")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file type
    allowed_extensions = {".png", ".jpg", ".jpeg"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Generate unique filename
    filename = f"patient_{patient_id}_{view_type}{ext}"
    file_path = UPLOADS_DIR / filename

    # Delete old file if replacing
    images = db.get_patient_images(patient_id)
    for img in images:
        if img['view_type'] == view_type and img['filename'] != filename:
            old_path = UPLOADS_DIR / img['filename']
            if old_path.exists():
                old_path.unlink()

    # Save file
    content = await file.read()
    file_path.write_bytes(content)

    # Add to database
    image_id = db.add_image(patient_id, view_type, filename)

    return {"image_id": image_id, "filename": filename}


@app.get("/images/{filename}")
async def get_image(filename: str, username: str = Depends(verify_credentials)):
    """Serve an uploaded image."""
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


# Annotation endpoints
@app.post("/annotations")
async def save_annotation(
    data: AnnotationData,
    username: str = Depends(verify_credentials),
):
    """Save annotation for a patient's image."""
    # Get image for this patient/view
    images = db.get_patient_images(data.patient_id)
    image = next((i for i in images if i['view_type'] == data.view_type), None)

    if not image:
        raise HTTPException(status_code=404, detail=f"No {data.view_type} image found for this patient")

    # Convert landmarks to dict format
    landmarks = {k: {"x": v.x, "y": v.y} for k, v in data.landmarks.items()}

    annotation_id = db.save_annotation(image['id'], landmarks, username)
    return {"annotation_id": annotation_id, "message": "Annotation saved"}


@app.get("/annotations/{patient_id}/{view_type}")
async def get_annotation(
    patient_id: int,
    view_type: str,
    username: str = Depends(verify_credentials),
):
    """Get annotation for a patient's specific view."""
    annotation = db.get_patient_annotation(patient_id, view_type)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return annotation


# Status endpoint
@app.get("/status")
async def get_status(username: str = Depends(verify_credentials)):
    """Get overall annotation statistics."""
    stats = db.get_stats()
    patients = db.list_patients()
    return {"stats": stats, "patients": patients}
