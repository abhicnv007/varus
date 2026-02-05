"""
FastAPI backend for calTAD landmark annotation tool.
"""
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configuration
USERNAME = "caltad"
PASSWORD = "annotate123"  # Change before deployment

# Paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
ANNOTATIONS_DIR.mkdir(exist_ok=True)

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


class AnnotationData(BaseModel):
    image_name: str
    view_type: str  # "ap" or "lateral"
    landmarks: dict[str, Landmark]
    annotator: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(username: str = Depends(verify_credentials)):
    """Serve the main annotation interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials),
):
    """Upload an X-ray image."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate file type
    allowed_extensions = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save file
    file_path = UPLOADS_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    return {"filename": file.filename, "message": "Upload successful"}


@app.get("/images")
async def list_images(username: str = Depends(verify_credentials)):
    """List all uploaded images."""
    images = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        images.extend([f.name for f in UPLOADS_DIR.glob(ext)])
    return {"images": sorted(images)}


@app.get("/images/{name}")
async def get_image(name: str, username: str = Depends(verify_credentials)):
    """Serve an uploaded image."""
    file_path = UPLOADS_DIR / name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@app.post("/annotations")
async def save_annotation(
    data: AnnotationData,
    username: str = Depends(verify_credentials),
):
    """Save an annotation."""
    now = datetime.utcnow().isoformat()

    # Create annotation ID from image name and view type
    annotation_id = f"{Path(data.image_name).stem}_{data.view_type}"
    file_path = ANNOTATIONS_DIR / f"{annotation_id}.json"

    # Check if updating existing annotation
    if file_path.exists():
        existing = json.loads(file_path.read_text())
        data.created_at = existing.get("created_at", now)
    else:
        data.created_at = now

    data.updated_at = now
    data.annotator = username

    # Save annotation
    file_path.write_text(json.dumps(data.model_dump(), indent=2))

    return {"annotation_id": annotation_id, "message": "Annotation saved"}


@app.get("/annotations/{image_name}")
async def get_annotation(
    image_name: str,
    view_type: str,
    username: str = Depends(verify_credentials),
):
    """Get annotation for a specific image and view type."""
    annotation_id = f"{Path(image_name).stem}_{view_type}"
    file_path = ANNOTATIONS_DIR / f"{annotation_id}.json"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Annotation not found")

    return json.loads(file_path.read_text())


@app.get("/annotations")
async def list_annotations(username: str = Depends(verify_credentials)):
    """List all annotations."""
    annotations = []
    for f in ANNOTATIONS_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        annotations.append({
            "id": f.stem,
            "image_name": data.get("image_name"),
            "view_type": data.get("view_type"),
            "annotator": data.get("annotator"),
            "updated_at": data.get("updated_at"),
        })
    return {"annotations": annotations}
