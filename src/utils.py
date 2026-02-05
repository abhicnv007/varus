"""
Utility functions for image loading and preprocessing.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Union

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file (supports DICOM, PNG, JPG, etc.).

    Returns grayscale image normalized to 0-255 uint8.
    """
    path = Path(path)

    if path.suffix.lower() in ['.dcm', '.dicom']:
        if not HAS_PYDICOM:
            raise ImportError("pydicom required for DICOM files: pip install pydicom")
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)
        # Normalize to 0-255
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {path}")

    return img


def preprocess_xray(img: np.ndarray,
                    target_size: Optional[Tuple[int, int]] = None,
                    enhance_contrast: bool = True) -> np.ndarray:
    """
    Preprocess X-ray image for analysis.

    Args:
        img: Grayscale input image
        target_size: Optional (width, height) to resize to
        enhance_contrast: Apply CLAHE for contrast enhancement

    Returns:
        Preprocessed image
    """
    processed = img.copy()

    # Resize if specified
    if target_size is not None:
        processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_LINEAR)

    # Contrast enhancement with CLAHE
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    return processed


def get_pixel_spacing(dicom_path: Union[str, Path]) -> Optional[Tuple[float, float]]:
    """
    Extract pixel spacing from DICOM metadata.

    Returns (row_spacing, col_spacing) in mm, or None if not available.
    """
    if not HAS_PYDICOM:
        return None

    try:
        ds = pydicom.dcmread(str(dicom_path))
        if hasattr(ds, 'PixelSpacing'):
            return tuple(float(x) for x in ds.PixelSpacing)
        if hasattr(ds, 'ImagerPixelSpacing'):
            return tuple(float(x) for x in ds.ImagerPixelSpacing)
    except Exception:
        pass

    return None


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def point_to_line_distance(point: Tuple[float, float],
                           line_point1: Tuple[float, float],
                           line_point2: Tuple[float, float]) -> float:
    """
    Calculate perpendicular distance from a point to a line defined by two points.
    """
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2

    # Line equation: ax + by + c = 0
    # From two points: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1

    # Distance formula
    distance = abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2 + 1e-8)
    return distance
