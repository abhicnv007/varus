"""
ROI (Region of Interest) Detection for Hip X-rays.

Stage 1 of the calTAD pipeline: Localize the proximal femur region.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ROI:
    """Detected region of interest."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        """Return center point."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def crop(self, image: np.ndarray, padding: int = 0) -> np.ndarray:
        """Crop the ROI from an image with optional padding."""
        h, w = image.shape[:2]
        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(w, self.x + self.width + padding)
        y2 = min(h, self.y + self.height + padding)
        return image[y1:y2, x1:x2]


class ROIDetector:
    """
    Detect proximal femur ROI using classical computer vision.

    This uses intensity-based detection assuming metallic implants
    appear bright on radiographs. Can be replaced with YOLO/Faster R-CNN
    for production use.
    """

    def __init__(self,
                 min_implant_area: int = 1000,
                 padding_ratio: float = 0.3):
        """
        Args:
            min_implant_area: Minimum area in pixels for implant detection
            padding_ratio: Padding around detected implant as ratio of ROI size
        """
        self.min_implant_area = min_implant_area
        self.padding_ratio = padding_ratio

    def detect(self, image: np.ndarray) -> Optional[ROI]:
        """
        Detect the proximal femur ROI containing the implant.

        Args:
            image: Grayscale X-ray image

        Returns:
            ROI object or None if detection fails
        """
        # Metallic implants appear very bright - threshold high intensities
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Additional threshold for very bright regions (metal)
        high_threshold = np.percentile(image, 98)
        metal_mask = (image > high_threshold).astype(np.uint8) * 255

        # Morphological operations to connect implant components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: use center of image
            h, w = image.shape
            return ROI(w // 4, h // 4, w // 2, h // 2, confidence=0.3)

        # Find largest contour (assumed to be implant)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < self.min_implant_area:
            # Fallback for weak detection
            h, w = image.shape
            return ROI(w // 4, h // 4, w // 2, h // 2, confidence=0.3)

        # Get bounding box with padding
        x, y, w, h = cv2.boundingRect(largest_contour)
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)

        img_h, img_w = image.shape
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(img_w - x, w + 2 * pad_x)
        h = min(img_h - y, h + 2 * pad_y)

        confidence = min(1.0, area / (self.min_implant_area * 10))

        return ROI(x, y, w, h, confidence)

    def detect_implant_region(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect metallic implant regions for screw detection.

        Returns list of contours representing metal regions.
        """
        # High threshold for metal
        high_threshold = np.percentile(image, 95)
        metal_mask = (image > high_threshold).astype(np.uint8) * 255

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [c for c in contours if cv2.contourArea(c) > self.min_implant_area // 10]
