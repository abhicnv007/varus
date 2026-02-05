"""
Visualization utilities for calTAD pipeline.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
from .landmark_detection import Landmark, LandmarkType, DetectedCircle, DetectedLine
from .roi_detection import ROI
from .measurement import CalTADResult


# Color scheme (BGR format for OpenCV)
COLORS = {
    LandmarkType.SCREW_TIP: (0, 0, 255),           # Red
    LandmarkType.SCREW_SHAFT_1: (0, 128, 255),     # Orange
    LandmarkType.SCREW_SHAFT_2: (0, 128, 255),     # Orange
    LandmarkType.FEMORAL_HEAD_CENTER: (255, 0, 0), # Blue
    LandmarkType.FEMORAL_HEAD_APEX: (255, 255, 0), # Cyan
    LandmarkType.CALCAR_POINT_1: (0, 255, 0),      # Green
    LandmarkType.CALCAR_POINT_2: (0, 255, 0),      # Green
}


def draw_landmarks(image: np.ndarray,
                   landmarks: Dict[LandmarkType, Landmark],
                   radius: int = 8,
                   thickness: int = 2,
                   show_labels: bool = True) -> np.ndarray:
    """
    Draw detected landmarks on image.

    Args:
        image: Input image (grayscale or BGR)
        landmarks: Dictionary of detected landmarks
        radius: Marker radius
        thickness: Line thickness
        show_labels: Whether to show landmark labels

    Returns:
        Image with landmarks drawn
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for landmark_type, landmark in landmarks.items():
        color = COLORS.get(landmark_type, (255, 255, 255))
        pt = landmark.position_int

        # Draw marker
        cv2.circle(vis, pt, radius, color, thickness)
        cv2.circle(vis, pt, 2, color, -1)  # Center dot

        # Draw label
        if show_labels:
            label = landmark_type.value.replace("_", " ").title()
            conf_str = f" ({landmark.confidence:.0%})"
            cv2.putText(vis, label + conf_str, (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw calcar line if both points exist
    if (LandmarkType.CALCAR_POINT_1 in landmarks and
        LandmarkType.CALCAR_POINT_2 in landmarks):
        p1 = landmarks[LandmarkType.CALCAR_POINT_1].position_int
        p2 = landmarks[LandmarkType.CALCAR_POINT_2].position_int
        cv2.line(vis, p1, p2, COLORS[LandmarkType.CALCAR_POINT_1], thickness)

    return vis


def draw_femoral_head(image: np.ndarray,
                      circle: DetectedCircle,
                      color: Tuple[int, int, int] = (255, 0, 0),
                      thickness: int = 2) -> np.ndarray:
    """Draw detected femoral head circle."""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    center = (int(circle.center_x), int(circle.center_y))
    radius = int(circle.radius)

    cv2.circle(vis, center, radius, color, thickness)
    cv2.circle(vis, center, 3, color, -1)

    return vis


def draw_screw(image: np.ndarray,
               line: DetectedLine,
               color: Tuple[int, int, int] = (0, 0, 255),
               thickness: int = 2) -> np.ndarray:
    """Draw detected lag screw line."""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    p1 = (int(line.x1), int(line.y1))
    p2 = (int(line.x2), int(line.y2))

    cv2.line(vis, p1, p2, color, thickness)
    cv2.circle(vis, p1, 5, color, -1)
    cv2.circle(vis, p2, 5, color, -1)

    return vis


def draw_roi(image: np.ndarray,
             roi: ROI,
             color: Tuple[int, int, int] = (0, 255, 255),
             thickness: int = 2) -> np.ndarray:
    """Draw ROI bounding box."""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    x, y, w, h = roi.bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

    # Add confidence label
    label = f"ROI ({roi.confidence:.0%})"
    cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis


def draw_measurement(image: np.ndarray,
                     landmarks: Dict[LandmarkType, Landmark],
                     result: CalTADResult,
                     show_distances: bool = True) -> np.ndarray:
    """
    Draw calTAD measurement visualization.

    Shows landmarks, measurement lines, and result annotation.
    """
    # Start with landmark visualization
    vis = draw_landmarks(image, landmarks)

    # Draw measurement lines
    if LandmarkType.SCREW_TIP in landmarks:
        tip = landmarks[LandmarkType.SCREW_TIP].position_int

        # Draw line to calcar
        if (LandmarkType.CALCAR_POINT_1 in landmarks and
            LandmarkType.CALCAR_POINT_2 in landmarks and
            result.x_calcar_px is not None):

            p1 = landmarks[LandmarkType.CALCAR_POINT_1].position_int
            p2 = landmarks[LandmarkType.CALCAR_POINT_2].position_int

            # Find perpendicular point on calcar line
            # Using projection formula
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            t = ((tip[0] - p1[0]) * dx + (tip[1] - p1[1]) * dy) / (dx*dx + dy*dy + 1e-8)
            proj_x = int(p1[0] + t * dx)
            proj_y = int(p1[1] + t * dy)

            cv2.line(vis, tip, (proj_x, proj_y), (255, 0, 255), 2, cv2.LINE_AA)

            if show_distances and result.x_calcar_mm:
                mid = ((tip[0] + proj_x) // 2, (tip[1] + proj_y) // 2)
                cv2.putText(vis, f"{result.x_calcar_mm:.1f}mm", mid,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Draw line to apex
        if (LandmarkType.FEMORAL_HEAD_APEX in landmarks and
            result.x_lat_px is not None):

            apex = landmarks[LandmarkType.FEMORAL_HEAD_APEX].position_int
            cv2.line(vis, tip, apex, (0, 255, 255), 2, cv2.LINE_AA)

            if show_distances and result.x_lat_mm:
                mid = ((tip[0] + apex[0]) // 2, (tip[1] + apex[1]) // 2)
                cv2.putText(vis, f"{result.x_lat_mm:.1f}mm", mid,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Add result annotation
    if result.is_valid:
        color = (0, 255, 0) if result.is_within_threshold else (0, 0, 255)
        status = "SAFE" if result.is_within_threshold else "AT RISK"

        text = f"calTAD: {result.caltad_mm:.1f}mm ({status})"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        conf_text = f"Confidence: {result.confidence:.0%}"
        cv2.putText(vis, conf_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def create_summary_figure(image: np.ndarray,
                          roi: Optional[ROI],
                          landmarks: Dict[LandmarkType, Landmark],
                          result: CalTADResult) -> np.ndarray:
    """
    Create a summary figure with multiple panels.

    Layout:
    [Original | ROI crop with landmarks | Measurement visualization]
    """
    h, w = image.shape[:2]

    # Panel 1: Original with ROI
    panel1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    if roi:
        panel1 = draw_roi(panel1, roi)
    cv2.putText(panel1, "Original + ROI", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 2: ROI crop with landmarks
    if roi:
        crop = roi.crop(image, padding=20)
        panel2 = draw_landmarks(crop, landmarks)
    else:
        panel2 = draw_landmarks(image, landmarks)
    cv2.putText(panel2, "Landmarks", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Panel 3: Measurement visualization
    panel3 = draw_measurement(image if roi is None else roi.crop(image, padding=20),
                              landmarks, result)

    # Resize panels to same height
    target_h = min(h, 600)

    def resize_panel(panel, target_height):
        ph, pw = panel.shape[:2]
        scale = target_height / ph
        new_w = int(pw * scale)
        return cv2.resize(panel, (new_w, target_height))

    panel1 = resize_panel(panel1, target_h)
    panel2 = resize_panel(panel2, target_h)
    panel3 = resize_panel(panel3, target_h)

    # Concatenate horizontally
    summary = np.hstack([panel1, panel2, panel3])

    return summary


def save_visualization(image: np.ndarray, path: str):
    """Save visualization to file."""
    cv2.imwrite(path, image)
