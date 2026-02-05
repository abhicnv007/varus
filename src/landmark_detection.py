"""
Landmark Detection for calTAD Measurement.

Stage 2 of the pipeline: Detect key anatomical and implant landmarks.

Landmarks needed:
- Lag screw tip
- Femoral head apex (for lateral view)
- Calcar reference points (for AP view)
- Screw shaft points (for diameter measurement)
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum


class LandmarkType(Enum):
    SCREW_TIP = "screw_tip"
    SCREW_SHAFT_1 = "screw_shaft_1"  # For diameter measurement
    SCREW_SHAFT_2 = "screw_shaft_2"
    FEMORAL_HEAD_CENTER = "femoral_head_center"
    FEMORAL_HEAD_APEX = "femoral_head_apex"
    CALCAR_POINT_1 = "calcar_point_1"  # Medial cortex reference
    CALCAR_POINT_2 = "calcar_point_2"


@dataclass
class Landmark:
    """Detected landmark with position and confidence."""
    type: LandmarkType
    x: float
    y: float
    confidence: float = 1.0

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def position_int(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


@dataclass
class DetectedCircle:
    """Detected circle (femoral head candidate)."""
    center_x: float
    center_y: float
    radius: float
    confidence: float = 1.0


@dataclass
class DetectedLine:
    """Detected line segment (screw candidate)."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    @property
    def angle(self) -> float:
        """Angle in degrees from horizontal."""
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class LandmarkDetector:
    """
    Detect calTAD-relevant landmarks using classical computer vision.

    Uses Hough transforms for initial detection:
    - Circular Hough Transform for femoral head
    - Probabilistic Hough Line Transform for lag screw

    Can be replaced with HRNet-based detection for production.
    """

    def __init__(self,
                 min_femoral_head_radius: int = 30,
                 max_femoral_head_radius: int = 150,
                 min_screw_length: int = 50,
                 screw_angle_range: Tuple[float, float] = (100, 170)):
        """
        Args:
            min_femoral_head_radius: Minimum expected femoral head radius (pixels)
            max_femoral_head_radius: Maximum expected femoral head radius (pixels)
            min_screw_length: Minimum expected screw length (pixels)
            screw_angle_range: Expected screw angle range (degrees from horizontal)
        """
        self.min_femoral_head_radius = min_femoral_head_radius
        self.max_femoral_head_radius = max_femoral_head_radius
        self.min_screw_length = min_screw_length
        self.screw_angle_range = screw_angle_range

    def detect_all(self, image: np.ndarray) -> Dict[LandmarkType, Landmark]:
        """
        Detect all landmarks needed for calTAD measurement.

        Args:
            image: Grayscale X-ray image (or ROI crop)

        Returns:
            Dictionary mapping LandmarkType to detected Landmark
        """
        landmarks = {}

        # Detect femoral head
        femoral_head = self.detect_femoral_head(image)
        if femoral_head:
            landmarks[LandmarkType.FEMORAL_HEAD_CENTER] = Landmark(
                LandmarkType.FEMORAL_HEAD_CENTER,
                femoral_head.center_x,
                femoral_head.center_y,
                femoral_head.confidence
            )
            # Apex is the most superior point of the femoral head
            landmarks[LandmarkType.FEMORAL_HEAD_APEX] = Landmark(
                LandmarkType.FEMORAL_HEAD_APEX,
                femoral_head.center_x,
                femoral_head.center_y - femoral_head.radius,
                femoral_head.confidence
            )

        # Detect lag screw
        screw = self.detect_lag_screw(image)
        if screw:
            # Screw tip is the end closer to femoral head center
            if femoral_head:
                dist1 = np.sqrt((screw.x1 - femoral_head.center_x)**2 +
                               (screw.y1 - femoral_head.center_y)**2)
                dist2 = np.sqrt((screw.x2 - femoral_head.center_x)**2 +
                               (screw.y2 - femoral_head.center_y)**2)
                if dist1 < dist2:
                    tip_x, tip_y = screw.x1, screw.y1
                    shaft_x, shaft_y = screw.x2, screw.y2
                else:
                    tip_x, tip_y = screw.x2, screw.y2
                    shaft_x, shaft_y = screw.x1, screw.y1
            else:
                # Assume tip is on the right (medial) side for AP view
                if screw.x1 > screw.x2:
                    tip_x, tip_y = screw.x1, screw.y1
                    shaft_x, shaft_y = screw.x2, screw.y2
                else:
                    tip_x, tip_y = screw.x2, screw.y2
                    shaft_x, shaft_y = screw.x1, screw.y1

            landmarks[LandmarkType.SCREW_TIP] = Landmark(
                LandmarkType.SCREW_TIP, tip_x, tip_y, screw.confidence
            )

            # Shaft points for diameter measurement
            mid = screw.midpoint
            landmarks[LandmarkType.SCREW_SHAFT_1] = Landmark(
                LandmarkType.SCREW_SHAFT_1, mid[0], mid[1], screw.confidence
            )

        # Detect calcar reference (medial femoral neck cortex)
        calcar = self.detect_calcar_line(image, femoral_head)
        if calcar:
            landmarks[LandmarkType.CALCAR_POINT_1] = Landmark(
                LandmarkType.CALCAR_POINT_1, calcar[0][0], calcar[0][1], 0.7
            )
            landmarks[LandmarkType.CALCAR_POINT_2] = Landmark(
                LandmarkType.CALCAR_POINT_2, calcar[1][0], calcar[1][1], 0.7
            )

        return landmarks

    def detect_femoral_head(self, image: np.ndarray) -> Optional[DetectedCircle]:
        """
        Detect femoral head using Circular Hough Transform.

        The femoral head appears as an approximately circular structure.
        """
        # Preprocessing
        blurred = cv2.GaussianBlur(image, (9, 9), 2)

        # Circular Hough Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=self.min_femoral_head_radius,
            maxRadius=self.max_femoral_head_radius
        )

        if circles is None:
            return None

        # Select the most likely femoral head
        # Typically in the upper-medial region of the image
        circles = np.uint16(np.around(circles[0]))

        # Score circles based on position and size
        h, w = image.shape
        best_circle = None
        best_score = -float('inf')

        for circle in circles:
            cx, cy, r = circle
            # Prefer circles in the upper half (femoral head position)
            position_score = (h - cy) / h  # Higher is better
            # Prefer medium-sized circles
            size_score = 1.0 - abs(r - (self.min_femoral_head_radius + self.max_femoral_head_radius) / 2) / self.max_femoral_head_radius

            score = position_score * 0.6 + size_score * 0.4

            if score > best_score:
                best_score = score
                best_circle = circle

        if best_circle is not None:
            return DetectedCircle(
                float(best_circle[0]),
                float(best_circle[1]),
                float(best_circle[2]),
                confidence=min(1.0, best_score + 0.5)
            )

        return None

    def detect_lag_screw(self, image: np.ndarray) -> Optional[DetectedLine]:
        """
        Detect lag screw using Probabilistic Hough Line Transform.

        The lag screw appears as a bright linear structure.
        """
        # Edge detection on bright regions (metal)
        high_threshold = np.percentile(image, 90)
        bright_mask = (image > high_threshold).astype(np.uint8) * 255

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(bright_mask, 50, 150)

        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=self.min_screw_length,
            maxLineGap=10
        )

        if lines is None:
            return None

        # Filter by angle and select longest
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)  # Convert to 0-180 range

            if self.screw_angle_range[0] <= angle <= self.screw_angle_range[1]:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                valid_lines.append((x1, y1, x2, y2, length, angle))

        if not valid_lines:
            # Try with relaxed angle constraints
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                valid_lines.append((x1, y1, x2, y2, length, angle))

        if not valid_lines:
            return None

        # Select longest line
        best = max(valid_lines, key=lambda x: x[4])

        return DetectedLine(
            float(best[0]), float(best[1]),
            float(best[2]), float(best[3]),
            confidence=min(1.0, best[4] / (self.min_screw_length * 2))
        )

    def detect_calcar_line(self, image: np.ndarray,
                          femoral_head: Optional[DetectedCircle]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Detect the calcar (medial femoral neck cortex) line.

        This is the most challenging detection - the calcar is the
        medial cortical bone line of the femoral neck.

        Returns two points defining the calcar tangent line.
        """
        if femoral_head is None:
            return None

        # The calcar line is typically inferior-medial to the femoral head
        # Start by finding strong edges in that region
        h, w = image.shape

        # Define search region (inferior to femoral head)
        search_y_start = int(femoral_head.center_y)
        search_y_end = min(h, int(femoral_head.center_y + femoral_head.radius * 2))
        search_x_start = max(0, int(femoral_head.center_x - femoral_head.radius * 2))
        search_x_end = int(femoral_head.center_x)

        # Extract region
        region = image[search_y_start:search_y_end, search_x_start:search_x_end]
        if region.size == 0:
            return None

        # Edge detection
        edges = cv2.Canny(region, 50, 150)

        # Hough lines for cortex
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=5)

        if lines is None:
            # Fallback: estimate calcar as line tangent to inferior femoral head
            cx, cy = femoral_head.center_x, femoral_head.center_y
            r = femoral_head.radius
            # Approximate calcar tangent point
            angle = np.radians(135)  # 135 degrees from horizontal (inferior-medial)
            p1 = (cx + r * np.cos(angle), cy + r * np.sin(angle))
            p2 = (cx + r * 1.5 * np.cos(angle), cy + r * 1.5 * np.sin(angle))
            return (p1, p2)

        # Find the most inferior-medial edge
        best_line = None
        best_score = -float('inf')

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Convert to global coordinates
            x1 += search_x_start
            y1 += search_y_start
            x2 += search_x_start
            y2 += search_y_start

            # Score based on position (inferior-medial preferred)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            score = mid_y - mid_x  # Prefer lower-left

            if score > best_score:
                best_score = score
                best_line = (x1, y1, x2, y2)

        if best_line:
            return ((best_line[0], best_line[1]), (best_line[2], best_line[3]))

        return None

    def measure_screw_diameter(self, image: np.ndarray,
                               screw: DetectedLine) -> Optional[float]:
        """
        Measure the apparent screw diameter in pixels.

        This is needed for magnification correction in the calTAD formula.
        """
        # Get perpendicular profile across screw at midpoint
        mid = screw.midpoint
        angle = np.radians(screw.angle + 90)  # Perpendicular

        # Sample points along perpendicular
        profile_length = 50
        x_offsets = np.arange(-profile_length, profile_length + 1) * np.cos(angle)
        y_offsets = np.arange(-profile_length, profile_length + 1) * np.sin(angle)

        h, w = image.shape
        profile = []

        for dx, dy in zip(x_offsets, y_offsets):
            x = int(mid[0] + dx)
            y = int(mid[1] + dy)
            if 0 <= x < w and 0 <= y < h:
                profile.append(image[y, x])
            else:
                profile.append(0)

        profile = np.array(profile)

        # Find width at half maximum
        threshold = (profile.max() + profile.min()) / 2
        above_threshold = profile > threshold

        # Find edges
        edges = np.diff(above_threshold.astype(int))
        rising = np.where(edges == 1)[0]
        falling = np.where(edges == -1)[0]

        if len(rising) > 0 and len(falling) > 0:
            diameter = falling[0] - rising[0]
            return float(diameter)

        return None
