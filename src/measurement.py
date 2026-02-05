"""
calTAD Measurement Calculation.

Stage 3 of the pipeline: Calculate calibrated tip-apex distance.

Formula: calTAD = [Xcalcar × (Dtrue/Dap)] + [Xlat × (Dtrue/Dlat)]

Where:
- Xcalcar: Distance from screw tip to calcar-tangent line (AP view)
- Xlat: Distance from screw tip to femoral head apex (Lateral view)
- Dtrue: Known true screw diameter (manufacturer spec)
- Dap: Measured screw diameter on AP view
- Dlat: Measured screw diameter on Lateral view
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from .landmark_detection import Landmark, LandmarkType
from .utils import euclidean_distance, point_to_line_distance


@dataclass
class CalTADResult:
    """Result of calTAD measurement."""
    # Combined calTAD value
    caltad_mm: Optional[float] = None

    # Component measurements (in mm after magnification correction)
    x_calcar_mm: Optional[float] = None  # AP view: tip to calcar line
    x_lat_mm: Optional[float] = None     # Lateral view: tip to apex

    # Raw pixel measurements
    x_calcar_px: Optional[float] = None
    x_lat_px: Optional[float] = None

    # Screw diameter measurements
    d_ap_px: Optional[float] = None      # Measured on AP
    d_lat_px: Optional[float] = None     # Measured on Lateral
    d_true_mm: float = 12.5              # Known true diameter

    # Magnification factors
    mag_ap: Optional[float] = None
    mag_lat: Optional[float] = None

    # Quality metrics
    confidence: float = 0.0
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @property
    def is_valid(self) -> bool:
        """Check if measurement is valid."""
        return self.caltad_mm is not None

    @property
    def is_within_threshold(self) -> bool:
        """Check if calTAD is within safe threshold (≤25mm)."""
        if self.caltad_mm is None:
            return False
        return self.caltad_mm <= 25.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.is_valid:
            return "calTAD measurement failed: insufficient landmarks detected"

        status = "SAFE" if self.is_within_threshold else "AT RISK"

        lines = [
            f"calTAD: {self.caltad_mm:.1f} mm ({status})",
            f"  - X_calcar (AP): {self.x_calcar_mm:.1f} mm" if self.x_calcar_mm else "  - X_calcar (AP): N/A",
            f"  - X_lat (Lateral): {self.x_lat_mm:.1f} mm" if self.x_lat_mm else "  - X_lat (Lateral): N/A",
            f"  - Confidence: {self.confidence:.0%}",
        ]

        if self.warnings:
            lines.append("  - Warnings: " + "; ".join(self.warnings))

        return "\n".join(lines)


class CalTADCalculator:
    """
    Calculate calibrated tip-apex distance from detected landmarks.
    """

    # Common lag screw diameters (mm)
    STANDARD_SCREW_DIAMETERS = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5]

    def __init__(self,
                 true_screw_diameter: float = 12.5,
                 pixel_spacing: Optional[float] = None):
        """
        Args:
            true_screw_diameter: Known screw diameter in mm (from manufacturer)
            pixel_spacing: Optional pixel spacing in mm/pixel (from DICOM)
        """
        self.true_screw_diameter = true_screw_diameter
        self.pixel_spacing = pixel_spacing

    def calculate_ap_view(self,
                          landmarks: Dict[LandmarkType, Landmark],
                          measured_screw_diameter_px: Optional[float] = None) -> CalTADResult:
        """
        Calculate calTAD component from AP view only.

        Measures distance from screw tip to calcar tangent line.
        """
        result = CalTADResult(d_true_mm=self.true_screw_diameter)

        # Check required landmarks
        if LandmarkType.SCREW_TIP not in landmarks:
            result.warnings.append("Screw tip not detected")
            return result

        screw_tip = landmarks[LandmarkType.SCREW_TIP]

        # Calculate X_calcar (distance to calcar line)
        if (LandmarkType.CALCAR_POINT_1 in landmarks and
            LandmarkType.CALCAR_POINT_2 in landmarks):

            calcar_p1 = landmarks[LandmarkType.CALCAR_POINT_1]
            calcar_p2 = landmarks[LandmarkType.CALCAR_POINT_2]

            result.x_calcar_px = point_to_line_distance(
                screw_tip.position,
                calcar_p1.position,
                calcar_p2.position
            )
        elif LandmarkType.FEMORAL_HEAD_APEX in landmarks:
            # Fallback: use distance to apex (standard TAD method)
            apex = landmarks[LandmarkType.FEMORAL_HEAD_APEX]
            result.x_calcar_px = euclidean_distance(screw_tip.position, apex.position)
            result.warnings.append("Using apex distance instead of calcar (fallback)")
        else:
            result.warnings.append("Cannot calculate X_calcar: missing calcar or apex landmarks")
            return result

        # Apply magnification correction
        if measured_screw_diameter_px is not None and measured_screw_diameter_px > 0:
            result.d_ap_px = measured_screw_diameter_px
            result.mag_ap = self.true_screw_diameter / measured_screw_diameter_px
            result.x_calcar_mm = result.x_calcar_px * result.mag_ap
        elif self.pixel_spacing is not None:
            result.x_calcar_mm = result.x_calcar_px * self.pixel_spacing
            result.warnings.append("Using DICOM pixel spacing instead of screw magnification")
        else:
            # Cannot convert to mm without magnification reference
            result.x_calcar_mm = result.x_calcar_px  # Report in pixels
            result.warnings.append("No magnification correction - value in pixels")

        # For AP-only measurement, calTAD = X_calcar
        result.caltad_mm = result.x_calcar_mm

        # Calculate confidence based on landmark detection confidence
        confidences = [screw_tip.confidence]
        if LandmarkType.CALCAR_POINT_1 in landmarks:
            confidences.append(landmarks[LandmarkType.CALCAR_POINT_1].confidence)
        result.confidence = np.mean(confidences)

        return result

    def calculate_lateral_view(self,
                               landmarks: Dict[LandmarkType, Landmark],
                               measured_screw_diameter_px: Optional[float] = None) -> CalTADResult:
        """
        Calculate TAD component from lateral view.

        Measures distance from screw tip to femoral head apex.
        """
        result = CalTADResult(d_true_mm=self.true_screw_diameter)

        # Check required landmarks
        if LandmarkType.SCREW_TIP not in landmarks:
            result.warnings.append("Screw tip not detected")
            return result

        if LandmarkType.FEMORAL_HEAD_APEX not in landmarks:
            result.warnings.append("Femoral head apex not detected")
            return result

        screw_tip = landmarks[LandmarkType.SCREW_TIP]
        apex = landmarks[LandmarkType.FEMORAL_HEAD_APEX]

        # Calculate X_lat
        result.x_lat_px = euclidean_distance(screw_tip.position, apex.position)

        # Apply magnification correction
        if measured_screw_diameter_px is not None and measured_screw_diameter_px > 0:
            result.d_lat_px = measured_screw_diameter_px
            result.mag_lat = self.true_screw_diameter / measured_screw_diameter_px
            result.x_lat_mm = result.x_lat_px * result.mag_lat
        elif self.pixel_spacing is not None:
            result.x_lat_mm = result.x_lat_px * self.pixel_spacing
            result.warnings.append("Using DICOM pixel spacing instead of screw magnification")
        else:
            result.x_lat_mm = result.x_lat_px
            result.warnings.append("No magnification correction - value in pixels")

        result.caltad_mm = result.x_lat_mm
        result.confidence = np.mean([screw_tip.confidence, apex.confidence])

        return result

    def calculate_combined(self,
                           ap_landmarks: Dict[LandmarkType, Landmark],
                           lat_landmarks: Dict[LandmarkType, Landmark],
                           ap_screw_diameter_px: Optional[float] = None,
                           lat_screw_diameter_px: Optional[float] = None) -> CalTADResult:
        """
        Calculate full calTAD from both AP and lateral views.

        calTAD = [Xcalcar × (Dtrue/Dap)] + [Xlat × (Dtrue/Dlat)]
        """
        ap_result = self.calculate_ap_view(ap_landmarks, ap_screw_diameter_px)
        lat_result = self.calculate_lateral_view(lat_landmarks, lat_screw_diameter_px)

        # Combine results
        result = CalTADResult(d_true_mm=self.true_screw_diameter)
        result.x_calcar_px = ap_result.x_calcar_px
        result.x_calcar_mm = ap_result.x_calcar_mm
        result.x_lat_px = lat_result.x_lat_px
        result.x_lat_mm = lat_result.x_lat_mm
        result.d_ap_px = ap_result.d_ap_px
        result.d_lat_px = lat_result.d_lat_px
        result.mag_ap = ap_result.mag_ap
        result.mag_lat = lat_result.mag_lat
        result.warnings = ap_result.warnings + lat_result.warnings

        # Calculate combined calTAD
        if ap_result.x_calcar_mm is not None and lat_result.x_lat_mm is not None:
            result.caltad_mm = ap_result.x_calcar_mm + lat_result.x_lat_mm
            result.confidence = (ap_result.confidence + lat_result.confidence) / 2
        elif ap_result.x_calcar_mm is not None:
            result.caltad_mm = ap_result.x_calcar_mm
            result.confidence = ap_result.confidence
            result.warnings.append("Lateral view unavailable - using AP only")
        elif lat_result.x_lat_mm is not None:
            result.caltad_mm = lat_result.x_lat_mm
            result.confidence = lat_result.confidence
            result.warnings.append("AP view unavailable - using lateral only")

        return result


def estimate_magnification(measured_diameter_px: float,
                          known_diameter_mm: float = 12.5) -> float:
    """
    Estimate magnification factor from screw diameter measurement.

    Returns conversion factor: mm_per_pixel = known_diameter_mm / measured_diameter_px
    """
    return known_diameter_mm / measured_diameter_px
