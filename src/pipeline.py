"""
Main calTAD Measurement Pipeline.

Orchestrates the three-stage process:
1. ROI Detection
2. Landmark Detection
3. calTAD Calculation

Requires both AP and lateral views for complete measurement.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass, field

from .utils import load_image, preprocess_xray, get_pixel_spacing
from .roi_detection import ROIDetector, ROI
from .landmark_detection import LandmarkDetector, Landmark, LandmarkType
from .measurement import CalTADCalculator, CalTADResult
from .visualization import (
    draw_landmarks, draw_measurement, draw_roi,
    save_visualization
)


@dataclass
class ViewResult:
    """Result from processing a single view."""
    image_path: str
    view_type: str
    roi: Optional[ROI] = None
    landmarks: Dict[LandmarkType, Landmark] = field(default_factory=dict)
    processed_image: Optional[np.ndarray] = None
    visualization: Optional[np.ndarray] = None
    screw_diameter_px: Optional[float] = None


@dataclass
class PipelineResult:
    """Complete result from calTAD pipeline (both views)."""
    # View results
    ap_result: Optional[ViewResult] = None
    lateral_result: Optional[ViewResult] = None

    # Combined measurement
    measurement: Optional[CalTADResult] = None

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.measurement is not None and self.measurement.is_valid

    @property
    def ap_visualization(self) -> Optional[np.ndarray]:
        return self.ap_result.visualization if self.ap_result else None

    @property
    def lateral_visualization(self) -> Optional[np.ndarray]:
        return self.lateral_result.visualization if self.lateral_result else None

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = ["calTAD Pipeline Results"]
        lines.append("=" * 40)

        # AP view info
        if self.ap_result:
            lines.append(f"\nAP View: {self.ap_result.image_path}")
            if self.ap_result.roi:
                lines.append(f"  ROI: {self.ap_result.roi.bbox} (conf: {self.ap_result.roi.confidence:.0%})")
            lines.append(f"  Landmarks: {len(self.ap_result.landmarks)}")
            for lm_type, lm in self.ap_result.landmarks.items():
                lines.append(f"    - {lm_type.value}: ({lm.x:.1f}, {lm.y:.1f})")

        # Lateral view info
        if self.lateral_result:
            lines.append(f"\nLateral View: {self.lateral_result.image_path}")
            if self.lateral_result.roi:
                lines.append(f"  ROI: {self.lateral_result.roi.bbox} (conf: {self.lateral_result.roi.confidence:.0%})")
            lines.append(f"  Landmarks: {len(self.lateral_result.landmarks)}")
            for lm_type, lm in self.lateral_result.landmarks.items():
                lines.append(f"    - {lm_type.value}: ({lm.x:.1f}, {lm.y:.1f})")

        # Measurement
        if self.measurement:
            lines.append("\n" + "-" * 40)
            lines.append(self.measurement.summary())

        return "\n".join(lines)


class CalTADPipeline:
    """
    End-to-end pipeline for automated calTAD measurement.

    Requires both AP and lateral views for complete measurement.

    Usage:
        pipeline = CalTADPipeline(screw_diameter=12.5)
        result = pipeline.process("ap.png", "lateral.png")
        print(result.summary())
    """

    def __init__(self,
                 screw_diameter: float = 12.5,
                 enhance_contrast: bool = True,
                 use_roi_detection: bool = True):
        """
        Initialize the calTAD pipeline.

        Args:
            screw_diameter: Known true screw diameter in mm
            enhance_contrast: Apply CLAHE preprocessing
            use_roi_detection: Whether to detect and crop ROI first
        """
        self.screw_diameter = screw_diameter
        self.enhance_contrast = enhance_contrast
        self.use_roi_detection = use_roi_detection

        # Initialize components
        self.roi_detector = ROIDetector()
        self.landmark_detector = LandmarkDetector()
        self.calculator = CalTADCalculator(true_screw_diameter=screw_diameter)

    def _process_single_view(self,
                             image_path: Union[str, Path],
                             view_type: str) -> ViewResult:
        """
        Process a single view (internal method).

        Args:
            image_path: Path to X-ray image
            view_type: "ap" or "lateral"

        Returns:
            ViewResult with landmarks and visualization
        """
        result = ViewResult(
            image_path=str(image_path),
            view_type=view_type
        )

        # Load and preprocess image
        image = load_image(image_path)
        processed = preprocess_xray(image, enhance_contrast=self.enhance_contrast)
        result.processed_image = processed

        # Try to get pixel spacing from DICOM
        pixel_spacing = get_pixel_spacing(image_path)
        if pixel_spacing:
            self.calculator.pixel_spacing = pixel_spacing[0]

        # Stage 1: ROI Detection
        working_image = processed
        roi_offset_x, roi_offset_y = 0, 0

        if self.use_roi_detection:
            roi = self.roi_detector.detect(processed)
            result.roi = roi
            if roi and roi.confidence > 0.5:
                padding = 30
                working_image = roi.crop(processed, padding=padding)
                roi_offset_x = roi.x - padding
                roi_offset_y = roi.y - padding

        # Stage 2: Landmark Detection
        landmarks = self.landmark_detector.detect_all(working_image)

        # Adjust landmark coordinates back to original image space
        if roi_offset_x != 0 or roi_offset_y != 0:
            adjusted_landmarks = {}
            for lm_type, lm in landmarks.items():
                adjusted_landmarks[lm_type] = Landmark(
                    lm_type,
                    lm.x + roi_offset_x,
                    lm.y + roi_offset_y,
                    lm.confidence
                )
            landmarks = adjusted_landmarks

        result.landmarks = landmarks

        # Measure screw diameter for magnification correction
        if LandmarkType.SCREW_TIP in landmarks:
            screw = self.landmark_detector.detect_lag_screw(working_image)
            if screw:
                result.screw_diameter_px = self.landmark_detector.measure_screw_diameter(
                    working_image, screw
                )

        # Generate visualization
        result.visualization = draw_measurement(
            processed,
            landmarks,
            CalTADResult()  # Empty result for now, just show landmarks
        )

        return result

    def process(self,
                ap_image_path: Union[str, Path],
                lateral_image_path: Union[str, Path]) -> PipelineResult:
        """
        Process paired AP and lateral views for complete calTAD measurement.

        Args:
            ap_image_path: Path to AP view X-ray
            lateral_image_path: Path to lateral view X-ray

        Returns:
            PipelineResult with measurements from both views
        """
        result = PipelineResult()

        # Process each view
        result.ap_result = self._process_single_view(ap_image_path, "ap")
        result.lateral_result = self._process_single_view(lateral_image_path, "lateral")

        # Stage 3: Combined measurement calculation
        result.measurement = self.calculator.calculate_combined(
            result.ap_result.landmarks,
            result.lateral_result.landmarks,
            result.ap_result.screw_diameter_px,
            result.lateral_result.screw_diameter_px
        )

        # Update visualizations with measurement results
        if result.ap_result.processed_image is not None:
            result.ap_result.visualization = draw_measurement(
                result.ap_result.processed_image,
                result.ap_result.landmarks,
                result.measurement
            )

        if result.lateral_result.processed_image is not None:
            result.lateral_result.visualization = draw_measurement(
                result.lateral_result.processed_image,
                result.lateral_result.landmarks,
                result.measurement
            )

        return result

    def create_summary(self, result: PipelineResult) -> np.ndarray:
        """Create side-by-side summary figure with both views."""
        import cv2

        panels = []

        # AP panel
        if result.ap_result and result.ap_result.visualization is not None:
            ap_vis = result.ap_result.visualization.copy()
            cv2.putText(ap_vis, "AP View", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panels.append(ap_vis)

        # Lateral panel
        if result.lateral_result and result.lateral_result.visualization is not None:
            lat_vis = result.lateral_result.visualization.copy()
            cv2.putText(lat_vis, "Lateral View", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panels.append(lat_vis)

        if not panels:
            raise ValueError("No visualizations available")

        # Resize to same height
        target_h = max(p.shape[0] for p in panels)
        resized = []
        for panel in panels:
            if panel.shape[0] != target_h:
                scale = target_h / panel.shape[0]
                new_w = int(panel.shape[1] * scale)
                panel = cv2.resize(panel, (new_w, target_h))
            resized.append(panel)

        # Concatenate horizontally
        summary = np.hstack(resized)

        # Add measurement result bar at bottom
        if result.measurement and result.measurement.is_valid:
            bar_height = 60
            bar = np.zeros((bar_height, summary.shape[1], 3), dtype=np.uint8)

            color = (0, 200, 0) if result.measurement.is_within_threshold else (0, 0, 200)
            status = "SAFE (<= 25mm)" if result.measurement.is_within_threshold else "AT RISK (> 25mm)"

            text = f"calTAD: {result.measurement.caltad_mm:.1f} mm - {status}"
            cv2.putText(bar, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            summary = np.vstack([summary, bar])

        return summary

    def save_result(self, result: PipelineResult, output_dir: Union[str, Path]):
        """Save all visualizations and summary to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if result.ap_visualization is not None:
            save_visualization(result.ap_visualization,
                             str(output_dir / "ap_result.png"))

        if result.lateral_visualization is not None:
            save_visualization(result.lateral_visualization,
                             str(output_dir / "lateral_result.png"))

        # Summary figure
        summary = self.create_summary(result)
        save_visualization(summary, str(output_dir / "caltad_summary.png"))

        # Text summary
        with open(output_dir / "results.txt", 'w') as f:
            f.write(result.summary())
