"""
Tests for calTAD pipeline.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import euclidean_distance, point_to_line_distance
from src.roi_detection import ROIDetector, ROI
from src.landmark_detection import LandmarkDetector, LandmarkType
from src.measurement import CalTADCalculator, CalTADResult
from src.pipeline import CalTADPipeline


def test_euclidean_distance():
    """Test distance calculation."""
    d = euclidean_distance((0, 0), (3, 4))
    assert abs(d - 5.0) < 1e-6, f"Expected 5.0, got {d}"
    print("✓ euclidean_distance")


def test_point_to_line_distance():
    """Test perpendicular distance to line."""
    # Point (0, 1) to line through (0, 0) and (1, 0) should be 1
    d = point_to_line_distance((0, 1), (0, 0), (1, 0))
    assert abs(d - 1.0) < 1e-6, f"Expected 1.0, got {d}"

    # Point (1, 1) to line y=x should be 0
    d = point_to_line_distance((1, 1), (0, 0), (2, 2))
    assert abs(d) < 1e-6, f"Expected 0, got {d}"
    print("✓ point_to_line_distance")


def test_roi_detection():
    """Test ROI detector with synthetic image."""
    # Create synthetic image with bright region (simulating implant)
    img = np.zeros((500, 500), dtype=np.uint8)
    # Draw bright rectangle (implant)
    img[150:350, 200:300] = 255

    detector = ROIDetector(min_implant_area=100)
    roi = detector.detect(img)

    assert roi is not None, "ROI should be detected"
    assert roi.x < 250, f"ROI x={roi.x} should be left of center"
    assert roi.y < 250, f"ROI y={roi.y} should be above center"
    print("✓ ROI detection")


def test_landmark_detector_femoral_head():
    """Test femoral head detection with synthetic circle."""
    # Create synthetic image with circle (simulating femoral head)
    img = np.zeros((400, 400), dtype=np.uint8)
    # Draw filled circle
    import cv2
    cv2.circle(img, (200, 150), 60, 200, -1)

    detector = LandmarkDetector(
        min_femoral_head_radius=40,
        max_femoral_head_radius=80
    )
    head = detector.detect_femoral_head(img)

    assert head is not None, "Femoral head should be detected"
    assert abs(head.center_x - 200) < 20, f"Center X should be near 200, got {head.center_x}"
    assert abs(head.center_y - 150) < 20, f"Center Y should be near 150, got {head.center_y}"
    assert abs(head.radius - 60) < 15, f"Radius should be near 60, got {head.radius}"
    print("✓ Femoral head detection")


def test_landmark_detector_screw():
    """Test lag screw detection with synthetic line."""
    import cv2

    # Create synthetic image with bright line (simulating screw)
    img = np.zeros((400, 400), dtype=np.uint8)
    # Draw thick line at ~135 degrees
    cv2.line(img, (100, 300), (300, 100), 255, 10)

    detector = LandmarkDetector(
        min_screw_length=30,
        screw_angle_range=(30, 170)
    )
    screw = detector.detect_lag_screw(img)

    assert screw is not None, "Screw should be detected"
    assert screw.length > 100, f"Length should be >100, got {screw.length}"
    print("✓ Lag screw detection")


def test_caltad_calculation():
    """Test calTAD formula calculation."""
    from src.landmark_detection import Landmark

    # Create mock landmarks
    landmarks = {
        LandmarkType.SCREW_TIP: Landmark(LandmarkType.SCREW_TIP, 150, 150, 0.9),
        LandmarkType.CALCAR_POINT_1: Landmark(LandmarkType.CALCAR_POINT_1, 100, 200, 0.8),
        LandmarkType.CALCAR_POINT_2: Landmark(LandmarkType.CALCAR_POINT_2, 200, 250, 0.8),
        LandmarkType.FEMORAL_HEAD_APEX: Landmark(LandmarkType.FEMORAL_HEAD_APEX, 160, 80, 0.85),
    }

    calculator = CalTADCalculator(true_screw_diameter=12.5)

    # Test AP view calculation
    result = calculator.calculate_ap_view(landmarks, measured_screw_diameter_px=25)
    assert result.x_calcar_mm is not None, "X_calcar should be calculated"
    assert result.mag_ap is not None, "Magnification should be calculated"
    assert abs(result.mag_ap - 0.5) < 0.01, f"Expected mag=0.5, got {result.mag_ap}"
    print("✓ calTAD calculation (AP)")

    # Test lateral view calculation
    result = calculator.calculate_lateral_view(landmarks, measured_screw_diameter_px=25)
    assert result.x_lat_mm is not None, "X_lat should be calculated"
    print("✓ calTAD calculation (Lateral)")


def test_caltad_threshold():
    """Test clinical threshold logic."""
    result = CalTADResult(caltad_mm=20.0)
    assert result.is_within_threshold, "20mm should be within threshold"

    result = CalTADResult(caltad_mm=30.0)
    assert not result.is_within_threshold, "30mm should exceed threshold"
    print("✓ Clinical threshold")


def run_all_tests():
    """Run all tests."""
    print("Running calTAD pipeline tests...")
    print("=" * 40)

    test_euclidean_distance()
    test_point_to_line_distance()
    test_roi_detection()
    test_landmark_detector_femoral_head()
    test_landmark_detector_screw()
    test_caltad_calculation()
    test_caltad_threshold()

    print("=" * 40)
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
