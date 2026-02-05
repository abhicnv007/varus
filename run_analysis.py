#!/usr/bin/env python3
"""
calTAD Analysis Runner

Usage:
    python run_analysis.py <ap_image> <lateral_image> [--screw-diameter 12.5]

Example:
    python run_analysis.py samples/ap.png samples/lateral.png
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import CalTADPipeline
from src.visualization import save_visualization


def main():
    parser = argparse.ArgumentParser(
        description="Automated calTAD measurement from hip X-rays"
    )
    parser.add_argument(
        "ap_image",
        type=str,
        help="Path to AP (anteroposterior) view X-ray"
    )
    parser.add_argument(
        "lateral_image",
        type=str,
        help="Path to lateral view X-ray"
    )
    parser.add_argument(
        "--screw-diameter", "-d",
        type=float,
        default=12.5,
        help="Known screw diameter in mm (default: 12.5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results (default: same as input)"
    )
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Disable ROI detection (process full image)"
    )

    args = parser.parse_args()

    # Validate inputs
    ap_path = Path(args.ap_image)
    lateral_path = Path(args.lateral_image)

    if not ap_path.exists():
        print(f"Error: AP image not found: {ap_path}")
        sys.exit(1)

    if not lateral_path.exists():
        print(f"Error: Lateral image not found: {lateral_path}")
        sys.exit(1)

    # Initialize pipeline
    pipeline = CalTADPipeline(
        screw_diameter=args.screw_diameter,
        use_roi_detection=not args.no_roi
    )

    print(f"AP image: {ap_path}")
    print(f"Lateral image: {lateral_path}")
    print(f"Screw diameter: {args.screw_diameter} mm")
    print("-" * 40)

    # Process paired images
    result = pipeline.process(ap_path, lateral_path)

    # Print results
    print()
    print(result.summary())
    print()

    # Save visualization
    output_dir = Path(args.output) if args.output else ap_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if result.ap_visualization is not None:
        ap_output = output_dir / f"{ap_path.stem}_result.png"
        save_visualization(result.ap_visualization, str(ap_output))
        print(f"AP visualization saved: {ap_output}")

    if result.lateral_visualization is not None:
        lat_output = output_dir / f"{lateral_path.stem}_result.png"
        save_visualization(result.lateral_visualization, str(lat_output))
        print(f"Lateral visualization saved: {lat_output}")

    # Save combined summary
    summary = pipeline.create_summary(result)
    summary_path = output_dir / "caltad_summary.png"
    save_visualization(summary, str(summary_path))
    print(f"Summary saved: {summary_path}")

    # Clinical interpretation
    if result.measurement and result.measurement.is_valid:
        print()
        if result.measurement.is_within_threshold:
            print("✓ calTAD ≤ 25mm: Low risk of lag screw cut-out")
        else:
            print("⚠ calTAD > 25mm: Elevated risk of lag screw cut-out")
            print("  Consider clinical correlation and possible revision")


if __name__ == "__main__":
    main()
