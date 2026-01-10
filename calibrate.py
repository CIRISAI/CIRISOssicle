#!/usr/bin/env python3
"""
Strain Sensor Calibration

Establishes baseline correlation statistics for detection.
Run this with the sensor STILL to characterize normal operation.

Usage:
    python calibrate.py --duration 60       # 1-minute calibration
    python calibrate.py --duration 3600     # 1-hour calibration (recommended)
    python calibrate.py --output baseline.json

Author: CIRIS L3C (Eric Moore)
License: BSL 1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
from pathlib import Path
from strain_sensor import StrainSensor, SensorConfig


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate strain sensor baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python calibrate.py --duration 60           # Quick 1-minute calibration
    python calibrate.py --duration 3600         # Full 1-hour calibration
    python calibrate.py -d 300 -o my_baseline.json
        """
    )
    parser.add_argument('--duration', '-d', type=float, default=60,
                        help='Calibration duration in seconds (default: 60)')
    parser.add_argument('--output', '-o', type=str, default='data/baseline.json',
                        help='Output file for baseline (default: data/baseline.json)')
    parser.add_argument('--threshold', '-t', type=float, default=3.0,
                        help='Detection threshold in sigma (default: 3.0)')

    args = parser.parse_args()

    print("="*60)
    print("STRAIN SENSOR CALIBRATION")
    print("="*60)
    print()
    print("Architecture: 3-Crystal Schützhold Detector")
    print("  Crystal A: r=3.70")
    print("  Crystal B: r=3.73")
    print("  Crystal C: r=3.76")
    print()
    print(f"Duration: {args.duration}s ({args.duration/60:.1f} min)")
    print(f"Output: {args.output}")
    print()
    print("*** KEEP THE SENSOR STILL DURING CALIBRATION ***")
    print()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure and calibrate
    config = SensorConfig(z_threshold=args.threshold)
    sensor = StrainSensor(config)

    baseline = sensor.calibrate(duration=args.duration)
    baseline.save(str(output_path))

    print()
    print("="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"Baseline saved to: {output_path}")
    print()
    print("To run detection:")
    print(f"    python detect.py --baseline {output_path}")


if __name__ == "__main__":
    main()
