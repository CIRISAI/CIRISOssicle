#!/usr/bin/env python3
"""
Strain Sensor Demonstration

Shows the 3-crystal Schützhold detector architecture and takes sample readings.

Usage:
    python run_demo.py
    python run_demo.py --samples 20

Author: CIRIS L3C (Eric Moore)
License: BSL 1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import time
import numpy as np
from strain_sensor import StrainSensor, SensorConfig


def main():
    parser = argparse.ArgumentParser(description='Strain sensor demonstration')
    parser.add_argument('--samples', '-n', type=int, default=10,
                        help='Number of samples to collect (default: 10)')

    args = parser.parse_args()

    print("="*60)
    print("STRAIN SENSOR - 3-CRYSTAL SCHÜTZHOLD DETECTOR")
    print("="*60)
    print()
    print("Architecture:")
    print("  Crystal A: r=3.70 (reference)")
    print("  Crystal B: r=3.73 (60° equivalent)")
    print("  Crystal C: r=3.76 (120° equivalent)")
    print()
    print("Detection metric: Pairwise correlations ρ(A,B), ρ(B,C), ρ(A,C)")
    print("Strain causes correlation variance to increase ~6×")
    print()

    print("-"*60)
    print("COLLECTING SAMPLES")
    print("-"*60)
    print()

    sensor = StrainSensor()

    readings = []
    for i in range(args.samples):
        rho_ab, rho_bc, rho_ac = sensor.read_raw()
        readings.append((rho_ab, rho_bc, rho_ac))
        print(f"  [{i+1:2d}] ρ(A,B)={rho_ab:+.4f}  ρ(B,C)={rho_bc:+.4f}  ρ(A,C)={rho_ac:+.4f}")

    print()
    print("-"*60)
    print("STATISTICS")
    print("-"*60)
    print()

    rho_ab = [r[0] for r in readings]
    rho_bc = [r[1] for r in readings]
    rho_ac = [r[2] for r in readings]

    print(f"  ρ(A,B): mean={np.mean(rho_ab):+.4f}  std={np.std(rho_ab):.4f}")
    print(f"  ρ(B,C): mean={np.mean(rho_bc):+.4f}  std={np.std(rho_bc):.4f}")
    print(f"  ρ(A,C): mean={np.mean(rho_ac):+.4f}  std={np.std(rho_ac):.4f}")
    print()

    total_var = np.var(rho_ab) + np.var(rho_bc) + np.var(rho_ac)
    print(f"  Total variance: {total_var:.6f}")
    print()

    print("="*60)
    print("USAGE")
    print("="*60)
    print("""
  To calibrate baseline (keep sensor STILL):
      python calibrate.py --duration 60

  To run detection (shake to test):
      python detect.py --threshold 3.0

  To compare baseline vs shaking:
      python triple_rotation_test.py baseline 30
      python triple_rotation_test.py rotated 30  # shake during this
      python triple_rotation_test.py compare
    """)


if __name__ == "__main__":
    main()
