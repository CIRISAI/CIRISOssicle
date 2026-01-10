#!/usr/bin/env python3
"""
Strain Detection - Real-time Monitoring

Monitors for strain events using calibrated baseline.

Usage:
    python detect.py --baseline data/baseline.json
    python detect.py --threshold 3.0
    python detect.py --duration 3600    # Monitor for 1 hour

Author: CIRIS L3C (Eric Moore)
License: BSL 1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from strain_sensor import StrainSensor, SensorConfig, SensorReading


def format_reading(reading: SensorReading, show_correlations: bool = True) -> str:
    """Format a reading for display."""
    ts = datetime.fromtimestamp(reading.timestamp).strftime('%H:%M:%S')

    if reading.detected:
        if reading.max_z >= 5.0:
            status = f"*** {reading.max_z:.1f}σ ALERT ***"
        else:
            status = f"**  {reading.max_z:.1f}σ DETECTED **"
    else:
        status = f"    {reading.max_z:.1f}σ"

    if show_correlations:
        return (f"[{ts}] "
                f"ρ(A,B)={reading.rho_ab:+.3f} "
                f"ρ(B,C)={reading.rho_bc:+.3f} "
                f"ρ(A,C)={reading.rho_ac:+.3f} "
                f"{status}")
    else:
        return f"[{ts}] {status}"


def main():
    parser = argparse.ArgumentParser(
        description='Real-time strain detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python detect.py                                # Use default baseline
    python detect.py --baseline my_baseline.json
    python detect.py --threshold 5.0               # Higher threshold (fewer false positives)
    python detect.py --duration 3600               # Monitor for 1 hour
    python detect.py --log events.json             # Log events to file
        """
    )
    parser.add_argument('--baseline', '-b', type=str, default='data/baseline.json',
                        help='Baseline file (default: data/baseline.json)')
    parser.add_argument('--threshold', '-t', type=float, default=3.0,
                        help='Detection threshold in sigma (default: 3.0)')
    parser.add_argument('--duration', '-d', type=float, default=None,
                        help='Monitoring duration in seconds (default: indefinite)')
    parser.add_argument('--log', '-l', type=str, default=None,
                        help='Log events to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Only print detected events')

    args = parser.parse_args()

    print("="*60)
    print("STRAIN DETECTION - REAL-TIME MONITOR")
    print("="*60)
    print()

    # Check baseline exists
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"ERROR: Baseline file not found: {baseline_path}")
        print()
        print("Run calibration first:")
        print(f"    python calibrate.py --output {baseline_path}")
        sys.exit(1)

    # Load sensor
    config = SensorConfig(z_threshold=args.threshold)
    sensor = StrainSensor(config)
    sensor.load_baseline(str(baseline_path))

    print()
    print(f"Detection threshold: {args.threshold}σ")
    if args.duration:
        print(f"Duration: {args.duration}s ({args.duration/60:.1f} min)")
    else:
        print("Duration: indefinite (Ctrl+C to stop)")
    print()
    print("-"*60)

    # Event log
    events = []

    start_time = time.time()
    sample_count = 0
    event_count = 0

    try:
        while args.duration is None or (time.time() - start_time) < args.duration:
            reading = sensor.read()
            sample_count += 1

            if reading.detected:
                event_count += 1
                events.append({
                    'timestamp': reading.timestamp,
                    'iso_time': datetime.fromtimestamp(reading.timestamp).isoformat(),
                    'rho_ab': reading.rho_ab,
                    'rho_bc': reading.rho_bc,
                    'rho_ac': reading.rho_ac,
                    'z_ab': reading.z_ab,
                    'z_bc': reading.z_bc,
                    'z_ac': reading.z_ac,
                    'max_z': reading.max_z,
                })

            if not args.quiet or reading.detected:
                print(format_reading(reading))

    except KeyboardInterrupt:
        print()

    # Summary
    elapsed = time.time() - start_time
    print("-"*60)
    print()
    print("="*60)
    print("MONITORING SUMMARY")
    print("="*60)
    print(f"Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Samples: {sample_count}")
    print(f"Events detected: {event_count}")
    if sample_count > 0:
        print(f"Event rate: {100*event_count/sample_count:.2f}%")

    # Save events log
    if args.log and events:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'duration_sec': elapsed,
            'samples': sample_count,
            'threshold': args.threshold,
            'baseline_file': str(baseline_path),
            'events': events
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"\nEvents logged to: {log_path}")

    elif args.log and not events:
        print(f"\nNo events to log.")


if __name__ == "__main__":
    main()
