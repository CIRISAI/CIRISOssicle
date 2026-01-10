#!/usr/bin/env python3
"""
Experiment 16: Relative Detection with Tetrahedral Sensor

Implements adaptive/relative detection to handle correlation instability:
1. Running baseline with exponential moving average
2. Drift detection using relative z-scores
3. Attack/recovery transition detection
4. Compare absolute vs relative thresholds

Key insight from Exp14: Correlations are NOT stable between runs.
→ Need relative detection (change from recent baseline) not absolute thresholds.

Author: CIRIS L3C
License: BSL 1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cupy as cp
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from collections import deque

from tetrahedral_sensor import TetrahedralSensor, TetraConfig


class WorkloadGenerator:
    """Generate GPU workloads."""

    def __init__(self, workload_type: str, intensity: float = 0.7):
        self.workload_type = workload_type
        self.intensity = intensity
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        size = int(1024 * 8 * self.intensity)

        if self.workload_type == "idle":
            while self.running:
                time.sleep(0.01)

        elif self.workload_type == "crypto":
            data = cp.random.randint(0, 2**32, size=(size,), dtype=cp.uint32)
            while self.running:
                for _ in range(200):
                    data = data ^ (data << 13)
                    data = data ^ (data >> 17)
                    data = data ^ (data << 5)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.002)


class RelativeDetector:
    """
    Implements relative detection with adaptive baseline.

    Uses exponential moving average (EMA) for smooth baseline tracking,
    with separate slow (baseline) and fast (signal) EMAs.
    """

    def __init__(self,
                 baseline_alpha: float = 0.02,  # Slow EMA for baseline
                 signal_alpha: float = 0.2,     # Fast EMA for current signal
                 threshold_sigma: float = 3.0,
                 warmup_samples: int = 50):
        self.baseline_alpha = baseline_alpha
        self.signal_alpha = signal_alpha
        self.threshold_sigma = threshold_sigma
        self.warmup_samples = warmup_samples

        # State
        self.baseline_ema = None
        self.signal_ema = None
        self.variance_ema = None
        self.sample_count = 0

        # History for analysis
        self.history = []

    def update(self, correlation_mean: float) -> dict:
        """Update detector with new correlation reading."""
        self.sample_count += 1

        # Initialize EMAs
        if self.baseline_ema is None:
            self.baseline_ema = correlation_mean
            self.signal_ema = correlation_mean
            self.variance_ema = 0.001  # Small initial variance

        # Update EMAs
        self.baseline_ema = (1 - self.baseline_alpha) * self.baseline_ema + self.baseline_alpha * correlation_mean
        self.signal_ema = (1 - self.signal_alpha) * self.signal_ema + self.signal_alpha * correlation_mean

        # Update variance estimate (of baseline EMA)
        deviation = correlation_mean - self.baseline_ema
        self.variance_ema = (1 - self.baseline_alpha) * self.variance_ema + self.baseline_alpha * (deviation ** 2)

        # Compute relative z-score
        std = np.sqrt(self.variance_ema) + 1e-10
        z_score = (self.signal_ema - self.baseline_ema) / std

        # Detection logic
        if self.sample_count < self.warmup_samples:
            state = 'calibrating'
            alert = False
        else:
            state = 'detecting'
            alert = abs(z_score) > self.threshold_sigma

        result = {
            'sample': self.sample_count,
            'state': state,
            'raw': float(correlation_mean),
            'baseline': float(self.baseline_ema),
            'signal': float(self.signal_ema),
            'std': float(std),
            'z_score': float(z_score),
            'alert': bool(alert)
        }

        self.history.append(result)
        return result

    def reset(self):
        """Reset detector state."""
        self.baseline_ema = None
        self.signal_ema = None
        self.variance_ema = None
        self.sample_count = 0
        self.history = []


class AbsoluteDetector:
    """Simple absolute threshold detector for comparison."""

    def __init__(self, threshold: float = -0.1, above: bool = True):
        self.threshold = threshold
        self.above = above
        self.history = []
        self.sample_count = 0

    def update(self, correlation_mean: float) -> dict:
        self.sample_count += 1

        if self.above:
            alert = correlation_mean > self.threshold
        else:
            alert = correlation_mean < self.threshold

        result = {
            'sample': self.sample_count,
            'raw': float(correlation_mean),
            'threshold': self.threshold,
            'alert': bool(alert)
        }
        self.history.append(result)
        return result

    def reset(self):
        self.history = []
        self.sample_count = 0


def run_relative_detection_test():
    """Run relative vs absolute detection comparison."""

    print("="*70)
    print("EXPERIMENT 16: RELATIVE DETECTION")
    print("="*70)
    print()
    print("Testing adaptive baseline for stable attack detection")
    print()

    results = {
        'experiment': 'exp16_relative_detection',
        'timestamp': datetime.now().isoformat()
    }

    # Initialize sensor
    sensor = TetrahedralSensor(TetraConfig(n_iterations=5000))

    # Initialize detectors
    relative_detector = RelativeDetector(
        baseline_alpha=0.02,  # ~50 sample time constant
        signal_alpha=0.2,     # ~5 sample time constant
        threshold_sigma=3.0,
        warmup_samples=50
    )

    absolute_detector = AbsoluteDetector(threshold=0.1, above=True)

    # =========================================================================
    # SCENARIO: IDLE → ATTACK → IDLE → ATTACK → IDLE
    # =========================================================================
    print("[SCENARIO] Multi-phase attack/recovery test")
    print("-"*50)

    phases = [
        ('idle', 30.0),
        ('attack', 30.0),
        ('idle', 30.0),
        ('attack', 20.0),
        ('idle', 20.0)
    ]

    timeline = []
    phase_start = 0

    for phase_name, duration in phases:
        print(f"\n[{phase_name.upper()}] Running for {duration:.0f}s...")

        if phase_name == 'idle':
            workload = WorkloadGenerator("idle")
        else:
            workload = WorkloadGenerator("crypto", intensity=0.8)

        workload.start()
        sensor.reset_history()
        time.sleep(1)  # Stabilize

        phase_alerts_rel = 0
        phase_alerts_abs = 0
        phase_samples = 0

        start = time.time()
        while time.time() - start < duration:
            reading = sensor.read()

            # Only process when we have valid correlations
            if reading.mean_correlation != 0:
                rel_result = relative_detector.update(reading.mean_correlation)
                abs_result = absolute_detector.update(reading.mean_correlation)

                if rel_result['alert']:
                    phase_alerts_rel += 1
                if abs_result['alert']:
                    phase_alerts_abs += 1
                phase_samples += 1

                timeline.append({
                    'time': phase_start + (time.time() - start),
                    'phase': phase_name,
                    'correlation': float(reading.mean_correlation),
                    'rel_z': rel_result['z_score'],
                    'rel_alert': rel_result['alert'],
                    'abs_alert': abs_result['alert']
                })

        workload.stop()

        phase_start += duration

        print(f"  Samples: {phase_samples}")
        print(f"  Relative alerts: {phase_alerts_rel} ({phase_alerts_rel/max(phase_samples,1)*100:.1f}%)")
        print(f"  Absolute alerts: {phase_alerts_abs} ({phase_alerts_abs/max(phase_samples,1)*100:.1f}%)")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Compute per-phase statistics
    phase_stats = {}
    for phase in ['idle', 'attack']:
        phase_data = [t for t in timeline if t['phase'] == phase]
        if phase_data:
            corrs = [t['correlation'] for t in phase_data]
            rel_alerts = sum(1 for t in phase_data if t['rel_alert'])
            abs_alerts = sum(1 for t in phase_data if t['abs_alert'])

            phase_stats[phase] = {
                'samples': len(phase_data),
                'mean_corr': float(np.mean(corrs)),
                'std_corr': float(np.std(corrs)),
                'rel_alert_rate': rel_alerts / len(phase_data),
                'abs_alert_rate': abs_alerts / len(phase_data)
            }

    print("\nPhase Statistics:")
    print(f"| Phase | Mean ρ | σ | Rel Alert % | Abs Alert % |")
    print(f"|-------|--------|---|-------------|-------------|")
    for phase, stats in phase_stats.items():
        print(f"| {phase:5} | {stats['mean_corr']:+.4f} | {stats['std_corr']:.4f} | {stats['rel_alert_rate']*100:5.1f}% | {stats['abs_alert_rate']*100:5.1f}% |")

    # Detection quality metrics
    if 'idle' in phase_stats and 'attack' in phase_stats:
        # True positive rate (attacks detected)
        tpr_rel = phase_stats['attack']['rel_alert_rate']
        tpr_abs = phase_stats['attack']['abs_alert_rate']

        # False positive rate (idle falsely alerted)
        fpr_rel = phase_stats['idle']['rel_alert_rate']
        fpr_abs = phase_stats['idle']['abs_alert_rate']

        print(f"\nDetection Quality:")
        print(f"| Metric | Relative | Absolute |")
        print(f"|--------|----------|----------|")
        print(f"| TPR (attack) | {tpr_rel*100:.1f}% | {tpr_abs*100:.1f}% |")
        print(f"| FPR (idle) | {fpr_rel*100:.1f}% | {fpr_abs*100:.1f}% |")

        # F1-like score
        if tpr_rel + (1 - fpr_rel) > 0:
            f1_rel = 2 * tpr_rel * (1 - fpr_rel) / (tpr_rel + (1 - fpr_rel))
        else:
            f1_rel = 0

        if tpr_abs + (1 - fpr_abs) > 0:
            f1_abs = 2 * tpr_abs * (1 - fpr_abs) / (tpr_abs + (1 - fpr_abs))
        else:
            f1_abs = 0

        print(f"| F1-like | {f1_rel:.3f} | {f1_abs:.3f} |")

        results['detection_quality'] = {
            'relative': {'tpr': tpr_rel, 'fpr': fpr_rel, 'f1': f1_rel},
            'absolute': {'tpr': tpr_abs, 'fpr': fpr_abs, 'f1': f1_abs}
        }

    results['phase_stats'] = phase_stats

    # =========================================================================
    # TRANSITION ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("TRANSITION ANALYSIS")
    print("-"*50)

    # Find transitions
    transitions = []
    for i in range(1, len(timeline)):
        if timeline[i]['phase'] != timeline[i-1]['phase']:
            transitions.append({
                'time': timeline[i]['time'],
                'from': timeline[i-1]['phase'],
                'to': timeline[i]['phase'],
                'index': i
            })

    print(f"Found {len(transitions)} transitions:")

    for trans in transitions:
        # Find time to first detection after transition
        if trans['to'] == 'attack':
            # Look for first alert after transition
            detection_delay = None
            for j in range(trans['index'], min(trans['index'] + 100, len(timeline))):
                if timeline[j]['rel_alert']:
                    detection_delay = timeline[j]['time'] - trans['time']
                    break
            print(f"  {trans['from']} → {trans['to']} at t={trans['time']:.1f}s: detection delay = {detection_delay:.2f}s" if detection_delay else f"  {trans['from']} → {trans['to']} at t={trans['time']:.1f}s: NOT DETECTED")

        else:  # Recovery
            # Look for first non-alert after transition
            recovery_delay = None
            for j in range(trans['index'], min(trans['index'] + 100, len(timeline))):
                if not timeline[j]['rel_alert']:
                    recovery_delay = timeline[j]['time'] - trans['time']
                    break
            print(f"  {trans['from']} → {trans['to']} at t={trans['time']:.1f}s: recovery delay = {recovery_delay:.2f}s" if recovery_delay else f"  {trans['from']} → {trans['to']} at t={trans['time']:.1f}s: STUCK IN ALERT")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("""
RELATIVE DETECTION ADVANTAGES:
  1. Adapts to correlation drift between runs
  2. No manual threshold tuning required
  3. Handles baseline shifts gracefully
  4. EMA smoothing reduces noise false positives

ABSOLUTE DETECTION DISADVANTAGES:
  1. Fixed threshold may be wrong for current conditions
  2. Baseline drift causes false positives/negatives
  3. Requires manual calibration per GPU

RECOMMENDED CONFIGURATION:
  - baseline_alpha = 0.02 (slow adaptation, ~50 sample window)
  - signal_alpha = 0.2 (fast response, ~5 sample window)
  - threshold = 3σ for high confidence
""")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp16_relative_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_relative_detection_test()
