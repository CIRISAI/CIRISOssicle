#!/usr/bin/env python3
"""
Experiment 9: Real-Time Tamper Detection System

Statistical Process Control approach:
1. Establish baseline fingerprint during "clean" operation
2. Monitor correlation in real-time during execution
3. Detect when correlation deviates from baseline (CUSUM, control charts)

This mimics how you'd actually use this in production:
- Train on known-good workload
- Alert when something unexpected happens

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
from strain_sensor import StrainSensor, SensorConfig


class TamperDetector:
    """Real-time tamper detection using correlation fingerprints."""

    def __init__(self, sensor: StrainSensor, window_size: int = 500):
        self.sensor = sensor
        self.window_size = window_size

        # Baseline statistics (set during training)
        self.baseline_mean = None
        self.baseline_std = None

        # Detection parameters
        self.threshold_sigma = 3.0  # Alert at 3σ deviation
        self.cusum_threshold = 5.0  # CUSUM alert threshold

        # Runtime state
        self.means_a = []
        self.means_b = []
        self.correlations = []
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.alerts = []

    def train(self, duration: float = 30.0):
        """Train on known-good workload to establish baseline."""
        print(f"Training baseline for {duration}s...")

        self.means_a = []
        self.means_b = []
        correlations = []

        start = time.time()
        while time.time() - start < duration:
            mean_a, mean_b, _ = self.sensor.read_raw()
            self.means_a.append(mean_a)
            self.means_b.append(mean_b)

            n = len(self.means_a)
            if n >= self.window_size and n % (self.window_size // 4) == 0:
                rho = np.corrcoef(
                    self.means_a[-self.window_size:],
                    self.means_b[-self.window_size:]
                )[0, 1]
                correlations.append(rho)

        self.baseline_mean = np.mean(correlations)
        self.baseline_std = np.std(correlations)
        self.correlations = correlations

        print(f"  Baseline: ρ = {self.baseline_mean:.4f} ± {self.baseline_std:.4f}")
        print(f"  Alert threshold: {self.baseline_mean:.4f} ± {self.threshold_sigma * self.baseline_std:.4f}")

        return self.baseline_mean, self.baseline_std

    def monitor_step(self):
        """Take one monitoring step. Returns (rho, alert_level, alert_type)."""
        if self.baseline_mean is None:
            raise ValueError("Must train before monitoring")

        # Collect sample
        mean_a, mean_b, _ = self.sensor.read_raw()
        self.means_a.append(mean_a)
        self.means_b.append(mean_b)

        # Keep buffer bounded
        if len(self.means_a) > self.window_size * 4:
            self.means_a = self.means_a[-self.window_size * 2:]
            self.means_b = self.means_b[-self.window_size * 2:]

        # Only compute correlation every window_size/4 samples
        n = len(self.means_a)
        if n < self.window_size or n % (self.window_size // 4) != 0:
            return None, 0, None

        # Compute correlation on current window
        rho = np.corrcoef(
            self.means_a[-self.window_size:],
            self.means_b[-self.window_size:]
        )[0, 1]

        self.correlations.append(rho)

        # Normalized deviation
        z = (rho - self.baseline_mean) / (self.baseline_std + 1e-10)

        # CUSUM update (only on window-level measurements)
        self.cusum_pos = max(0, self.cusum_pos + z - 0.5)
        self.cusum_neg = max(0, self.cusum_neg - z - 0.5)

        # Determine alert level
        alert_level = 0
        alert_type = None

        if abs(z) > self.threshold_sigma:
            alert_level = 2
            alert_type = "THRESHOLD"
        elif self.cusum_pos > self.cusum_threshold:
            alert_level = 1
            alert_type = "CUSUM_HIGH"
        elif self.cusum_neg > self.cusum_threshold:
            alert_level = 1
            alert_type = "CUSUM_LOW"

        if alert_level > 0:
            self.alerts.append({
                'time': time.time(),
                'rho': float(rho),
                'z': float(z),
                'alert_type': alert_type,
                'cusum_pos': float(self.cusum_pos),
                'cusum_neg': float(self.cusum_neg),
            })

        return rho, alert_level, alert_type

    def get_stats(self):
        """Get current monitoring statistics."""
        if len(self.correlations) < 5:
            return {}

        recent = self.correlations[-20:]
        return {
            'current_mean': float(np.mean(recent)),
            'current_std': float(np.std(recent)),
            'drift': float(np.mean(recent) - self.baseline_mean),
            'cusum_pos': float(self.cusum_pos),
            'cusum_neg': float(self.cusum_neg),
            'n_alerts': len(self.alerts),
        }


class WorkloadGenerator:
    """Generate GPU workloads."""

    def __init__(self, workload_type: str, intensity: float = 0.7):
        self.workload_type = workload_type
        self.intensity = intensity
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)

    def _run(self):
        size = int(1024 + 3072 * self.intensity)

        if self.workload_type == "transformer":
            q = cp.random.randn(size, 64, dtype=cp.float32)
            k = cp.random.randn(size, 64, dtype=cp.float32)
            v = cp.random.randn(size, 64, dtype=cp.float32)
            while self.running:
                attn = cp.matmul(q, k.T)
                attn = cp.exp(attn - cp.max(attn, axis=-1, keepdims=True))
                attn = attn / cp.sum(attn, axis=-1, keepdims=True)
                out = cp.matmul(attn, v)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.005)

        elif self.workload_type == "mining":
            data = cp.random.randint(0, 2**32, size=(size * 4,), dtype=cp.uint32)
            while self.running:
                for _ in range(200):
                    data = data ^ (data << 13)
                    data = data ^ (data >> 17)
                    data = data ^ (data << 5)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.002)

        elif self.workload_type == "memory_attack":
            # Aggressive memory bandwidth attack
            a = cp.random.randn(size * size * 8, dtype=cp.float32)
            b = cp.zeros_like(a)
            while self.running:
                cp.copyto(b, a)
                cp.copyto(a, b)
                cp.cuda.Stream.null.synchronize()


def run_tamper_detection_demo():
    """Run live tamper detection demonstration."""
    print("="*70)
    print("EXPERIMENT 9: REAL-TIME TAMPER DETECTION")
    print("="*70)
    print()
    print("Demonstrating tamper-evident GPU computing:")
    print("1. Train baseline on 'transformer' workload")
    print("2. Monitor during clean operation")
    print("3. Inject 'mining' attack and observe detection")
    print()

    config = SensorConfig(n_iterations=5000)
    sensor = StrainSensor(config=config)
    detector = TamperDetector(sensor, window_size=500)

    # Start known workload
    print("[PHASE 1] STARTING KNOWN WORKLOAD")
    print("-"*50)
    workload = WorkloadGenerator("transformer", intensity=0.7)
    workload.start()
    time.sleep(2)
    print("Transformer workload running...")

    # Train baseline
    print("\n[PHASE 2] TRAINING BASELINE")
    print("-"*50)
    detector.train(duration=30.0)

    # Monitor clean operation
    print("\n[PHASE 3] MONITORING CLEAN OPERATION (30s)")
    print("-"*50)

    clean_alerts = 0
    start = time.time()
    last_print = start

    while time.time() - start < 30:
        rho, alert_level, alert_type = detector.monitor_step()

        if rho is not None:
            if alert_level > 0:
                clean_alerts += 1

            # Print status every 5 seconds
            now = time.time()
            if now - last_print >= 5.0:
                stats = detector.get_stats()
                status = "ALERT!" if alert_level > 0 else "OK"
                print(f"  t={now-start:.0f}s: ρ={rho:.4f} drift={stats.get('drift', 0):+.4f} [{status}]")
                last_print = now

    print(f"\nClean operation: {clean_alerts} alerts (false positives)")

    # Inject attack
    print("\n[PHASE 4] INJECTING ATTACK (mining)")
    print("-"*50)
    print("Starting mining workload alongside transformer...")

    attacker = WorkloadGenerator("mining", intensity=0.8)
    attacker.start()
    time.sleep(1)

    tamper_alerts = 0
    alert_times = []
    start = time.time()
    last_print = start
    first_alert_time = None

    while time.time() - start < 30:
        rho, alert_level, alert_type = detector.monitor_step()

        if rho is not None:
            if alert_level > 0:
                tamper_alerts += 1
                if first_alert_time is None:
                    first_alert_time = time.time() - start
                alert_times.append(time.time() - start)

            now = time.time()
            if now - last_print >= 5.0:
                stats = detector.get_stats()
                status = f"*** {alert_type} ***" if alert_level > 0 else "OK"
                print(f"  t={now-start:.0f}s: ρ={rho:.4f} drift={stats.get('drift', 0):+.4f} [{status}]")
                last_print = now

    attacker.stop()
    workload.stop()

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"Baseline: ρ = {detector.baseline_mean:.4f} ± {detector.baseline_std:.4f}")
    print()
    print(f"Clean operation (30s):")
    print(f"  False positive alerts: {clean_alerts}")
    print()
    print(f"Under attack (30s):")
    print(f"  Alerts triggered: {tamper_alerts}")
    if first_alert_time:
        print(f"  Time to first alert: {first_alert_time:.1f}s")
    print()

    if tamper_alerts > clean_alerts * 3 and tamper_alerts > 5:
        print("="*70)
        print("*** TAMPER DETECTION SUCCESSFUL ***")
        print("="*70)
        print()
        print(f"The mining attack was detected with {tamper_alerts} alerts")
        print(f"vs {clean_alerts} false positives during clean operation.")
        print()
        print("This demonstrates that GPU workloads can be made tamper-evident")
        print("using correlation-based fingerprinting!")
        success = True
    else:
        print("Tamper detection was not conclusive.")
        print(f"Alerts during attack ({tamper_alerts}) not significantly higher")
        print(f"than false positives ({clean_alerts}).")
        success = False

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp9_tamper_{timestamp}.json"

    output = {
        'experiment': 'exp9_tamper_detection',
        'timestamp': datetime.now().isoformat(),
        'baseline_mean': float(detector.baseline_mean),
        'baseline_std': float(detector.baseline_std),
        'clean_alerts': clean_alerts,
        'tamper_alerts': tamper_alerts,
        'first_alert_time': first_alert_time,
        'detection_successful': success,
        'all_alerts': detector.alerts[-50:] if detector.alerts else [],  # Last 50
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return success


if __name__ == "__main__":
    run_tamper_detection_demo()
