#!/usr/bin/env python3
"""
Experiment 18: Attack Intensity Sweep

Tests detection at different attack intensities to find:
1. Minimum detectable attack level
2. Relationship between GPU utilization and Δρ
3. Optimal attack parameters for future testing

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
import subprocess
from datetime import datetime
from pathlib import Path

from tetrahedral_sensor import TetrahedralSensor, TetraConfig


class IntensiveAttack:
    """More intensive GPU attack workload."""

    def __init__(self, attack_type: str = 'matmul', size: int = 4096):
        self.attack_type = attack_type
        self.size = size
        self.running = False
        self.thread = None
        self.gpu_util = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        if self.attack_type == 'matmul':
            # Large matrix multiply - very GPU intensive
            a = cp.random.randn(self.size, self.size, dtype=cp.float32)
            b = cp.random.randn(self.size, self.size, dtype=cp.float32)
            while self.running:
                c = cp.matmul(a, b)
                cp.cuda.Stream.null.synchronize()

        elif self.attack_type == 'crypto_heavy':
            # More intensive crypto-like workload
            size = self.size * 16
            data = cp.random.randint(0, 2**32, size=(size,), dtype=cp.uint32)
            while self.running:
                for _ in range(500):  # More iterations
                    data = data ^ (data << 13)
                    data = data ^ (data >> 17)
                    data = data ^ (data << 5)
                    data = data + (data << 7)  # Additional ops
                    data = data ^ (data >> 11)
                cp.cuda.Stream.null.synchronize()

        elif self.attack_type == 'sfu':
            # Special Function Unit (trig functions)
            x = cp.random.randn(self.size * self.size, dtype=cp.float32)
            while self.running:
                y = cp.sin(x) * cp.cos(x) * cp.exp(-x*x/2)
                x = y + 0.01
                cp.cuda.Stream.null.synchronize()

        elif self.attack_type == 'memory':
            # Memory bandwidth attack
            size = self.size * self.size * 4
            src = cp.random.randn(size, dtype=cp.float32)
            dst = cp.zeros(size, dtype=cp.float32)
            while self.running:
                dst[:] = src
                src[:] = dst
                cp.cuda.Stream.null.synchronize()


def get_gpu_utilization():
    """Get current GPU utilization percentage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        return int(result.stdout.strip())
    except:
        return 0


def collect_with_monitoring(sensor: TetrahedralSensor, duration: float):
    """Collect data while monitoring GPU utilization."""
    correlations = []
    utilizations = []

    start = time.time()
    last_util_check = start

    while time.time() - start < duration:
        reading = sensor.read()

        if reading.rho_ab != 0:
            correlations.append({
                'mean': reading.mean_correlation,
                'rho_ab': reading.rho_ab,
                'rho_cd': reading.rho_cd,  # Most distant pair
            })

        # Check utilization every second
        if time.time() - last_util_check > 1.0:
            utilizations.append(get_gpu_utilization())
            last_util_check = time.time()

    return correlations, utilizations


def run_intensity_sweep():
    """Run attack intensity sweep."""

    print("="*70)
    print("EXPERIMENT 18: ATTACK INTENSITY SWEEP")
    print("="*70)
    print()
    print("Finding relationship between attack intensity and detection signal")
    print()

    results = {
        'experiment': 'exp18_intensity_sweep',
        'timestamp': datetime.now().isoformat()
    }

    sensor = TetrahedralSensor(TetraConfig(n_iterations=5000))

    # =========================================================================
    # PHASE 1: Baseline collection
    # =========================================================================
    print("[PHASE 1] BASELINE COLLECTION")
    print("-"*50)

    sensor.reset_history()
    baseline, util_base = collect_with_monitoring(sensor, duration=30.0)

    baseline_mean = np.mean([c['mean'] for c in baseline])
    baseline_std = np.std([c['mean'] for c in baseline])

    print(f"Samples: {len(baseline)}")
    print(f"Mean ρ: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"GPU util: {np.mean(util_base):.0f}%")

    results['baseline'] = {
        'samples': len(baseline),
        'mean': float(baseline_mean),
        'std': float(baseline_std),
        'gpu_util': float(np.mean(util_base))
    }

    # =========================================================================
    # PHASE 2: Attack intensity sweep
    # =========================================================================
    print("\n" + "="*70)
    print("[PHASE 2] ATTACK INTENSITY SWEEP")
    print("-"*50)

    attacks = [
        ('crypto_heavy', 1024),   # Light
        ('crypto_heavy', 4096),   # Medium
        ('crypto_heavy', 8192),   # Heavy
        ('matmul', 1024),         # Small matmul
        ('matmul', 2048),         # Medium matmul
        ('matmul', 4096),         # Large matmul
        ('sfu', 2048),            # SFU attack
        ('memory', 4096),         # Memory bandwidth
    ]

    sweep_results = []

    for attack_type, size in attacks:
        print(f"\nTesting {attack_type} (size={size})...")

        attack = IntensiveAttack(attack_type, size)
        attack.start()
        time.sleep(2)  # Warm up

        sensor.reset_history()
        attack_data, util_attack = collect_with_monitoring(sensor, duration=20.0)

        attack.stop()

        if attack_data:
            attack_mean = np.mean([c['mean'] for c in attack_data])
            delta = attack_mean - baseline_mean
            z_score = abs(delta) / (baseline_std + 1e-10)
            mean_util = np.mean(util_attack) if util_attack else 0

            result = {
                'attack_type': attack_type,
                'size': size,
                'samples': len(attack_data),
                'mean': float(attack_mean),
                'delta': float(delta),
                'z_score': float(z_score),
                'gpu_util': float(mean_util),
                'detectable': bool(z_score > 3)
            }
            sweep_results.append(result)

            detect_mark = " ***" if z_score > 3 else ""
            print(f"  GPU util: {mean_util:.0f}%, Δρ={delta:+.4f}, z={z_score:.2f}σ{detect_mark}")

    results['attacks'] = sweep_results

    # =========================================================================
    # PHASE 3: Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Sort by z-score
    sweep_results.sort(key=lambda x: x['z_score'], reverse=True)

    print("\n| Attack | Size | GPU% | Δρ | z-score | Detect |")
    print("|--------|------|------|-----|---------|--------|")
    for r in sweep_results:
        detect = "YES" if r['detectable'] else "no"
        print(f"| {r['attack_type']:12} | {r['size']:4} | {r['gpu_util']:3.0f}% | {r['delta']:+.4f} | {r['z_score']:5.2f}σ | {detect:6} |")

    # Correlation between GPU util and z-score
    utils = [r['gpu_util'] for r in sweep_results]
    z_scores = [r['z_score'] for r in sweep_results]

    if len(utils) > 2:
        corr = np.corrcoef(utils, z_scores)[0, 1]
        print(f"\nCorrelation(GPU%, z-score): {corr:.3f}")

        if corr > 0.5:
            print("→ Higher GPU utilization correlates with stronger signal")
        elif corr < -0.5:
            print("→ Lower GPU utilization correlates with stronger signal")
        else:
            print("→ Weak correlation between GPU utilization and signal strength")

    # Best attack for detection
    if sweep_results:
        best = sweep_results[0]
        print(f"\nBest attack for detection: {best['attack_type']} (size={best['size']})")
        print(f"  z-score: {best['z_score']:.2f}σ")
        print(f"  GPU utilization: {best['gpu_util']:.0f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    detectable = [r for r in sweep_results if r['detectable']]
    if detectable:
        print(f"\n*** {len(detectable)} ATTACK(S) EXCEED 3σ THRESHOLD ***")
        for r in detectable:
            print(f"    {r['attack_type']} (size={r['size']}): z={r['z_score']:.2f}σ")
    else:
        print("\n*** NO ATTACKS EXCEED 3σ THRESHOLD ***")
        print(f"    Highest: {sweep_results[0]['attack_type']} at z={sweep_results[0]['z_score']:.2f}σ")

    print(f"\nKey findings:")
    print(f"  - Baseline σ: {baseline_std:.4f}")
    print(f"  - Threshold (3σ): {3*baseline_std:.4f}")
    print(f"  - Best |Δρ|: {max(abs(r['delta']) for r in sweep_results):.4f}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp18_intensity_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_intensity_sweep()
