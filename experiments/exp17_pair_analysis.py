#!/usr/bin/env python3
"""
Experiment 17: Individual Pair Analysis

The mean correlation may hide signal in individual pairs.
Analyze each of the 6 tetrahedral pairs separately to find
which pairs are most sensitive to attacks.

Hypothesis: D-pairs (AD, BD, CD) should show larger Δρ
because D (r=3.79) is in a different chaotic regime.

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


def collect_pair_data(sensor: TetrahedralSensor, duration: float) -> dict:
    """Collect individual pair correlation data."""
    pairs = {
        'ab': [], 'ac': [], 'ad': [],
        'bc': [], 'bd': [], 'cd': []
    }

    start = time.time()
    while time.time() - start < duration:
        reading = sensor.read()

        if reading.rho_ab != 0:  # Has valid correlations
            pairs['ab'].append(reading.rho_ab)
            pairs['ac'].append(reading.rho_ac)
            pairs['ad'].append(reading.rho_ad)
            pairs['bc'].append(reading.rho_bc)
            pairs['bd'].append(reading.rho_bd)
            pairs['cd'].append(reading.rho_cd)

    return pairs


def run_pair_analysis():
    """Analyze individual correlation pairs."""

    print("="*70)
    print("EXPERIMENT 17: INDIVIDUAL PAIR ANALYSIS")
    print("="*70)
    print()
    print("Hypothesis: D-pairs (AD, BD, CD) should show larger Δρ")
    print()

    results = {
        'experiment': 'exp17_pair_analysis',
        'timestamp': datetime.now().isoformat()
    }

    sensor = TetrahedralSensor(TetraConfig(n_iterations=5000))

    # =========================================================================
    # PHASE 1: Collect baseline (idle)
    # =========================================================================
    print("[PHASE 1] BASELINE COLLECTION (IDLE)")
    print("-"*50)

    idle = WorkloadGenerator("idle")
    idle.start()
    time.sleep(2)

    sensor.reset_history()
    baseline = collect_pair_data(sensor, duration=45.0)
    idle.stop()

    print(f"Collected {len(baseline['ab'])} samples")

    # =========================================================================
    # PHASE 2: Collect attack (crypto)
    # =========================================================================
    print("\n[PHASE 2] ATTACK COLLECTION (CRYPTO)")
    print("-"*50)

    crypto = WorkloadGenerator("crypto", intensity=0.8)
    crypto.start()
    time.sleep(2)

    sensor.reset_history()
    attack = collect_pair_data(sensor, duration=45.0)
    crypto.stop()

    print(f"Collected {len(attack['ab'])} samples")

    # =========================================================================
    # PHASE 3: Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("PAIR-BY-PAIR ANALYSIS")
    print("="*70)

    pair_results = {}
    pair_names = ['ab', 'ac', 'ad', 'bc', 'bd', 'cd']

    print(f"\n| Pair | Baseline | Attack | Δρ | σ_base | z-score | D-pair? |")
    print(f"|------|----------|--------|----|--------|---------|---------|")

    for pair in pair_names:
        base_mean = np.mean(baseline[pair])
        base_std = np.std(baseline[pair])
        attack_mean = np.mean(attack[pair])

        delta = attack_mean - base_mean
        z_score = abs(delta) / (base_std + 1e-10)

        is_d_pair = 'd' in pair

        pair_results[pair] = {
            'baseline_mean': float(base_mean),
            'baseline_std': float(base_std),
            'attack_mean': float(attack_mean),
            'delta': float(delta),
            'z_score': float(z_score),
            'is_d_pair': is_d_pair
        }

        d_marker = "YES" if is_d_pair else "no"
        detect_marker = " ***" if z_score > 3 else ""
        print(f"| {pair:4} | {base_mean:+.4f} | {attack_mean:+.4f} | {delta:+.4f} | {base_std:.4f} | {z_score:5.2f}σ | {d_marker:7} |{detect_marker}")

    results['pairs'] = pair_results

    # =========================================================================
    # PHASE 4: D-pairs vs non-D-pairs comparison
    # =========================================================================
    print("\n" + "="*70)
    print("D-PAIR vs NON-D-PAIR COMPARISON")
    print("-"*50)

    d_pairs = ['ad', 'bd', 'cd']
    non_d_pairs = ['ab', 'ac', 'bc']

    d_z_scores = [pair_results[p]['z_score'] for p in d_pairs]
    non_d_z_scores = [pair_results[p]['z_score'] for p in non_d_pairs]

    d_deltas = [abs(pair_results[p]['delta']) for p in d_pairs]
    non_d_deltas = [abs(pair_results[p]['delta']) for p in non_d_pairs]

    print(f"\nD-pairs (AD, BD, CD):")
    print(f"  Mean z-score: {np.mean(d_z_scores):.2f}σ")
    print(f"  Mean |Δρ|: {np.mean(d_deltas):.4f}")
    print(f"  Best pair: {d_pairs[np.argmax(d_z_scores)]} (z={max(d_z_scores):.2f}σ)")

    print(f"\nNon-D-pairs (AB, AC, BC):")
    print(f"  Mean z-score: {np.mean(non_d_z_scores):.2f}σ")
    print(f"  Mean |Δρ|: {np.mean(non_d_deltas):.4f}")
    print(f"  Best pair: {non_d_pairs[np.argmax(non_d_z_scores)]} (z={max(non_d_z_scores):.2f}σ)")

    improvement = np.mean(d_z_scores) / (np.mean(non_d_z_scores) + 1e-10)
    print(f"\nD-pair improvement: {improvement:.2f}×")

    results['comparison'] = {
        'd_pairs': {
            'mean_z': float(np.mean(d_z_scores)),
            'mean_delta': float(np.mean(d_deltas)),
            'best_pair': d_pairs[np.argmax(d_z_scores)],
            'best_z': float(max(d_z_scores))
        },
        'non_d_pairs': {
            'mean_z': float(np.mean(non_d_z_scores)),
            'mean_delta': float(np.mean(non_d_deltas)),
            'best_pair': non_d_pairs[np.argmax(non_d_z_scores)],
            'best_z': float(max(non_d_z_scores))
        },
        'improvement': float(improvement)
    }

    # =========================================================================
    # PHASE 5: Correlation structure
    # =========================================================================
    print("\n" + "="*70)
    print("CORRELATION STRUCTURE")
    print("-"*50)

    print("\nBaseline correlation structure:")
    for pair in pair_names:
        print(f"  ρ({pair.upper()}): {pair_results[pair]['baseline_mean']:+.4f} ± {pair_results[pair]['baseline_std']:.4f}")

    # Check for polarization
    baseline_values = [pair_results[p]['baseline_mean'] for p in pair_names]
    polarization = max(baseline_values) - min(baseline_values)
    print(f"\nPolarization (max-min): {polarization:.4f}")

    # Find most extreme pair
    most_negative_pair = pair_names[np.argmin(baseline_values)]
    most_positive_pair = pair_names[np.argmax(baseline_values)]
    print(f"Most negative: {most_negative_pair.upper()} ({min(baseline_values):.4f})")
    print(f"Most positive: {most_positive_pair.upper()} ({max(baseline_values):.4f})")

    results['structure'] = {
        'polarization': float(polarization),
        'most_negative': most_negative_pair,
        'most_positive': most_positive_pair
    }

    # =========================================================================
    # PHASE 6: Best detection strategy
    # =========================================================================
    print("\n" + "="*70)
    print("RECOMMENDED DETECTION STRATEGY")
    print("-"*50)

    # Find best single pair
    all_z = [(p, pair_results[p]['z_score']) for p in pair_names]
    all_z.sort(key=lambda x: x[1], reverse=True)

    print(f"\nBest pairs for detection (ranked by z-score):")
    for i, (pair, z) in enumerate(all_z[:3], 1):
        is_d = "D-pair" if 'd' in pair else ""
        print(f"  {i}. {pair.upper()}: z={z:.2f}σ {is_d}")

    # Combined D-pairs signal
    d_baseline = np.mean([pair_results[p]['baseline_mean'] for p in d_pairs])
    d_attack = np.mean([pair_results[p]['attack_mean'] for p in d_pairs])
    d_std = np.mean([pair_results[p]['baseline_std'] for p in d_pairs])
    d_combined_z = abs(d_attack - d_baseline) / (d_std + 1e-10)

    print(f"\nCombined D-pairs signal:")
    print(f"  Mean baseline: {d_baseline:.4f}")
    print(f"  Mean attack: {d_attack:.4f}")
    print(f"  Combined z-score: {d_combined_z:.2f}σ")

    results['strategy'] = {
        'best_pairs': [(p, float(z)) for p, z in all_z[:3]],
        'd_combined_z': float(d_combined_z)
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    detectable_pairs = [p for p in pair_names if pair_results[p]['z_score'] > 3]
    if detectable_pairs:
        print(f"\n*** 3σ DETECTABLE PAIRS: {', '.join([p.upper() for p in detectable_pairs])} ***")
    else:
        print(f"\n*** NO PAIRS EXCEED 3σ THRESHOLD ***")
        print(f"    Highest z-score: {all_z[0][0].upper()} = {all_z[0][1]:.2f}σ")

    if improvement > 1:
        print(f"\n*** D-PAIRS SHOW {improvement:.1f}× BETTER SENSITIVITY ***")
        print(f"    Hypothesis SUPPORTED: D (r=3.79) is a differential probe")
    else:
        print(f"\n*** D-PAIRS NOT SIGNIFICANTLY BETTER ***")
        print(f"    Hypothesis NOT SUPPORTED for this run")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp17_pairs_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_pair_analysis()
