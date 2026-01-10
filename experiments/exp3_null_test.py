#!/usr/bin/env python3
"""
Experiment 3: Null Hypothesis Test - Sham Shake

Goal: Rule out software artifacts by running the exact same protocol
      but WITHOUT physical shaking during the "shake" phase.

If the effect is real:
  - Baseline A variance ≈ "Sham shake" variance ≈ Baseline B variance
  - F-ratio ≈ 1.0

If the effect is a software artifact:
  - "Sham shake" variance would be elevated even without physical shaking

This is a critical control experiment.

Author: CIRIS L3C
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path
from strain_sensor import StrainSensor


def get_gpu_temp() -> float:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def collect_phase(sensor: StrainSensor, duration: float, name: str) -> dict:
    """Collect data for one phase.

    CORRECTED: Collects time series of crystal means, then computes
    correlations over the time series (not spatial correlations).
    """
    print(f"\n[{name}] Collecting for {duration}s...")

    temp_start = get_gpu_temp()
    means_a, means_b, means_c = [], [], []

    start = time.time()
    sample_count = 0
    while time.time() - start < duration:
        # read_raw returns crystal MEANS
        mean_a, mean_b, mean_c = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)
        means_c.append(mean_c)
        sample_count += 1

        if sample_count % 50 == 0:
            elapsed = time.time() - start
            print(f"  {sample_count} samples, {elapsed:.0f}s, temp={get_gpu_temp()}°C")

    temp_end = get_gpu_temp()
    elapsed = time.time() - start

    # Compute correlations over the TIME SERIES
    means_a = np.array(means_a)
    means_b = np.array(means_b)
    means_c = np.array(means_c)

    rho_ab = np.corrcoef(means_a, means_b)[0, 1]
    rho_bc = np.corrcoef(means_b, means_c)[0, 1]
    rho_ac = np.corrcoef(means_a, means_c)[0, 1]

    print(f"  Done: {sample_count} samples")
    print(f"  ρ(A,B)={rho_ab:+.4f}  ρ(B,C)={rho_bc:+.4f}  ρ(A,C)={rho_ac:+.4f}")

    return {
        'name': name,
        'n_samples': sample_count,
        'duration_sec': elapsed,
        'temp_start': temp_start,
        'temp_end': temp_end,
        'rho_ab': float(rho_ab),
        'rho_bc': float(rho_bc),
        'rho_ac': float(rho_ac),
        'mean_a_avg': float(np.mean(means_a)),
        'mean_b_avg': float(np.mean(means_b)),
        'mean_c_avg': float(np.mean(means_c)),
    }


def run_null_test(phase_duration: float = 60.0):
    """Run the null hypothesis test (sham shake)."""
    print("="*70)
    print("EXPERIMENT 3: NULL HYPOTHESIS TEST (SHAM SHAKE)")
    print("="*70)
    print()
    print("This test runs the EXACT same protocol as Exp 1, but")
    print("WITHOUT any physical shaking during the 'shake' phase.")
    print()
    print("If our detection is real physical coupling:")
    print("  → All three phases should have similar variance")
    print("  → F-ratio ≈ 1.0")
    print()
    print("If there's a software artifact:")
    print("  → 'Sham shake' phase would show elevated variance")
    print("  → That would invalidate our detection method")
    print()
    print(f"Each phase: {phase_duration}s")
    print()

    sensor = StrainSensor()

    # Phase 1: Baseline A
    print("-"*60)
    print("PHASE 1: BASELINE A")
    print("Keep device STILL.")
    print("-"*60)
    input("Press ENTER when ready...")
    baseline_a = collect_phase(sensor, phase_duration, "BASELINE_A")

    # Phase 2: Sham shake (still stationary!)
    print()
    print("-"*60)
    print("PHASE 2: SHAM SHAKE")
    print("*** DO NOT SHAKE - Keep device STILL ***")
    print("(This is the control - we're testing if the")
    print(" protocol itself causes any artifact)")
    print("-"*60)
    input("Press ENTER when ready (remember: stay STILL)...")
    sham_shake = collect_phase(sensor, phase_duration, "SHAM_SHAKE")

    # Phase 3: Baseline B
    print()
    print("-"*60)
    print("PHASE 3: BASELINE B")
    print("Keep device STILL.")
    print("-"*60)
    input("Press ENTER when ready...")
    baseline_b = collect_phase(sensor, phase_duration, "BASELINE_B")

    # Analysis
    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    # Compare correlations across phases
    # If null hypothesis holds, all phases should have similar correlations
    mean_baseline_ab = (baseline_a['rho_ab'] + baseline_b['rho_ab']) / 2
    mean_baseline_bc = (baseline_a['rho_bc'] + baseline_b['rho_bc']) / 2
    mean_baseline_ac = (baseline_a['rho_ac'] + baseline_b['rho_ac']) / 2

    delta_ab = abs(sham_shake['rho_ab'] - mean_baseline_ab)
    delta_bc = abs(sham_shake['rho_bc'] - mean_baseline_bc)
    delta_ac = abs(sham_shake['rho_ac'] - mean_baseline_ac)
    max_delta = max(delta_ab, delta_bc, delta_ac)

    print("CORRELATION COMPARISON:")
    print(f"  {'Phase':<12} {'ρ(A,B)':>10} {'ρ(B,C)':>10} {'ρ(A,C)':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Baseline A':<12} {baseline_a['rho_ab']:>+10.4f} {baseline_a['rho_bc']:>+10.4f} {baseline_a['rho_ac']:>+10.4f}")
    print(f"  {'Sham shake':<12} {sham_shake['rho_ab']:>+10.4f} {sham_shake['rho_bc']:>+10.4f} {sham_shake['rho_ac']:>+10.4f}")
    print(f"  {'Baseline B':<12} {baseline_b['rho_ab']:>+10.4f} {baseline_b['rho_bc']:>+10.4f} {baseline_b['rho_ac']:>+10.4f}")
    print()
    print(f"  Max |Δρ| from baseline: {max_delta:.4f}")
    print()

    # Verdict
    print("="*70)
    print("VERDICT")
    print("="*70)
    print()

    # Threshold: correlations should NOT change without physical perturbation
    # A change > 0.05 during sham shake would indicate an artifact
    if max_delta < 0.05:
        print("✓ NULL HYPOTHESIS CONFIRMED")
        print(f"  Max |Δρ| = {max_delta:.4f} (threshold: 0.05)")
        print("  No software artifact detected.")
        print("  Correlations remained stable without physical perturbation.")
        print()
        print("  This validates that any correlation changes during actual")
        print("  shaking are due to real physical coupling.")
        verdict = "null_confirmed"
    elif max_delta < 0.10:
        print("? MARGINAL RESULT")
        print(f"  Max |Δρ| = {max_delta:.4f} (threshold: 0.05)")
        print("  Some correlation drift detected, but below detection threshold.")
        print()
        print("  Consider longer baselines or investigating thermal drift.")
        verdict = "marginal"
    else:
        print("⚠ WARNING: POSSIBLE ARTIFACT")
        print(f"  Max |Δρ| = {max_delta:.4f} (threshold: 0.05)")
        print("  The 'sham shake' phase showed different correlations")
        print("  even without physical shaking.")
        print()
        print("  Possible causes:")
        print("  - Thermal drift during experiment")
        print("  - Software timing artifact")
        print("  - GPU state changes")
        print()
        print("  Consider investigating before trusting detection results.")
        verdict = "possible_artifact"

    print()

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp3_null_test_{timestamp}.json"

    results = {
        'experiment': 'exp3_null_test',
        'timestamp': datetime.now().isoformat(),
        'phase_duration_sec': phase_duration,
        'max_delta': max_delta,
        'delta_ab': delta_ab,
        'delta_bc': delta_bc,
        'delta_ac': delta_ac,
        'verdict': verdict,
        'phases': {
            'baseline_a': baseline_a,
            'sham_shake': sham_shake,
            'baseline_b': baseline_b,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 3: Null Test (Sham Shake)')
    parser.add_argument('--duration', '-d', type=float, default=15,
                        help='Duration per phase in seconds (default: 15)')
    args = parser.parse_args()

    run_null_test(phase_duration=args.duration)
