#!/usr/bin/env python3
"""
Experiment 1: Baseline vs Shake - Controlled Comparison (CORRECTED)

CRITICAL: Correlations are computed over TIME SERIES of crystal means,
not spatial correlation within a single snapshot.

Protocol:
1. Baseline A - collect time series of means, compute correlations
2. Shake - collect time series during shaking
3. Baseline B - verify return to baseline

Detection metric: Variance of correlations σ²_ρ increases during shaking.
From ACCELEROMETER_THEORY.md: 6× variance ratio (0.05² vs 0.02²)

Sample rate options (n_iterations):
  - 5000: ~1.0 kHz (original, most sensitive)
  - 2000: ~2.4 kHz
  - 1000: ~4.1 kHz
  - 500:  ~6.6 kHz (fastest, may reduce sensitivity)

Window size: 1000 samples (~1 second at 1kHz) for stable correlation estimates.

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
from dataclasses import dataclass, asdict
from typing import List
from strain_sensor import StrainSensor, SensorConfig


def get_gpu_temp() -> float:
    """Get GPU temperature (called only at phase boundaries, not in hot loop)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=2
        )
        return float(result.stdout.strip())
    except:
        return 0.0


@dataclass
class PhaseResult:
    """Results from one experimental phase."""
    name: str
    duration_sec: float
    n_samples: int
    temp_start: float
    temp_end: float

    # Mean correlations (computed over full time series)
    rho_ab_mean: float
    rho_bc_mean: float
    rho_ac_mean: float

    # VARIANCE OF CORRELATIONS (key metric! - from sliding windows)
    # This is what increases during shaking per ACCELEROMETER_THEORY.md
    rho_ab_var: float
    rho_bc_var: float
    rho_ac_var: float
    rho_var_total: float

    # Raw means (for debugging)
    mean_a_avg: float
    mean_b_avg: float
    mean_c_avg: float


def collect_phase(sensor: StrainSensor, duration: float, name: str,
                  prompt: str = None, window_size: int = 1000) -> PhaseResult:
    """Collect data for one experimental phase.

    Args:
        sensor: StrainSensor instance
        duration: Collection duration in seconds
        name: Phase name for logging
        prompt: Optional prompt to display before collection
        window_size: Samples per correlation window (default 1000 for ~1s windows)
    """
    if prompt:
        print(f"\n{'='*60}")
        print(prompt)
        print(f"{'='*60}")
        input("Press ENTER when ready...")

    print(f"\n[{name}] Collecting for {duration}s (window_size={window_size})...")

    # Get temp only at phase boundaries (not in hot loop)
    temp_start = get_gpu_temp()

    # Collect time series of means - NO subprocess calls in this loop!
    means_a, means_b, means_c = [], [], []

    start = time.time()
    sample_count = 0
    last_print = start

    while time.time() - start < duration:
        mean_a, mean_b, mean_c = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)
        means_c.append(mean_c)
        sample_count += 1

        # Progress every 1 second (no subprocess!)
        now = time.time()
        if now - last_print >= 1.0:
            rate = sample_count / (now - start)
            print(f"  {sample_count} samples, {now - start:.0f}s, {rate:.0f}/sec")
            last_print = now

    temp_end = get_gpu_temp()
    elapsed = time.time() - start

    print(f"  Done: {sample_count} samples in {elapsed:.1f}s ({sample_count/elapsed:.0f}/sec)")

    # Compute correlations over the TIME SERIES
    means_a = np.array(means_a)
    means_b = np.array(means_b)
    means_c = np.array(means_c)

    # Compute correlations over SLIDING WINDOWS
    # Window size of 1000 at 1kHz = ~1 second per window (stable estimates)
    # Step by half window for 50% overlap
    rho_ab_windows = []
    rho_bc_windows = []
    rho_ac_windows = []

    step = window_size // 2
    for i in range(0, len(means_a) - window_size + 1, step):
        window_a = means_a[i:i + window_size]
        window_b = means_b[i:i + window_size]
        window_c = means_c[i:i + window_size]

        rho_ab_windows.append(np.corrcoef(window_a, window_b)[0, 1])
        rho_bc_windows.append(np.corrcoef(window_b, window_c)[0, 1])
        rho_ac_windows.append(np.corrcoef(window_a, window_c)[0, 1])

    # Mean and variance of correlations
    rho_ab_mean = np.mean(rho_ab_windows) if rho_ab_windows else 0
    rho_bc_mean = np.mean(rho_bc_windows) if rho_bc_windows else 0
    rho_ac_mean = np.mean(rho_ac_windows) if rho_ac_windows else 0

    rho_ab_var = np.var(rho_ab_windows) if rho_ab_windows else 0
    rho_bc_var = np.var(rho_bc_windows) if rho_bc_windows else 0
    rho_ac_var = np.var(rho_ac_windows) if rho_ac_windows else 0
    rho_var_total = rho_ab_var + rho_bc_var + rho_ac_var

    result = PhaseResult(
        name=name,
        duration_sec=elapsed,
        n_samples=sample_count,
        temp_start=temp_start,
        temp_end=temp_end,
        rho_ab_mean=float(rho_ab_mean),
        rho_bc_mean=float(rho_bc_mean),
        rho_ac_mean=float(rho_ac_mean),
        rho_ab_var=float(rho_ab_var),
        rho_bc_var=float(rho_bc_var),
        rho_ac_var=float(rho_ac_var),
        rho_var_total=float(rho_var_total),
        mean_a_avg=float(np.mean(means_a)),
        mean_b_avg=float(np.mean(means_b)),
        mean_c_avg=float(np.mean(means_c)),
    )

    n_windows = len(rho_ab_windows)
    print(f"  Done: {sample_count} samples, {n_windows} windows")
    print(f"  ρ(A,B)={rho_ab_mean:+.4f}±{np.sqrt(rho_ab_var):.4f}")
    print(f"  ρ(B,C)={rho_bc_mean:+.4f}±{np.sqrt(rho_bc_var):.4f}")
    print(f"  ρ(A,C)={rho_ac_mean:+.4f}±{np.sqrt(rho_ac_var):.4f}")
    print(f"  σ²_ρ_total={rho_var_total:.6f}")

    return result


def run_experiment(phase_duration: float = 60.0, n_iterations: int = 5000,
                   window_size: int = 1000):
    """Run the full baseline vs shake experiment.

    Args:
        phase_duration: Duration of each phase in seconds
        n_iterations: Kernel iterations (5000=1kHz, 1000=4kHz, 500=6.6kHz)
        window_size: Samples per correlation window (1000 recommended)
    """
    # Calculate expected sample rate
    rate_table = {5000: 1000, 2000: 2400, 1000: 4100, 500: 6600}
    expected_rate = rate_table.get(n_iterations, 1000)

    print("="*70)
    print("EXPERIMENT 1: BASELINE VS SHAKE")
    print("="*70)
    print()
    print("Detection metric: Variance of correlations σ²_ρ")
    print("Expected: 6× increase during shaking (per ACCELEROMETER_THEORY.md)")
    print()
    print(f"Configuration:")
    print(f"  n_iterations: {n_iterations} (~{expected_rate} samples/sec)")
    print(f"  window_size:  {window_size} samples (~{window_size/expected_rate:.1f}s per window)")
    print(f"  phase_duration: {phase_duration}s")
    print()
    print("Phases:")
    print("  1. Baseline A: Stationary")
    print("  2. Shake: Vigorous shaking (high jerk)")
    print("  3. Baseline B: Stationary (verify return)")
    print()

    config = SensorConfig(n_iterations=n_iterations)
    sensor = StrainSensor(config=config)

    # Thermal check
    print("-"*60)
    print("THERMAL CHECK")
    print("-"*60)
    temp1 = get_gpu_temp()
    print(f"Current GPU temp: {temp1}°C")
    time.sleep(5)
    temp2 = get_gpu_temp()
    print(f"After 5s: {temp2}°C (drift: {abs(temp2-temp1):.1f}°C)")

    # Phase 1: Baseline A
    baseline_a = collect_phase(
        sensor, phase_duration, "BASELINE_A",
        "PHASE 1: BASELINE A\nKeep device COMPLETELY STILL.",
        window_size=window_size
    )

    # Phase 2: Shake
    shake = collect_phase(
        sensor, phase_duration, "SHAKE",
        "PHASE 2: SHAKE\nShake VIGOROUSLY with sharp, sudden movements.\n(High jerk = better signal)",
        window_size=window_size
    )

    # Phase 3: Baseline B
    baseline_b = collect_phase(
        sensor, phase_duration, "BASELINE_B",
        "PHASE 3: BASELINE B\nSTOP. Keep device COMPLETELY STILL.",
        window_size=window_size
    )

    # Analysis
    print()
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    # VARIANCE OF CORRELATIONS (primary detection metric per ACCELEROMETER_THEORY.md)
    # This measures how much the correlations FLUCTUATE during each phase
    # Shaking causes correlations to become MORE variable (F-ratio > 1)
    var_baseline = (baseline_a.rho_var_total + baseline_b.rho_var_total) / 2
    var_shake = shake.rho_var_total
    F_ratio = var_shake / (var_baseline + 1e-20)

    print("CORRELATION VARIANCE (primary metric - from ACCELEROMETER_THEORY.md):")
    print(f"  {'Phase':<12} {'σ²_ρ(A,B)':>12} {'σ²_ρ(B,C)':>12} {'σ²_ρ(A,C)':>12} {'σ²_total':>12}")
    print(f"  {'-'*62}")
    print(f"  {'Baseline A':<12} {baseline_a.rho_ab_var:>12.6f} {baseline_a.rho_bc_var:>12.6f} {baseline_a.rho_ac_var:>12.6f} {baseline_a.rho_var_total:>12.6f}")
    print(f"  {'Shake':<12} {shake.rho_ab_var:>12.6f} {shake.rho_bc_var:>12.6f} {shake.rho_ac_var:>12.6f} {shake.rho_var_total:>12.6f}")
    print(f"  {'Baseline B':<12} {baseline_b.rho_ab_var:>12.6f} {baseline_b.rho_bc_var:>12.6f} {baseline_b.rho_ac_var:>12.6f} {baseline_b.rho_var_total:>12.6f}")
    print()
    print(f"  Mean baseline σ²_ρ: {var_baseline:.6f}")
    print(f"  Shake σ²_ρ:         {var_shake:.6f}")
    print(f"  F-ratio (shake/baseline): {F_ratio:.2f}×  (expect ~6× if effect present)")
    print()

    print("MEAN CORRELATIONS:")
    print(f"  {'Phase':<12} {'ρ(A,B)':>10} {'ρ(B,C)':>10} {'ρ(A,C)':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Baseline A':<12} {baseline_a.rho_ab_mean:>+10.4f} {baseline_a.rho_bc_mean:>+10.4f} {baseline_a.rho_ac_mean:>+10.4f}")
    print(f"  {'Shake':<12} {shake.rho_ab_mean:>+10.4f} {shake.rho_bc_mean:>+10.4f} {shake.rho_ac_mean:>+10.4f}")
    print(f"  {'Baseline B':<12} {baseline_b.rho_ab_mean:>+10.4f} {baseline_b.rho_bc_mean:>+10.4f} {baseline_b.rho_ac_mean:>+10.4f}")
    print()

    # Compute mean correlation changes
    delta_ab = shake.rho_ab_mean - (baseline_a.rho_ab_mean + baseline_b.rho_ab_mean) / 2
    delta_bc = shake.rho_bc_mean - (baseline_a.rho_bc_mean + baseline_b.rho_bc_mean) / 2
    delta_ac = shake.rho_ac_mean - (baseline_a.rho_ac_mean + baseline_b.rho_ac_mean) / 2

    print("MEAN CORRELATION CHANGE (Δρ = shake - mean_baseline):")
    print(f"  Δρ(A,B): {delta_ab:+.4f}")
    print(f"  Δρ(B,C): {delta_bc:+.4f}")
    print(f"  Δρ(A,C): {delta_ac:+.4f}")
    print()

    max_delta = max(abs(delta_ab), abs(delta_bc), abs(delta_ac))

    print("THERMAL ANALYSIS:")
    print(f"  Baseline A: {baseline_a.temp_start}°C → {baseline_a.temp_end}°C")
    print(f"  Shake:      {shake.temp_start}°C → {shake.temp_end}°C")
    print(f"  Baseline B: {baseline_b.temp_start}°C → {baseline_b.temp_end}°C")
    print()

    # Verdict - F-ratio is primary, correlation change is secondary
    print("="*70)
    print("VERDICT")
    print("="*70)
    print()

    # Primary: Variance increase (F-ratio)
    # ACCELEROMETER_THEORY.md: "6× variance increase during shaking"
    if F_ratio > 2.0:
        print(f"✓ VARIANCE INCREASE DETECTED")
        print(f"  F-ratio = {F_ratio:.2f}× (threshold: 2.0×)")
        print()
        print("  Shaking caused increased variance in crystal means.")
        print("  This indicates physical perturbation affecting collapse rates.")
        verdict = "significant_variance"
    elif F_ratio > 1.5:
        print(f"? MARGINAL VARIANCE INCREASE")
        print(f"  F-ratio = {F_ratio:.2f}× (threshold: 2.0×)")
        verdict = "marginal_variance"
    elif max_delta > 0.10:
        print(f"✓ CORRELATION CHANGE DETECTED")
        print(f"  Maximum |Δρ| = {max_delta:.4f} (threshold: 0.10)")
        print()
        print("  Shaking caused a measurable change in correlation structure.")
        verdict = "significant_correlation"
    elif max_delta > 0.05 or F_ratio > 1.2:
        print(f"? MARGINAL CHANGE")
        print(f"  F-ratio = {F_ratio:.2f}×, Max |Δρ| = {max_delta:.4f}")
        verdict = "marginal"
    else:
        print(f"✗ NO SIGNIFICANT CHANGE")
        print(f"  F-ratio = {F_ratio:.2f}× (threshold: 2.0×)")
        print(f"  Max |Δρ| = {max_delta:.4f} (threshold: 0.10)")
        print()
        print("  Correlations remained stable during shaking.")
        print("  Check: Was shaking vigorous enough?")
        verdict = "none"

    print()

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp1_baseline_vs_shake_{timestamp}.json"

    results = {
        'experiment': 'exp1_baseline_vs_shake',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'phase_duration_sec': phase_duration,
            'n_iterations': n_iterations,
            'window_size': window_size,
            'expected_rate_hz': expected_rate,
        },
        'F_ratio': F_ratio,
        'sigma2_rho_baseline': var_baseline,
        'sigma2_rho_shake': var_shake,
        'max_delta_rho': max_delta,
        'delta_rho_ab': delta_ab,
        'delta_rho_bc': delta_bc,
        'delta_rho_ac': delta_ac,
        'verdict': verdict,
        'phases': {
            'baseline_a': asdict(baseline_a),
            'shake': asdict(shake),
            'baseline_b': asdict(baseline_b),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Experiment 1: Baseline vs Shake',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sample rate options (n_iterations):
  5000  ~1.0 kHz (original, most sensitive)
  2000  ~2.4 kHz
  1000  ~4.1 kHz
  500   ~6.6 kHz (fastest)

Examples:
  python exp1_baseline_vs_shake.py -d 30 -i 1000 -w 1000
    30s phases, 4kHz sampling, 1000-sample windows (~0.25s each)

  python exp1_baseline_vs_shake.py -d 15 -i 5000 -w 1000
    15s phases, 1kHz sampling, 1000-sample windows (~1s each)
""")
    parser.add_argument('--duration', '-d', type=float, default=15,
                        help='Duration per phase in seconds (default: 15)')
    parser.add_argument('--iterations', '-i', type=int, default=5000,
                        help='Kernel iterations: 5000=1kHz, 1000=4kHz, 500=6.6kHz (default: 5000)')
    parser.add_argument('--window', '-w', type=int, default=1000,
                        help='Window size for correlation (default: 1000 samples)')
    args = parser.parse_args()

    run_experiment(
        phase_duration=args.duration,
        n_iterations=args.iterations,
        window_size=args.window
    )
