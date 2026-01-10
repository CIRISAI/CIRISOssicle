#!/usr/bin/env python3
"""
Experiment 7: Noise Floor and Sensitivity Analysis

Characterize:
1. Baseline noise floor (stability when nothing changes)
2. Temporal autocorrelation (how long until measurements decorrelate)
3. Minimum detectable perturbation
4. Signal-to-noise ratio for various load types

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
from datetime import datetime
from pathlib import Path
from scipy import stats
from strain_sensor import StrainSensor, SensorConfig


def collect_baseline(sensor, duration: float = 60.0, window_size: int = 500):
    """Collect baseline data with no perturbation."""
    print(f"Collecting baseline for {duration}s...")

    means_a, means_b, means_c = [], [], []
    correlations = []
    timestamps = []

    start = time.time()
    while time.time() - start < duration:
        mean_a, mean_b, mean_c = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)
        means_c.append(mean_c)
        timestamps.append(time.time() - start)

        # Compute windowed correlation periodically
        n = len(means_a)
        if n >= window_size and n % (window_size // 4) == 0:
            wa = np.array(means_a[-window_size:])
            wb = np.array(means_b[-window_size:])
            rho = np.corrcoef(wa, wb)[0, 1]
            correlations.append({
                'time': timestamps[-1],
                'rho_ab': float(rho),
                'n': n,
            })

    return {
        'means_a': np.array(means_a),
        'means_b': np.array(means_b),
        'means_c': np.array(means_c),
        'timestamps': np.array(timestamps),
        'correlations': correlations,
    }


def analyze_noise_floor(baseline_data):
    """Analyze noise characteristics."""
    corrs = baseline_data['correlations']
    rhos = np.array([c['rho_ab'] for c in corrs])
    times = np.array([c['time'] for c in corrs])

    results = {
        'n_samples': len(baseline_data['means_a']),
        'n_windows': len(rhos),
        'duration': times[-1] if len(times) > 0 else 0,
        'sample_rate': len(baseline_data['means_a']) / (times[-1] + 1e-10) if len(times) > 0 else 0,
    }

    if len(rhos) < 5:
        print("  Not enough data for analysis")
        return results

    # Basic statistics
    results['rho_mean'] = float(np.mean(rhos))
    results['rho_std'] = float(np.std(rhos))
    results['rho_min'] = float(np.min(rhos))
    results['rho_max'] = float(np.max(rhos))
    results['rho_range'] = float(np.max(rhos) - np.min(rhos))

    # Temporal autocorrelation (how quickly do measurements decorrelate?)
    if len(rhos) > 20:
        lags = range(1, min(20, len(rhos) // 4))
        autocorrs = []
        for lag in lags:
            ac = np.corrcoef(rhos[:-lag], rhos[lag:])[0, 1]
            autocorrs.append(ac)
        results['autocorr_lag1'] = float(autocorrs[0]) if autocorrs else None
        results['autocorr_lag5'] = float(autocorrs[4]) if len(autocorrs) > 4 else None
        results['autocorr_lag10'] = float(autocorrs[9]) if len(autocorrs) > 9 else None

    # Trend analysis (is there drift over time?)
    slope, intercept, r_value, p_value, std_err = stats.linregress(times, rhos)
    results['drift_slope'] = float(slope)  # rho units per second
    results['drift_r2'] = float(r_value ** 2)
    results['drift_p_value'] = float(p_value)

    # Stationarity check (compare first half to second half)
    mid = len(rhos) // 2
    first_half = rhos[:mid]
    second_half = rhos[mid:]
    t_stat, t_p_value = stats.ttest_ind(first_half, second_half)
    results['stationarity_t'] = float(t_stat)
    results['stationarity_p'] = float(t_p_value)
    results['is_stationary'] = t_p_value > 0.05

    return results


def compute_sensitivity(sensor, perturbation_type: str = "compute",
                        intensity: float = 0.5, duration: float = 20.0,
                        window_size: int = 500):
    """Measure sensitivity to a specific perturbation."""
    import threading

    print(f"Testing sensitivity to {perturbation_type} (intensity={intensity})...")

    # Collect baseline first
    baseline_corrs = []
    means_a, means_b = [], []

    start = time.time()
    print("  Collecting pre-perturbation baseline (10s)...")
    while time.time() - start < 10:
        mean_a, mean_b, _ = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)

        n = len(means_a)
        if n >= window_size and n % (window_size // 4) == 0:
            rho = np.corrcoef(means_a[-window_size:], means_b[-window_size:])[0, 1]
            baseline_corrs.append(rho)

    baseline_mean = np.mean(baseline_corrs)
    baseline_std = np.std(baseline_corrs)

    # Start perturbation
    def run_load():
        size = int(2048 + 6144 * intensity)
        if perturbation_type == "compute":
            a = cp.random.randn(size * size, dtype=cp.float32)
            while getattr(run_load, 'running', True):
                b = cp.sin(a) * cp.cos(a)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.01)
        elif perturbation_type == "memory":
            a = cp.random.randn(size * size, dtype=cp.float32)
            while getattr(run_load, 'running', True):
                b = a.copy()
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.01)
        elif perturbation_type == "matmul":
            a = cp.random.randn(size, size, dtype=cp.float32)
            while getattr(run_load, 'running', True):
                b = cp.matmul(a, a)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.01)

    run_load.running = True
    load_thread = threading.Thread(target=run_load)
    load_thread.start()
    time.sleep(1)  # Stabilize

    # Collect during perturbation
    perturb_corrs = []
    means_a, means_b = [], []

    print(f"  Collecting during perturbation ({duration}s)...")
    start = time.time()
    while time.time() - start < duration:
        mean_a, mean_b, _ = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)

        n = len(means_a)
        if n >= window_size and n % (window_size // 4) == 0:
            rho = np.corrcoef(means_a[-window_size:], means_b[-window_size:])[0, 1]
            perturb_corrs.append(rho)

    run_load.running = False
    load_thread.join(timeout=5)

    perturb_mean = np.mean(perturb_corrs)
    perturb_std = np.std(perturb_corrs)

    # Compute effect size
    delta = perturb_mean - baseline_mean
    pooled_std = np.sqrt((baseline_std**2 + perturb_std**2) / 2)
    effect_size = delta / pooled_std if pooled_std > 0 else 0

    # Statistical significance
    t_stat, p_value = stats.ttest_ind(baseline_corrs, perturb_corrs)

    return {
        'perturbation_type': perturbation_type,
        'intensity': intensity,
        'baseline_mean': float(baseline_mean),
        'baseline_std': float(baseline_std),
        'perturb_mean': float(perturb_mean),
        'perturb_std': float(perturb_std),
        'delta_rho': float(delta),
        'effect_size_d': float(effect_size),  # Cohen's d
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'snr': float(abs(delta) / baseline_std) if baseline_std > 0 else 0,
    }


def run_noise_sensitivity_experiment():
    """Run full noise floor and sensitivity analysis."""
    print("="*70)
    print("EXPERIMENT 7: NOISE FLOOR & SENSITIVITY")
    print("="*70)
    print()

    config = SensorConfig(n_iterations=5000)
    sensor = StrainSensor(config=config)

    results = {}

    # Part 1: Noise floor characterization
    print("\n[PART 1] NOISE FLOOR CHARACTERIZATION")
    print("-"*50)
    baseline_data = collect_baseline(sensor, duration=60.0, window_size=500)
    results['noise_floor'] = analyze_noise_floor(baseline_data)

    # Convert numpy bools to Python bools for JSON
    for key, val in results['noise_floor'].items():
        if isinstance(val, (np.bool_, np.generic)):
            results['noise_floor'][key] = val.item()

    nf = results['noise_floor']
    print(f"\nNoise Floor Results:")
    print(f"  Sample rate: {nf.get('sample_rate', 0):.0f} Hz")
    print(f"  ρ mean: {nf.get('rho_mean', 0):.4f}")
    print(f"  ρ std (noise floor): {nf.get('rho_std', 0):.4f}")
    print(f"  ρ range: {nf.get('rho_range', 0):.4f}")
    print(f"  Autocorr lag1: {nf.get('autocorr_lag1', 'N/A')}")
    print(f"  Drift slope: {nf.get('drift_slope', 0):.6f}/s")
    print(f"  Stationary: {nf.get('is_stationary', 'N/A')}")

    # Part 2: Sensitivity to different perturbations
    print("\n[PART 2] SENSITIVITY ANALYSIS")
    print("-"*50)

    perturbations = [
        ("compute", 0.25),
        ("compute", 0.5),
        ("compute", 1.0),
        ("memory", 0.5),
        ("matmul", 0.5),
    ]

    results['sensitivities'] = []
    for ptype, intensity in perturbations:
        sens = compute_sensitivity(sensor, ptype, intensity, duration=15.0)
        results['sensitivities'].append(sens)

        sig_marker = "***" if sens['p_value'] < 0.001 else "**" if sens['p_value'] < 0.01 else "*" if sens['p_value'] < 0.05 else ""
        print(f"\n  {ptype} (intensity={intensity}):")
        print(f"    Δρ = {sens['delta_rho']:+.4f}")
        print(f"    Effect size (d) = {sens['effect_size_d']:.2f}")
        print(f"    SNR = {sens['snr']:.1f}")
        print(f"    p-value = {sens['p_value']:.4f} {sig_marker}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    noise_floor = nf.get('rho_std', 0)
    print(f"\nNoise floor: σ = {noise_floor:.4f}")
    print(f"Minimum detectable Δρ (2σ): {2 * noise_floor:.4f}")
    print(f"Minimum detectable Δρ (3σ): {3 * noise_floor:.4f}")

    print("\nSensitivity by perturbation type:")
    print(f"{'Type':<15} {'Intensity':<10} {'Δρ':<10} {'SNR':<8} {'Detected?':<10}")
    print("-"*55)
    for s in results['sensitivities']:
        detected = "YES" if s['snr'] > 2 else "marginal" if s['snr'] > 1 else "no"
        print(f"{s['perturbation_type']:<15} {s['intensity']:<10} {s['delta_rho']:+.4f}    {s['snr']:<8.1f} {detected:<10}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp7_noise_sensitivity_{timestamp}.json"

    output = {
        'experiment': 'exp7_noise_sensitivity',
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_noise_sensitivity_experiment()
