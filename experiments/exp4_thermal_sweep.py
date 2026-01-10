#!/usr/bin/env python3
"""
Experiment 4: Thermal Calibration of Correlation-Based Temperature Sensor

Uses correlation structure between coupled chaotic oscillators on GPU
as a software-only thermal probe.

Prior Art:
- GPU race condition TRNGs (2015): Temperature as entropy INPUT
- Our approach: Temperature as measured OUTPUT via correlation drift

Measured sensitivity: dρ/dT ≈ 0.015 per °C
Detection threshold: ~2-3°C (2σ)

Protocol:
1. Start cold (idle GPU)
2. Heat to target temperature WHILE collecting correlation data
3. Continue collecting during cooldown
4. Compute calibration curve: ρ(A,B) vs T across full thermal sweep
5. Estimate sensitivity dρ/dT

Author: CIRIS L3C
License: BSL 1.1
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
from strain_sensor import StrainSensor, SensorConfig


def get_gpu_temp() -> float:
    """Get GPU temperature via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def warm_gpu_with_data(sensor: 'StrainSensor', target_temp: float = 75.0,
                        timeout: float = 180.0, window_size: int = 100):
    """
    Warm up GPU while collecting correlation data.

    This captures the thermal transient that shows correlation drift.
    Returns data_points collected during warmup.

    Note: Uses smaller window_size (default 100) since sample rate during
    warmup is ~4-5/s due to matmul compute overhead.
    """
    import cupy as cp

    print(f"\n[WARMUP] Heating GPU to {target_temp}°C while collecting data...")
    print(f"  (Using window_size={window_size} for warmup phase)")
    start = time.time()
    initial_temp = get_gpu_temp()
    print(f"  Starting at {initial_temp}°C")

    # Create large arrays for compute load
    a = cp.random.randn(8192, 8192, dtype=cp.float32)

    # Raw sample storage
    means_a, means_b, means_c = [], [], []
    sample_temps = []
    data_points = []

    last_print = start
    last_temp_check = start
    current_temp = initial_temp

    while current_temp < target_temp and (time.time() - start) < timeout:
        # Interleave: heat burst, then sample
        # Heat burst (fewer iterations to allow sampling)
        for _ in range(3):
            b = cp.matmul(a, a)
        cp.cuda.Stream.null.synchronize()

        # Collect correlation sample
        mean_a, mean_b, mean_c = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)
        means_c.append(mean_c)
        sample_temps.append(current_temp)

        # Check temperature every 0.5s
        now = time.time()
        if now - last_temp_check >= 0.5:
            current_temp = get_gpu_temp()
            last_temp_check = now

        # Compute windowed correlation
        n = len(means_a)
        if n >= window_size and n % (window_size // 2) == 0:
            wa = np.array(means_a[-window_size:])
            wb = np.array(means_b[-window_size:])
            wc = np.array(means_c[-window_size:])
            wt = sample_temps[-window_size:]

            rho_ab = np.corrcoef(wa, wb)[0, 1]
            rho_bc = np.corrcoef(wb, wc)[0, 1]
            rho_ac = np.corrcoef(wa, wc)[0, 1]

            avg_temp = np.mean(wt)

            data_points.append({
                'time': now - start,
                'temp': avg_temp,
                'rho_ab': float(rho_ab),
                'rho_bc': float(rho_bc),
                'rho_ac': float(rho_ac),
                'n_samples': n,
                'phase': 'warmup',
            })

        # Progress every 5 seconds
        if now - last_print >= 5.0:
            rate = len(means_a) / (now - start) if (now - start) > 0 else 0
            n_windows = len(data_points)
            print(f"  {now - start:.0f}s: T={current_temp}°C  "
                  f"{len(means_a)} samples ({rate:.0f}/s)  {n_windows} windows")
            last_print = now

    final_temp = get_gpu_temp()
    elapsed = time.time() - start
    print(f"  Reached {final_temp}°C in {elapsed:.0f}s")
    print(f"  Collected {len(means_a)} samples, {len(data_points)} windows during warmup")

    del a
    if 'b' in dir():
        del b
    cp.get_default_memory_pool().free_all_blocks()

    return final_temp, data_points, means_a, means_b, means_c, sample_temps


def collect_calibration_data(sensor: StrainSensor, duration: float,
                              window_size: int = 1000, phase: str = "cooling"):
    """
    Collect correlation data with temperature readings.

    Uses window_size=1000 for stable correlation estimates (matches original
    triple_rotation methodology).
    """
    print(f"\n[{phase.upper()}] Collecting data for {duration}s...")

    # Raw sample storage
    means_a, means_b, means_c = [], [], []
    sample_temps = []

    # Windowed correlation data points
    data_points = []

    start = time.time()
    last_print = start
    last_temp_check = start
    current_temp = get_gpu_temp()

    while time.time() - start < duration:
        # Collect sample
        mean_a, mean_b, mean_c = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)
        means_c.append(mean_c)
        sample_temps.append(current_temp)

        # Check temperature every 0.5s (avoid nvidia-smi overhead in hot loop)
        now = time.time()
        if now - last_temp_check >= 0.5:
            current_temp = get_gpu_temp()
            last_temp_check = now

        # Compute windowed correlation every window_size/2 samples
        n = len(means_a)
        if n >= window_size and n % (window_size // 2) == 0:
            # Use last window_size samples
            wa = np.array(means_a[-window_size:])
            wb = np.array(means_b[-window_size:])
            wc = np.array(means_c[-window_size:])
            wt = sample_temps[-window_size:]

            rho_ab = np.corrcoef(wa, wb)[0, 1]
            rho_bc = np.corrcoef(wb, wc)[0, 1]
            rho_ac = np.corrcoef(wa, wc)[0, 1]

            avg_temp = np.mean(wt)

            data_points.append({
                'time': now - start,
                'temp': avg_temp,
                'rho_ab': float(rho_ab),
                'rho_bc': float(rho_bc),
                'rho_ac': float(rho_ac),
                'n_samples': n,
            })

        # Progress every 5 seconds
        if now - last_print >= 5.0:
            rate = len(means_a) / (now - start)
            n_windows = len(data_points)
            print(f"  {now - start:.0f}s: T={current_temp}°C  "
                  f"{len(means_a)} samples ({rate:.0f}/s)  {n_windows} windows")
            last_print = now

    # Final stats
    print(f"  Collected {len(means_a)} samples, {len(data_points)} correlation windows")

    return data_points, means_a, means_b, means_c, sample_temps


def compute_calibration(data_points):
    """
    Compute thermal calibration: ρ vs T relationship.

    Returns sensitivity (dρ/dT) and correlation coefficient.
    """
    if len(data_points) < 5:
        print("Not enough data points for calibration")
        return {}

    temps = np.array([d['temp'] for d in data_points])
    rho_ab = np.array([d['rho_ab'] for d in data_points])
    rho_bc = np.array([d['rho_bc'] for d in data_points])
    rho_ac = np.array([d['rho_ac'] for d in data_points])

    # Remove any NaN values
    valid = ~(np.isnan(rho_ab) | np.isnan(rho_bc) | np.isnan(rho_ac) | np.isnan(temps))
    temps = temps[valid]
    rho_ab = rho_ab[valid]
    rho_bc = rho_bc[valid]
    rho_ac = rho_ac[valid]

    if len(temps) < 5:
        print("Not enough valid data points after filtering NaN")
        return {}

    temp_range = temps.max() - temps.min()

    # Linear regression: ρ = slope * T + intercept
    slope_ab, intercept_ab = np.polyfit(temps, rho_ab, 1)
    slope_bc, intercept_bc = np.polyfit(temps, rho_bc, 1)
    slope_ac, intercept_ac = np.polyfit(temps, rho_ac, 1)

    # Correlation coefficient (how well does ρ track T?)
    corr_t_ab = np.corrcoef(temps, rho_ab)[0, 1]
    corr_t_bc = np.corrcoef(temps, rho_bc)[0, 1]
    corr_t_ac = np.corrcoef(temps, rho_ac)[0, 1]

    # Residual noise (after removing linear trend)
    residual_ab = rho_ab - (slope_ab * temps + intercept_ab)
    residual_bc = rho_bc - (slope_bc * temps + intercept_bc)
    residual_ac = rho_ac - (slope_ac * temps + intercept_ac)

    noise_ab = np.std(residual_ab)
    noise_bc = np.std(residual_bc)
    noise_ac = np.std(residual_ac)

    # Detection threshold (2σ noise / sensitivity)
    if abs(slope_ab) > 1e-6:
        min_detectable_ab = 2 * noise_ab / abs(slope_ab)
    else:
        min_detectable_ab = float('inf')

    return {
        'temp_range': float(temp_range),
        'temp_min': float(temps.min()),
        'temp_max': float(temps.max()),
        'n_points': len(temps),

        # Sensitivity (dρ/dT)
        'sensitivity_ab': float(slope_ab),
        'sensitivity_bc': float(slope_bc),
        'sensitivity_ac': float(slope_ac),

        # Intercepts
        'intercept_ab': float(intercept_ab),
        'intercept_bc': float(intercept_bc),
        'intercept_ac': float(intercept_ac),

        # Correlation with temperature
        'corr_t_ab': float(corr_t_ab),
        'corr_t_bc': float(corr_t_bc),
        'corr_t_ac': float(corr_t_ac),

        # Noise floor
        'noise_ab': float(noise_ab),
        'noise_bc': float(noise_bc),
        'noise_ac': float(noise_ac),

        # Detection capability
        'min_detectable_degC': float(min_detectable_ab),
    }


def print_calibration_results(cal):
    """Print formatted calibration results."""
    print("\n" + "="*70)
    print("THERMAL CALIBRATION RESULTS")
    print("="*70)
    print()
    print(f"Temperature range: {cal['temp_min']:.1f}°C → {cal['temp_max']:.1f}°C "
          f"(ΔT = {cal['temp_range']:.1f}°C)")
    print(f"Data points: {cal['n_points']}")
    print()

    print("SENSITIVITY (dρ/dT):")
    print(f"  ρ(A,B): {cal['sensitivity_ab']:+.5f} per °C")
    print(f"  ρ(B,C): {cal['sensitivity_bc']:+.5f} per °C")
    print(f"  ρ(A,C): {cal['sensitivity_ac']:+.5f} per °C")
    print()

    print("CORRELATION WITH TEMPERATURE:")
    print(f"  corr(T, ρ_AB): {cal['corr_t_ab']:+.4f}")
    print(f"  corr(T, ρ_BC): {cal['corr_t_bc']:+.4f}")
    print(f"  corr(T, ρ_AC): {cal['corr_t_ac']:+.4f}")
    print()

    print("NOISE FLOOR (after detrending):")
    print(f"  σ(ρ_AB): {cal['noise_ab']:.4f}")
    print(f"  σ(ρ_BC): {cal['noise_bc']:.4f}")
    print(f"  σ(ρ_AC): {cal['noise_ac']:.4f}")
    print()

    print("DETECTION CAPABILITY:")
    print(f"  Minimum detectable ΔT: {cal['min_detectable_degC']:.1f}°C (2σ threshold)")
    print()

    # Verdict
    print("="*70)
    print("VERDICT")
    print("="*70)
    print()

    max_corr = max(abs(cal['corr_t_ab']), abs(cal['corr_t_bc']), abs(cal['corr_t_ac']))

    if cal['temp_range'] < 5:
        print("⚠️  INSUFFICIENT TEMPERATURE RANGE")
        print(f"   Only {cal['temp_range']:.1f}°C swing - need >10°C for reliable calibration")
        print("   Try: --warm 80 --cool-duration 300")
        verdict = "insufficient_range"
    elif max_corr > 0.7:
        print("✓ STRONG THERMAL CORRELATION")
        print(f"  Max |corr(T, ρ)| = {max_corr:.3f}")
        print(f"  Sensitivity: {cal['sensitivity_ab']:+.5f} per °C")
        print(f"  Resolution: ~{cal['min_detectable_degC']:.1f}°C")
        print()
        print("  The correlation structure reliably tracks GPU temperature.")
        verdict = "calibrated"
    elif max_corr > 0.4:
        print("? MODERATE THERMAL CORRELATION")
        print(f"  Max |corr(T, ρ)| = {max_corr:.3f}")
        print("  May need larger temperature range or longer collection.")
        verdict = "moderate"
    else:
        print("✗ WEAK THERMAL CORRELATION")
        print(f"  Max |corr(T, ρ)| = {max_corr:.3f}")
        print("  Temperature tracking not reliable in this range.")
        verdict = "weak"

    print()
    return verdict


def run_thermal_calibration(warm_target: float = 80.0,
                             cool_duration: float = 300.0,
                             window_size: int = 1000,
                             n_iterations: int = 5000):
    """
    Run full thermal calibration experiment.

    Args:
        warm_target: Target temperature to warm GPU to (°C)
        cool_duration: Duration to collect data while cooling (seconds)
        window_size: Samples per correlation window (1000 recommended)
        n_iterations: Kernel iterations (5000=1kHz, 1000=4kHz)
    """
    print("="*70)
    print("EXPERIMENT 4: THERMAL CALIBRATION (FULL SWEEP)")
    print("="*70)
    print()
    print("Calibrating correlation-based temperature sensor.")
    print()
    print("Prior art: GPU race condition TRNGs use temperature as entropy INPUT")
    print("Our approach: Use correlation structure to MEASURE temperature")
    print()
    print(f"Protocol:")
    print(f"  1. Record cold baseline temperature")
    print(f"  2. Warm GPU to {warm_target}°C WHILE collecting data (captures thermal drift!)")
    print(f"  3. Continue collecting for {cool_duration}s during cooling")
    print(f"  4. Compute calibration curve: ρ(A,B) vs T across full sweep")
    print()
    print(f"Parameters:")
    print(f"  Window size: {window_size} samples")
    print(f"  Kernel iterations: {n_iterations}")
    print()

    # Record cold baseline
    cold_temp = get_gpu_temp()
    print(f"Cold baseline: {cold_temp}°C")

    # Initialize sensor
    config = SensorConfig(n_iterations=n_iterations)
    sensor = StrainSensor(config=config)

    # Warm up GPU WHILE collecting data (this is where thermal drift happens!)
    # Use smaller window for warmup since sample rate is ~4/s during heating
    warmup_window = min(100, window_size)
    hot_temp, warmup_data, warmup_a, warmup_b, warmup_c, warmup_temps = warm_gpu_with_data(
        sensor, target_temp=warm_target, timeout=180.0, window_size=warmup_window
    )

    # Collect data during cooling
    cooling_data, cooling_a, cooling_b, cooling_c, cooling_temps = collect_calibration_data(
        sensor, cool_duration, window_size=window_size, phase="cooling"
    )

    # Mark cooling phase
    for dp in cooling_data:
        dp['phase'] = 'cooling'

    # Combine warmup + cooling data
    all_data_points = warmup_data + cooling_data
    all_means_a = warmup_a + cooling_a
    all_means_b = warmup_b + cooling_b
    all_means_c = warmup_c + cooling_c
    all_temps = warmup_temps + cooling_temps

    final_temp = get_gpu_temp()
    print(f"\nFinal temperature: {final_temp}°C")
    print(f"Total data: {len(warmup_data)} warmup + {len(cooling_data)} cooling = {len(all_data_points)} windows")

    # Compute calibration on combined warmup + cooling data
    calibration = compute_calibration(all_data_points)

    if calibration:
        verdict = print_calibration_results(calibration)
        calibration['verdict'] = verdict
    else:
        verdict = "failed"

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp4_thermal_calibration_{timestamp}.json"

    results = {
        'experiment': 'exp4_thermal_calibration_full_sweep',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'warm_target': warm_target,
            'cool_duration': cool_duration,
            'window_size': window_size,
            'n_iterations': n_iterations,
        },
        'temperatures': {
            'cold_baseline': cold_temp,
            'hot_peak': hot_temp,
            'final': final_temp,
        },
        'calibration': calibration,
        'data_points': all_data_points,
        'warmup_windows': len(warmup_data),
        'cooling_windows': len(cooling_data),
        'raw_sample_count': len(all_means_a),
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    # Print calibration equation
    if calibration and calibration.get('verdict') == 'calibrated':
        print()
        print("="*70)
        print("CALIBRATION EQUATION")
        print("="*70)
        print()
        print(f"  T ≈ (ρ_AB - {calibration['intercept_ab']:.4f}) / {calibration['sensitivity_ab']:.5f}")
        print()
        print("  Or inversely:")
        print(f"  ρ_AB(T) = {calibration['sensitivity_ab']:.5f} × T + {calibration['intercept_ab']:.4f}")
        print()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Experiment 4: Thermal Calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp4_thermal_sweep.py --warm 80 --duration 300
  python exp4_thermal_sweep.py --warm 85 --duration 600 --window 1000
        """
    )
    parser.add_argument('--warm', '-w', type=float, default=80,
                        help='Target warm temperature in °C (default: 80)')
    parser.add_argument('--duration', '-d', type=float, default=300,
                        help='Cooling duration in seconds (default: 300)')
    parser.add_argument('--window', type=int, default=1000,
                        help='Window size for correlation (default: 1000)')
    parser.add_argument('--iterations', '-i', type=int, default=5000,
                        help='Kernel iterations: 5000=1kHz, 1000=4kHz (default: 5000)')
    args = parser.parse_args()

    run_thermal_calibration(
        warm_target=args.warm,
        cool_duration=args.duration,
        window_size=args.window,
        n_iterations=args.iterations
    )
