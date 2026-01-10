#!/usr/bin/env python3
"""
Experiment 5: GPU Load Sensitivity of Correlation Structure

Discovery: The correlation coefficient ρ between coupled chaotic oscillators
changes based on concurrent GPU compute load, not just temperature.

Findings from exp4:
- With matmul load: ρ(A,B) ≈ -0.52
- Without load:     ρ(A,B) ≈ -0.28
- Delta: ~0.24 (huge signal!)

This experiment systematically characterizes the load effect by:
1. Measuring baseline correlation (no concurrent load)
2. Measuring correlation with various load types/intensities
3. Computing load sensitivity: Δρ vs load level

Potential applications:
- Detect concurrent GPU processes
- Detect interference/contention
- Software-only GPU activity sensor

Prior art comparison:
- ML-based interference detection uses timing metrics (Silveira et al. 2020)
- Our approach: correlation structure as intrinsic sensor (potentially novel)

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
import threading
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


def get_gpu_utilization() -> float:
    """Get GPU utilization percentage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except:
        return 0.0


class LoadGenerator:
    """Generate controllable GPU load in background with yielding.

    Key insight: Without yielding, heavy load monopolizes GPU and blocks
    the sensor kernel from running. We add periodic yields to allow
    interleaved execution.
    """

    def __init__(self, load_type: str = "matmul", intensity: float = 1.0,
                 yield_interval: float = 0.01):
        self.load_type = load_type
        self.intensity = intensity
        self.yield_interval = yield_interval  # Seconds to yield for sensor
        self.running = False
        self.thread = None

    def start(self):
        """Start load generation in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_load)
        self.thread.start()

    def stop(self):
        """Stop load generation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run_load(self):
        """Background load generation loop with periodic yields."""
        import cupy as cp

        # Size based on intensity (0.0 = 1024, 1.0 = 8192)
        size = int(1024 + 7168 * self.intensity)

        # Burst count - do N iterations then yield
        burst_count = max(1, int(10 * self.intensity))

        if self.load_type == "matmul":
            a = cp.random.randn(size, size, dtype=cp.float32)
            while self.running:
                for _ in range(burst_count):
                    if not self.running:
                        break
                    b = cp.matmul(a, a)
                cp.cuda.Stream.null.synchronize()
                time.sleep(self.yield_interval)  # Yield for sensor

        elif self.load_type == "memory":
            # Memory bandwidth stress
            a = cp.random.randn(size * size, dtype=cp.float32)
            while self.running:
                for _ in range(burst_count):
                    if not self.running:
                        break
                    b = a.copy()
                cp.cuda.Stream.null.synchronize()
                time.sleep(self.yield_interval)

        elif self.load_type == "compute":
            # Pure compute (element-wise)
            a = cp.random.randn(size * size, dtype=cp.float32)
            while self.running:
                for _ in range(burst_count):
                    if not self.running:
                        break
                    b = cp.sin(a) * cp.cos(a) * cp.exp(-a * a)
                cp.cuda.Stream.null.synchronize()
                time.sleep(self.yield_interval)


def collect_correlation_data(sensor: StrainSensor, duration: float,
                              window_size: int = 500, label: str = ""):
    """Collect correlation data for specified duration."""
    means_a, means_b, means_c = [], [], []
    temps = []
    utils = []

    data_points = []

    start = time.time()
    last_check = start
    current_temp = get_gpu_temp()
    current_util = get_gpu_utilization()

    while time.time() - start < duration:
        mean_a, mean_b, mean_c = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)
        means_c.append(mean_c)
        temps.append(current_temp)
        utils.append(current_util)

        now = time.time()
        if now - last_check >= 0.5:
            current_temp = get_gpu_temp()
            current_util = get_gpu_utilization()
            last_check = now

        # Compute windowed correlation
        n = len(means_a)
        if n >= window_size and n % (window_size // 2) == 0:
            wa = np.array(means_a[-window_size:])
            wb = np.array(means_b[-window_size:])
            wc = np.array(means_c[-window_size:])

            rho_ab = np.corrcoef(wa, wb)[0, 1]
            rho_bc = np.corrcoef(wb, wc)[0, 1]
            rho_ac = np.corrcoef(wa, wc)[0, 1]

            data_points.append({
                'time': now - start,
                'temp': np.mean(temps[-window_size:]),
                'util': np.mean(utils[-window_size:]),
                'rho_ab': float(rho_ab),
                'rho_bc': float(rho_bc),
                'rho_ac': float(rho_ac),
                'label': label,
            })

    return data_points


def run_load_sensitivity_experiment(duration_per_phase: float = 30.0,
                                     window_size: int = 500,
                                     n_iterations: int = 5000):
    """
    Run load sensitivity experiment.

    Protocol:
    1. Baseline (no load)
    2. Matmul load (various intensities)
    3. Memory load
    4. Compute load
    5. Return to baseline
    """
    print("="*70)
    print("EXPERIMENT 5: GPU LOAD SENSITIVITY")
    print("="*70)
    print()
    print("Testing how correlation structure changes with GPU load.")
    print()
    print("Prior art: ML-based interference detection uses timing metrics")
    print("Our approach: Correlation structure as intrinsic load sensor")
    print()

    # Initialize sensor
    config = SensorConfig(n_iterations=n_iterations)
    sensor = StrainSensor(config=config)

    all_data = []

    phases = [
        ("baseline_pre", None, 0.0),
        ("matmul_low", "matmul", 0.25),
        ("matmul_med", "matmul", 0.5),
        ("matmul_high", "matmul", 1.0),
        ("memory", "memory", 1.0),
        ("compute", "compute", 1.0),
        ("baseline_post", None, 0.0),
    ]

    for phase_name, load_type, intensity in phases:
        print(f"\n[{phase_name.upper()}]")

        if load_type:
            print(f"  Starting {load_type} load (intensity={intensity})...")
            load = LoadGenerator(load_type, intensity)
            load.start()
            time.sleep(1)  # Let load stabilize
        else:
            load = None
            print("  No concurrent load")

        temp = get_gpu_temp()
        util = get_gpu_utilization()
        print(f"  GPU: {temp}°C, {util}% utilization")
        print(f"  Collecting for {duration_per_phase}s...")

        phase_data = collect_correlation_data(
            sensor, duration_per_phase,
            window_size=window_size, label=phase_name
        )
        all_data.extend(phase_data)

        if phase_data:
            rhos = [d['rho_ab'] for d in phase_data]
            print(f"  ρ(A,B) = {np.mean(rhos):.4f} ± {np.std(rhos):.4f}")

        if load:
            load.stop()
            time.sleep(1)  # Let GPU settle

    # Analyze results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print()

    # Group by phase
    phase_stats = {}
    for phase_name, _, _ in phases:
        phase_points = [d for d in all_data if d['label'] == phase_name]
        if phase_points:
            rhos = [d['rho_ab'] for d in phase_points]
            temps = [d['temp'] for d in phase_points]
            utils = [d['util'] for d in phase_points]
            phase_stats[phase_name] = {
                'rho_mean': np.mean(rhos),
                'rho_std': np.std(rhos),
                'temp_mean': np.mean(temps),
                'util_mean': np.mean(utils),
                'n_windows': len(phase_points),
            }

    print(f"{'Phase':<20} {'ρ(A,B)':<20} {'Temp':<10} {'Util':<10}")
    print("-"*60)

    baseline_rho = phase_stats.get('baseline_pre', {}).get('rho_mean', 0)

    for phase_name, stats in phase_stats.items():
        delta = stats['rho_mean'] - baseline_rho
        print(f"{phase_name:<20} {stats['rho_mean']:+.4f} ± {stats['rho_std']:.4f}  "
              f"{stats['temp_mean']:.0f}°C    {stats['util_mean']:.0f}%  "
              f"(Δ={delta:+.4f})")

    # Key findings
    print()
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)
    print()

    if 'baseline_pre' in phase_stats and 'matmul_high' in phase_stats:
        delta_load = phase_stats['matmul_high']['rho_mean'] - phase_stats['baseline_pre']['rho_mean']
        print(f"Load effect (no load → matmul): Δρ = {delta_load:+.4f}")
        print()

        if abs(delta_load) > 0.05:
            print("✓ SIGNIFICANT LOAD SENSITIVITY DETECTED")
            print(f"  The correlation structure changes by {abs(delta_load):.2f}")
            print("  when concurrent GPU compute is running.")
            print()
            print("  Potential applications:")
            print("  - Detect other processes using GPU")
            print("  - Detect interference/contention")
            print("  - Software-only GPU activity monitor")
        else:
            print("? Weak or no load sensitivity detected")

    # Check temperature confound
    if 'baseline_pre' in phase_stats and 'baseline_post' in phase_stats:
        temp_delta = phase_stats['baseline_post']['temp_mean'] - phase_stats['baseline_pre']['temp_mean']
        rho_delta = phase_stats['baseline_post']['rho_mean'] - phase_stats['baseline_pre']['rho_mean']
        print()
        print(f"Temperature confound check:")
        print(f"  Pre→Post baseline temp change: {temp_delta:+.1f}°C")
        print(f"  Pre→Post baseline ρ change:    {rho_delta:+.4f}")

        if abs(temp_delta) > 5 and abs(rho_delta) > 0.02:
            print("  ⚠️  Temperature may be confounding load effect")
        else:
            print("  ✓ Temperature confound appears minimal")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp5_load_sensitivity_{timestamp}.json"

    results = {
        'experiment': 'exp5_load_sensitivity',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'duration_per_phase': duration_per_phase,
            'window_size': window_size,
            'n_iterations': n_iterations,
        },
        'phase_stats': phase_stats,
        'data_points': all_data,
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Experiment 5: GPU Load Sensitivity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp5_load_sensitivity.py --duration 30
  python exp5_load_sensitivity.py --duration 60 --window 1000
        """
    )
    parser.add_argument('--duration', '-d', type=float, default=30,
                        help='Duration per phase in seconds (default: 30)')
    parser.add_argument('--window', type=int, default=500,
                        help='Window size for correlation (default: 500)')
    parser.add_argument('--iterations', '-i', type=int, default=5000,
                        help='Kernel iterations (default: 5000)')
    args = parser.parse_args()

    run_load_sensitivity_experiment(
        duration_per_phase=args.duration,
        window_size=args.window,
        n_iterations=args.iterations
    )
