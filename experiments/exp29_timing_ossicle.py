#!/usr/bin/env python3
"""
Experiment 29: TimingOssicle - Explicit Timing Extraction

Based on RATCHET findings and cross-GPU coherence test results:
- Correlation structure is 100% algorithmic
- Detection works via timing effects (p=0.007)
- Raw timing has high entropy (8 bits/byte in RATCHET)

This experiment makes TIMING EXPLICIT:
1. Capture kernel execution time for each step
2. Use timing variance as primary strain signal
3. Compare timing-based detection vs correlation-based detection

ARCHITECTURE:
                    ┌─────────────────┐
   Kernel ────────► │  Timing LSBs   │────► TRNG output
   Execution        │  (ns precision) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Timing Variance │────► Strain gauge
                    │   (σ shift)     │      (workload detection)
                    └─────────────────┘

HYPOTHESIS:
- Timing variance increases under workload (scheduling contention)
- This is the ACTUAL signal causing correlation shifts
- Direct timing measurement may be more sensitive than correlations

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


class TimingOssicle:
    """
    Ossicle that explicitly captures kernel timing.

    Two outputs:
    1. Timing statistics (the physical signal)
    2. Correlations (the strain gauge, for comparison)
    """

    KERNEL_CODE = r'''
    extern "C" __global__ void ossicle_step(
        float* state_a, float* state_b, float* state_c,
        float r_a, float r_b, float r_c,
        float twist_ab, float twist_bc,
        float coupling, int n, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float a = state_a[idx];
        float b = state_b[idx];
        float c = state_c[idx];

        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        for (int iter = 0; iter < iterations; iter++) {
            float na = state_a[left] + state_a[right];
            float nb = state_b[left] + state_b[right];
            float nc = state_c[left] + state_c[right];

            float interference_ab = b * cosf(twist_ab) + a * cosf(-twist_ab);
            float interference_bc = c * cosf(twist_bc) + b * cosf(-twist_bc);

            float new_a = r_a * a * (1.0f - a)
                        + coupling * (na - 2.0f * a)
                        + coupling * 0.1f * interference_ab;

            float new_b = r_b * b * (1.0f - b)
                        + coupling * (nb - 2.0f * b)
                        + coupling * 0.1f * (interference_ab + interference_bc);

            float new_c = r_c * c * (1.0f - c)
                        + coupling * (nc - 2.0f * c)
                        + coupling * 0.1f * interference_bc;

            a = fminf(fmaxf(new_a, 0.0001f), 0.9999f);
            b = fminf(fmaxf(new_b, 0.0001f), 0.9999f);
            c = fminf(fmaxf(new_c, 0.0001f), 0.9999f);

            state_a[idx] = a;
            state_b[idx] = b;
            state_c[idx] = c;
        }
    }
    '''

    def __init__(self, n_cells=64, n_iterations=500):
        self.n_cells = n_cells
        self.n_iterations = n_iterations

        self.r_a = 3.70
        self.r_b = 3.73
        self.r_c = 3.76

        self.twist_ab = np.radians(1.1)
        self.twist_bc = np.radians(1.1)
        self.coupling = 0.05

        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('ossicle_step')

        self.block_size = 64
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size

        self.reset()

    def reset(self):
        """Reset oscillator states."""
        self.state_a = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_b = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_c = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)

    def step_timed(self):
        """
        Execute one step and return both timing and state means.

        Returns:
            (timing_ns, mean_a, mean_b, mean_c)
        """
        # High-precision timing
        start = time.perf_counter_ns()

        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.state_a, self.state_b, self.state_c,
             cp.float32(self.r_a), cp.float32(self.r_b), cp.float32(self.r_c),
             cp.float32(self.twist_ab), cp.float32(self.twist_bc),
             cp.float32(self.coupling), cp.int32(self.n_cells),
             cp.int32(self.n_iterations))
        )
        cp.cuda.Stream.null.synchronize()

        end = time.perf_counter_ns()
        timing_ns = end - start

        return (
            timing_ns,
            float(cp.mean(self.state_a)),
            float(cp.mean(self.state_b)),
            float(cp.mean(self.state_c))
        )


class TimingStrainGauge:
    """
    Strain gauge based on kernel timing variance.

    Detection method:
    1. Baseline: Measure timing variance under idle conditions
    2. Monitor: Detect when timing variance shifts significantly

    This is the direct signal - no chaotic mixing needed.
    """

    def __init__(self, window_size=100):
        self.sensor = TimingOssicle()
        self.window_size = window_size

        # Baseline statistics
        self.baseline_timing_mean = None
        self.baseline_timing_std = None
        self.baseline_corr_mean = None
        self.baseline_corr_std = None

        # Rolling windows
        self.timing_window = deque(maxlen=window_size)
        self.corr_window = deque(maxlen=window_size)

    def calibrate(self, duration=10.0):
        """Calibrate baseline timing and correlation statistics."""
        print(f"Calibrating for {duration}s... (keep system IDLE)")

        timings = []
        correlations = []

        history_a, history_b, history_c = [], [], []

        start = time.time()
        while time.time() - start < duration:
            timing_ns, a, b, c = self.sensor.step_timed()
            timings.append(timing_ns)
            history_a.append(a)
            history_b.append(b)
            history_c.append(c)

        # Timing statistics
        self.baseline_timing_mean = np.mean(timings)
        self.baseline_timing_std = np.std(timings)

        # Correlation statistics (over full window)
        arr_a = np.array(history_a)
        arr_b = np.array(history_b)
        arr_c = np.array(history_c)

        def safe_corr(x, y):
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                return 0.0
            r = np.corrcoef(x, y)[0, 1]
            return r if not np.isnan(r) else 0.0

        rho_ab = safe_corr(arr_a, arr_b)
        rho_bc = safe_corr(arr_b, arr_c)
        rho_ac = safe_corr(arr_a, arr_c)

        mean_corr = (rho_ab + rho_bc + rho_ac) / 3

        self.baseline_corr_mean = mean_corr
        self.baseline_corr_std = np.std([rho_ab, rho_bc, rho_ac])

        # Ensure non-zero std
        if self.baseline_timing_std < 1:
            self.baseline_timing_std = 1  # ns minimum
        if self.baseline_corr_std < 0.001:
            self.baseline_corr_std = 0.001

        print(f"Baseline timing: {self.baseline_timing_mean/1000:.1f} ± {self.baseline_timing_std/1000:.1f} μs")
        print(f"Baseline correlation: {self.baseline_corr_mean:+.6f} ± {self.baseline_corr_std:.6f}")
        print(f"Samples: {len(timings)}")

        return {
            'timing_mean': self.baseline_timing_mean,
            'timing_std': self.baseline_timing_std,
            'corr_mean': self.baseline_corr_mean,
            'corr_std': self.baseline_corr_std,
            'n_samples': len(timings)
        }

    def measure(self, duration=5.0):
        """
        Measure both timing and correlation strain.

        Returns dict with both signal types for comparison.
        """
        if self.baseline_timing_mean is None:
            raise RuntimeError("Call calibrate() first")

        self.sensor.reset()

        timings = []
        history_a, history_b, history_c = [], [], []

        start = time.time()
        while time.time() - start < duration:
            timing_ns, a, b, c = self.sensor.step_timed()
            timings.append(timing_ns)
            history_a.append(a)
            history_b.append(b)
            history_c.append(c)

        # Timing analysis
        timing_mean = np.mean(timings)
        timing_std = np.std(timings)
        timing_z = abs(timing_mean - self.baseline_timing_mean) / self.baseline_timing_std
        timing_var_ratio = timing_std / self.baseline_timing_std

        # Correlation analysis
        arr_a = np.array(history_a)
        arr_b = np.array(history_b)
        arr_c = np.array(history_c)

        def safe_corr(x, y):
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                return 0.0
            r = np.corrcoef(x, y)[0, 1]
            return r if not np.isnan(r) else 0.0

        rho_ab = safe_corr(arr_a, arr_b)
        rho_bc = safe_corr(arr_b, arr_c)
        rho_ac = safe_corr(arr_a, arr_c)

        mean_corr = (rho_ab + rho_bc + rho_ac) / 3
        corr_z = abs(mean_corr - self.baseline_corr_mean) / self.baseline_corr_std

        # Extract timing LSBs for entropy analysis
        timing_lsbs = [t & 0xFF for t in timings]
        lsb_entropy = self._estimate_entropy(timing_lsbs)

        return {
            # Timing-based detection
            'timing_mean_us': timing_mean / 1000,
            'timing_std_us': timing_std / 1000,
            'timing_z': timing_z,
            'timing_var_ratio': timing_var_ratio,
            'timing_detected': timing_z > 2.0 or timing_var_ratio > 2.0,

            # Correlation-based detection (original method)
            'corr_mean': mean_corr,
            'corr_z': corr_z,
            'corr_detected': corr_z > 2.0,

            # Entropy from timing LSBs
            'timing_lsb_entropy': lsb_entropy,

            # Meta
            'n_samples': len(timings),
            'sample_rate': len(timings) / duration
        }

    def _estimate_entropy(self, data):
        """Estimate entropy in bits/byte from data."""
        if len(data) < 10:
            return 0.0

        # Count byte frequencies
        counts = np.bincount(data, minlength=256)
        probs = counts / len(data)
        probs = probs[probs > 0]  # Remove zeros

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        return entropy


class WorkloadSimulator:
    """Simple workload generator for testing."""

    COMPUTE_KERNEL = r'''
    extern "C" __global__ void compute_load(float* data, int n, int rounds) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float x = data[idx];
        for (int r = 0; r < rounds; r++) {
            x = sinf(x) * cosf(x) + 0.1f;
            x = sqrtf(fabsf(x) + 0.01f);
        }
        data[idx] = x;
    }
    '''

    def __init__(self, intensity=0.5):
        self.intensity = intensity
        self.running = False
        self.thread = None

        self.module = cp.RawModule(code=self.COMPUTE_KERNEL)
        self.kernel = self.module.get_function('compute_load')

        self.size = int(512 * 512 * intensity)
        self.rounds = int(50 * intensity)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        data = cp.random.randn(self.size, dtype=cp.float32)
        block_size = 256
        grid_size = (self.size + block_size - 1) // block_size

        while self.running:
            self.kernel(
                (grid_size,), (block_size,),
                (data, cp.int32(self.size), cp.int32(self.rounds))
            )
            cp.cuda.Stream.null.synchronize()


def run_comparison_test():
    """Compare timing-based vs correlation-based detection."""
    print("=" * 70)
    print("EXPERIMENT 29: TIMING OSSICLE")
    print("=" * 70)
    print()
    print("Comparing TIMING-based vs CORRELATION-based detection")
    print()
    print("HYPOTHESIS: Timing is the primary signal; correlations are secondary.")
    print()

    gauge = TimingStrainGauge()

    # Calibrate
    print("-" * 50)
    baseline = gauge.calibrate(duration=10.0)
    print()

    results = {
        'experiment': 'exp29_timing_ossicle',
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline,
        'tests': []
    }

    # Test conditions
    test_configs = [
        {'name': 'idle', 'intensity': 0},
        {'name': 'load_30%', 'intensity': 0.3},
        {'name': 'load_50%', 'intensity': 0.5},
        {'name': 'load_70%', 'intensity': 0.7},
        {'name': 'load_90%', 'intensity': 0.9},
    ]

    print("=" * 70)
    print("DETECTION COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Condition':<12} {'Timing z':>10} {'Timing Det':>12} {'Corr z':>10} {'Corr Det':>10} {'LSB Ent':>10}")
    print("-" * 70)

    for config in test_configs:
        workload = None
        if config['intensity'] > 0:
            workload = WorkloadSimulator(intensity=config['intensity'])
            workload.start()
            time.sleep(0.5)  # Stabilize

        measurement = gauge.measure(duration=5.0)

        if workload:
            workload.stop()

        timing_det = "YES" if measurement['timing_detected'] else "no"
        corr_det = "YES" if measurement['corr_detected'] else "no"

        print(f"{config['name']:<12} "
              f"{measurement['timing_z']:>10.2f} "
              f"{timing_det:>12} "
              f"{measurement['corr_z']:>10.2f} "
              f"{corr_det:>10} "
              f"{measurement['timing_lsb_entropy']:>10.2f}")

        # Convert numpy/bool types for JSON serialization
        test_result = {
            'name': config['name'],
            'intensity': config['intensity'],
        }
        for k, v in measurement.items():
            if isinstance(v, (np.bool_, bool)):
                test_result[k] = bool(v)
            elif isinstance(v, (np.integer, np.floating)):
                test_result[k] = float(v)
            else:
                test_result[k] = v
        results['tests'].append(test_result)

        time.sleep(0.5)  # Cool down

    print("-" * 70)
    print()

    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    timing_detections = sum(1 for t in results['tests'] if t['timing_detected'] and t['intensity'] > 0)
    corr_detections = sum(1 for t in results['tests'] if t['corr_detected'] and t['intensity'] > 0)
    total_workload_tests = sum(1 for t in results['tests'] if t['intensity'] > 0)

    print(f"\nTiming-based detection rate: {timing_detections}/{total_workload_tests}")
    print(f"Correlation-based detection rate: {corr_detections}/{total_workload_tests}")

    # Compare sensitivity
    workload_tests = [t for t in results['tests'] if t['intensity'] > 0]
    if workload_tests:
        avg_timing_z = np.mean([t['timing_z'] for t in workload_tests])
        avg_corr_z = np.mean([t['corr_z'] for t in workload_tests])

        print(f"\nAverage timing z-score under load: {avg_timing_z:.2f}")
        print(f"Average correlation z-score under load: {avg_corr_z:.2f}")

        if avg_timing_z > avg_corr_z:
            print(f"\n>>> TIMING is {avg_timing_z/avg_corr_z:.1f}x MORE SENSITIVE than correlations")
        else:
            print(f"\n>>> CORRELATION is {avg_corr_z/avg_timing_z:.1f}x MORE SENSITIVE than timing")

    # Entropy analysis
    print("\n" + "=" * 70)
    print("TIMING LSB ENTROPY")
    print("=" * 70)

    idle_entropy = results['tests'][0]['timing_lsb_entropy']
    print(f"\nIdle entropy: {idle_entropy:.2f} bits/byte")
    print(f"Max possible: 8.00 bits/byte")
    print(f"Efficiency: {idle_entropy/8*100:.1f}%")

    if idle_entropy > 6.0:
        print("\n>>> Timing LSBs have HIGH ENTROPY - good for TRNG")
    elif idle_entropy > 4.0:
        print("\n>>> Timing LSBs have MODERATE ENTROPY")
    else:
        print("\n>>> Timing LSBs have LOW ENTROPY - needs conditioning")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp29_timing_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if avg_timing_z > avg_corr_z:
        print("""
    TIMING-BASED DETECTION is more sensitive.

    Recommended architecture:

        Kernel ──► Timing (ns) ──┬──► LSB Extract ──► TRNG (entropy)
                                 │
                                 └──► Variance ──► Strain Gauge (detection)

    The chaotic oscillator is OPTIONAL - timing alone may be sufficient
    for both entropy generation and workload detection.
        """)
    else:
        print("""
    CORRELATION-BASED DETECTION is more sensitive.

    This is unexpected given cross-GPU coherence results.
    Possible explanations:
    - Timing affects sample positions in chaotic trajectory
    - Correlations aggregate timing effects over many samples
    - Correlations may be more robust to measurement noise

    Recommended: Keep current architecture but add timing capture
    for multi-modal detection.
        """)

    return results


if __name__ == "__main__":
    run_comparison_test()
