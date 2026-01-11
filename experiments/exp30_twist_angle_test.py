#!/usr/bin/env python3
"""
Experiment 30: Does Twist Angle Matter?

HYPOTHESIS: If timing is the real detection signal (not PDN coupling),
then the twist angle should be IRRELEVANT to detection sensitivity.

TEST: Run detection at different twist angles under load.
If detection is identical across angles, we can remove that complexity.

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


class ParameterizedOssicle:
    """Ossicle with configurable twist angle."""

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

    def __init__(self, twist_deg=1.1, n_cells=64, n_iterations=500):
        self.twist_deg = twist_deg
        self.n_cells = n_cells
        self.n_iterations = n_iterations

        self.r_a = 3.70
        self.r_b = 3.73
        self.r_c = 3.76

        self.twist_ab = np.radians(twist_deg)
        self.twist_bc = np.radians(twist_deg)
        self.coupling = 0.05

        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('ossicle_step')

        self.block_size = 64
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size
        self.reset()

    def reset(self):
        self.state_a = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_b = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_c = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)

    def step_timed(self):
        """Step with timing measurement."""
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

        return (
            end - start,
            float(cp.mean(self.state_a)),
            float(cp.mean(self.state_b)),
            float(cp.mean(self.state_c))
        )


class Workload:
    """Simple GPU workload."""

    KERNEL = r'''
    extern "C" __global__ void compute(float* data, int n, int rounds) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        float x = data[idx];
        for (int r = 0; r < rounds; r++) {
            x = sinf(x) * cosf(x) + 0.1f;
        }
        data[idx] = x;
    }
    '''

    def __init__(self, intensity=0.7):
        self.intensity = intensity
        self.running = False
        self.thread = None
        self.module = cp.RawModule(code=self.KERNEL)
        self.kernel = self.module.get_function('compute')
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
        bs = 256
        gs = (self.size + bs - 1) // bs
        while self.running:
            self.kernel((gs,), (bs,), (data, cp.int32(self.size), cp.int32(self.rounds)))
            cp.cuda.Stream.null.synchronize()


def measure(sensor, duration=5.0):
    """Measure timing and correlations."""
    sensor.reset()

    timings = []
    history_a, history_b, history_c = [], [], []

    start = time.time()
    while time.time() - start < duration:
        t, a, b, c = sensor.step_timed()
        timings.append(t)
        history_a.append(a)
        history_b.append(b)
        history_c.append(c)

    # Timing stats
    timing_mean = np.mean(timings)
    timing_std = np.std(timings)

    # Correlation stats
    arr_a, arr_b, arr_c = np.array(history_a), np.array(history_b), np.array(history_c)

    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return r if not np.isnan(r) else 0.0

    rho_ab = safe_corr(arr_a, arr_b)
    rho_bc = safe_corr(arr_b, arr_c)
    rho_ac = safe_corr(arr_a, arr_c)
    mean_corr = (rho_ab + rho_bc + rho_ac) / 3

    return {
        'timing_mean_us': timing_mean / 1000,
        'timing_std_us': timing_std / 1000,
        'mean_corr': mean_corr,
        'n_samples': len(timings)
    }


def run_test():
    """Test detection across different twist angles."""
    print("=" * 70)
    print("EXPERIMENT 30: DOES TWIST ANGLE MATTER?")
    print("=" * 70)
    print()
    print("HYPOTHESIS: If timing is the signal, twist angle is irrelevant.")
    print()

    # Test angles
    twist_angles = [0, 0.5, 1.1, 5.0, 45.0, 90.0]

    results = {
        'experiment': 'exp30_twist_angle_test',
        'timestamp': datetime.now().isoformat(),
        'angles': []
    }

    # Measure baseline and load for each angle
    print("Testing each angle: baseline then 70% load")
    print()
    print(f"{'Angle':>8} | {'Baseline':>20} | {'Under Load':>20} | {'Timing Δ':>10} | {'Corr Δ':>10}")
    print(f"{'':>8} | {'timing_μs / corr':>20} | {'timing_μs / corr':>20} | {'(load-base)':>10} | {'(load-base)':>10}")
    print("-" * 85)

    for angle in twist_angles:
        sensor = ParameterizedOssicle(twist_deg=angle)

        # Baseline measurement
        baseline = measure(sensor, duration=5.0)

        # Load measurement
        workload = Workload(intensity=0.7)
        workload.start()
        time.sleep(0.5)
        loaded = measure(sensor, duration=5.0)
        workload.stop()

        # Compute deltas
        timing_delta = loaded['timing_mean_us'] - baseline['timing_mean_us']
        corr_delta = loaded['mean_corr'] - baseline['mean_corr']

        print(f"{angle:>7.1f}° | "
              f"{baseline['timing_mean_us']:>8.1f} / {baseline['mean_corr']:>+8.5f} | "
              f"{loaded['timing_mean_us']:>8.1f} / {loaded['mean_corr']:>+8.5f} | "
              f"{timing_delta:>+10.1f} | "
              f"{corr_delta:>+10.5f}")

        results['angles'].append({
            'twist_deg': angle,
            'baseline_timing_us': baseline['timing_mean_us'],
            'baseline_corr': baseline['mean_corr'],
            'loaded_timing_us': loaded['timing_mean_us'],
            'loaded_corr': loaded['mean_corr'],
            'timing_delta': timing_delta,
            'corr_delta': corr_delta
        })

        time.sleep(0.5)

    print("-" * 85)

    # Analysis
    timing_deltas = [r['timing_delta'] for r in results['angles']]
    corr_deltas = [abs(r['corr_delta']) for r in results['angles']]

    timing_delta_std = np.std(timing_deltas)
    corr_delta_std = np.std(corr_deltas)

    print()
    print("ANALYSIS:")
    print(f"  Timing delta std across angles: {timing_delta_std:.2f} μs")
    print(f"  Correlation delta std across angles: {corr_delta_std:.6f}")
    print()

    # Conclusion
    if timing_delta_std < 5.0 and corr_delta_std < 0.01:
        print(">>> TWIST ANGLE DOES NOT MATTER")
        print("    Detection sensitivity is consistent across all angles.")
        print("    The twist angle can be REMOVED from the design.")
        results['conclusion'] = 'angle_irrelevant'
    else:
        print(">>> TWIST ANGLE MAY MATTER")
        print("    Detection varies significantly with angle.")
        print("    Further investigation needed.")
        results['conclusion'] = 'angle_may_matter'

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp30_twist_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_test()
