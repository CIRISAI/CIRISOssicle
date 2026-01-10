#!/usr/bin/env python3
"""
Experiment 22: 7-Oscillator Prime Configuration

n=7 is prime, DOF=21 (also between Fibonacci numbers!)
Testing if prime oscillator counts have special properties.

DOF = n(n-1)/2 = 7*6/2 = 21

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
from itertools import combinations


class PrimeSensorKernel:
    """7-oscillator sensor with 21 correlation pairs."""

    KERNEL_CODE = r'''
    extern "C" __global__ void prime7_step(
        float* states, float* r_vals, float* twists,
        float coupling, int n, int n_osc, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float s[7];
        for (int o = 0; o < 7; o++) {
            s[o] = states[o * n + idx];
        }

        float r[7], tw[7];
        for (int o = 0; o < 7; o++) {
            r[o] = r_vals[o];
            tw[o] = twists[o];
        }

        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        for (int iter = 0; iter < iterations; iter++) {
            float new_s[7];

            for (int o = 0; o < 7; o++) {
                float neighbor = states[o * n + left] + states[o * n + right];

                float cross = 0.0f;
                for (int p = 0; p < 7; p++) {
                    if (p != o) {
                        float twist_diff = tw[p] - tw[o];
                        cross += s[p] * cosf(twist_diff) * 0.025f;
                    }
                }

                new_s[o] = r[o] * s[o] * (1.0f - s[o])
                         + coupling * (neighbor - 2.0f * s[o])
                         + coupling * cross;

                new_s[o] = fminf(fmaxf(new_s[o], 0.0001f), 0.9999f);
            }

            for (int o = 0; o < 7; o++) {
                s[o] = new_s[o];
                states[o * n + idx] = s[o];
            }
        }
    }
    '''

    def __init__(self, r_base: float = 3.70, spacing: float = 0.02,
                 twist_angle_deg: float = 0.0,
                 n_cells: int = 1024, coupling: float = 0.05,
                 n_iterations: int = 5000):

        self.n_osc = 7
        self.n_cells = n_cells
        self.coupling = coupling
        self.n_iterations = n_iterations

        self.r_vals = np.array([r_base + i * spacing for i in range(7)], dtype=np.float32)

        # Twist on last oscillator (probe)
        self.twists = np.zeros(7, dtype=np.float32)
        self.twists[6] = np.radians(twist_angle_deg)

        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('prime7_step')

        self.states_gpu = None
        self.r_vals_gpu = cp.asarray(self.r_vals)
        self.twists_gpu = cp.asarray(self.twists)

        self._init_states()

        self.block_size = 256
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size

        self.dof = self.n_osc * (self.n_osc - 1) // 2
        print(f"7-oscillator sensor: DOF = {self.dof}")

    def _init_states(self):
        states = np.random.uniform(0.1, 0.9, (self.n_osc, self.n_cells)).astype(np.float32)
        self.states_gpu = cp.asarray(states)

    def step(self):
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.states_gpu, self.r_vals_gpu, self.twists_gpu,
             cp.float32(self.coupling), cp.int32(self.n_cells),
             cp.int32(self.n_osc), cp.int32(self.n_iterations))
        )
        cp.cuda.Stream.null.synchronize()
        return [float(cp.mean(self.states_gpu[i])) for i in range(self.n_osc)]

    def reset(self):
        self._init_states()


class WorkloadGenerator:
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
        elif self.workload_type == "memory":
            src = cp.random.randn(size * size, dtype=cp.float32)
            dst = cp.zeros(size * size, dtype=cp.float32)
            while self.running:
                dst[:] = src
                src[:] = dst
                cp.cuda.Stream.null.synchronize()


def collect_correlations(sensor, duration: float, window: int = 100):
    history = [[] for _ in range(sensor.n_osc)]
    correlations = []
    pairs = list(combinations(range(sensor.n_osc), 2))

    start = time.time()
    while time.time() - start < duration:
        means = sensor.step()
        for i, m in enumerate(means):
            history[i].append(m)

        if len(history[0]) >= window and len(history[0]) % (window // 4) == 0:
            arrays = [np.array(h[-window:]) for h in history]

            def safe_corr(x, y):
                if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                    return 0.0
                r = np.corrcoef(x, y)[0, 1]
                return r if not np.isnan(r) else 0.0

            pair_corrs = {f'rho_{i}{j}': safe_corr(arrays[i], arrays[j]) for i, j in pairs}
            pair_corrs['mean'] = np.mean(list(pair_corrs.values()))
            correlations.append(pair_corrs)

    return correlations


def run_prime_test():
    print("="*70)
    print("EXPERIMENT 22: 7-OSCILLATOR (PRIME) CONFIGURATION")
    print("="*70)
    print()
    print("n=7 (prime), DOF=21")
    print("Testing twist angles to find optimal...")
    print()

    results = {'experiment': 'exp22_7osc_prime', 'timestamp': datetime.now().isoformat()}

    # Twist sweep including graphene magic angle and fractions
    twists = [0.0, 0.275, 0.55, 0.786, 1.1, 1.57, 2.2, 5.0, 15.0, 45.0, 90.0]

    config_results = []

    for twist in twists:
        print(f"\nTesting twist = {twist} deg...")

        sensor = PrimeSensorKernel(r_base=3.70, spacing=0.02, twist_angle_deg=twist)

        # Baseline
        idle = WorkloadGenerator("idle")
        idle.start()
        time.sleep(1)
        sensor.reset()
        baseline = collect_correlations(sensor, duration=25.0)
        idle.stop()

        if not baseline:
            continue

        baseline_means = [c['mean'] for c in baseline]
        baseline_mean = np.mean(baseline_means)
        baseline_std = np.std(baseline_means)

        # Attack
        attack_wl = WorkloadGenerator("memory", intensity=0.8)
        attack_wl.start()
        time.sleep(1)
        sensor.reset()
        attack = collect_correlations(sensor, duration=25.0)
        attack_wl.stop()

        if not attack:
            continue

        attack_mean = np.mean([c['mean'] for c in attack])
        delta = attack_mean - baseline_mean
        z_score = abs(delta) / (baseline_std + 1e-10)

        result = {
            'twist': twist,
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'attack_mean': float(attack_mean),
            'delta': float(delta),
            'z_score': float(z_score)
        }
        config_results.append(result)

        mark = " ***" if z_score > 3 else ""
        print(f"  z = {z_score:.2f}{mark}")

    results['configurations'] = config_results

    # Analysis
    print("\n" + "="*70)
    print("7-OSCILLATOR (DOF=21) RESULTS")
    print("="*70)

    if config_results:
        baseline_z = next((r['z_score'] for r in config_results if r['twist'] == 0.0), 0.01)

        print("\n| Twist (deg) | z-score | Amplification |")
        print("|-------------|---------|---------------|")
        for r in sorted(config_results, key=lambda x: x['z_score'], reverse=True):
            amp = r['z_score'] / (baseline_z + 1e-10)
            mark = " <-- BEST" if r == max(config_results, key=lambda x: x['z_score']) else ""
            print(f"| {r['twist']:11.3f} | {r['z_score']:7.2f} | {amp:13.1f}x |{mark}")

        best = max(config_results, key=lambda x: x['z_score'])
        print(f"\n*** OPTIMAL TWIST AT DOF=21: {best['twist']} deg ***")
        print(f"    z-score: {best['z_score']:.2f}")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp22_7osc_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_prime_test()
