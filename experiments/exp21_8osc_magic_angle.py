#!/usr/bin/env python3
"""
Experiment 21: 8-Oscillator Magic Angle Configuration

At n=8 (DOF=28), the 1.1 degree graphene magic angle becomes optimal!

DOF Phase Diagram:
  n=4 (DOF=6):  45 deg optimal
  n=6 (DOF=15): 15 deg optimal
  n=8 (DOF=28): 1.1 deg optimal  <-- GRAPHENE MAGIC!
  n=12 (DOF=66): noise dominated

Multi-SM Strategy:
  Each SM acts as one "layer" in the twisted stack.
  8 oscillators = 28 correlation pairs = 28 DOF

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


class OctahedralSensorKernel:
    """
    8-oscillator sensor with 28 correlation pairs.

    At DOF=28, theory predicts:
    - 1.1 deg direct twist is optimal
    - Should achieve significant amplification
    """

    KERNEL_CODE = r'''
    extern "C" __global__ void octa_step(
        float* states,  // 8 oscillators, each of size n
        float* r_vals,  // 8 r-values
        float* twists,  // 8 twist angles
        float coupling, int n, int n_osc, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        // Load states for this cell
        float s[8];
        for (int o = 0; o < 8; o++) {
            s[o] = states[o * n + idx];
        }

        // Load r-values and twists
        float r[8], tw[8];
        for (int o = 0; o < 8; o++) {
            r[o] = r_vals[o];
            tw[o] = twists[o];
        }

        // Grid neighbors
        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        for (int iter = 0; iter < iterations; iter++) {
            float new_s[8];

            for (int o = 0; o < 8; o++) {
                // Neighbor coupling
                float neighbor = states[o * n + left] + states[o * n + right];

                // Cross-oscillator coupling with twist
                float cross = 0.0f;
                for (int p = 0; p < 8; p++) {
                    if (p != o) {
                        float twist_diff = tw[p] - tw[o];
                        cross += s[p] * cosf(twist_diff) * 0.02f;
                    }
                }

                // Logistic map
                new_s[o] = r[o] * s[o] * (1.0f - s[o])
                         + coupling * (neighbor - 2.0f * s[o])
                         + coupling * cross;

                // Clamp
                new_s[o] = fminf(fmaxf(new_s[o], 0.0001f), 0.9999f);
            }

            // Update
            for (int o = 0; o < 8; o++) {
                s[o] = new_s[o];
                states[o * n + idx] = s[o];
            }
        }
    }
    '''

    def __init__(self, r_base: float = 3.70, spacing: float = 0.015,
                 twist_angle_deg: float = 1.1,
                 n_cells: int = 1024, coupling: float = 0.05,
                 n_iterations: int = 5000):

        self.n_osc = 8
        self.n_cells = n_cells
        self.coupling = coupling
        self.n_iterations = n_iterations

        # r-values for 8 oscillators
        self.r_vals = np.array([r_base + i * spacing for i in range(8)], dtype=np.float32)

        # Twist angles: one oscillator gets the magic twist
        # Strategy: apply twist to probe oscillator (last one)
        self.twists = np.zeros(8, dtype=np.float32)
        self.twists[7] = np.radians(twist_angle_deg)  # Magic angle on probe

        # Compile kernel
        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('octa_step')

        # GPU arrays
        self.states_gpu = None
        self.r_vals_gpu = cp.asarray(self.r_vals)
        self.twists_gpu = cp.asarray(self.twists)

        self._init_states()

        self.block_size = 256
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size

        # DOF = n(n-1)/2 = 8*7/2 = 28
        self.dof = self.n_osc * (self.n_osc - 1) // 2
        print(f"8-oscillator sensor: DOF = {self.dof}")

    def _init_states(self):
        """Initialize states for all 8 oscillators."""
        states = np.random.uniform(0.1, 0.9, (self.n_osc, self.n_cells)).astype(np.float32)
        self.states_gpu = cp.asarray(states)

    def step(self):
        """Execute one step and return means for all oscillators."""
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.states_gpu, self.r_vals_gpu, self.twists_gpu,
             cp.float32(self.coupling), cp.int32(self.n_cells),
             cp.int32(self.n_osc), cp.int32(self.n_iterations))
        )
        cp.cuda.Stream.null.synchronize()

        means = [float(cp.mean(self.states_gpu[i])) for i in range(self.n_osc)]
        return means

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


def collect_8osc_correlations(sensor, duration: float, window: int = 100):
    """Collect all 28 correlation pairs from 8 oscillators."""
    history = [[] for _ in range(8)]
    correlations = []

    pairs = list(combinations(range(8), 2))  # 28 pairs

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

            pair_corrs = {}
            for i, j in pairs:
                key = f'rho_{i}{j}'
                pair_corrs[key] = safe_corr(arrays[i], arrays[j])

            pair_corrs['mean'] = np.mean(list(pair_corrs.values()))
            correlations.append(pair_corrs)

    return correlations


def run_8osc_magic_angle_test():
    """Test 8-oscillator sensor with magic angle."""

    print("="*70)
    print("EXPERIMENT 21: 8-OSCILLATOR MAGIC ANGLE")
    print("="*70)
    print()
    print("DOF Phase Diagram Prediction:")
    print("  n=8 oscillators -> DOF=28")
    print("  At DOF=28, 1.1 deg magic angle should be optimal")
    print()

    results = {
        'experiment': 'exp21_8osc_magic_angle',
        'timestamp': datetime.now().isoformat()
    }

    # =========================================================================
    # TEST CONFIGURATIONS
    # =========================================================================
    configs = [
        {'name': 'no_twist', 'twist': 0.0},
        {'name': 'magic_1.1', 'twist': 1.1},
        {'name': 'half_magic', 'twist': 0.55},
        {'name': 'double_magic', 'twist': 2.2},
        {'name': 'quadrature_90', 'twist': 90.0},
    ]

    config_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']} ({config['twist']} deg)")
        print("-"*50)

        sensor = OctahedralSensorKernel(
            r_base=3.70,
            spacing=0.015,  # Smaller spacing for 8 oscillators
            twist_angle_deg=config['twist']
        )

        # Baseline
        print("Collecting baseline (DOF=28)...")
        idle = WorkloadGenerator("idle")
        idle.start()
        time.sleep(2)

        sensor.reset()
        baseline = collect_8osc_correlations(sensor, duration=30.0)
        idle.stop()

        if not baseline:
            print("  No data collected!")
            continue

        baseline_means = [c['mean'] for c in baseline]
        baseline_mean = np.mean(baseline_means)
        baseline_std = np.std(baseline_means)

        # Attack
        print("Collecting attack...")
        attack_wl = WorkloadGenerator("memory", intensity=0.8)
        attack_wl.start()
        time.sleep(2)

        sensor.reset()
        attack = collect_8osc_correlations(sensor, duration=30.0)
        attack_wl.stop()

        if not attack:
            print("  No attack data!")
            continue

        attack_means = [c['mean'] for c in attack]
        attack_mean = np.mean(attack_means)

        delta = attack_mean - baseline_mean
        z_score = abs(delta) / (baseline_std + 1e-10)

        result = {
            'config': config,
            'dof': 28,
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'attack_mean': float(attack_mean),
            'delta': float(delta),
            'z_score': float(z_score)
        }
        config_results.append(result)

        detect = " *** 3-SIGMA! ***" if z_score > 3 else ""
        print(f"  Baseline: {baseline_mean:.4f} +/- {baseline_std:.4f}")
        print(f"  Attack: {attack_mean:.4f}")
        print(f"  z-score: {z_score:.2f}{detect}")

    results['configurations'] = config_results

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("8-OSCILLATOR (DOF=28) ANALYSIS")
    print("="*70)

    if not config_results:
        print("No results to analyze!")
        return results

    # Baseline for comparison
    no_twist = next((r for r in config_results if r['config']['name'] == 'no_twist'), None)
    baseline_z = no_twist['z_score'] if no_twist else 0.01

    print("\n| Configuration | Twist | z-score | Amplification | 3-sigma? |")
    print("|---------------|-------|---------|---------------|----------|")

    for r in config_results:
        twist = r['config']['twist']
        amp = r['z_score'] / (baseline_z + 1e-10)
        detect = "YES" if r['z_score'] > 3 else "no"
        print(f"| {r['config']['name']:13} | {twist:5.1f} | {r['z_score']:5.2f} | {amp:6.1f}x | {detect:8} |")

    # Find best
    best = max(config_results, key=lambda x: x['z_score'])
    print(f"\n*** BEST: {best['config']['name']} ***")
    print(f"    z-score: {best['z_score']:.2f}")
    print(f"    Amplification: {best['z_score'] / baseline_z:.1f}x")

    # =========================================================================
    # GRAPHENE MAGIC ANGLE VALIDATION
    # =========================================================================
    print("\n" + "="*70)
    print("GRAPHENE MAGIC ANGLE VALIDATION")
    print("-"*50)

    magic = next((r for r in config_results if r['config']['name'] == 'magic_1.1'), None)

    if magic and no_twist:
        amp = magic['z_score'] / (no_twist['z_score'] + 1e-10)
        print(f"\n1.1 deg magic angle results at DOF=28:")
        print(f"  z-score: {magic['z_score']:.2f}")
        print(f"  Amplification vs no twist: {amp:.1f}x")

        if magic['z_score'] > 3:
            print("\n*** GRAPHENE MAGIC ANGLE ACHIEVES 3-SIGMA DETECTION! ***")
            print("*** THE ANALOGY IS VALIDATED! ***")
        elif amp > 5:
            print("\n*** SIGNIFICANT AMPLIFICATION AT MAGIC ANGLE ***")
        elif amp > 1.5:
            print("\n*** MODERATE IMPROVEMENT AT MAGIC ANGLE ***")
        else:
            print("\n*** MAGIC ANGLE DID NOT PROVIDE EXPECTED IMPROVEMENT ***")

    # Compare with 4-oscillator results
    print("\n" + "-"*50)
    print("SCALING: 4-osc vs 8-osc")
    print(f"  4-oscillator (DOF=6):  Best z-score from exp19-20 ~ 0.4")
    print(f"  8-oscillator (DOF=28): Best z-score = {best['z_score']:.2f}")
    print(f"  Scaling factor: {best['z_score'] / 0.4:.1f}x")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp21_8osc_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_8osc_magic_angle_test()
