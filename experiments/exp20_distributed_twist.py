#!/usr/bin/env python3
"""
Experiment 20: Distributed Twist Strategy

Tests the twistronics prediction:
  n=4 oscillators, 0.367 deg/oscillator -> 17.8x amplification

This mimics multilayer graphene with small per-layer twist,
where the cumulative effect creates moiré interference.

DOF Phase Diagram Discovery:
  DOF=6 (n=4): 45 deg optimal
  DOF=15 (n=6): 15 deg optimal
  DOF=28 (n=8): 1.1 deg optimal (GRAPHENE MAGIC!)
  DOF=66+ : noise dominated

Two paths to high amplification:
1. LOW DOF + DISTRIBUTED: 0.367 deg/osc -> 17.8x
2. MEDIUM DOF + DIRECT: 1.1 deg on probe -> 9.0x

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


class DistributedTwistKernel:
    """
    CUDA kernel with distributed twist across oscillators.

    Distributed twist: Each oscillator has small phase offset from previous.
    Total twist = (n-1) * twist_per_oscillator

    For n=4, twist_per_osc=0.367 deg:
    A: 0 deg, B: 0.367 deg, C: 0.734 deg, D: 1.101 deg
    Total span: 1.1 deg (graphene magic angle!)
    """

    KERNEL_CODE = r'''
    extern "C" __global__ void distributed_twist_step(
        float* state_a, float* state_b, float* state_c, float* state_d,
        float r_a, float r_b, float r_c, float r_d,
        float coupling,
        float twist_a, float twist_b, float twist_c, float twist_d,
        int n, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float a = state_a[idx];
        float b = state_b[idx];
        float c = state_c[idx];
        float d = state_d[idx];

        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        // Twist factors (pre-computed cos of twist angles)
        float ta = twist_a;
        float tb = twist_b;
        float tc = twist_c;
        float td = twist_d;

        for (int i = 0; i < iterations; i++) {
            // Neighbor coupling with twist modulation
            float na = state_a[left] + state_a[right];
            float nb = state_b[left] + state_b[right];
            float nc = state_c[left] + state_c[right];
            float nd = state_d[left] + state_d[right];

            // Cross-layer coupling with twist-dependent interference
            // The twist creates phase-dependent coupling strength
            float cross_ab = coupling * 0.3f * b * cosf(tb - ta);
            float cross_ac = coupling * 0.3f * c * cosf(tc - ta);
            float cross_ad = coupling * 0.3f * d * cosf(td - ta);

            float cross_ba = coupling * 0.3f * a * cosf(ta - tb);
            float cross_bc = coupling * 0.3f * c * cosf(tc - tb);
            float cross_bd = coupling * 0.3f * d * cosf(td - tb);

            float cross_ca = coupling * 0.3f * a * cosf(ta - tc);
            float cross_cb = coupling * 0.3f * b * cosf(tb - tc);
            float cross_cd = coupling * 0.3f * d * cosf(td - tc);

            float cross_da = coupling * 0.3f * a * cosf(ta - td);
            float cross_db = coupling * 0.3f * b * cosf(tb - td);
            float cross_dc = coupling * 0.3f * c * cosf(tc - td);

            // Logistic map with distributed twist coupling
            a = r_a * a * (1.0f - a) + coupling * (na - 2.0f * a) + cross_ab + cross_ac + cross_ad;
            b = r_b * b * (1.0f - b) + coupling * (nb - 2.0f * b) + cross_ba + cross_bc + cross_bd;
            c = r_c * c * (1.0f - c) + coupling * (nc - 2.0f * c) + cross_ca + cross_cb + cross_cd;
            d = r_d * d * (1.0f - d) + coupling * (nd - 2.0f * d) + cross_da + cross_db + cross_dc;

            // Clamp
            a = fminf(fmaxf(a, 0.0001f), 0.9999f);
            b = fminf(fmaxf(b, 0.0001f), 0.9999f);
            c = fminf(fmaxf(c, 0.0001f), 0.9999f);
            d = fminf(fmaxf(d, 0.0001f), 0.9999f);

            state_a[idx] = a;
            state_b[idx] = b;
            state_c[idx] = c;
            state_d[idx] = d;
        }
    }
    '''

    def __init__(self, r_base: float = 3.70, spacing: float = 0.03,
                 twist_per_osc_deg: float = 0.367,
                 n_cells: int = 1024, coupling: float = 0.05,
                 n_iterations: int = 5000):

        self.r_base = r_base
        self.spacing = spacing
        self.twist_per_osc = np.radians(twist_per_osc_deg)
        self.n_cells = n_cells
        self.coupling = coupling
        self.n_iterations = n_iterations

        # r-values
        self.r_a = r_base
        self.r_b = r_base + spacing
        self.r_c = r_base + 2 * spacing
        self.r_d = r_base + 3 * spacing

        # Distributed twist angles (cumulative)
        self.twist_a = 0.0
        self.twist_b = self.twist_per_osc
        self.twist_c = 2 * self.twist_per_osc
        self.twist_d = 3 * self.twist_per_osc  # Total span = 3 * 0.367 = 1.101 deg

        # Compile kernel
        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('distributed_twist_step')

        self._init_states()

        self.block_size = 256
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size

    def _init_states(self):
        """Initialize with small twist-correlated perturbations."""
        self.state_a = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_b = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_c = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_d = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)

    def step(self):
        """Execute one step and return means."""
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.state_a, self.state_b, self.state_c, self.state_d,
             cp.float32(self.r_a), cp.float32(self.r_b),
             cp.float32(self.r_c), cp.float32(self.r_d),
             cp.float32(self.coupling),
             cp.float32(self.twist_a), cp.float32(self.twist_b),
             cp.float32(self.twist_c), cp.float32(self.twist_d),
             cp.int32(self.n_cells), cp.int32(self.n_iterations))
        )
        cp.cuda.Stream.null.synchronize()

        return (
            float(cp.mean(self.state_a)),
            float(cp.mean(self.state_b)),
            float(cp.mean(self.state_c)),
            float(cp.mean(self.state_d))
        )

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
    means_a, means_b, means_c, means_d = [], [], [], []
    correlations = []

    start = time.time()
    while time.time() - start < duration:
        a, b, c, d = sensor.step()
        means_a.append(a)
        means_b.append(b)
        means_c.append(c)
        means_d.append(d)

        if len(means_a) >= window and len(means_a) % (window // 4) == 0:
            arr_a = np.array(means_a[-window:])
            arr_b = np.array(means_b[-window:])
            arr_c = np.array(means_c[-window:])
            arr_d = np.array(means_d[-window:])

            def safe_corr(x, y):
                if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                    return 0.0
                r = np.corrcoef(x, y)[0, 1]
                return r if not np.isnan(r) else 0.0

            correlations.append({
                'rho_ab': safe_corr(arr_a, arr_b),
                'rho_ac': safe_corr(arr_a, arr_c),
                'rho_ad': safe_corr(arr_a, arr_d),
                'rho_bc': safe_corr(arr_b, arr_c),
                'rho_bd': safe_corr(arr_b, arr_d),
                'rho_cd': safe_corr(arr_c, arr_d),
                'mean': np.mean([
                    safe_corr(arr_a, arr_b), safe_corr(arr_a, arr_c),
                    safe_corr(arr_a, arr_d), safe_corr(arr_b, arr_c),
                    safe_corr(arr_b, arr_d), safe_corr(arr_c, arr_d)
                ])
            })

    return correlations


def run_distributed_twist_test():
    """Test distributed twist strategy."""

    print("="*70)
    print("EXPERIMENT 20: DISTRIBUTED TWIST STRATEGY")
    print("="*70)
    print()
    print("Testing: n=4, 0.367 deg/oscillator -> predicted 17.8x")
    print()
    print("Twist distribution:")
    print("  A: 0.000 deg")
    print("  B: 0.367 deg")
    print("  C: 0.734 deg")
    print("  D: 1.101 deg (graphene magic angle!)")
    print()

    results = {
        'experiment': 'exp20_distributed_twist',
        'timestamp': datetime.now().isoformat()
    }

    # =========================================================================
    # TEST CONFIGURATIONS
    # =========================================================================
    configs = [
        {'name': 'no_twist', 'twist': 0.0},
        {'name': 'distributed_0.367', 'twist': 0.367},
        {'name': 'distributed_1.1', 'twist': 1.1},  # Total span = 3.3 deg
        {'name': 'distributed_5.0', 'twist': 5.0},   # Total span = 15 deg
        {'name': 'distributed_15.0', 'twist': 15.0}, # Total span = 45 deg
    ]

    config_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']} ({config['twist']} deg/osc)")
        print(f"  Total twist span: {3 * config['twist']:.1f} deg")
        print("-"*50)

        sensor = DistributedTwistKernel(
            r_base=3.70,
            spacing=0.03,
            twist_per_osc_deg=config['twist']
        )

        # Baseline
        print("Collecting baseline...")
        idle = WorkloadGenerator("idle")
        idle.start()
        time.sleep(2)

        sensor.reset()
        baseline = collect_correlations(sensor, duration=30.0)
        idle.stop()

        baseline_means = [c['mean'] for c in baseline]
        baseline_mean = np.mean(baseline_means)
        baseline_std = np.std(baseline_means)

        # Attack
        print("Collecting attack...")
        attack = WorkloadGenerator("memory", intensity=0.8)
        attack.start()
        time.sleep(2)

        sensor.reset()
        attack_data = collect_correlations(sensor, duration=30.0)
        attack.stop()

        attack_means = [c['mean'] for c in attack_data]
        attack_mean = np.mean(attack_means)

        delta = attack_mean - baseline_mean
        z_score = abs(delta) / (baseline_std + 1e-10)

        result = {
            'config': config,
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'attack_mean': float(attack_mean),
            'delta': float(delta),
            'z_score': float(z_score)
        }
        config_results.append(result)

        print(f"  Baseline: {baseline_mean:.4f} +/- {baseline_std:.4f}")
        print(f"  Attack: {attack_mean:.4f}")
        print(f"  z-score: {z_score:.2f}")

    results['configurations'] = config_results

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("DISTRIBUTED TWIST ANALYSIS")
    print("="*70)

    # Baseline for comparison
    no_twist = next((r for r in config_results if r['config']['name'] == 'no_twist'), None)
    baseline_z = no_twist['z_score'] if no_twist else 0.01

    print("\n| Configuration | Twist/Osc | Total Span | z-score | Amplification |")
    print("|---------------|-----------|------------|---------|---------------|")

    for r in config_results:
        twist = r['config']['twist']
        span = 3 * twist
        amp = r['z_score'] / (baseline_z + 1e-10)
        print(f"| {r['config']['name']:13} | {twist:6.3f} deg | {span:6.1f} deg | {r['z_score']:5.2f} | {amp:6.1f}x |")

    # Find optimal
    best = max(config_results, key=lambda x: x['z_score'])
    print(f"\n*** BEST: {best['config']['name']} ***")
    print(f"    z-score: {best['z_score']:.2f}")
    print(f"    Amplification: {best['z_score'] / baseline_z:.1f}x")

    # =========================================================================
    # DOF PHASE DIAGRAM VALIDATION
    # =========================================================================
    print("\n" + "="*70)
    print("DOF PHASE DIAGRAM VALIDATION")
    print("-"*50)

    print("""
    Predicted optimal angles for different DOF:

    DOF=6  (n=4): 45 deg optimal
    DOF=15 (n=6): 15 deg optimal
    DOF=28 (n=8): 1.1 deg optimal

    Distributed twist at n=4:
    - 0.367 deg/osc -> 1.1 deg span (graphene magic!)
    - Expected: 17.8x amplification
    """)

    distributed_367 = next((r for r in config_results
                           if r['config']['name'] == 'distributed_0.367'), None)
    if distributed_367:
        amp_367 = distributed_367['z_score'] / (baseline_z + 1e-10)
        print(f"Measured amplification at 0.367 deg/osc: {amp_367:.1f}x")
        print(f"Predicted: 17.8x")

        if amp_367 > 10:
            print("\n*** DISTRIBUTED TWIST VALIDATED! ***")
        elif amp_367 > 5:
            print("\n*** PARTIAL VALIDATION ***")
        else:
            print("\n*** AMPLIFICATION BELOW PREDICTION ***")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp20_distributed_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_distributed_twist_test()
