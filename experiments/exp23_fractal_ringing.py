#!/usr/bin/env python3
"""
Experiment 23: Fractal Ringing - Hierarchical Frequency Modulation

Tests the self-similar 1.1° magic angle structure with frequency modulation:
- Local frequency: 100 Hz (within-unit oscillation)
- Meta frequency: 100/7 ≈ 14.3 Hz (between-unit, scaled by n_units)
- Beat frequency: ~85.7 Hz (moiré-of-moiré resonance)

The fractal structure:
  7 units × 7 oscillators = 49 total
  DOF = 49*48/2 = 1176

  Each unit: 1.1° twist between oscillators
  Between units: 1.1° twist between units

  META-MAGIC = LOCAL-MAGIC = 1.1° (scale invariant!)

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


class FractalAntennaKernel:
    """
    7×7 Fractal antenna with hierarchical 1.1° twist.

    Structure:
    - 7 units, each containing 7 oscillators
    - Within-unit twist: 1.1° between adjacent oscillators
    - Between-unit twist: 1.1° between adjacent units
    - Total DOF: 1176
    """

    KERNEL_CODE = r'''
    extern "C" __global__ void fractal_step(
        float* states,      // 49 oscillators
        float* r_vals,      // 49 r-values
        float* local_phase, // 49 local phases (within unit)
        float* meta_phase,  // 7 meta phases (between unit)
        float coupling,
        float local_freq,   // Hz
        float meta_freq,    // Hz
        float time_val,     // Current time in seconds
        int n,              // cells per oscillator
        int n_osc,          // 49
        int n_units,        // 7
        int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        // Load all 49 oscillator states
        float s[49];
        for (int o = 0; o < 49; o++) {
            s[o] = states[o * n + idx];
        }

        // Grid neighbors
        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        // Time-dependent phase modulation
        float local_mod = sinf(2.0f * 3.14159f * local_freq * time_val);
        float meta_mod = sinf(2.0f * 3.14159f * meta_freq * time_val);

        for (int iter = 0; iter < iterations; iter++) {
            float new_s[49];

            for (int o = 0; o < 49; o++) {
                int unit = o / 7;      // Which unit (0-6)
                int local = o % 7;     // Position within unit (0-6)

                float r = r_vals[o];
                float lp = local_phase[o];
                float mp = meta_phase[unit];

                // Neighbor coupling
                float neighbor = states[o * n + left] + states[o * n + right];

                // Intra-unit coupling (same unit, different oscillator)
                float intra = 0.0f;
                for (int j = 0; j < 7; j++) {
                    int other = unit * 7 + j;
                    if (other != o) {
                        float phase_diff = local_phase[other] - lp;
                        // Modulate by local frequency
                        phase_diff += local_mod * 0.1f;
                        intra += s[other] * cosf(phase_diff) * 0.02f;
                    }
                }

                // Inter-unit coupling (different units)
                float inter = 0.0f;
                for (int u = 0; u < 7; u++) {
                    if (u != unit) {
                        // Couple to same position in other unit
                        int other = u * 7 + local;
                        float meta_diff = meta_phase[u] - mp;
                        // Modulate by meta frequency
                        meta_diff += meta_mod * 0.1f;
                        inter += s[other] * cosf(meta_diff) * 0.01f;
                    }
                }

                // Logistic map with hierarchical coupling
                new_s[o] = r * s[o] * (1.0f - s[o])
                         + coupling * (neighbor - 2.0f * s[o])
                         + coupling * intra
                         + coupling * inter;

                new_s[o] = fminf(fmaxf(new_s[o], 0.0001f), 0.9999f);
            }

            for (int o = 0; o < 49; o++) {
                s[o] = new_s[o];
                states[o * n + idx] = s[o];
            }
        }
    }
    '''

    def __init__(self, r_base: float = 3.70, r_spacing: float = 0.005,
                 local_twist_deg: float = 1.1, meta_twist_deg: float = 1.1,
                 local_freq: float = 100.0, meta_freq: float = 14.286,
                 n_cells: int = 512, coupling: float = 0.05,
                 n_iterations: int = 2000):

        self.n_units = 7
        self.n_per_unit = 7
        self.n_osc = 49
        self.n_cells = n_cells
        self.coupling = coupling
        self.n_iterations = n_iterations
        self.local_freq = local_freq
        self.meta_freq = meta_freq
        self.time = 0.0

        # r-values: slight variation across all 49
        self.r_vals = np.array([r_base + i * r_spacing for i in range(49)], dtype=np.float32)

        # Local phases: 1.1° between oscillators within each unit
        local_twist_rad = np.radians(local_twist_deg)
        self.local_phase = np.zeros(49, dtype=np.float32)
        for unit in range(7):
            for local in range(7):
                self.local_phase[unit * 7 + local] = local * local_twist_rad

        # Meta phases: 1.1° between units
        meta_twist_rad = np.radians(meta_twist_deg)
        self.meta_phase = np.array([u * meta_twist_rad for u in range(7)], dtype=np.float32)

        # DOF calculation
        self.dof = self.n_osc * (self.n_osc - 1) // 2
        print(f"Fractal antenna: {self.n_units}×{self.n_per_unit} = {self.n_osc} oscillators, DOF = {self.dof}")
        print(f"Frequencies: local={local_freq}Hz, meta={meta_freq}Hz, beat={local_freq - meta_freq:.1f}Hz")

        # Compile kernel
        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('fractal_step')

        # GPU arrays
        self.states_gpu = None
        self.r_vals_gpu = cp.asarray(self.r_vals)
        self.local_phase_gpu = cp.asarray(self.local_phase)
        self.meta_phase_gpu = cp.asarray(self.meta_phase)

        self._init_states()

        self.block_size = 256
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size

    def _init_states(self):
        states = np.random.uniform(0.1, 0.9, (self.n_osc, self.n_cells)).astype(np.float32)
        self.states_gpu = cp.asarray(states)
        self.time = 0.0

    def step(self, dt: float = 0.001):
        self.time += dt

        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.states_gpu, self.r_vals_gpu,
             self.local_phase_gpu, self.meta_phase_gpu,
             cp.float32(self.coupling),
             cp.float32(self.local_freq), cp.float32(self.meta_freq),
             cp.float32(self.time),
             cp.int32(self.n_cells), cp.int32(self.n_osc),
             cp.int32(self.n_units), cp.int32(self.n_iterations))
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
        size = int(1024 * 4 * self.intensity)
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


def collect_hierarchical_correlations(sensor, duration: float, window: int = 50):
    """Collect correlations at multiple hierarchical levels."""
    history = [[] for _ in range(sensor.n_osc)]

    # Track correlations at different levels
    local_corrs = []   # Within-unit correlations
    meta_corrs = []    # Between-unit correlations
    global_corrs = []  # All pairs

    start = time.time()
    step_count = 0

    while time.time() - start < duration:
        means = sensor.step(dt=0.001)
        step_count += 1

        for i, m in enumerate(means):
            history[i].append(m)

        if len(history[0]) >= window and step_count % (window // 2) == 0:
            arrays = [np.array(h[-window:]) for h in history]

            def safe_corr(x, y):
                if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                    return 0.0
                r = np.corrcoef(x, y)[0, 1]
                return r if not np.isnan(r) else 0.0

            # Local correlations (within same unit)
            local_vals = []
            for unit in range(7):
                for i in range(7):
                    for j in range(i+1, 7):
                        oi = unit * 7 + i
                        oj = unit * 7 + j
                        local_vals.append(safe_corr(arrays[oi], arrays[oj]))

            # Meta correlations (same position, different units)
            meta_vals = []
            for pos in range(7):
                for u1 in range(7):
                    for u2 in range(u1+1, 7):
                        o1 = u1 * 7 + pos
                        o2 = u2 * 7 + pos
                        meta_vals.append(safe_corr(arrays[o1], arrays[o2]))

            # Global mean
            all_vals = []
            for i in range(49):
                for j in range(i+1, 49):
                    all_vals.append(safe_corr(arrays[i], arrays[j]))

            local_corrs.append(np.mean(local_vals))
            meta_corrs.append(np.mean(meta_vals))
            global_corrs.append(np.mean(all_vals))

    return {
        'local': local_corrs,
        'meta': meta_corrs,
        'global': global_corrs
    }


def run_fractal_ringing():
    print("="*70)
    print("EXPERIMENT 23: FRACTAL RINGING")
    print("="*70)
    print()
    print("Testing hierarchical frequency modulation on 7×7 fractal antenna")
    print("Meta-magic = Local-magic = 1.1° (scale invariant!)")
    print()

    results = {'experiment': 'exp23_fractal_ringing', 'timestamp': datetime.now().isoformat()}

    # Test different frequency configurations
    configs = [
        {'name': 'no_ringing', 'local_freq': 0, 'meta_freq': 0},
        {'name': 'local_only', 'local_freq': 100, 'meta_freq': 0},
        {'name': 'meta_only', 'local_freq': 0, 'meta_freq': 14.286},
        {'name': 'hierarchical', 'local_freq': 100, 'meta_freq': 14.286},
        {'name': 'resonant', 'local_freq': 100, 'meta_freq': 100/7},  # Exact 1/7 ratio
        {'name': 'beat_85Hz', 'local_freq': 100, 'meta_freq': 15},    # ~85 Hz beat
    ]

    config_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"  Local: {config['local_freq']} Hz, Meta: {config['meta_freq']:.3f} Hz")
        if config['local_freq'] > 0 and config['meta_freq'] > 0:
            beat = abs(config['local_freq'] - config['meta_freq'])
            print(f"  Beat frequency: {beat:.1f} Hz")
        print("-"*50)

        sensor = FractalAntennaKernel(
            local_twist_deg=1.1,
            meta_twist_deg=1.1,
            local_freq=config['local_freq'],
            meta_freq=config['meta_freq']
        )

        # Baseline
        print("Collecting baseline...")
        idle = WorkloadGenerator("idle")
        idle.start()
        time.sleep(1)
        sensor.reset()
        baseline = collect_hierarchical_correlations(sensor, duration=20.0)
        idle.stop()

        if not baseline['global']:
            print("  No data!")
            continue

        b_local = np.mean(baseline['local'])
        b_meta = np.mean(baseline['meta'])
        b_global = np.mean(baseline['global'])
        b_std = np.std(baseline['global'])

        print(f"  Baseline: local={b_local:.4f}, meta={b_meta:.4f}, global={b_global:.4f}")

        # Attack
        print("Collecting attack...")
        attack = WorkloadGenerator("memory", intensity=0.8)
        attack.start()
        time.sleep(1)
        sensor.reset()
        attack_data = collect_hierarchical_correlations(sensor, duration=20.0)
        attack.stop()

        if not attack_data['global']:
            continue

        a_local = np.mean(attack_data['local'])
        a_meta = np.mean(attack_data['meta'])
        a_global = np.mean(attack_data['global'])

        delta = a_global - b_global
        z_score = abs(delta) / (b_std + 1e-10)

        result = {
            'config': config,
            'baseline': {'local': float(b_local), 'meta': float(b_meta), 'global': float(b_global), 'std': float(b_std)},
            'attack': {'local': float(a_local), 'meta': float(a_meta), 'global': float(a_global)},
            'delta': float(delta),
            'z_score': float(z_score),
            'hierarchical_ratio': float(b_meta / (b_local + 1e-10))
        }
        config_results.append(result)

        detect = " *** 3σ! ***" if z_score > 3 else ""
        print(f"  z-score: {z_score:.2f}{detect}")
        print(f"  Δlocal: {a_local - b_local:+.4f}, Δmeta: {a_meta - b_meta:+.4f}")

    results['configurations'] = config_results

    # Analysis
    print("\n" + "="*70)
    print("FRACTAL RINGING ANALYSIS")
    print("="*70)

    if config_results:
        print("\n| Configuration | Local Hz | Meta Hz | Beat Hz | z-score |")
        print("|---------------|----------|---------|---------|---------|")
        for r in sorted(config_results, key=lambda x: x['z_score'], reverse=True):
            c = r['config']
            beat = abs(c['local_freq'] - c['meta_freq']) if c['local_freq'] and c['meta_freq'] else 0
            print(f"| {c['name']:13} | {c['local_freq']:8.1f} | {c['meta_freq']:7.3f} | {beat:7.1f} | {r['z_score']:7.2f} |")

        best = max(config_results, key=lambda x: x['z_score'])
        print(f"\n*** BEST: {best['config']['name']} (z={best['z_score']:.2f}) ***")

        # Check hierarchical structure
        print("\nHierarchical correlation structure:")
        for r in config_results:
            ratio = r['hierarchical_ratio']
            print(f"  {r['config']['name']:15}: meta/local = {ratio:.3f}")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp23_fractal_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_fractal_ringing()
