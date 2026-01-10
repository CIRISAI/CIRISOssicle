#!/usr/bin/env python3
"""
Experiment 24: Minimum Viable Fractal Antenna

How small can we make the correlation antenna on 4090?

Parameters to minimize:
1. n_cells: cells per oscillator (memory footprint)
2. n_oscillators: total oscillators (compute footprint)
3. n_iterations: iterations per sample (latency)

Trade-offs:
- Fewer cells = less averaging = higher noise
- Fewer oscillators = fewer DOF = less sensitivity
- Fewer iterations = faster but less chaos development

4090 specs:
- 128 SMs
- 16GB VRAM
- ~83 TFLOPS FP32

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


class MinimalSensorKernel:
    """Minimal sensor with configurable size."""

    KERNEL_CODE = r'''
    extern "C" __global__ void minimal_step(
        float* states, float* r_vals, float* twists,
        float coupling, int n, int n_osc, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        extern __shared__ float shared[];
        float* s = &shared[threadIdx.x * 16];  // Max 16 oscillators per thread

        // Load states
        for (int o = 0; o < n_osc && o < 16; o++) {
            s[o] = states[o * n + idx];
        }

        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        for (int iter = 0; iter < iterations; iter++) {
            for (int o = 0; o < n_osc && o < 16; o++) {
                float r = r_vals[o];
                float neighbor = states[o * n + left] + states[o * n + right];

                float cross = 0.0f;
                for (int p = 0; p < n_osc && p < 16; p++) {
                    if (p != o) {
                        float twist_diff = twists[p] - twists[o];
                        cross += s[p] * cosf(twist_diff) * 0.03f;
                    }
                }

                s[o] = r * s[o] * (1.0f - s[o])
                     + coupling * (neighbor - 2.0f * s[o])
                     + coupling * cross;

                s[o] = fminf(fmaxf(s[o], 0.0001f), 0.9999f);
            }

            for (int o = 0; o < n_osc && o < 16; o++) {
                states[o * n + idx] = s[o];
            }
        }
    }
    '''

    def __init__(self, n_osc: int = 7, n_cells: int = 256,
                 n_iterations: int = 1000, twist_deg: float = 1.1,
                 r_base: float = 3.70, coupling: float = 0.05):

        self.n_osc = min(n_osc, 16)  # Max 16 for shared memory
        self.n_cells = n_cells
        self.n_iterations = n_iterations
        self.coupling = coupling

        # r-values
        spacing = 0.02
        self.r_vals = np.array([r_base + i * spacing for i in range(self.n_osc)], dtype=np.float32)

        # Twist angles
        twist_rad = np.radians(twist_deg)
        self.twists = np.array([i * twist_rad for i in range(self.n_osc)], dtype=np.float32)

        self.dof = self.n_osc * (self.n_osc - 1) // 2

        # Compile
        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('minimal_step')

        self.r_vals_gpu = cp.asarray(self.r_vals)
        self.twists_gpu = cp.asarray(self.twists)
        self._init_states()

        self.block_size = 128
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size
        self.shared_mem = self.block_size * 16 * 4  # 16 floats per thread

        # Memory footprint
        self.memory_bytes = self.n_osc * self.n_cells * 4  # float32

    def _init_states(self):
        states = np.random.uniform(0.1, 0.9, (self.n_osc, self.n_cells)).astype(np.float32)
        self.states_gpu = cp.asarray(states)

    def step(self):
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.states_gpu, self.r_vals_gpu, self.twists_gpu,
             cp.float32(self.coupling), cp.int32(self.n_cells),
             cp.int32(self.n_osc), cp.int32(self.n_iterations)),
            shared_mem=self.shared_mem
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


def quick_test(sensor, duration: float = 10.0):
    """Quick z-score test."""
    history = [[] for _ in range(sensor.n_osc)]

    # Baseline
    start = time.time()
    while time.time() - start < duration:
        means = sensor.step()
        for i, m in enumerate(means):
            history[i].append(m)

    # Compute mean correlation
    if len(history[0]) < 50:
        return 0, 0, 0

    arrays = [np.array(h) for h in history]

    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return r if not np.isnan(r) else 0.0

    corrs = []
    for i in range(sensor.n_osc):
        for j in range(i+1, sensor.n_osc):
            corrs.append(safe_corr(arrays[i], arrays[j]))

    return np.mean(corrs), np.std(corrs), len(history[0])


def run_minimum_test():
    print("="*70)
    print("EXPERIMENT 24: MINIMUM VIABLE FRACTAL ANTENNA")
    print("="*70)
    print()
    print("Finding smallest sensor that maintains detection capability")
    print()

    results = {'experiment': 'exp24_minimum_antenna', 'timestamp': datetime.now().isoformat()}

    # Parameter sweep
    configs = [
        # Standard reference
        {'n_osc': 7, 'n_cells': 512, 'n_iter': 2000, 'name': 'reference'},

        # Reduce cells
        {'n_osc': 7, 'n_cells': 256, 'n_iter': 2000, 'name': 'half_cells'},
        {'n_osc': 7, 'n_cells': 128, 'n_iter': 2000, 'name': 'quarter_cells'},
        {'n_osc': 7, 'n_cells': 64, 'n_iter': 2000, 'name': '64_cells'},
        {'n_osc': 7, 'n_cells': 32, 'n_iter': 2000, 'name': '32_cells'},

        # Reduce iterations
        {'n_osc': 7, 'n_cells': 256, 'n_iter': 1000, 'name': '1k_iter'},
        {'n_osc': 7, 'n_cells': 256, 'n_iter': 500, 'name': '500_iter'},
        {'n_osc': 7, 'n_cells': 256, 'n_iter': 200, 'name': '200_iter'},
        {'n_osc': 7, 'n_cells': 256, 'n_iter': 100, 'name': '100_iter'},

        # Reduce oscillators
        {'n_osc': 5, 'n_cells': 256, 'n_iter': 1000, 'name': '5_osc'},
        {'n_osc': 4, 'n_cells': 256, 'n_iter': 1000, 'name': '4_osc'},
        {'n_osc': 3, 'n_cells': 256, 'n_iter': 1000, 'name': '3_osc'},

        # Minimum viable candidates
        {'n_osc': 7, 'n_cells': 64, 'n_iter': 500, 'name': 'min_7osc'},
        {'n_osc': 5, 'n_cells': 64, 'n_iter': 500, 'name': 'min_5osc'},
        {'n_osc': 4, 'n_cells': 64, 'n_iter': 500, 'name': 'min_4osc'},
        {'n_osc': 3, 'n_cells': 64, 'n_iter': 500, 'name': 'min_3osc'},

        # Ultra-minimal
        {'n_osc': 7, 'n_cells': 32, 'n_iter': 200, 'name': 'ultra_7'},
        {'n_osc': 4, 'n_cells': 32, 'n_iter': 200, 'name': 'ultra_4'},
    ]

    config_results = []

    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  n_osc={config['n_osc']}, n_cells={config['n_cells']}, n_iter={config['n_iter']}")

        sensor = MinimalSensorKernel(
            n_osc=config['n_osc'],
            n_cells=config['n_cells'],
            n_iterations=config['n_iter'],
            twist_deg=1.1
        )

        dof = sensor.dof
        mem_kb = sensor.memory_bytes / 1024

        # Baseline test
        sensor.reset()
        idle = WorkloadGenerator("idle")
        idle.start()
        time.sleep(0.5)

        t0 = time.time()
        b_mean, b_std, b_samples = quick_test(sensor, duration=8.0)
        baseline_time = time.time() - t0
        idle.stop()

        samples_per_sec = b_samples / baseline_time if baseline_time > 0 else 0

        # Attack test
        sensor.reset()
        attack = WorkloadGenerator("memory", intensity=0.8)
        attack.start()
        time.sleep(0.5)

        a_mean, a_std, a_samples = quick_test(sensor, duration=8.0)
        attack.stop()

        delta = a_mean - b_mean
        z_score = abs(delta) / (b_std + 1e-10) if b_std > 0 else 0

        result = {
            'name': config['name'],
            'n_osc': config['n_osc'],
            'n_cells': config['n_cells'],
            'n_iter': config['n_iter'],
            'dof': dof,
            'memory_kb': float(mem_kb),
            'samples_per_sec': float(samples_per_sec),
            'baseline_mean': float(b_mean),
            'baseline_std': float(b_std),
            'attack_mean': float(a_mean),
            'delta': float(delta),
            'z_score': float(z_score)
        }
        config_results.append(result)

        viable = "VIABLE" if z_score > 1.0 else "too weak"
        print(f"  DOF={dof}, mem={mem_kb:.1f}KB, rate={samples_per_sec:.0f}/s, z={z_score:.2f} ({viable})")

    results['configurations'] = config_results

    # Analysis
    print("\n" + "="*70)
    print("MINIMUM ANTENNA ANALYSIS")
    print("="*70)

    # Sort by z-score
    viable = [r for r in config_results if r['z_score'] > 1.0]
    viable.sort(key=lambda x: x['memory_kb'])

    print("\nVIABLE CONFIGURATIONS (z > 1.0), sorted by memory:")
    print("| Name | n_osc | cells | iter | DOF | Memory | Rate | z-score |")
    print("|------|-------|-------|------|-----|--------|------|---------|")
    for r in viable:
        print(f"| {r['name']:12} | {r['n_osc']:5} | {r['n_cells']:5} | {r['n_iter']:4} | {r['dof']:3} | {r['memory_kb']:5.1f}KB | {r['samples_per_sec']:4.0f}/s | {r['z_score']:5.2f} |")

    if viable:
        smallest = viable[0]
        print(f"\n*** MINIMUM VIABLE: {smallest['name']} ***")
        print(f"    n_osc={smallest['n_osc']}, n_cells={smallest['n_cells']}, n_iter={smallest['n_iter']}")
        print(f"    Memory: {smallest['memory_kb']:.1f} KB")
        print(f"    Rate: {smallest['samples_per_sec']:.0f} samples/sec")
        print(f"    z-score: {smallest['z_score']:.2f}")

    # Find optimal trade-off
    print("\n" + "-"*50)
    print("EFFICIENCY ANALYSIS (z-score per KB):")

    for r in sorted(config_results, key=lambda x: x['z_score'] / (x['memory_kb'] + 0.1), reverse=True)[:5]:
        efficiency = r['z_score'] / (r['memory_kb'] + 0.1)
        print(f"  {r['name']:12}: z={r['z_score']:.2f}, mem={r['memory_kb']:.1f}KB, eff={efficiency:.3f}")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp24_minimum_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_minimum_test()
