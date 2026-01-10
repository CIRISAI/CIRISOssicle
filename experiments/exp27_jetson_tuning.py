#!/usr/bin/env python3
"""
Experiment 27: Jetson Orin Parameter Tuning

Find optimal ossicle parameters for Jetson Orin's PDN characteristics.

Jetson Orin specs:
- 8 SMs (vs 128 on RTX 4090)
- 7 GB unified memory
- Tegra Ampere architecture
- Lower TDP = different PDN noise profile

Parameters to tune:
- n_cells: spatial averaging
- n_iterations: chaos development time
- n_oscillators: DOF for correlation
- twist_deg: magic angle may differ
- r_base/spacing: chaotic regime

Author: CIRIS L3C
License: BSL 1.1
"""

import numpy as np
import cupy as cp
import time
import json
import threading
from datetime import datetime
from pathlib import Path


class TunableOssicle:
    """Ossicle with tunable parameters for Jetson optimization."""

    KERNEL_CODE = r'''
    extern "C" __global__ void ossicle_step(
        float* states, float* r_vals, float* twists,
        float coupling, int n, int n_osc, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        extern __shared__ float shared[];
        float* s = &shared[threadIdx.x * 16];

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

    def __init__(self, n_osc: int = 3, n_cells: int = 64,
                 n_iterations: int = 500, twist_deg: float = 1.1,
                 r_base: float = 3.70, spacing: float = 0.03,
                 coupling: float = 0.05):

        self.n_osc = min(n_osc, 16)
        self.n_cells = n_cells
        self.n_iterations = n_iterations
        self.coupling = coupling

        self.r_vals = np.array([r_base + i * spacing for i in range(self.n_osc)], dtype=np.float32)

        twist_rad = np.radians(twist_deg)
        self.twists = np.array([i * twist_rad for i in range(self.n_osc)], dtype=np.float32)

        self.dof = self.n_osc * (self.n_osc - 1) // 2

        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('ossicle_step')

        self.r_vals_gpu = cp.asarray(self.r_vals)
        self.twists_gpu = cp.asarray(self.twists)
        self._init_states()

        self.block_size = min(64, n_cells)
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size
        self.shared_mem = self.block_size * 16 * 4

        self.memory_bytes = self.n_osc * self.n_cells * 4

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


class MemoryWorkload:
    """Memory bandwidth workload for testing."""

    def __init__(self, intensity: float = 0.7):
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
        # Smaller size for Jetson's limited memory
        size = int(512 * self.intensity)
        src = cp.random.randn(size * size, dtype=cp.float32)
        dst = cp.zeros(size * size, dtype=cp.float32)
        while self.running:
            dst[:] = src
            src[:] = dst
            cp.cuda.Stream.null.synchronize()


def quick_test(sensor, duration: float = 6.0):
    """Quick correlation test."""
    history = [[] for _ in range(sensor.n_osc)]

    start = time.time()
    while time.time() - start < duration:
        means = sensor.step()
        for i, m in enumerate(means):
            history[i].append(m)

    if len(history[0]) < 30:
        return 0, 0, 0, 0

    arrays = [np.array(h) for h in history]

    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return r if not np.isnan(r) else 0.0

    corrs = []
    for i in range(sensor.n_osc):
        for j in range(i + 1, sensor.n_osc):
            corrs.append(safe_corr(arrays[i], arrays[j]))

    mean_corr = np.mean(corrs)
    std_corr = np.std(corrs)

    elapsed = time.time() - start
    sample_rate = len(history[0]) / elapsed

    return mean_corr, std_corr, len(history[0]), sample_rate


def test_config(config, test_duration=6.0):
    """Test a single configuration."""
    sensor = TunableOssicle(**config)

    # Baseline
    sensor.reset()
    b_mean, b_std, b_samples, b_rate = quick_test(sensor, test_duration)

    # Attack (memory workload)
    workload = MemoryWorkload(intensity=0.7)
    workload.start()
    time.sleep(0.3)

    sensor.reset()
    a_mean, a_std, a_samples, a_rate = quick_test(sensor, test_duration)
    workload.stop()

    delta = a_mean - b_mean
    z_score = abs(delta) / (b_std + 1e-10)

    return {
        'config': config,
        'baseline_mean': float(b_mean),
        'baseline_std': float(b_std),
        'baseline_samples': b_samples,
        'baseline_rate': float(b_rate),
        'attack_mean': float(a_mean),
        'attack_samples': a_samples,
        'delta': float(delta),
        'z_score': float(z_score),
        'memory_kb': sensor.memory_bytes / 1024,
        'dof': sensor.dof
    }


def run_jetson_tuning():
    print("=" * 70)
    print("EXPERIMENT 27: JETSON ORIN PARAMETER TUNING")
    print("=" * 70)
    print()
    print("Finding optimal ossicle parameters for Jetson Orin")
    print()

    results = {
        'experiment': 'exp27_jetson_tuning',
        'timestamp': datetime.now().isoformat(),
        'platform': 'Jetson Orin'
    }

    # Parameter configurations to test
    configs = [
        # Original ossicle (reference)
        {'n_osc': 3, 'n_cells': 64, 'n_iterations': 500, 'twist_deg': 1.1, 'name': 'original'},

        # More iterations (chaos development)
        {'n_osc': 3, 'n_cells': 64, 'n_iterations': 1000, 'twist_deg': 1.1, 'name': 'iter_1000'},
        {'n_osc': 3, 'n_cells': 64, 'n_iterations': 2000, 'twist_deg': 1.1, 'name': 'iter_2000'},
        {'n_osc': 3, 'n_cells': 64, 'n_iterations': 5000, 'twist_deg': 1.1, 'name': 'iter_5000'},

        # More cells (spatial averaging)
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'name': 'cells_128'},
        {'n_osc': 3, 'n_cells': 256, 'n_iterations': 1000, 'twist_deg': 1.1, 'name': 'cells_256'},
        {'n_osc': 3, 'n_cells': 512, 'n_iterations': 1000, 'twist_deg': 1.1, 'name': 'cells_512'},

        # More oscillators (more DOF)
        {'n_osc': 4, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'name': '4osc_128'},
        {'n_osc': 5, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'name': '5osc_128'},
        {'n_osc': 7, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'name': '7osc_128'},

        # Different twist angles
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 0.55, 'name': 'twist_0.55'},
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 2.2, 'name': 'twist_2.2'},
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 5.0, 'name': 'twist_5.0'},
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 45.0, 'name': 'twist_45'},
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 90.0, 'name': 'twist_90'},

        # Different r-values (chaotic regime)
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'r_base': 3.80, 'name': 'r_3.80'},
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'r_base': 3.85, 'name': 'r_3.85'},
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'r_base': 3.90, 'name': 'r_3.90'},

        # Larger spacing
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'spacing': 0.05, 'name': 'space_0.05'},
        {'n_osc': 3, 'n_cells': 128, 'n_iterations': 1000, 'twist_deg': 1.1, 'spacing': 0.08, 'name': 'space_0.08'},

        # Best guess combinations
        {'n_osc': 4, 'n_cells': 256, 'n_iterations': 2000, 'twist_deg': 90.0, 'name': 'quad_large'},
        {'n_osc': 3, 'n_cells': 256, 'n_iterations': 2000, 'twist_deg': 1.1, 'r_base': 3.85, 'name': 'big_chaos'},
        {'n_osc': 5, 'n_cells': 256, 'n_iterations': 1000, 'twist_deg': 45.0, 'name': '5osc_45deg'},
    ]

    test_results = []

    for i, cfg in enumerate(configs):
        name = cfg.pop('name', f'config_{i}')
        print(f"\n[{i+1}/{len(configs)}] Testing: {name}")
        print(f"  params: {cfg}")

        try:
            result = test_config(cfg, test_duration=6.0)
            result['name'] = name

            z = result['z_score']
            detected = "STRONG!" if z > 3 else ("YES" if z > 2 else ("weak" if z > 1 else "no"))
            print(f"  z-score: {z:.2f} ({detected})")
            print(f"  rate: {result['baseline_rate']:.0f}/s, mem: {result['memory_kb']:.1f}KB")

            test_results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    results['tests'] = test_results

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Sort by z-score
    sorted_results = sorted(test_results, key=lambda x: x['z_score'], reverse=True)

    print("\n| Rank | Config | z-score | Rate | Memory | Detection |")
    print("|------|--------|---------|------|--------|-----------|")

    for i, r in enumerate(sorted_results[:15]):
        detected = "STRONG" if r['z_score'] > 3 else ("YES" if r['z_score'] > 2 else ("weak" if r['z_score'] > 1 else "no"))
        print(f"| {i+1:4} | {r['name']:14} | {r['z_score']:7.2f} | {r['baseline_rate']:4.0f} | {r['memory_kb']:5.1f}KB | {detected:9} |")

    # Best config
    if sorted_results:
        best = sorted_results[0]
        print(f"\n*** BEST CONFIGURATION: {best['name']} ***")
        print(f"    z-score: {best['z_score']:.2f}")
        print(f"    Sample rate: {best['baseline_rate']:.0f}/s")
        print(f"    Memory: {best['memory_kb']:.1f} KB")
        print(f"    Config: {best['config']}")

        results['best'] = best

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp27_jetson_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_jetson_tuning()
