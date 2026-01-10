#!/usr/bin/env python3
"""
Experiment 2: Parameter Sweep - Find Optimal Configuration

Goal: Sweep through parameter space to find configuration with maximum sensitivity.

Parameters to sweep:
- r-value separation (Δr between crystals)
- Base r-value (center of the three crystals)
- Coupling strength
- Grid size
- Number of iterations

For each configuration, measure:
- Baseline variance (no perturbation)
- Correlation stability (CV = std/mean)
- Sample rate (samples/sec)

The optimal configuration minimizes baseline variance while maintaining
correlation stability, allowing smaller perturbations to be detected.

Author: CIRIS L3C
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cupy as cp
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import itertools


# Kernel (same as strain_sensor.py)
TRIPLE_CRYSTAL_KERNEL = r'''
extern "C" __global__
void triple_crystal_kernel(
    float* out_a, float* out_b, float* out_c,
    float* state_a, float* state_b, float* state_c,
    const int width, const int height,
    const int n_iterations,
    const float r_a, const float r_b, const float r_c,
    const float coupling,
    const unsigned int seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    unsigned int rng_a = idx + seed;
    unsigned int rng_b = idx + seed + 1000000;
    unsigned int rng_c = idx + seed + 2000000;

    rng_a = rng_a * 1103515245u + 12345u;
    rng_b = rng_b * 1103515245u + 12345u;
    rng_c = rng_c * 1103515245u + 12345u;

    state_a[idx] = 0.3f + 0.4f * ((float)(rng_a & 0xFFFF) / 65536.0f);
    state_b[idx] = 0.3f + 0.4f * ((float)(rng_b & 0xFFFF) / 65536.0f);
    state_c[idx] = 0.3f + 0.4f * ((float)(rng_c & 0xFFFF) / 65536.0f);

    for (int iter = 0; iter < n_iterations; iter++) {
        float my_a = state_a[idx];
        float neighbors_a = 0.0f;
        int n = 0;
        if (x > 0) { neighbors_a += state_a[idx-1]; n++; }
        if (x < width-1) { neighbors_a += state_a[idx+1]; n++; }
        if (y > 0) { neighbors_a += state_a[idx-width]; n++; }
        if (y < height-1) { neighbors_a += state_a[idx+width]; n++; }
        float coupled_a = (n > 0) ? (1.0f - coupling*n)*my_a + coupling*neighbors_a : my_a;
        float new_a = r_a * coupled_a * (1.0f - coupled_a);
        new_a = fminf(0.999f, fmaxf(0.001f, new_a));
        rng_a = rng_a * 1103515245u + 12345u;
        new_a += ((float)(rng_a & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_a[idx] = fminf(0.999f, fmaxf(0.001f, new_a));

        float my_b = state_b[idx];
        float neighbors_b = 0.0f;
        if (x > 0) { neighbors_b += state_b[idx-1]; }
        if (x < width-1) { neighbors_b += state_b[idx+1]; }
        if (y > 0) { neighbors_b += state_b[idx-width]; }
        if (y < height-1) { neighbors_b += state_b[idx+width]; }
        float coupled_b = (n > 0) ? (1.0f - coupling*n)*my_b + coupling*neighbors_b : my_b;
        float new_b = r_b * coupled_b * (1.0f - coupled_b);
        new_b = fminf(0.999f, fmaxf(0.001f, new_b));
        rng_b = rng_b * 1103515245u + 12345u;
        new_b += ((float)(rng_b & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_b[idx] = fminf(0.999f, fmaxf(0.001f, new_b));

        float my_c = state_c[idx];
        float neighbors_c = 0.0f;
        if (x > 0) { neighbors_c += state_c[idx-1]; }
        if (x < width-1) { neighbors_c += state_c[idx+1]; }
        if (y > 0) { neighbors_c += state_c[idx-width]; }
        if (y < height-1) { neighbors_c += state_c[idx+width]; }
        float coupled_c = (n > 0) ? (1.0f - coupling*n)*my_c + coupling*neighbors_c : my_c;
        float new_c = r_c * coupled_c * (1.0f - coupled_c);
        new_c = fminf(0.999f, fmaxf(0.001f, new_c));
        rng_c = rng_c * 1103515245u + 12345u;
        new_c += ((float)(rng_c & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_c[idx] = fminf(0.999f, fmaxf(0.001f, new_c));
    }

    out_a[idx] = state_a[idx];
    out_b[idx] = state_b[idx];
    out_c[idx] = state_c[idx];
}
'''

_module = cp.RawModule(code=TRIPLE_CRYSTAL_KERNEL)
_kernel = _module.get_function('triple_crystal_kernel')


@dataclass
class ConfigResult:
    """Results for one configuration."""
    r_base: float
    delta_r: float
    coupling: float
    grid_size: int
    n_iterations: int

    n_samples: int
    samples_per_sec: float

    # Correlation means
    mean_ab: float
    mean_bc: float
    mean_ac: float

    # Correlation variances
    var_ab: float
    var_bc: float
    var_ac: float
    var_total: float

    # Stability metrics
    cv_ab: float  # Coefficient of variation
    cv_bc: float
    cv_ac: float


def run_config(r_base: float, delta_r: float, coupling: float,
               grid_size: int, n_iterations: int, n_samples: int = 50) -> ConfigResult:
    """Run one configuration and collect statistics.

    CORRECTED: Collects time series of crystal MEANS, then computes
    TEMPORAL correlations (not spatial correlations within snapshots).
    """
    width = height = grid_size
    n = width * height

    r_a = r_base - delta_r
    r_b = r_base
    r_c = r_base + delta_r

    # Collect time series of crystal means
    means_a, means_b, means_c = [], [], []

    start = time.time()

    for _ in range(n_samples):
        out_a = cp.zeros(n, dtype=cp.float32)
        out_b = cp.zeros(n, dtype=cp.float32)
        out_c = cp.zeros(n, dtype=cp.float32)
        state_a = cp.zeros(n, dtype=cp.float32)
        state_b = cp.zeros(n, dtype=cp.float32)
        state_c = cp.zeros(n, dtype=cp.float32)

        block = (8, 8)
        grid = ((width + 7) // 8, (height + 7) // 8)
        # CONSTANT seed - variation comes from race conditions, not initial state
        seed = 42

        _kernel(
            grid, block,
            (out_a, out_b, out_c, state_a, state_b, state_c,
             np.int32(width), np.int32(height),
             np.int32(n_iterations),
             np.float32(r_a), np.float32(r_b), np.float32(r_c),
             np.float32(coupling),
             np.uint32(seed))
        )
        cp.cuda.Stream.null.synchronize()

        # Collect MEANS (not spatial correlations)
        means_a.append(float(out_a.mean()))
        means_b.append(float(out_b.mean()))
        means_c.append(float(out_c.mean()))

    elapsed = time.time() - start

    # Compute TEMPORAL correlations over the time series
    means_a = np.array(means_a)
    means_b = np.array(means_b)
    means_c = np.array(means_c)

    rho_ab = np.corrcoef(means_a, means_b)[0, 1]
    rho_bc = np.corrcoef(means_b, means_c)[0, 1]
    rho_ac = np.corrcoef(means_a, means_c)[0, 1]

    # Variance of means (affects sensitivity)
    var_a = np.var(means_a)
    var_b = np.var(means_b)
    var_c = np.var(means_c)
    var_total = var_a + var_b + var_c

    return ConfigResult(
        r_base=r_base,
        delta_r=delta_r,
        coupling=coupling,
        grid_size=grid_size,
        n_iterations=n_iterations,
        n_samples=n_samples,
        samples_per_sec=n_samples / elapsed,
        mean_ab=float(rho_ab),  # Temporal correlation
        mean_bc=float(rho_bc),
        mean_ac=float(rho_ac),
        var_ab=float(var_a),  # Variance of crystal A means
        var_bc=float(var_b),  # Variance of crystal B means
        var_ac=float(var_c),  # Variance of crystal C means
        var_total=float(var_total),
        cv_ab=float(np.std(means_a) / (abs(np.mean(means_a)) + 1e-10)),
        cv_bc=float(np.std(means_b) / (abs(np.mean(means_b)) + 1e-10)),
        cv_ac=float(np.std(means_c) / (abs(np.mean(means_c)) + 1e-10)),
    )


def run_sweep(quick: bool = False):
    """Run parameter sweep."""
    print("="*70)
    print("EXPERIMENT 2: PARAMETER SWEEP")
    print("="*70)
    print()

    # Define parameter ranges
    if quick:
        r_bases = [3.70, 3.75, 3.80]
        delta_rs = [0.02, 0.03, 0.05]
        couplings = [0.02, 0.05, 0.10]
        grid_sizes = [32]
        n_iterations_list = [5000]
        n_samples = 30
    else:
        r_bases = [3.65, 3.70, 3.73, 3.75, 3.78, 3.80, 3.85]
        delta_rs = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
        couplings = [0.01, 0.02, 0.05, 0.10, 0.20]
        grid_sizes = [16, 32, 64]
        n_iterations_list = [2000, 5000, 10000]
        n_samples = 50

    # Generate all configurations
    configs = list(itertools.product(r_bases, delta_rs, couplings, grid_sizes, n_iterations_list))
    total = len(configs)

    print(f"Sweeping {total} configurations...")
    print(f"  r_base: {r_bases}")
    print(f"  delta_r: {delta_rs}")
    print(f"  coupling: {couplings}")
    print(f"  grid_size: {grid_sizes}")
    print(f"  iterations: {n_iterations_list}")
    print(f"  samples per config: {n_samples}")
    print()

    results = []
    start_time = time.time()

    for i, (r_base, delta_r, coupling, grid_size, n_iter) in enumerate(configs):
        # Progress
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0

        print(f"[{i+1}/{total}] r={r_base:.2f} Δr={delta_r:.2f} c={coupling:.2f} "
              f"g={grid_size} i={n_iter} ... ", end="", flush=True)

        try:
            result = run_config(r_base, delta_r, coupling, grid_size, n_iter, n_samples)
            results.append(result)
            print(f"var={result.var_total:.6f} rate={result.samples_per_sec:.1f}/s")
        except Exception as e:
            print(f"ERROR: {e}")

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    # Sort by variance (lower is better for sensitivity)
    results.sort(key=lambda r: r.var_total)

    print("TOP 10 CONFIGURATIONS (lowest baseline variance):")
    print()
    print(f"{'Rank':<5} {'r_base':<7} {'Δr':<6} {'coup':<6} {'grid':<5} {'iter':<6} "
          f"{'var_total':<12} {'rate/s':<8}")
    print("-"*70)

    for i, r in enumerate(results[:10]):
        print(f"{i+1:<5} {r.r_base:<7.2f} {r.delta_r:<6.2f} {r.coupling:<6.2f} "
              f"{r.grid_size:<5} {r.n_iterations:<6} {r.var_total:<12.6f} "
              f"{r.samples_per_sec:<8.1f}")

    print()
    print("WORST 5 CONFIGURATIONS (highest baseline variance):")
    print()
    for i, r in enumerate(results[-5:]):
        print(f"  r={r.r_base:.2f} Δr={r.delta_r:.2f} c={r.coupling:.2f} "
              f"g={r.grid_size} → var={r.var_total:.6f}")

    # Best configuration
    best = results[0]
    print()
    print("="*70)
    print("OPTIMAL CONFIGURATION")
    print("="*70)
    print()
    print(f"  r_base:       {best.r_base}")
    print(f"  delta_r:      {best.delta_r}")
    print(f"  coupling:     {best.coupling}")
    print(f"  grid_size:    {best.grid_size}")
    print(f"  n_iterations: {best.n_iterations}")
    print()
    print(f"  Baseline variance: {best.var_total:.6f}")
    print(f"  Sample rate:       {best.samples_per_sec:.1f}/s")
    print()
    print("  Correlation means:")
    print(f"    ρ(A,B): {best.mean_ab:+.4f}")
    print(f"    ρ(B,C): {best.mean_bc:+.4f}")
    print(f"    ρ(A,C): {best.mean_ac:+.4f}")

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp2_parameter_sweep_{timestamp}.json"

    output_data = {
        'experiment': 'exp2_parameter_sweep',
        'timestamp': datetime.now().isoformat(),
        'n_configs': len(results),
        'optimal': asdict(best),
        'all_results': [asdict(r) for r in results]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 2: Parameter Sweep')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick sweep (fewer configs)')
    args = parser.parse_args()

    run_sweep(quick=args.quick)
