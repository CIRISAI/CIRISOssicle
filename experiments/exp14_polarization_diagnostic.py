#!/usr/bin/env python3
"""
Experiment 14: Correlation Polarization Diagnostic

Investigates why ρ(A,B) ≈ -0.45 while ρ(B,C), ρ(A,C) ≈ 0

Hypotheses:
1. TEMPORAL: Sequential execution (A→B→C) creates temporal correlation asymmetry
2. PARAMETER: r-value spacing (3.70, 3.73, 3.76) creates different chaotic regimes
3. SPATIAL: PDN topology - A and B share power domain, C is isolated

Tests:
1. Reverse execution order (C→B→A) - if temporal, ρ(C,B) should become high
2. Equalize r-values (all 3.73) - if parameter-based, correlations should equalize
3. Interleave execution (A,C,B) - tests ordering effect

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
from datetime import datetime
from pathlib import Path


# Original order kernel (A→B→C)
KERNEL_ABC = r'''
extern "C" __global__
void crystal_abc(
    float* out_a, float* out_b, float* out_c,
    float* state_a, float* state_b, float* state_c,
    const int width, const int height,
    const int n_iterations,
    const float r_a, const float r_b, const float r_c,
    const float coupling, const unsigned int seed
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
        // A FIRST
        float my_a = state_a[idx];
        float neighbors_a = 0.0f; int n = 0;
        if (x > 0) { neighbors_a += state_a[idx-1]; n++; }
        if (x < width-1) { neighbors_a += state_a[idx+1]; n++; }
        if (y > 0) { neighbors_a += state_a[idx-width]; n++; }
        if (y < height-1) { neighbors_a += state_a[idx+width]; n++; }
        float coupled_a = (n > 0) ? (1.0f - coupling*n)*my_a + coupling*neighbors_a : my_a;
        float new_a = r_a * coupled_a * (1.0f - coupled_a);
        rng_a = rng_a * 1103515245u + 12345u;
        new_a += ((float)(rng_a & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_a[idx] = fminf(0.999f, fmaxf(0.001f, new_a));

        // B SECOND
        float my_b = state_b[idx];
        float neighbors_b = 0.0f;
        if (x > 0) { neighbors_b += state_b[idx-1]; }
        if (x < width-1) { neighbors_b += state_b[idx+1]; }
        if (y > 0) { neighbors_b += state_b[idx-width]; }
        if (y < height-1) { neighbors_b += state_b[idx+width]; }
        float coupled_b = (n > 0) ? (1.0f - coupling*n)*my_b + coupling*neighbors_b : my_b;
        float new_b = r_b * coupled_b * (1.0f - coupled_b);
        rng_b = rng_b * 1103515245u + 12345u;
        new_b += ((float)(rng_b & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_b[idx] = fminf(0.999f, fmaxf(0.001f, new_b));

        // C THIRD
        float my_c = state_c[idx];
        float neighbors_c = 0.0f;
        if (x > 0) { neighbors_c += state_c[idx-1]; }
        if (x < width-1) { neighbors_c += state_c[idx+1]; }
        if (y > 0) { neighbors_c += state_c[idx-width]; }
        if (y < height-1) { neighbors_c += state_c[idx+width]; }
        float coupled_c = (n > 0) ? (1.0f - coupling*n)*my_c + coupling*neighbors_c : my_c;
        float new_c = r_c * coupled_c * (1.0f - coupled_c);
        rng_c = rng_c * 1103515245u + 12345u;
        new_c += ((float)(rng_c & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_c[idx] = fminf(0.999f, fmaxf(0.001f, new_c));
    }

    out_a[idx] = state_a[idx];
    out_b[idx] = state_b[idx];
    out_c[idx] = state_c[idx];
}
'''

# Reversed order kernel (C→B→A)
KERNEL_CBA = r'''
extern "C" __global__
void crystal_cba(
    float* out_a, float* out_b, float* out_c,
    float* state_a, float* state_b, float* state_c,
    const int width, const int height,
    const int n_iterations,
    const float r_a, const float r_b, const float r_c,
    const float coupling, const unsigned int seed
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
        // C FIRST
        float my_c = state_c[idx];
        float neighbors_c = 0.0f; int n = 0;
        if (x > 0) { neighbors_c += state_c[idx-1]; n++; }
        if (x < width-1) { neighbors_c += state_c[idx+1]; n++; }
        if (y > 0) { neighbors_c += state_c[idx-width]; n++; }
        if (y < height-1) { neighbors_c += state_c[idx+width]; n++; }
        float coupled_c = (n > 0) ? (1.0f - coupling*n)*my_c + coupling*neighbors_c : my_c;
        float new_c = r_c * coupled_c * (1.0f - coupled_c);
        rng_c = rng_c * 1103515245u + 12345u;
        new_c += ((float)(rng_c & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_c[idx] = fminf(0.999f, fmaxf(0.001f, new_c));

        // B SECOND
        float my_b = state_b[idx];
        float neighbors_b = 0.0f;
        if (x > 0) { neighbors_b += state_b[idx-1]; }
        if (x < width-1) { neighbors_b += state_b[idx+1]; }
        if (y > 0) { neighbors_b += state_b[idx-width]; }
        if (y < height-1) { neighbors_b += state_b[idx+width]; }
        float coupled_b = (n > 0) ? (1.0f - coupling*n)*my_b + coupling*neighbors_b : my_b;
        float new_b = r_b * coupled_b * (1.0f - coupled_b);
        rng_b = rng_b * 1103515245u + 12345u;
        new_b += ((float)(rng_b & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_b[idx] = fminf(0.999f, fmaxf(0.001f, new_b));

        // A THIRD
        float my_a = state_a[idx];
        float neighbors_a = 0.0f;
        if (x > 0) { neighbors_a += state_a[idx-1]; }
        if (x < width-1) { neighbors_a += state_a[idx+1]; }
        if (y > 0) { neighbors_a += state_a[idx-width]; }
        if (y < height-1) { neighbors_a += state_a[idx+width]; }
        float coupled_a = (n > 0) ? (1.0f - coupling*n)*my_a + coupling*neighbors_a : my_a;
        float new_a = r_a * coupled_a * (1.0f - coupled_a);
        rng_a = rng_a * 1103515245u + 12345u;
        new_a += ((float)(rng_a & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_a[idx] = fminf(0.999f, fmaxf(0.001f, new_a));
    }

    out_a[idx] = state_a[idx];
    out_b[idx] = state_b[idx];
    out_c[idx] = state_c[idx];
}
'''


class OrderedSensor:
    """Sensor with configurable execution order."""

    def __init__(self, order: str = "ABC", width: int = 32, height: int = 32,
                 n_iterations: int = 5000, r_values: tuple = (3.70, 3.73, 3.76)):
        self.order = order
        self.width = width
        self.height = height
        self.n_iterations = n_iterations
        self.r_a, self.r_b, self.r_c = r_values
        self.coupling = 0.05

        if order == "ABC":
            module = cp.RawModule(code=KERNEL_ABC)
            self._kernel = module.get_function('crystal_abc')
        else:  # CBA
            module = cp.RawModule(code=KERNEL_CBA)
            self._kernel = module.get_function('crystal_cba')

    def read_raw(self):
        n = self.width * self.height

        out_a = cp.zeros(n, dtype=cp.float32)
        out_b = cp.zeros(n, dtype=cp.float32)
        out_c = cp.zeros(n, dtype=cp.float32)
        state_a = cp.zeros(n, dtype=cp.float32)
        state_b = cp.zeros(n, dtype=cp.float32)
        state_c = cp.zeros(n, dtype=cp.float32)

        block = (8, 8)
        grid = ((self.width + 7) // 8, (self.height + 7) // 8)

        self._kernel(
            grid, block,
            (out_a, out_b, out_c, state_a, state_b, state_c,
             np.int32(self.width), np.int32(self.height),
             np.int32(self.n_iterations),
             np.float32(self.r_a), np.float32(self.r_b), np.float32(self.r_c),
             np.float32(self.coupling), np.uint32(42))
        )

        cp.cuda.Stream.null.synchronize()

        return (
            float(out_a.get().mean()),
            float(out_b.get().mean()),
            float(out_c.get().mean())
        )


def collect_correlations(sensor, duration: float, window: int = 100):
    """Collect correlation samples."""
    means_a, means_b, means_c = [], [], []
    correlations = []

    start = time.time()
    while time.time() - start < duration:
        a, b, c = sensor.read_raw()
        means_a.append(a)
        means_b.append(b)
        means_c.append(c)

        if len(means_a) >= window and len(means_a) % (window // 4) == 0:
            a_arr = np.array(means_a[-window:])
            b_arr = np.array(means_b[-window:])
            c_arr = np.array(means_c[-window:])

            def safe_corr(x, y):
                if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                    return 0.0
                r = np.corrcoef(x, y)[0, 1]
                return r if not np.isnan(r) else 0.0

            correlations.append({
                'rho_ab': safe_corr(a_arr, b_arr),
                'rho_bc': safe_corr(b_arr, c_arr),
                'rho_ac': safe_corr(a_arr, c_arr)
            })

    return correlations


def run_diagnostic():
    """Run polarization diagnostic."""

    print("="*70)
    print("EXPERIMENT 14: CORRELATION POLARIZATION DIAGNOSTIC")
    print("="*70)
    print()
    print("Question: Why is ρ(A,B) ≈ -0.45 while ρ(B,C), ρ(A,C) ≈ 0?")
    print()
    print("Hypotheses:")
    print("  1. TEMPORAL: Sequential execution order (A→B→C)")
    print("  2. PARAMETER: r-value differences (3.70, 3.73, 3.76)")
    print()

    results = {
        'experiment': 'exp14_polarization_diagnostic',
        'timestamp': datetime.now().isoformat()
    }

    # =========================================================================
    # TEST 1: Original order (A→B→C)
    # =========================================================================
    print("[TEST 1] ORIGINAL ORDER (A→B→C)")
    print("-"*50)

    sensor_abc = OrderedSensor(order="ABC", r_values=(3.70, 3.73, 3.76))
    corr_abc = collect_correlations(sensor_abc, duration=20.0)

    rho_ab_abc = np.mean([c['rho_ab'] for c in corr_abc])
    rho_bc_abc = np.mean([c['rho_bc'] for c in corr_abc])
    rho_ac_abc = np.mean([c['rho_ac'] for c in corr_abc])

    print(f"  ρ(A,B): {rho_ab_abc:.4f}  (1st-2nd)")
    print(f"  ρ(B,C): {rho_bc_abc:.4f}  (2nd-3rd)")
    print(f"  ρ(A,C): {rho_ac_abc:.4f}  (1st-3rd)")

    results['abc_order'] = {
        'rho_ab': float(rho_ab_abc),
        'rho_bc': float(rho_bc_abc),
        'rho_ac': float(rho_ac_abc)
    }

    # =========================================================================
    # TEST 2: Reversed order (C→B→A)
    # =========================================================================
    print("\n[TEST 2] REVERSED ORDER (C→B→A)")
    print("-"*50)
    print("If temporal, ρ(C,B) should now be strongest...")

    sensor_cba = OrderedSensor(order="CBA", r_values=(3.70, 3.73, 3.76))
    corr_cba = collect_correlations(sensor_cba, duration=20.0)

    rho_ab_cba = np.mean([c['rho_ab'] for c in corr_cba])
    rho_bc_cba = np.mean([c['rho_bc'] for c in corr_cba])
    rho_ac_cba = np.mean([c['rho_ac'] for c in corr_cba])

    print(f"  ρ(A,B): {rho_ab_cba:.4f}  (3rd-2nd)")
    print(f"  ρ(B,C): {rho_bc_cba:.4f}  (2nd-1st) ← Should be strongest if temporal")
    print(f"  ρ(A,C): {rho_ac_cba:.4f}  (3rd-1st)")

    results['cba_order'] = {
        'rho_ab': float(rho_ab_cba),
        'rho_bc': float(rho_bc_cba),
        'rho_ac': float(rho_ac_cba)
    }

    # =========================================================================
    # TEST 3: Equal r-values (all 3.73)
    # =========================================================================
    print("\n[TEST 3] EQUAL R-VALUES (all 3.73)")
    print("-"*50)
    print("If parameter-based, correlations should equalize...")

    sensor_equal = OrderedSensor(order="ABC", r_values=(3.73, 3.73, 3.73))
    corr_equal = collect_correlations(sensor_equal, duration=20.0)

    rho_ab_eq = np.mean([c['rho_ab'] for c in corr_equal])
    rho_bc_eq = np.mean([c['rho_bc'] for c in corr_equal])
    rho_ac_eq = np.mean([c['rho_ac'] for c in corr_equal])

    print(f"  ρ(A,B): {rho_ab_eq:.4f}")
    print(f"  ρ(B,C): {rho_bc_eq:.4f}")
    print(f"  ρ(A,C): {rho_ac_eq:.4f}")

    results['equal_r'] = {
        'rho_ab': float(rho_ab_eq),
        'rho_bc': float(rho_bc_eq),
        'rho_ac': float(rho_ac_eq)
    }

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Check temporal hypothesis
    print("\nTEMPORAL HYPOTHESIS:")
    # In ABC order, 1st-2nd should be strongest
    # In CBA order, 1st-2nd (which is C-B) should be strongest
    abc_strongest = max(abs(rho_ab_abc), abs(rho_bc_abc), abs(rho_ac_abc))
    cba_adjacent_corr = abs(rho_bc_cba)  # C and B are adjacent in CBA order

    if abs(rho_ab_abc) == abc_strongest and cba_adjacent_corr > abs(rho_ac_cba):
        print("  SUPPORTED: Correlation follows execution adjacency")
        print(f"    ABC: ρ(A,B)={rho_ab_abc:.4f} is strongest (A-B adjacent)")
        print(f"    CBA: ρ(B,C)={rho_bc_cba:.4f} is {'strongest' if cba_adjacent_corr >= abs(rho_ab_cba) else 'not strongest'} (C-B adjacent)")
    else:
        print("  NOT SUPPORTED: Correlation doesn't follow execution order")
        print(f"    ABC strongest: {'AB' if abs(rho_ab_abc) == abc_strongest else 'BC' if abs(rho_bc_abc) == abc_strongest else 'AC'}")

    # Check parameter hypothesis
    print("\nPARAMETER HYPOTHESIS:")
    equal_spread = max(abs(rho_ab_eq), abs(rho_bc_eq), abs(rho_ac_eq)) - \
                   min(abs(rho_ab_eq), abs(rho_bc_eq), abs(rho_ac_eq))
    original_spread = max(abs(rho_ab_abc), abs(rho_bc_abc), abs(rho_ac_abc)) - \
                      min(abs(rho_ab_abc), abs(rho_bc_abc), abs(rho_ac_abc))

    if equal_spread < original_spread * 0.5:
        print("  SUPPORTED: Equal r-values reduce correlation spread")
        print(f"    Original spread: {original_spread:.4f}")
        print(f"    Equal-r spread:  {equal_spread:.4f}")
    else:
        print("  NOT SUPPORTED: Equal r-values don't equalize correlations")
        print(f"    Original spread: {original_spread:.4f}")
        print(f"    Equal-r spread:  {equal_spread:.4f}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp14_polarization_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_diagnostic()
