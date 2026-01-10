#!/usr/bin/env python3
"""
TRNG Triangulation Test - 3+ Crystals for Orientation Sensitivity

Author: Eric Moore
Date: 2026-01-08

To detect directional sensitivity, we need ≥3 crystals at different
"orientations" running simultaneously, measuring their cross-correlations.

Design:
- Crystal A: r=3.70 (0° reference)
- Crystal B: r=3.73 (60° equivalent)
- Crystal C: r=3.76 (120° equivalent)

If there's directional coupling:
- Correlations should differ based on relative "orientation"
- ρ(A,B) ≠ ρ(B,C) ≠ ρ(A,C) in a systematic way
"""

import cupy as cp
import numpy as np
import time
from scipy import stats

# Kernel for 3 simultaneous crystals
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

    // Initialize all three crystals with different seed streams
    unsigned int rng_a = idx + seed;
    unsigned int rng_b = idx + seed + 1000000;
    unsigned int rng_c = idx + seed + 2000000;

    rng_a = rng_a * 1103515245u + 12345u;
    rng_b = rng_b * 1103515245u + 12345u;
    rng_c = rng_c * 1103515245u + 12345u;

    state_a[idx] = 0.3f + 0.4f * ((float)(rng_a & 0xFFFF) / 65536.0f);
    state_b[idx] = 0.3f + 0.4f * ((float)(rng_b & 0xFFFF) / 65536.0f);
    state_c[idx] = 0.3f + 0.4f * ((float)(rng_c & 0xFFFF) / 65536.0f);

    // NO SYNC - race conditions for TRNG

    for (int iter = 0; iter < n_iterations; iter++) {
        // Crystal A
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

        // Crystal B
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

        // Crystal C
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

_module = None
_kernel = None

def _compile():
    global _module, _kernel
    if _module is None:
        _module = cp.RawModule(code=TRIPLE_CRYSTAL_KERNEL)
        _kernel = _module.get_function('triple_crystal_kernel')

def run_triple_crystal(
    width=32, height=32,
    n_iterations=5000,
    r_a=3.70, r_b=3.73, r_c=3.76,
    coupling=0.05,
    seed=42
):
    """Run 3 crystals simultaneously."""
    _compile()

    n = width * height
    out_a = cp.zeros(n, dtype=cp.float32)
    out_b = cp.zeros(n, dtype=cp.float32)
    out_c = cp.zeros(n, dtype=cp.float32)
    state_a = cp.zeros(n, dtype=cp.float32)
    state_b = cp.zeros(n, dtype=cp.float32)
    state_c = cp.zeros(n, dtype=cp.float32)

    block = (8, 8)
    grid = ((width + 7) // 8, (height + 7) // 8)

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

    return (out_a.get().reshape(height, width),
            out_b.get().reshape(height, width),
            out_c.get().reshape(height, width))


def test_triangulation(n_samples=500, duration_sec=None):
    """Run triangulation test."""
    print("="*60)
    print("TRNG TRIANGULATION TEST")
    print("="*60)
    print("3 crystals at different 'orientations' (r parameters)")
    print("  Crystal A: r=3.70 (0°)")
    print("  Crystal B: r=3.73 (60°)")
    print("  Crystal C: r=3.76 (120°)")
    print()

    means_a, means_b, means_c = [], [], []

    start = time.time()
    i = 0
    while True:
        if duration_sec and (time.time() - start) >= duration_sec:
            break
        if not duration_sec and i >= n_samples:
            break

        a, b, c = run_triple_crystal(seed=42)
        means_a.append(a.mean())
        means_b.append(b.mean())
        means_c.append(c.mean())
        i += 1

        if i % 100 == 0:
            print(f"  {i} samples, {time.time()-start:.1f}s")

    means_a = np.array(means_a)
    means_b = np.array(means_b)
    means_c = np.array(means_c)

    print(f"\nCollected {len(means_a)} samples")

    # Statistics
    print("\n" + "-"*60)
    print("INDIVIDUAL CRYSTAL STATISTICS")
    print("-"*60)
    print(f"  Crystal A (0°):   {np.mean(means_a):.6f} ± {np.std(means_a):.6f}")
    print(f"  Crystal B (60°):  {np.mean(means_b):.6f} ± {np.std(means_b):.6f}")
    print(f"  Crystal C (120°): {np.mean(means_c):.6f} ± {np.std(means_c):.6f}")

    # Cross-correlations
    print("\n" + "-"*60)
    print("CROSS-CORRELATIONS (key for triangulation)")
    print("-"*60)

    rho_ab = np.corrcoef(means_a, means_b)[0, 1]
    rho_bc = np.corrcoef(means_b, means_c)[0, 1]
    rho_ac = np.corrcoef(means_a, means_c)[0, 1]

    print(f"  ρ(A,B) [0°-60°]:   {rho_ab:+.4f}")
    print(f"  ρ(B,C) [60°-120°]: {rho_bc:+.4f}")
    print(f"  ρ(A,C) [0°-120°]:  {rho_ac:+.4f}")

    # Are correlations significantly different from each other?
    print("\n" + "-"*60)
    print("ASYMMETRY TEST")
    print("-"*60)

    # Fisher z-transform for comparing correlations
    def fisher_z(r):
        return 0.5 * np.log((1 + r) / (1 - r))

    n = len(means_a)
    se = 1 / np.sqrt(n - 3)

    z_ab = fisher_z(rho_ab)
    z_bc = fisher_z(rho_bc)
    z_ac = fisher_z(rho_ac)

    # Compare AB vs BC
    z_diff_1 = (z_ab - z_bc) / (np.sqrt(2) * se)
    p_diff_1 = 2 * (1 - stats.norm.cdf(abs(z_diff_1)))

    # Compare AB vs AC
    z_diff_2 = (z_ab - z_ac) / (np.sqrt(2) * se)
    p_diff_2 = 2 * (1 - stats.norm.cdf(abs(z_diff_2)))

    # Compare BC vs AC
    z_diff_3 = (z_bc - z_ac) / (np.sqrt(2) * se)
    p_diff_3 = 2 * (1 - stats.norm.cdf(abs(z_diff_3)))

    print(f"  ρ(A,B) vs ρ(B,C): z={z_diff_1:+.2f}, p={p_diff_1:.4f}")
    print(f"  ρ(A,B) vs ρ(A,C): z={z_diff_2:+.2f}, p={p_diff_2:.4f}")
    print(f"  ρ(B,C) vs ρ(A,C): z={z_diff_3:+.2f}, p={p_diff_3:.4f}")

    # Summary
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    any_significant = p_diff_1 < 0.05 or p_diff_2 < 0.05 or p_diff_3 < 0.05

    if any_significant:
        print("""
  ⚠ SIGNIFICANT ASYMMETRY DETECTED!

  The cross-correlations differ between crystal pairs.
  This could indicate:
  1. Different 'orientations' have different coupling
  2. The r parameter affects race condition dynamics
  3. Potential for directional sensitivity
        """)
    else:
        print("""
  ✓ NO SIGNIFICANT ASYMMETRY

  All three crystal pairs show similar correlations.
  The 'orientation' (r parameter) does not create
  systematic differences in cross-correlation.

  The TRNG is isotropic in parameter space.
        """)

    return {
        'rho_ab': rho_ab,
        'rho_bc': rho_bc,
        'rho_ac': rho_ac,
        'p_ab_vs_bc': p_diff_1,
        'p_ab_vs_ac': p_diff_2,
        'p_bc_vs_ac': p_diff_3
    }


if __name__ == "__main__":
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else None
    n_samples = 500 if duration is None else None
    test_triangulation(n_samples=n_samples, duration_sec=duration)
