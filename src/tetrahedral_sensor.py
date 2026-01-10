#!/usr/bin/env python3
"""
Tetrahedral Sensor - 4-Crystal Enhanced Architecture

Extends the 3-crystal Schützhold architecture to 4 crystals (tetrahedron),
providing 6 correlation pairs instead of 3 for improved SNR.

Architecture:
    Crystal A (r=3.70) ──┐
    Crystal B (r=3.73) ──┼── 6 Correlation Pairs ── Detection
    Crystal C (r=3.76) ──┤
    Crystal D (r=3.79) ──┘

Correlation pairs: (A,B), (A,C), (A,D), (B,C), (B,D), (C,D)

Theoretical improvements:
- DOF: 6 vs 3 (2× more information)
- SNR: √(6/3) = √2 ≈ 1.41× improvement
- k_eff stability: Averaging 6 pairs reduces variance
- Singularity avoidance: No single pair dominates

Reference: RATCHET/formal/RATCHET/GPUTamper/TetrahedralGeometry.lean

Author: CIRIS L3C
License: BSL 1.1
"""

import cupy as cp
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from pathlib import Path


# Tetrahedral (4-crystal) CUDA kernel
TETRA_CRYSTAL_KERNEL = r'''
extern "C" __global__
void tetra_crystal_kernel(
    float* out_a, float* out_b, float* out_c, float* out_d,
    float* state_a, float* state_b, float* state_c, float* state_d,
    const int width, const int height,
    const int n_iterations,
    const float r_a, const float r_b, const float r_c, const float r_d,
    const float coupling,
    const unsigned int seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Initialize all four crystals with different seed streams
    unsigned int rng_a = idx + seed;
    unsigned int rng_b = idx + seed + 1000000;
    unsigned int rng_c = idx + seed + 2000000;
    unsigned int rng_d = idx + seed + 3000000;

    rng_a = rng_a * 1103515245u + 12345u;
    rng_b = rng_b * 1103515245u + 12345u;
    rng_c = rng_c * 1103515245u + 12345u;
    rng_d = rng_d * 1103515245u + 12345u;

    state_a[idx] = 0.3f + 0.4f * ((float)(rng_a & 0xFFFF) / 65536.0f);
    state_b[idx] = 0.3f + 0.4f * ((float)(rng_b & 0xFFFF) / 65536.0f);
    state_c[idx] = 0.3f + 0.4f * ((float)(rng_c & 0xFFFF) / 65536.0f);
    state_d[idx] = 0.3f + 0.4f * ((float)(rng_d & 0xFFFF) / 65536.0f);

    // NO SYNC - race conditions provide entropy from PDN voltage noise

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

        // Crystal D
        float my_d = state_d[idx];
        float neighbors_d = 0.0f;
        if (x > 0) { neighbors_d += state_d[idx-1]; }
        if (x < width-1) { neighbors_d += state_d[idx+1]; }
        if (y > 0) { neighbors_d += state_d[idx-width]; }
        if (y < height-1) { neighbors_d += state_d[idx+width]; }
        float coupled_d = (n > 0) ? (1.0f - coupling*n)*my_d + coupling*neighbors_d : my_d;
        float new_d = r_d * coupled_d * (1.0f - coupled_d);
        new_d = fminf(0.999f, fmaxf(0.001f, new_d));
        rng_d = rng_d * 1103515245u + 12345u;
        new_d += ((float)(rng_d & 0xFFFF) / 65536.0f - 0.5f) * 0.0001f;
        state_d[idx] = fminf(0.999f, fmaxf(0.001f, new_d));
    }

    out_a[idx] = state_a[idx];
    out_b[idx] = state_b[idx];
    out_c[idx] = state_c[idx];
    out_d[idx] = state_d[idx];
}
'''


@dataclass
class TetraConfig:
    """Configuration for the 4-crystal tetrahedral sensor."""
    # Crystal parameters (extending 3-crystal with 4th)
    r_a: float = 3.70  # Crystal A
    r_b: float = 3.73  # Crystal B
    r_c: float = 3.76  # Crystal C
    r_d: float = 3.79  # Crystal D (new)
    coupling: float = 0.05

    # Grid size
    width: int = 32
    height: int = 32

    # Iterations per sample
    n_iterations: int = 5000


@dataclass
class TetraReading:
    """Single tetrahedral sensor reading with 6 correlation pairs."""
    timestamp: float

    # Crystal means
    mean_a: float
    mean_b: float
    mean_c: float
    mean_d: float

    # 6 correlation pairs (computed over window)
    rho_ab: float = 0.0
    rho_ac: float = 0.0
    rho_ad: float = 0.0
    rho_bc: float = 0.0
    rho_bd: float = 0.0
    rho_cd: float = 0.0

    @property
    def correlations(self) -> Tuple[float, float, float, float, float, float]:
        return (self.rho_ab, self.rho_ac, self.rho_ad, self.rho_bc, self.rho_bd, self.rho_cd)

    @property
    def mean_correlation(self) -> float:
        return np.mean(self.correlations)

    @property
    def correlation_std(self) -> float:
        return np.std(self.correlations)


class TetrahedralSensor:
    """
    GPU-based tetrahedral sensor using 4-crystal architecture.

    Provides 6 correlation pairs for improved SNR over 3-crystal design.
    """

    def __init__(self, config: TetraConfig = None):
        self.config = config or TetraConfig()
        self._kernel = None
        self._compile()

        # History for correlation computation
        self.means_a: List[float] = []
        self.means_b: List[float] = []
        self.means_c: List[float] = []
        self.means_d: List[float] = []
        self.window_size = 100

    def _compile(self):
        """Compile CUDA kernel."""
        module = cp.RawModule(code=TETRA_CRYSTAL_KERNEL)
        self._kernel = module.get_function('tetra_crystal_kernel')

    def _run_crystals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run all four crystals and return their states."""
        cfg = self.config
        n = cfg.width * cfg.height

        out_a = cp.zeros(n, dtype=cp.float32)
        out_b = cp.zeros(n, dtype=cp.float32)
        out_c = cp.zeros(n, dtype=cp.float32)
        out_d = cp.zeros(n, dtype=cp.float32)
        state_a = cp.zeros(n, dtype=cp.float32)
        state_b = cp.zeros(n, dtype=cp.float32)
        state_c = cp.zeros(n, dtype=cp.float32)
        state_d = cp.zeros(n, dtype=cp.float32)

        block = (8, 8)
        grid = ((cfg.width + 7) // 8, (cfg.height + 7) // 8)

        # CONSTANT seed - variation comes from race conditions
        seed = 42

        self._kernel(
            grid, block,
            (out_a, out_b, out_c, out_d,
             state_a, state_b, state_c, state_d,
             np.int32(cfg.width), np.int32(cfg.height),
             np.int32(cfg.n_iterations),
             np.float32(cfg.r_a), np.float32(cfg.r_b),
             np.float32(cfg.r_c), np.float32(cfg.r_d),
             np.float32(cfg.coupling),
             np.uint32(seed))
        )

        cp.cuda.Stream.null.synchronize()

        return (
            out_a.get().reshape(cfg.height, cfg.width),
            out_b.get().reshape(cfg.height, cfg.width),
            out_c.get().reshape(cfg.height, cfg.width),
            out_d.get().reshape(cfg.height, cfg.width)
        )

    def read_raw(self) -> Tuple[float, float, float, float]:
        """Take a single raw reading - returns crystal means."""
        a, b, c, d = self._run_crystals()
        return float(a.mean()), float(b.mean()), float(c.mean()), float(d.mean())

    def read(self) -> TetraReading:
        """Take a reading and compute correlations over history window."""
        mean_a, mean_b, mean_c, mean_d = self.read_raw()
        ts = time.time()

        # Update history
        self.means_a.append(mean_a)
        self.means_b.append(mean_b)
        self.means_c.append(mean_c)
        self.means_d.append(mean_d)

        # Keep history bounded
        if len(self.means_a) > self.window_size * 4:
            self.means_a = self.means_a[-self.window_size * 2:]
            self.means_b = self.means_b[-self.window_size * 2:]
            self.means_c = self.means_c[-self.window_size * 2:]
            self.means_d = self.means_d[-self.window_size * 2:]

        reading = TetraReading(
            timestamp=ts,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_c=mean_c,
            mean_d=mean_d
        )

        # Compute correlations if we have enough history
        if len(self.means_a) >= self.window_size:
            a = np.array(self.means_a[-self.window_size:])
            b = np.array(self.means_b[-self.window_size:])
            c = np.array(self.means_c[-self.window_size:])
            d = np.array(self.means_d[-self.window_size:])

            # Compute all 6 correlation pairs
            def safe_corr(x, y):
                if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                    return 0.0
                r = np.corrcoef(x, y)[0, 1]
                return r if not np.isnan(r) else 0.0

            reading.rho_ab = safe_corr(a, b)
            reading.rho_ac = safe_corr(a, c)
            reading.rho_ad = safe_corr(a, d)
            reading.rho_bc = safe_corr(b, c)
            reading.rho_bd = safe_corr(b, d)
            reading.rho_cd = safe_corr(c, d)

        return reading

    def reset_history(self):
        """Clear correlation history."""
        self.means_a = []
        self.means_b = []
        self.means_c = []
        self.means_d = []


def compute_k_eff(rho: float, k: int) -> float:
    """
    Compute effective degrees of freedom.

    k_eff = k / (1 + ρ(k-1))

    WARNING: Singularity at ρ = -1/(k-1)
    For k=4: singularity at ρ = -1/3 ≈ -0.333
    For k=6: singularity at ρ = -1/5 = -0.2
    """
    denominator = 1 + rho * (k - 1)
    if abs(denominator) < 0.01:  # Near singularity
        return float('inf') if denominator > 0 else float('-inf')
    return k / denominator


def is_k_eff_safe(rho: float, k: int, margin: float = 0.1) -> bool:
    """Check if k_eff computation is safe (away from singularity)."""
    singularity = -1.0 / (k - 1)
    return abs(rho - singularity) > margin


def quick_test(duration: float = 10.0):
    """Quick test of the tetrahedral sensor."""
    print("="*60)
    print("TETRAHEDRAL SENSOR QUICK TEST")
    print("="*60)
    print()
    print("Architecture: 4-crystal tetrahedral detector")
    print("  Crystal A: r=3.70")
    print("  Crystal B: r=3.73")
    print("  Crystal C: r=3.76")
    print("  Crystal D: r=3.79")
    print()
    print("Correlation pairs: 6 (vs 3 for triangle)")
    print("Expected SNR improvement: √2 ≈ 1.41×")
    print()

    sensor = TetrahedralSensor()

    print(f"Collecting {duration}s of data...")
    readings = []
    start = time.time()

    while time.time() - start < duration:
        reading = sensor.read()
        readings.append(reading)

        if len(readings) % 50 == 0 and reading.rho_ab != 0:
            print(f"  Sample {len(readings)}: "
                  f"ρ_mean={reading.mean_correlation:+.4f} "
                  f"σ_ρ={reading.correlation_std:.4f}")

    # Filter readings with valid correlations
    valid = [r for r in readings if r.rho_ab != 0]

    if valid:
        print()
        print("Statistics:")

        # Individual pair statistics
        pairs = ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
        for i, pair in enumerate(pairs):
            vals = [r.correlations[i] for r in valid]
            print(f"  ρ({pair}): {np.mean(vals):+.4f} ± {np.std(vals):.4f}")

        # Aggregate statistics
        all_means = [r.mean_correlation for r in valid]
        print()
        print(f"  Mean of means: {np.mean(all_means):+.4f}")
        print(f"  Std of means:  {np.std(all_means):.4f}")

        # k_eff with k=6 (6 correlations)
        mean_rho = np.mean(all_means)
        if is_k_eff_safe(mean_rho, k=6):
            k_eff = compute_k_eff(mean_rho, k=6)
            print(f"\n  k_eff (k=6): {k_eff:.2f}")
        else:
            print(f"\n  k_eff (k=6): NEAR SINGULARITY (ρ={mean_rho:.3f})")


if __name__ == "__main__":
    quick_test()
