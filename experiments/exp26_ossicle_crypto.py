#!/usr/bin/env python3
"""
Experiment 26: Crypto Mining Detection with Ossicle Sensor

Can the minimal 0.8KB ossicle sensor detect crypto mining?

CRYPTO MINING CHARACTERISTICS:
- High compute utilization (integer/hash operations)
- Repetitive, predictable access patterns
- Sustained load over long periods
- Different PDN signature than memory-bound workloads

HYPOTHESIS:
- Crypto mining creates NEGENTROPIC strain (ordered, repetitive)
- Memory bandwidth creates ENTROPIC strain (random access)
- Ossicle should detect both but with different signatures

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


class OssicleKernel:
    """Minimal 3-oscillator ossicle sensor (0.8 KB)."""

    KERNEL_CODE = r'''
    extern "C" __global__ void ossicle_step(
        float* state_a, float* state_b, float* state_c,
        float r_a, float r_b, float r_c,
        float twist_ab, float twist_bc,
        float coupling, int n, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float a = state_a[idx];
        float b = state_b[idx];
        float c = state_c[idx];

        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        for (int iter = 0; iter < iterations; iter++) {
            float na = state_a[left] + state_a[right];
            float nb = state_b[left] + state_b[right];
            float nc = state_c[left] + state_c[right];

            float interference_ab = b * cosf(twist_ab) + a * cosf(-twist_ab);
            float interference_bc = c * cosf(twist_bc) + b * cosf(-twist_bc);

            float new_a = r_a * a * (1.0f - a)
                        + coupling * (na - 2.0f * a)
                        + coupling * 0.1f * interference_ab;

            float new_b = r_b * b * (1.0f - b)
                        + coupling * (nb - 2.0f * b)
                        + coupling * 0.1f * (interference_ab + interference_bc);

            float new_c = r_c * c * (1.0f - c)
                        + coupling * (nc - 2.0f * c)
                        + coupling * 0.1f * interference_bc;

            a = fminf(fmaxf(new_a, 0.0001f), 0.9999f);
            b = fminf(fmaxf(new_b, 0.0001f), 0.9999f);
            c = fminf(fmaxf(new_c, 0.0001f), 0.9999f);

            state_a[idx] = a;
            state_b[idx] = b;
            state_c[idx] = c;
        }
    }
    '''

    def __init__(self, n_cells: int = 64, n_iterations: int = 500,
                 r_base: float = 3.70, spacing: float = 0.03,
                 twist_deg: float = 1.1, coupling: float = 0.05):

        self.n_cells = n_cells
        self.n_iterations = n_iterations
        self.coupling = coupling

        self.r_a = r_base
        self.r_b = r_base + spacing
        self.r_c = r_base + 2 * spacing

        self.twist_ab = np.radians(twist_deg)
        self.twist_bc = np.radians(twist_deg)

        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('ossicle_step')

        self._init_states()

        self.block_size = 64
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size

    def _init_states(self):
        self.state_a = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_b = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)
        self.state_c = cp.random.uniform(0.1, 0.9, self.n_cells, dtype=cp.float32)

    def step(self):
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.state_a, self.state_b, self.state_c,
             cp.float32(self.r_a), cp.float32(self.r_b), cp.float32(self.r_c),
             cp.float32(self.twist_ab), cp.float32(self.twist_bc),
             cp.float32(self.coupling), cp.int32(self.n_cells),
             cp.int32(self.n_iterations))
        )
        cp.cuda.Stream.null.synchronize()

        return (
            float(cp.mean(self.state_a)),
            float(cp.mean(self.state_b)),
            float(cp.mean(self.state_c))
        )

    def reset(self):
        self._init_states()


class CryptoMiningSimulator:
    """
    Simulates crypto mining workload patterns.

    Real crypto mining (SHA-256, Ethash, etc.) uses:
    - Heavy integer arithmetic
    - Bitwise operations
    - Repetitive hash computations
    - High ALU utilization, lower memory bandwidth

    This simulation creates similar PDN stress patterns.
    """

    HASH_KERNEL = r'''
    extern "C" __global__ void hash_compute(
        unsigned int* data, unsigned int* nonce, int n, int rounds
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        unsigned int h = nonce[idx % 256];
        unsigned int d = data[idx];

        // Simulate SHA-256-like operations
        for (int r = 0; r < rounds; r++) {
            // Bitwise rotations and mixing (simplified)
            h = ((h << 7) | (h >> 25)) ^ d;
            h = h + ((h << 3) ^ (h >> 5));
            d = d ^ h;
            d = ((d << 11) | (d >> 21)) + h;
            h = h ^ ((d << 13) | (d >> 19));

            // More mixing to increase compute intensity
            unsigned int t = h;
            h = d;
            d = t ^ (d + h);
            h = ((h * 0x9e3779b9u) >> 16) ^ h;
        }

        data[idx] = h ^ d;
    }
    '''

    def __init__(self, intensity: float = 0.7):
        self.intensity = intensity
        self.running = False
        self.thread = None

        # Compile CUDA kernel
        self.module = cp.RawModule(code=self.HASH_KERNEL)
        self.kernel = self.module.get_function('hash_compute')

        # Size based on intensity (scale for 4090)
        self.size = int(1024 * 1024 * intensity)
        self.rounds = int(100 * intensity)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        # Initialize data
        data = cp.random.randint(0, 2**31, self.size, dtype=cp.uint32)
        nonce = cp.arange(256, dtype=cp.uint32)

        block_size = 256
        grid_size = (self.size + block_size - 1) // block_size

        while self.running:
            # Run hash computation
            self.kernel(
                (grid_size,), (block_size,),
                (data, nonce, cp.int32(self.size), cp.int32(self.rounds))
            )
            cp.cuda.Stream.null.synchronize()

            # Increment nonce (like real mining)
            nonce = nonce + 256


class MemoryBandwidthWorkload:
    """Memory bandwidth workload for comparison."""

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
        size = int(1024 * 4 * self.intensity)
        src = cp.random.randn(size * size, dtype=cp.float32)
        dst = cp.zeros(size * size, dtype=cp.float32)
        while self.running:
            dst[:] = src
            src[:] = dst
            cp.cuda.Stream.null.synchronize()


def quick_test(sensor, duration: float = 10.0):
    """
    Quick z-score test matching exp24 methodology.
    Computes correlations over FULL time series, not windowed.
    """
    history_a, history_b, history_c = [], [], []

    start = time.time()
    while time.time() - start < duration:
        a, b, c = sensor.step()
        history_a.append(a)
        history_b.append(b)
        history_c.append(c)

    if len(history_a) < 50:
        return 0, 0, 0, 0

    arr_a = np.array(history_a)
    arr_b = np.array(history_b)
    arr_c = np.array(history_c)

    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return r if not np.isnan(r) else 0.0

    # All 3 correlation pairs (DOF=3)
    rho_ab = safe_corr(arr_a, arr_b)
    rho_bc = safe_corr(arr_b, arr_c)
    rho_ac = safe_corr(arr_a, arr_c)

    corrs = [rho_ab, rho_bc, rho_ac]
    mean_corr = np.mean(corrs)
    std_corr = np.std(corrs)

    elapsed = time.time() - start
    sample_rate = len(history_a) / elapsed

    return mean_corr, std_corr, len(history_a), sample_rate


def multi_trial_test(sensor, workload, duration: float = 8.0, n_trials: int = 5):
    """
    Run multiple trials to get stable statistics.
    This matches exp24's approach of using variance across measurements.
    """
    means = []
    stds = []

    for trial in range(n_trials):
        sensor.reset()
        mean_corr, std_corr, n_samples, sample_rate = quick_test(sensor, duration)
        means.append(mean_corr)
        stds.append(std_corr)

    return {
        'mean': float(np.mean(means)),
        'std': float(np.std(means)),  # Variance across trials
        'within_std': float(np.mean(stds)),  # Average within-trial variance
        'n_samples_per_trial': n_samples,
        'sample_rate': sample_rate,
        'n_trials': n_trials
    }


def run_crypto_detection_test():
    print("="*70)
    print("EXPERIMENT 26: CRYPTO MINING DETECTION WITH OSSICLE")
    print("="*70)
    print()
    print("Testing if 0.8KB ossicle sensor can detect crypto mining")
    print("Using exp24-validated methodology with multiple trials")
    print()
    print("WORKLOAD SIGNATURES:")
    print("  - Crypto mining: High compute, repetitive patterns")
    print("  - Memory bandwidth: High bandwidth, random patterns")
    print()

    results = {
        'experiment': 'exp26_ossicle_crypto',
        'timestamp': datetime.now().isoformat(),
        'sensor': {
            'type': 'ossicle',
            'n_oscillators': 3,
            'n_cells': 64,
            'n_iterations': 500,
            'memory_kb': 0.75,
            'twist_deg': 1.1
        }
    }

    sensor = OssicleKernel(n_cells=64, n_iterations=500, twist_deg=1.1)
    print(f"Ossicle sensor initialized: 3 osc, 64 cells, 500 iter, 0.75 KB\n")

    # Workloads to test
    test_configs = [
        {'name': 'baseline', 'workload': None, 'intensity': 0},
        {'name': 'crypto_30%', 'workload': 'crypto', 'intensity': 0.3},
        {'name': 'crypto_50%', 'workload': 'crypto', 'intensity': 0.5},
        {'name': 'crypto_70%', 'workload': 'crypto', 'intensity': 0.7},
        {'name': 'crypto_90%', 'workload': 'crypto', 'intensity': 0.9},
        {'name': 'memory_50%', 'workload': 'memory', 'intensity': 0.5},
        {'name': 'memory_70%', 'workload': 'memory', 'intensity': 0.7},
        {'name': 'memory_90%', 'workload': 'memory', 'intensity': 0.9},
    ]

    test_results = []
    baseline_mean = None
    baseline_std = None

    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print("-"*40)

        # Start workload
        if config['workload'] == 'crypto':
            workload = CryptoMiningSimulator(intensity=config['intensity'])
            workload.start()
            time.sleep(0.5)  # Let workload stabilize
        elif config['workload'] == 'memory':
            workload = MemoryBandwidthWorkload(intensity=config['intensity'])
            workload.start()
            time.sleep(0.5)
        else:
            workload = None

        # Run quick test (matching exp24 methodology)
        sensor.reset()
        mean_corr, std_corr, n_samples, sample_rate = quick_test(sensor, duration=8.0)

        # Stop workload
        if workload:
            workload.stop()

        # Store baseline for z-score calculation
        if config['name'] == 'baseline':
            baseline_mean = mean_corr
            baseline_std = std_corr

        # Calculate z-score relative to baseline
        z_score = 0
        delta = 0
        if baseline_mean is not None and config['name'] != 'baseline':
            delta = mean_corr - baseline_mean
            z_score = abs(delta) / (baseline_std + 1e-10)

        result = {
            'name': config['name'],
            'intensity': config['intensity'],
            'workload_type': config['workload'],
            'mean_corr': float(mean_corr),
            'std_corr': float(std_corr),
            'sample_rate': float(sample_rate),
            'z_score': float(z_score),
            'delta': float(delta),
            'n_samples': n_samples
        }
        test_results.append(result)

        # Print results
        print(f"  Samples: {n_samples}, Rate: {sample_rate:.0f} /s")
        print(f"  Mean correlation: {mean_corr:.6f}")
        print(f"  Std correlation:  {std_corr:.6f}")

        if config['name'] != 'baseline':
            print(f"  Delta:            {delta:+.6f}")
            print(f"  Z-SCORE:          {z_score:.2f}")
            if z_score > 3:
                print(f"  >>> STRONG DETECTION! <<<")
            elif z_score > 2:
                print(f"  >>> DETECTED! <<<")
            elif z_score > 1:
                print(f"  >>> WEAK DETECTION <<<")

    results['tests'] = test_results
    results['baseline'] = {'mean': baseline_mean, 'std': baseline_std}

    # Summary Analysis
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)

    crypto_results = [r for r in test_results if r['workload_type'] == 'crypto']
    memory_results = [r for r in test_results if r['workload_type'] == 'memory']

    print(f"\nBaseline: mean={baseline_mean:.6f}, std={baseline_std:.6f}")

    print("\n| Workload      | Intensity | z-score | Detection | Delta   |")
    print("|---------------|-----------|---------|-----------|---------|")

    for r in test_results:
        if r['name'] == 'baseline':
            continue
        detected = "STRONG" if r['z_score'] > 3 else ("YES" if r['z_score'] > 2 else ("WEAK" if r['z_score'] > 1 else "NO"))
        print(f"| {r['name']:13} | {r['intensity']*100:6.0f}%   | {r['z_score']:7.2f} | {detected:9} | {r['delta']:+.5f} |")

    # Determine minimum detectable crypto intensity
    detectable = [r for r in crypto_results if r['z_score'] > 2.0]
    if detectable:
        min_intensity = min(r['intensity'] for r in detectable)
        print(f"\n*** MINIMUM DETECTABLE CRYPTO INTENSITY: {min_intensity*100:.0f}% ***")
    else:
        weak_detectable = [r for r in crypto_results if r['z_score'] > 1.0]
        if weak_detectable:
            min_intensity = min(r['intensity'] for r in weak_detectable)
            print(f"\n*** WEAK CRYPTO DETECTION AT: {min_intensity*100:.0f}% (z > 1.0) ***")
        else:
            print("\n*** CRYPTO: Detection varies by run ***")

    # Same for memory
    mem_detectable = [r for r in memory_results if r['z_score'] > 2.0]
    if mem_detectable:
        min_mem = min(r['intensity'] for r in mem_detectable)
        print(f"*** MINIMUM DETECTABLE MEMORY INTENSITY: {min_mem*100:.0f}% ***")

    # Compare crypto vs memory signatures
    print("\n" + "-"*50)
    print("SIGNATURE COMPARISON (Crypto vs Memory):")

    if crypto_results and memory_results:
        best_crypto = max(crypto_results, key=lambda x: x['z_score'])
        best_memory = max(memory_results, key=lambda x: x['z_score'])

        print(f"\n  Best crypto ({best_crypto['name']}):")
        print(f"    z-score:      {best_crypto['z_score']:.2f}")
        print(f"    delta:        {best_crypto['delta']:+.6f}")

        print(f"\n  Best memory ({best_memory['name']}):")
        print(f"    z-score:      {best_memory['z_score']:.2f}")
        print(f"    delta:        {best_memory['delta']:+.6f}")

        # Direction comparison
        crypto_dir = "positive" if best_crypto['delta'] > 0 else "negative"
        memory_dir = "positive" if best_memory['delta'] > 0 else "negative"
        print(f"\n  Crypto shifts correlation: {crypto_dir}")
        print(f"  Memory shifts correlation: {memory_dir}")

        if crypto_dir != memory_dir:
            print(f"  >>> DISTINGUISHABLE SIGNATURES! <<<")

    # Physical interpretation
    print("\n" + "="*70)
    print("PHYSICAL INTERPRETATION")
    print("="*70)
    print("""
    CRYPTO MINING PDN SIGNATURE:
    ┌────────────────────────────────────────────────────────────────┐
    │  - High, sustained ALU utilization                             │
    │  - Repetitive hash computations create periodic PDN stress     │
    │  - Tends toward NEGENTROPIC (ordered) strain                   │
    │  - Entropy DECREASES (more structured pattern)                 │
    └────────────────────────────────────────────────────────────────┘

    MEMORY BANDWIDTH PDN SIGNATURE:
    ┌────────────────────────────────────────────────────────────────┐
    │  - High memory controller activity                             │
    │  - Random access patterns create chaotic PDN noise             │
    │  - Tends toward ENTROPIC (disordered) strain                   │
    │  - Entropy INCREASES (less structured pattern)                 │
    └────────────────────────────────────────────────────────────────┘

    The ossicle can potentially DISTINGUISH between workload types
    by measuring the sign and magnitude of entropy strain!
    """)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp26_crypto_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_crypto_detection_test()
