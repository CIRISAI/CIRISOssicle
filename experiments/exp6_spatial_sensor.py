#!/usr/bin/env python3
"""
Experiment 6: Spatial Distribution Sensor

Deploy oscillators across DIFFERENT SMs to detect spatial voltage gradients.

Key insight from GPUVolt research:
- Voltage droops vary spatially across the GPU die
- Intercore interference means activity on one SM affects others
- Correlation between oscillators on different SMs encodes spatial gradient info

Approach:
- Launch multiple thread blocks (one per SM target)
- Each block runs a chaotic oscillator independently
- Record which SM each block runs on via __smid()
- Compute cross-SM correlations to detect spatial patterns

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

# CUDA kernel that records SM ID and runs oscillator
SPATIAL_KERNEL = r'''
extern "C" __global__ void spatial_oscillator(
    float* state,           // Per-block oscillator state
    float* output_means,    // Mean from each block
    int* sm_ids,            // SM ID for each block
    float r,
    float coupling,
    int n_iterations,
    unsigned int seed
) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    // Record which SM this block is running on
    if (tid == 0) {
        unsigned int smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        sm_ids[block_id] = smid;
    }
    __syncthreads();

    // Initialize state for this thread
    unsigned int local_seed = seed + block_id * 1024 + tid;
    float x = (float)(local_seed % 1000) / 1000.0f;

    // Run coupled logistic map iterations
    extern __shared__ float shared_state[];

    for (int iter = 0; iter < n_iterations; iter++) {
        // Logistic map step
        float x_new = r * x * (1.0f - x);

        // Store for coupling
        shared_state[tid] = x_new;
        __syncthreads();

        // Coupling with neighbors
        int left = (tid - 1 + blockDim.x) % blockDim.x;
        int right = (tid + 1) % blockDim.x;
        x = (1.0f - coupling) * x_new +
            coupling * 0.5f * (shared_state[left] + shared_state[right]);

        __syncthreads();
    }

    // Final state to shared for reduction
    shared_state[tid] = x;
    __syncthreads();

    // Reduce to compute mean
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_state[i];
        }
        output_means[block_id] = sum / blockDim.x;
    }
}
'''


class SpatialSensor:
    """Multi-SM spatial correlation sensor."""

    def __init__(self, n_blocks: int = 32, threads_per_block: int = 256,
                 r: float = 3.75, coupling: float = 0.5,
                 n_iterations: int = 5000, seed: int = 42):
        self.n_blocks = n_blocks
        self.threads_per_block = threads_per_block
        self.r = r
        self.coupling = coupling
        self.n_iterations = n_iterations
        self.seed = seed

        # Compile kernel
        self.module = cp.RawModule(code=SPATIAL_KERNEL)
        self.kernel = self.module.get_function('spatial_oscillator')

        # Allocate buffers
        self.d_state = cp.zeros(n_blocks * threads_per_block, dtype=cp.float32)
        self.d_means = cp.zeros(n_blocks, dtype=cp.float32)
        self.d_sm_ids = cp.zeros(n_blocks, dtype=cp.int32)

        # Shared memory size
        self.shared_mem = threads_per_block * 4  # float per thread

    def read(self, sample_idx: int = 0):
        """Read from all blocks, return means and SM IDs.

        Uses sample_idx to vary seed between reads for temporal variation.
        """
        # Vary seed each read to get temporal variation
        current_seed = self.seed + sample_idx * 13337

        self.kernel(
            (self.n_blocks,), (self.threads_per_block,),
            (self.d_state, self.d_means, self.d_sm_ids,
             cp.float32(self.r), cp.float32(self.coupling),
             cp.int32(self.n_iterations), cp.uint32(current_seed)),
            shared_mem=self.shared_mem
        )
        cp.cuda.Stream.null.synchronize()

        return self.d_means.get(), self.d_sm_ids.get()


def collect_spatial_data(sensor: SpatialSensor, duration: float, label: str = ""):
    """Collect spatial correlation data."""
    all_means = []
    all_sm_ids = []
    timestamps = []

    start = time.time()
    sample_idx = 0
    while time.time() - start < duration:
        means, sm_ids = sensor.read(sample_idx)
        all_means.append(means)
        all_sm_ids.append(sm_ids)
        timestamps.append(time.time() - start)
        sample_idx += 1

    return {
        'means': np.array(all_means),
        'sm_ids': np.array(all_sm_ids),
        'timestamps': np.array(timestamps),
        'label': label,
        'n_samples': sample_idx,
    }


def analyze_spatial_correlations(data):
    """Analyze correlations between different SMs."""
    means = data['means']  # Shape: (n_samples, n_blocks)
    sm_ids = data['sm_ids'][0]  # SM IDs (should be constant)

    n_samples, n_blocks = means.shape
    unique_sms = np.unique(sm_ids)

    print(f"  Samples: {n_samples}")
    print(f"  Blocks: {n_blocks}")
    print(f"  Unique SMs: {len(unique_sms)} - {sorted(unique_sms)}")

    # Group blocks by SM
    sm_to_blocks = {}
    for block_id, sm in enumerate(sm_ids):
        if sm not in sm_to_blocks:
            sm_to_blocks[sm] = []
        sm_to_blocks[sm].append(block_id)

    print(f"  Blocks per SM: {dict((k, len(v)) for k, v in sm_to_blocks.items())}")

    # Compute within-SM vs between-SM correlations
    within_sm_corrs = []
    between_sm_corrs = []

    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            corr = np.corrcoef(means[:, i], means[:, j])[0, 1]
            if np.isnan(corr):
                continue

            if sm_ids[i] == sm_ids[j]:
                within_sm_corrs.append(corr)
            else:
                between_sm_corrs.append(corr)

    results = {
        'n_unique_sms': len(unique_sms),
        'sm_ids': sorted(unique_sms.tolist()),
        'within_sm_corr_mean': np.mean(within_sm_corrs) if within_sm_corrs else None,
        'within_sm_corr_std': np.std(within_sm_corrs) if within_sm_corrs else None,
        'between_sm_corr_mean': np.mean(between_sm_corrs) if between_sm_corrs else None,
        'between_sm_corr_std': np.std(between_sm_corrs) if between_sm_corrs else None,
        'n_within_pairs': len(within_sm_corrs),
        'n_between_pairs': len(between_sm_corrs),
    }

    return results


def run_spatial_experiment(duration: float = 30.0, n_blocks: int = 64):
    """Run spatial correlation experiment."""
    print("="*70)
    print("EXPERIMENT 6: SPATIAL SENSOR")
    print("="*70)
    print()
    print("Deploying oscillators across multiple SMs to detect spatial gradients.")
    print()
    print("Physics basis (GPUVolt research):")
    print("  - Voltage droops vary spatially across GPU die")
    print("  - Activity on one SM affects voltage on other SMs")
    print("  - Correlation between SMs encodes spatial information")
    print()
    print(f"Parameters:")
    print(f"  Blocks: {n_blocks}")
    print(f"  Duration per phase: {duration}s")
    print()

    sensor = SpatialSensor(n_blocks=n_blocks)

    # Test SM distribution
    print("Testing SM distribution...")
    _, sm_ids = sensor.read()
    unique_sms = np.unique(sm_ids)
    print(f"  Blocks spread across {len(unique_sms)} SMs: {sorted(unique_sms)}")
    print()

    results = {}

    # Phase 1: Baseline (no load)
    print("[BASELINE] No concurrent load")
    baseline_data = collect_spatial_data(sensor, duration, "baseline")
    results['baseline'] = analyze_spatial_correlations(baseline_data)
    w = results['baseline']['within_sm_corr_mean']
    b = results['baseline']['between_sm_corr_mean']
    print(f"  Within-SM corr: {f'{w:.4f}' if w is not None else 'N/A'}")
    print(f"  Between-SM corr: {f'{b:.4f}' if b is not None else 'N/A'}")
    print()

    # Phase 2: Compute load
    print("[COMPUTE LOAD] Running sin/cos/exp stress")
    import threading

    def compute_load():
        a = cp.random.randn(4096 * 4096, dtype=cp.float32)
        while getattr(compute_load, 'running', True):
            b = cp.sin(a) * cp.cos(a)
            cp.cuda.Stream.null.synchronize()
            time.sleep(0.01)

    compute_load.running = True
    load_thread = threading.Thread(target=compute_load)
    load_thread.start()
    time.sleep(1)  # Let load stabilize

    load_data = collect_spatial_data(sensor, duration, "compute_load")

    compute_load.running = False
    load_thread.join(timeout=5)

    results['compute_load'] = analyze_spatial_correlations(load_data)
    w = results['compute_load']['within_sm_corr_mean']
    b = results['compute_load']['between_sm_corr_mean']
    print(f"  Within-SM corr: {f'{w:.4f}' if w is not None else 'N/A'}")
    print(f"  Between-SM corr: {f'{b:.4f}' if b is not None else 'N/A'}")
    print()

    # Analysis
    print("="*70)
    print("SPATIAL ANALYSIS")
    print("="*70)
    print()

    # Compare within vs between SM
    print("Within-SM vs Between-SM correlation:")
    for phase in ['baseline', 'compute_load']:
        r = results[phase]
        print(f"  {phase}:")
        if r['within_sm_corr_mean'] is not None and r['between_sm_corr_mean'] is not None:
            diff = r['within_sm_corr_mean'] - r['between_sm_corr_mean']
            print(f"    Within-SM:  {r['within_sm_corr_mean']:+.4f} ± {r['within_sm_corr_std']:.4f}")
            print(f"    Between-SM: {r['between_sm_corr_mean']:+.4f} ± {r['between_sm_corr_std']:.4f}")
            print(f"    Difference: {diff:+.4f}")

            if abs(diff) > 0.02:
                print(f"    → SPATIAL GRADIENT DETECTED")
        print()

    # Load effect on spatial pattern
    if results['baseline']['between_sm_corr_mean'] and results['compute_load']['between_sm_corr_mean']:
        delta = results['compute_load']['between_sm_corr_mean'] - results['baseline']['between_sm_corr_mean']
        print(f"Load effect on between-SM correlation: {delta:+.4f}")
        if abs(delta) > 0.05:
            print("  → Load creates spatial correlation changes across GPU!")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp6_spatial_{timestamp}.json"

    output = {
        'experiment': 'exp6_spatial_sensor',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_blocks': n_blocks,
            'duration': duration,
        },
        'results': results,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 6: Spatial Sensor')
    parser.add_argument('--duration', '-d', type=float, default=30,
                        help='Duration per phase (default: 30)')
    parser.add_argument('--blocks', '-b', type=int, default=64,
                        help='Number of blocks (default: 64)')
    args = parser.parse_args()

    run_spatial_experiment(duration=args.duration, n_blocks=args.blocks)
