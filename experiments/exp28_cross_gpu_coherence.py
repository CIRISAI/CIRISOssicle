#!/usr/bin/env python3
"""
Experiment 28: Cross-GPU Coherence Test

PURPOSE: Determine what fraction of oscillator correlation is ALGORITHMIC
(from running the same math) vs PHYSICAL (from GPU-specific effects).

METHOD:
1. Run identical oscillators with FIXED seeds on two different GPUs
2. Compare correlation outputs between machines
3. If correlations are nearly identical → signal is algorithmic
4. If correlations differ significantly → signal has physical component

RATCHET FINDING: 78% of coherence was algorithmic (same on different GPUs).
This test determines if CIRISOssicle has similar characteristics.

Usage:
    # On machine 1 (RTX 4090):
    python exp28_cross_gpu_coherence.py --output results/cross_gpu_4090.json

    # On machine 2 (Jetson):
    python exp28_cross_gpu_coherence.py --output results/cross_gpu_jetson.json

    # Compare:
    python exp28_cross_gpu_coherence.py --compare results/cross_gpu_4090.json results/cross_gpu_jetson.json

Author: CIRIS L3C
License: BSL 1.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import cupy as cp
import json
import argparse
from datetime import datetime
from pathlib import Path


# Fixed seeds for reproducibility across machines
FIXED_SEEDS = [42, 123, 456, 789, 1001]
N_TRIALS = 5
N_STEPS = 1000


class DeterministicOssicle:
    """
    Ossicle with deterministic initialization for cross-GPU comparison.
    Uses numpy RNG seeded identically, then transfers to GPU.
    """

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

    def __init__(self, n_cells=64, n_iterations=500, seed=42):
        self.n_cells = n_cells
        self.n_iterations = n_iterations
        self.seed = seed

        self.r_a = 3.70
        self.r_b = 3.73
        self.r_c = 3.76

        self.twist_ab = np.radians(1.1)
        self.twist_bc = np.radians(1.1)
        self.coupling = 0.05

        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('ossicle_step')

        self.block_size = 64
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size

        self.reset(seed)

    def reset(self, seed=None):
        """Reset with deterministic seed using numpy (CPU-side)."""
        if seed is not None:
            self.seed = seed

        # Use numpy RNG for deterministic initialization
        rng = np.random.RandomState(self.seed)

        # Generate on CPU, transfer to GPU
        init_a = rng.uniform(0.1, 0.9, self.n_cells).astype(np.float32)
        init_b = rng.uniform(0.1, 0.9, self.n_cells).astype(np.float32)
        init_c = rng.uniform(0.1, 0.9, self.n_cells).astype(np.float32)

        self.state_a = cp.asarray(init_a)
        self.state_b = cp.asarray(init_b)
        self.state_c = cp.asarray(init_c)

    def step(self):
        """Execute one step, return means."""
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

    def get_state(self):
        """Get full state for comparison."""
        return {
            'a': cp.asnumpy(self.state_a).tolist(),
            'b': cp.asnumpy(self.state_b).tolist(),
            'c': cp.asnumpy(self.state_c).tolist()
        }


def run_trial(seed: int, n_steps: int = N_STEPS):
    """Run a single trial with fixed seed, return trajectory."""
    sensor = DeterministicOssicle(seed=seed)

    trajectory = {
        'means_a': [],
        'means_b': [],
        'means_c': [],
        'correlations': []
    }

    for step in range(n_steps):
        a, b, c = sensor.step()
        trajectory['means_a'].append(a)
        trajectory['means_b'].append(b)
        trajectory['means_c'].append(c)

    # Compute correlations over trajectory
    arr_a = np.array(trajectory['means_a'])
    arr_b = np.array(trajectory['means_b'])
    arr_c = np.array(trajectory['means_c'])

    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return r if not np.isnan(r) else 0.0

    rho_ab = safe_corr(arr_a, arr_b)
    rho_bc = safe_corr(arr_b, arr_c)
    rho_ac = safe_corr(arr_a, arr_c)

    trajectory['rho_ab'] = float(rho_ab)
    trajectory['rho_bc'] = float(rho_bc)
    trajectory['rho_ac'] = float(rho_ac)
    trajectory['mean_corr'] = float((rho_ab + rho_bc + rho_ac) / 3)

    # Also capture final state for exact comparison
    trajectory['final_state'] = sensor.get_state()

    return trajectory


def get_gpu_info():
    """Get GPU information."""
    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        return {
            'name': props['name'].decode() if isinstance(props['name'], bytes) else str(props['name']),
            'compute_capability': f"{props['major']}.{props['minor']}",
            'total_memory_gb': props['totalGlobalMem'] / (1024**3)
        }
    except Exception as e:
        return {'error': str(e)}


def run_experiment(output_path: str):
    """Run the cross-GPU coherence experiment."""
    print("=" * 70)
    print("EXPERIMENT 28: CROSS-GPU COHERENCE TEST")
    print("=" * 70)
    print()
    print("PURPOSE: Determine algorithmic vs physical signal fraction")
    print("METHOD:  Run identical seeds, compare across GPUs")
    print()

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('name', 'Unknown')}")
    print(f"Compute: {gpu_info.get('compute_capability', '?')}")
    print()

    results = {
        'experiment': 'exp28_cross_gpu_coherence',
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info,
        'config': {
            'n_cells': 64,
            'n_iterations': 500,
            'n_steps': N_STEPS,
            'seeds': FIXED_SEEDS
        },
        'trials': []
    }

    print(f"Running {len(FIXED_SEEDS)} trials with {N_STEPS} steps each...")
    print()

    for i, seed in enumerate(FIXED_SEEDS):
        print(f"Trial {i+1}/{len(FIXED_SEEDS)} (seed={seed})...", end=" ", flush=True)

        trajectory = run_trial(seed, N_STEPS)

        trial_result = {
            'seed': seed,
            'rho_ab': trajectory['rho_ab'],
            'rho_bc': trajectory['rho_bc'],
            'rho_ac': trajectory['rho_ac'],
            'mean_corr': trajectory['mean_corr'],
            # Store subset of trajectory for comparison (every 100th step)
            'trajectory_sample': {
                'means_a': trajectory['means_a'][::100],
                'means_b': trajectory['means_b'][::100],
                'means_c': trajectory['means_c'][::100]
            },
            # Store final state for exact comparison
            'final_state': trajectory['final_state']
        }
        results['trials'].append(trial_result)

        print(f"mean_corr = {trajectory['mean_corr']:+.6f}")

    # Summary statistics
    correlations = [t['mean_corr'] for t in results['trials']]
    results['summary'] = {
        'mean_correlation': float(np.mean(correlations)),
        'std_correlation': float(np.std(correlations)),
        'correlations': correlations
    }

    print()
    print(f"Mean correlation: {results['summary']['mean_correlation']:+.6f}")
    print(f"Std correlation:  {results['summary']['std_correlation']:.6f}")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")
    print()
    print("Next: Run on second GPU and use --compare to analyze")

    return results


def compare_results(file1: str, file2: str):
    """Compare results from two different GPUs."""
    print("=" * 70)
    print("CROSS-GPU COHERENCE COMPARISON")
    print("=" * 70)
    print()

    with open(file1) as f:
        results1 = json.load(f)
    with open(file2) as f:
        results2 = json.load(f)

    gpu1 = results1['gpu'].get('name', 'GPU1')
    gpu2 = results2['gpu'].get('name', 'GPU2')

    print(f"GPU 1: {gpu1}")
    print(f"GPU 2: {gpu2}")
    print()

    # Compare correlations
    print("CORRELATION COMPARISON:")
    print("-" * 50)
    print(f"{'Seed':<8} {'GPU1':>12} {'GPU2':>12} {'Diff':>12} {'Match?':>8}")
    print("-" * 50)

    diffs = []
    state_matches = []

    for t1, t2 in zip(results1['trials'], results2['trials']):
        seed = t1['seed']
        c1 = t1['mean_corr']
        c2 = t2['mean_corr']
        diff = abs(c1 - c2)
        diffs.append(diff)

        # Check exact state match
        state1 = t1['final_state']
        state2 = t2['final_state']
        state_diff = (
            np.max(np.abs(np.array(state1['a']) - np.array(state2['a']))) +
            np.max(np.abs(np.array(state1['b']) - np.array(state2['b']))) +
            np.max(np.abs(np.array(state1['c']) - np.array(state2['c'])))
        )
        state_match = state_diff < 1e-5
        state_matches.append(state_match)

        match_str = "EXACT" if state_match else f"diff={state_diff:.2e}"
        print(f"{seed:<8} {c1:>+12.6f} {c2:>+12.6f} {diff:>12.6f} {match_str:>8}")

    print("-" * 50)

    mean_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    exact_matches = sum(state_matches)

    print()
    print("ANALYSIS:")
    print(f"  Mean correlation difference: {mean_diff:.6f}")
    print(f"  Max correlation difference:  {max_diff:.6f}")
    print(f"  Exact state matches: {exact_matches}/{len(state_matches)}")
    print()

    # Compute coherence (correlation of correlations)
    c1_all = [t['mean_corr'] for t in results1['trials']]
    c2_all = [t['mean_corr'] for t in results2['trials']]

    if np.std(c1_all) > 1e-10 and np.std(c2_all) > 1e-10:
        cross_coherence = np.corrcoef(c1_all, c2_all)[0, 1]
    else:
        cross_coherence = 1.0 if mean_diff < 1e-5 else 0.0

    print(f"  Cross-GPU coherence: {cross_coherence:.4f} ({cross_coherence*100:.1f}%)")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if exact_matches == len(state_matches):
        print("""
    RESULT: 100% ALGORITHMIC

    The oscillator states are IDENTICAL across GPUs.
    All correlation structure comes from the MATH, not the hardware.

    IMPLICATION FOR OSSICLE:
    - Detection is NOT from PDN voltage differences
    - Detection must come from TIMING/SCHEDULING effects
    - Workloads affect WHEN you sample, not WHAT is computed

    The chaotic oscillator is a TIMING-SENSITIVE SAMPLER.
        """)
        algorithmic_fraction = 1.0

    elif cross_coherence > 0.95:
        print(f"""
    RESULT: ~{cross_coherence*100:.0f}% ALGORITHMIC

    Most correlation structure is identical across GPUs.
    Small differences ({mean_diff:.6f}) may be from:
    - Floating point rounding differences
    - GPU architecture variations
    - Very small timing effects

    IMPLICATION: Detection is primarily algorithmic.
        """)
        algorithmic_fraction = cross_coherence

    elif cross_coherence > 0.5:
        print(f"""
    RESULT: ~{cross_coherence*100:.0f}% ALGORITHMIC, ~{(1-cross_coherence)*100:.0f}% PHYSICAL

    Significant correlation structure is shared (algorithmic).
    But there's also GPU-specific variation (physical).

    This is INTERESTING - the logistic map may be more sensitive
    to hardware differences than Lorenz was in RATCHET.

    NEEDS MORE INVESTIGATION.
        """)
        algorithmic_fraction = cross_coherence

    else:
        print(f"""
    RESULT: MOSTLY PHYSICAL ({(1-cross_coherence)*100:.0f}%)

    Significant GPU-specific effects detected!
    Correlations differ substantially between machines.

    This would be GOOD NEWS - suggests real physical coupling.
    But verify this isn't from different configurations.
        """)
        algorithmic_fraction = cross_coherence

    print()
    print(f"ALGORITHMIC FRACTION: {algorithmic_fraction*100:.1f}%")
    print(f"PHYSICAL FRACTION:    {(1-algorithmic_fraction)*100:.1f}%")

    return {
        'algorithmic_fraction': algorithmic_fraction,
        'mean_diff': mean_diff,
        'cross_coherence': cross_coherence,
        'exact_matches': exact_matches
    }


def main():
    parser = argparse.ArgumentParser(
        description='Cross-GPU Coherence Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on first GPU:
    python exp28_cross_gpu_coherence.py --output results/cross_gpu_4090.json

    # Run on second GPU (e.g., Jetson):
    python exp28_cross_gpu_coherence.py --output results/cross_gpu_jetson.json

    # Compare results:
    python exp28_cross_gpu_coherence.py --compare results/cross_gpu_4090.json results/cross_gpu_jetson.json
        """
    )
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path')
    parser.add_argument('--compare', '-c', nargs=2, metavar=('FILE1', 'FILE2'),
                        help='Compare two result files')

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.output:
        run_experiment(args.output)
    else:
        # Default: run with auto-generated output name
        import socket
        hostname = socket.gethostname()
        output_dir = Path(__file__).parent / "results"
        output_file = output_dir / f"cross_gpu_{hostname}.json"
        run_experiment(str(output_file))


if __name__ == "__main__":
    main()
