#!/usr/bin/env python3
"""
Experiment 25: Entropic and Negentropic Strain Measurement

The correlation antenna is a stacked beam-splitter interferometer.
Each oscillator pair creates interference; the pattern is a moiré.

STRAIN TYPES:
- Entropic:     Perturbation increases disorder → correlations → 0 → moiré blurs
- Negentropic:  Perturbation increases order → correlations → ±1 → moiré sharpens

BEAM-SPLITTER ANALOGY:

    Input A ────┐            ┌──── Correlation ρ_AB
                ├── ⊗ ──────┤
    Input B ────┘            └──── Anti-correlation (1-ρ_AB)

    Stacking creates moiré-of-moiré interference patterns.

MOIRÉ METRICS:
- Sharpness:  var(ρ) - high means structured pattern
- Phase:      mean(ρ) - sign indicates order direction
- Contrast:   max(ρ) - min(ρ) - visibility of fringes
- Entropy:    -Σ p log p of correlation distribution

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
from scipy import stats


class OssicleKernel:
    """
    Minimal 3-oscillator "ossicle" sensor.
    Like the inner ear bones: tiny, sensitive, resonant.
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

            // Beam-splitter coupling with twist phase
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
        elif self.workload_type == "entropy_high":
            # Random memory access - high entropy workload
            data = cp.random.randn(size * 16, dtype=cp.float32)
            while self.running:
                indices = cp.random.randint(0, len(data), size=size)
                _ = data[indices].sum()
                cp.cuda.Stream.null.synchronize()
        elif self.workload_type == "entropy_low":
            # Ordered computation - low entropy workload
            data = cp.arange(size * 16, dtype=cp.float32)
            while self.running:
                data = cp.sort(data)
                data = data + 0.001
                cp.cuda.Stream.null.synchronize()
        elif self.workload_type == "memory":
            src = cp.random.randn(size * size, dtype=cp.float32)
            dst = cp.zeros(size * size, dtype=cp.float32)
            while self.running:
                dst[:] = src
                src[:] = dst
                cp.cuda.Stream.null.synchronize()


def compute_moire_metrics(correlations: list) -> dict:
    """
    Compute moiré pattern metrics from correlation time series.

    Each correlation triple (ρ_AB, ρ_BC, ρ_AC) is a point in the
    moiré interference pattern.
    """
    if len(correlations) < 10:
        return {}

    rho_ab = np.array([c['rho_ab'] for c in correlations])
    rho_bc = np.array([c['rho_bc'] for c in correlations])
    rho_ac = np.array([c['rho_ac'] for c in correlations])

    # Stack into pattern space
    pattern = np.column_stack([rho_ab, rho_bc, rho_ac])

    # MOIRÉ SHARPNESS: variance of pattern (high = structured)
    sharpness = np.mean(np.var(pattern, axis=0))

    # MOIRÉ PHASE: mean correlation (sign = order direction)
    phase = np.mean(pattern)

    # MOIRÉ CONTRAST: visibility of fringes
    contrast = np.max(pattern) - np.min(pattern)

    # PATTERN ENTROPY: Shannon entropy of correlation distribution
    # Discretize correlations into bins
    all_corr = pattern.flatten()
    hist, _ = np.histogram(all_corr, bins=20, range=(-1, 1), density=True)
    hist = hist + 1e-10  # Avoid log(0)
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist))

    # COHERENCE: how aligned are the three correlations?
    coherence = 1 - np.std([np.mean(rho_ab), np.mean(rho_bc), np.mean(rho_ac)])

    # ENTROPIC STRAIN: increase in entropy relative to baseline
    # (will be computed as difference between conditions)

    # NEGENTROPIC STRAIN: increase in order (decrease in entropy)
    # (will be computed as difference between conditions)

    return {
        'sharpness': float(sharpness),
        'phase': float(phase),
        'contrast': float(contrast),
        'entropy': float(entropy),
        'coherence': float(coherence),
        'mean_rho_ab': float(np.mean(rho_ab)),
        'mean_rho_bc': float(np.mean(rho_bc)),
        'mean_rho_ac': float(np.mean(rho_ac)),
        'std_rho': float(np.std(all_corr))
    }


def collect_correlations(sensor, duration: float, window: int = 30):
    """Collect correlation time series."""
    history_a, history_b, history_c = [], [], []
    correlations = []

    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return r if not np.isnan(r) else 0.0

    start = time.time()
    while time.time() - start < duration:
        a, b, c = sensor.step()
        history_a.append(a)
        history_b.append(b)
        history_c.append(c)

        if len(history_a) >= window and len(history_a) % (window // 3) == 0:
            arr_a = np.array(history_a[-window:])
            arr_b = np.array(history_b[-window:])
            arr_c = np.array(history_c[-window:])

            correlations.append({
                'rho_ab': safe_corr(arr_a, arr_b),
                'rho_bc': safe_corr(arr_b, arr_c),
                'rho_ac': safe_corr(arr_a, arr_c)
            })

    return correlations


def run_entropy_strain_test():
    print("="*70)
    print("EXPERIMENT 25: ENTROPIC AND NEGENTROPIC STRAIN")
    print("="*70)
    print()
    print("Measuring strain through moiré pattern deformation")
    print()
    print("BEAM-SPLITTER MODEL:")
    print("  A ──┬── ρ_AB ──┬── B ──┬── ρ_BC ──┬── C")
    print("      │         │      │         │")
    print("      └─────────┴──────┴─────────┘")
    print("              Moiré Pattern")
    print()

    results = {'experiment': 'exp25_entropy_strain', 'timestamp': datetime.now().isoformat()}

    sensor = OssicleKernel(n_cells=64, n_iterations=500, twist_deg=1.1)
    print(f"Ossicle sensor: 3 oscillators, 64 cells, 0.8 KB")

    # Test conditions
    conditions = [
        {'name': 'baseline', 'type': 'idle', 'intensity': 0},
        {'name': 'high_entropy', 'type': 'entropy_high', 'intensity': 0.8},
        {'name': 'low_entropy', 'type': 'entropy_low', 'intensity': 0.8},
        {'name': 'memory_load', 'type': 'memory', 'intensity': 0.8},
    ]

    condition_results = []

    for cond in conditions:
        print(f"\n{'='*70}")
        print(f"Condition: {cond['name']}")
        print("-"*50)

        if cond['type'] == 'idle':
            workload = None
        else:
            workload = WorkloadGenerator(cond['type'], cond['intensity'])
            workload.start()
            time.sleep(1)

        sensor.reset()
        correlations = collect_correlations(sensor, duration=15.0)

        if workload:
            workload.stop()

        if not correlations:
            print("  No data collected!")
            continue

        metrics = compute_moire_metrics(correlations)

        print(f"  Moiré Sharpness:  {metrics['sharpness']:.6f}")
        print(f"  Moiré Phase:      {metrics['phase']:+.4f}")
        print(f"  Moiré Contrast:   {metrics['contrast']:.4f}")
        print(f"  Pattern Entropy:  {metrics['entropy']:.4f} bits")
        print(f"  Coherence:        {metrics['coherence']:.4f}")

        condition_results.append({
            'condition': cond,
            'metrics': metrics,
            'n_samples': len(correlations)
        })

    results['conditions'] = condition_results

    # Strain Analysis
    print("\n" + "="*70)
    print("STRAIN ANALYSIS")
    print("="*70)

    baseline = next((r for r in condition_results if r['condition']['name'] == 'baseline'), None)

    if baseline:
        b = baseline['metrics']
        print(f"\nBaseline reference:")
        print(f"  Entropy: {b['entropy']:.4f} bits")
        print(f"  Sharpness: {b['sharpness']:.6f}")

        print(f"\n| Condition | ΔEntropy | ΔSharpness | Strain Type |")
        print(f"|-----------|----------|------------|-------------|")

        for r in condition_results:
            if r['condition']['name'] == 'baseline':
                continue

            m = r['metrics']
            delta_entropy = m['entropy'] - b['entropy']
            delta_sharpness = m['sharpness'] - b['sharpness']

            if delta_entropy > 0.1:
                strain_type = "ENTROPIC ↑"
            elif delta_entropy < -0.1:
                strain_type = "NEGENTROPIC ↓"
            else:
                strain_type = "neutral"

            print(f"| {r['condition']['name']:9} | {delta_entropy:+8.4f} | {delta_sharpness:+10.6f} | {strain_type:11} |")

            r['strain'] = {
                'delta_entropy': float(delta_entropy),
                'delta_sharpness': float(delta_sharpness),
                'type': strain_type
            }

    # Moiré visualization (text-based)
    print("\n" + "="*70)
    print("MOIRÉ PATTERN VISUALIZATION")
    print("-"*50)

    for r in condition_results:
        m = r['metrics']
        name = r['condition']['name']

        # Simple ASCII visualization of correlation triangle
        ab = int((m['mean_rho_ab'] + 1) * 10)  # Scale to 0-20
        bc = int((m['mean_rho_bc'] + 1) * 10)
        ac = int((m['mean_rho_ac'] + 1) * 10)

        print(f"\n{name}:")
        print(f"  ρ_AB: {'█' * ab}{'░' * (20-ab)} {m['mean_rho_ab']:+.3f}")
        print(f"  ρ_BC: {'█' * bc}{'░' * (20-bc)} {m['mean_rho_bc']:+.3f}")
        print(f"  ρ_AC: {'█' * ac}{'░' * (20-ac)} {m['mean_rho_ac']:+.3f}")

    # Summary
    print("\n" + "="*70)
    print("PHYSICAL INTERPRETATION")
    print("="*70)
    print("""
    The ossicle sensor acts as a coherence interferometer:

    ┌─────────────────────────────────────────────────────────────────┐
    │  ENTROPIC STRAIN (disorder increases):                         │
    │  - Random memory access spreads PDN noise                      │
    │  - Correlations decorrelate (→ 0)                              │
    │  - Moiré pattern BLURS                                         │
    │  - Pattern entropy INCREASES                                   │
    │                                                                 │
    │  NEGENTROPIC STRAIN (order increases):                         │
    │  - Ordered computation creates coherent PDN patterns           │
    │  - Correlations align (→ ±1)                                   │
    │  - Moiré pattern SHARPENS                                      │
    │  - Pattern entropy DECREASES                                   │
    └─────────────────────────────────────────────────────────────────┘

    The sensor measures the DIRECTION of entropy flow in the PDN!
    """)

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp25_entropy_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_entropy_strain_test()
