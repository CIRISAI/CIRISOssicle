#!/usr/bin/env python3
"""
Experiment 19: Magic Configuration Validation

Tests the twistronics-derived magic configuration:
  r_base = 3.661
  spacing = 0.030
  twist = 90° (quadrature)

The twistronics analogy predicts this should give ~27× amplification
compared to our standard configuration (r=3.70, spacing=0.03).

Twist Implementation:
- 0° (in-phase): All oscillators start at same phase
- 90° (quadrature): 90° phase offset between oscillators
- 180° (anti-phase): Alternating phase

Phase is implemented via initial conditions and coupling structure.

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


class MagicConfigKernel:
    """
    CUDA kernel with magic configuration parameters.

    Magic config: r_base=3.661, spacing=0.030, quadrature coupling
    Standard config: r_base=3.70, spacing=0.03, in-phase coupling
    """

    KERNEL_CODE = r'''
    extern "C" __global__ void oscillator_step_magic(
        float* state_a, float* state_b, float* state_c, float* state_d,
        float r_a, float r_b, float r_c, float r_d,
        float coupling,
        float phase_a, float phase_b, float phase_c, float phase_d,
        int n, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float a = state_a[idx];
        float b = state_b[idx];
        float c = state_c[idx];
        float d = state_d[idx];

        // Grid neighbors
        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        // Phase-shifted coupling (quadrature = 90 deg = pi/2 offset)
        // This creates interference patterns similar to moire in graphene
        float cos_a = cosf(phase_a);
        float cos_b = cosf(phase_b);
        float cos_c = cosf(phase_c);
        float cos_d = cosf(phase_d);

        for (int i = 0; i < iterations; i++) {
            // Neighbor coupling with phase modulation
            float na = state_a[left] * cos_a + state_a[right] * cos_a;
            float nb = state_b[left] * cos_b + state_b[right] * cos_b;
            float nc = state_c[left] * cos_c + state_c[right] * cos_c;
            float nd = state_d[left] * cos_d + state_d[right] * cos_d;

            // Cross-coupling between layers (moire-like)
            float cross_a = coupling * 0.5f * (b * cos_b + c * cos_c + d * cos_d);
            float cross_b = coupling * 0.5f * (a * cos_a + c * cos_c + d * cos_d);
            float cross_c = coupling * 0.5f * (a * cos_a + b * cos_b + d * cos_d);
            float cross_d = coupling * 0.5f * (a * cos_a + b * cos_b + c * cos_c);

            // Logistic map with phase-modulated coupling
            a = r_a * a * (1.0f - a) + coupling * (na - 2.0f * a) + cross_a;
            b = r_b * b * (1.0f - b) + coupling * (nb - 2.0f * b) + cross_b;
            c = r_c * c * (1.0f - c) + coupling * (nc - 2.0f * c) + cross_c;
            d = r_d * d * (1.0f - d) + coupling * (nd - 2.0f * d) + cross_d;

            // Clamp
            a = fminf(fmaxf(a, 0.0001f), 0.9999f);
            b = fminf(fmaxf(b, 0.0001f), 0.9999f);
            c = fminf(fmaxf(c, 0.0001f), 0.9999f);
            d = fminf(fmaxf(d, 0.0001f), 0.9999f);

            state_a[idx] = a;
            state_b[idx] = b;
            state_c[idx] = c;
            state_d[idx] = d;
        }
    }
    '''

    def __init__(self, r_base: float = 3.661, spacing: float = 0.030,
                 twist_degrees: float = 90.0,
                 n_cells: int = 1024, coupling: float = 0.05,
                 n_iterations: int = 5000):

        self.r_base = r_base
        self.spacing = spacing
        self.twist_degrees = twist_degrees
        self.n_cells = n_cells
        self.coupling = coupling
        self.n_iterations = n_iterations

        # Compute r-values
        self.r_a = r_base
        self.r_b = r_base + spacing
        self.r_c = r_base + 2 * spacing
        self.r_d = r_base + 3 * spacing

        # Compute phase offsets (twist)
        # Quadrature: 0°, 90°, 180°, 270°
        twist_rad = np.radians(twist_degrees)
        self.phase_a = 0.0
        self.phase_b = twist_rad
        self.phase_c = 2 * twist_rad
        self.phase_d = 3 * twist_rad

        # Compile kernel
        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('oscillator_step_magic')

        # Initialize states with phase offset
        self._init_states()

        self.block_size = 256
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size

    def _init_states(self):
        """Initialize states with phase-offset initial conditions."""
        # Base random state
        base = np.random.uniform(0.1, 0.9, self.n_cells).astype(np.float32)

        # Phase-shifted initial conditions
        phase_shift = np.sin(np.linspace(0, 2*np.pi, self.n_cells))

        self.state_a = cp.asarray(base + 0.05 * phase_shift * np.cos(self.phase_a))
        self.state_b = cp.asarray(base + 0.05 * phase_shift * np.cos(self.phase_b))
        self.state_c = cp.asarray(base + 0.05 * phase_shift * np.cos(self.phase_c))
        self.state_d = cp.asarray(base + 0.05 * phase_shift * np.cos(self.phase_d))

        # Clamp to valid range
        self.state_a = cp.clip(self.state_a, 0.1, 0.9).astype(cp.float32)
        self.state_b = cp.clip(self.state_b, 0.1, 0.9).astype(cp.float32)
        self.state_c = cp.clip(self.state_c, 0.1, 0.9).astype(cp.float32)
        self.state_d = cp.clip(self.state_d, 0.1, 0.9).astype(cp.float32)

    def step(self):
        """Execute one step and return means."""
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.state_a, self.state_b, self.state_c, self.state_d,
             cp.float32(self.r_a), cp.float32(self.r_b),
             cp.float32(self.r_c), cp.float32(self.r_d),
             cp.float32(self.coupling),
             cp.float32(self.phase_a), cp.float32(self.phase_b),
             cp.float32(self.phase_c), cp.float32(self.phase_d),
             cp.int32(self.n_cells), cp.int32(self.n_iterations))
        )
        cp.cuda.Stream.null.synchronize()

        return (
            float(cp.mean(self.state_a)),
            float(cp.mean(self.state_b)),
            float(cp.mean(self.state_c)),
            float(cp.mean(self.state_d))
        )

    def reset(self):
        """Reset to fresh initial conditions."""
        self._init_states()


class WorkloadGenerator:
    """Generate GPU workloads."""

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
        size = int(1024 * 8 * self.intensity)

        if self.workload_type == "idle":
            while self.running:
                time.sleep(0.01)

        elif self.workload_type == "memory":
            # Memory bandwidth attack (strongest signal from exp18)
            src = cp.random.randn(size * size, dtype=cp.float32)
            dst = cp.zeros(size * size, dtype=cp.float32)
            while self.running:
                dst[:] = src
                src[:] = dst
                cp.cuda.Stream.null.synchronize()


def collect_correlations(sensor, duration: float, window: int = 100):
    """Collect correlation data."""
    means_a, means_b, means_c, means_d = [], [], [], []
    correlations = []

    start = time.time()
    while time.time() - start < duration:
        a, b, c, d = sensor.step()
        means_a.append(a)
        means_b.append(b)
        means_c.append(c)
        means_d.append(d)

        if len(means_a) >= window and len(means_a) % (window // 4) == 0:
            arr_a = np.array(means_a[-window:])
            arr_b = np.array(means_b[-window:])
            arr_c = np.array(means_c[-window:])
            arr_d = np.array(means_d[-window:])

            def safe_corr(x, y):
                if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                    return 0.0
                r = np.corrcoef(x, y)[0, 1]
                return r if not np.isnan(r) else 0.0

            # All 6 pairs
            rho_ab = safe_corr(arr_a, arr_b)
            rho_ac = safe_corr(arr_a, arr_c)
            rho_ad = safe_corr(arr_a, arr_d)
            rho_bc = safe_corr(arr_b, arr_c)
            rho_bd = safe_corr(arr_b, arr_d)
            rho_cd = safe_corr(arr_c, arr_d)

            mean_rho = np.mean([rho_ab, rho_ac, rho_ad, rho_bc, rho_bd, rho_cd])
            correlations.append({
                'rho_ab': rho_ab, 'rho_ac': rho_ac, 'rho_ad': rho_ad,
                'rho_bc': rho_bc, 'rho_bd': rho_bd, 'rho_cd': rho_cd,
                'mean': mean_rho
            })

    return correlations


def run_magic_configuration_test():
    """Test magic configuration vs standard configuration."""

    print("="*70)
    print("EXPERIMENT 19: MAGIC CONFIGURATION VALIDATION")
    print("="*70)
    print()
    print("Testing twistronics-derived magic configuration:")
    print("  r_base = 3.661 (vs standard 3.70)")
    print("  spacing = 0.030")
    print("  twist = 90° quadrature (vs in-phase)")
    print()

    results = {
        'experiment': 'exp19_magic_configuration',
        'timestamp': datetime.now().isoformat()
    }

    # =========================================================================
    # CONFIGURATION COMPARISON
    # =========================================================================
    configs = [
        {'name': 'standard', 'r_base': 3.70, 'spacing': 0.03, 'twist': 0},
        {'name': 'magic', 'r_base': 3.661, 'spacing': 0.030, 'twist': 90},
        {'name': 'quadrature_standard_r', 'r_base': 3.70, 'spacing': 0.03, 'twist': 90},
        {'name': 'anti_phase', 'r_base': 3.661, 'spacing': 0.030, 'twist': 180},
    ]

    config_results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"  r_base={config['r_base']}, spacing={config['spacing']}, twist={config['twist']}°")
        print("-"*50)

        sensor = MagicConfigKernel(
            r_base=config['r_base'],
            spacing=config['spacing'],
            twist_degrees=config['twist']
        )

        # Collect baseline
        print("Collecting baseline (idle)...")
        idle = WorkloadGenerator("idle")
        idle.start()
        time.sleep(2)

        sensor.reset()
        baseline = collect_correlations(sensor, duration=30.0)
        idle.stop()

        baseline_means = [c['mean'] for c in baseline]
        baseline_mean = np.mean(baseline_means)
        baseline_std = np.std(baseline_means)

        print(f"  Baseline: ρ = {baseline_mean:.4f} ± {baseline_std:.4f}")

        # Collect attack
        print("Collecting attack (memory bandwidth)...")
        attack = WorkloadGenerator("memory", intensity=0.8)
        attack.start()
        time.sleep(2)

        sensor.reset()
        attack_data = collect_correlations(sensor, duration=30.0)
        attack.stop()

        attack_means = [c['mean'] for c in attack_data]
        attack_mean = np.mean(attack_means)

        delta = attack_mean - baseline_mean
        z_score = abs(delta) / (baseline_std + 1e-10)

        print(f"  Attack: ρ = {attack_mean:.4f}")
        print(f"  Δρ = {delta:+.4f}, z = {z_score:.2f}σ")

        result = {
            'config': config,
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'attack_mean': float(attack_mean),
            'delta': float(delta),
            'z_score': float(z_score),
            'detectable': bool(z_score > 3)
        }
        config_results.append(result)

        detect_mark = " *** 3σ DETECTED! ***" if z_score > 3 else ""
        print(f"  Result: {z_score:.2f}σ{detect_mark}")

    results['configurations'] = config_results

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("CONFIGURATION COMPARISON")
    print("="*70)

    print("\n| Configuration | r_base | Twist | z-score | Amplification | 3σ? |")
    print("|---------------|--------|-------|---------|---------------|-----|")

    # Use standard as baseline for amplification
    standard_z = next((r['z_score'] for r in config_results if r['config']['name'] == 'standard'), 1)

    for r in config_results:
        config = r['config']
        amplification = r['z_score'] / (standard_z + 1e-10)
        detect = "YES" if r['detectable'] else "no"
        print(f"| {config['name']:13} | {config['r_base']:.3f} | {config['twist']:3}° | {r['z_score']:5.2f}σ | {amplification:5.1f}× | {detect:3} |")

    # Find best configuration
    best = max(config_results, key=lambda x: x['z_score'])
    print(f"\n*** BEST CONFIGURATION: {best['config']['name']} ***")
    print(f"    z-score: {best['z_score']:.2f}σ")
    print(f"    Amplification: {best['z_score'] / standard_z:.1f}× vs standard")

    results['best_config'] = best['config']['name']
    results['best_z_score'] = best['z_score']

    # =========================================================================
    # TWISTRONICS VALIDATION
    # =========================================================================
    print("\n" + "="*70)
    print("TWISTRONICS VALIDATION")
    print("-"*50)

    magic_result = next((r for r in config_results if r['config']['name'] == 'magic'), None)
    standard_result = next((r for r in config_results if r['config']['name'] == 'standard'), None)

    if magic_result and standard_result:
        improvement = magic_result['z_score'] / (standard_result['z_score'] + 1e-10)

        print(f"\nPredicted improvement: 26.9×")
        print(f"Measured improvement: {improvement:.1f}×")

        if improvement > 10:
            print(f"\n*** MAGIC CONFIGURATION VALIDATED! ***")
            print(f"    Twistronics analogy holds for GPU oscillators")
        elif improvement > 2:
            print(f"\n*** PARTIAL VALIDATION ***")
            print(f"    Significant improvement but below prediction")
        else:
            print(f"\n*** MAGIC CONFIGURATION NOT VALIDATED ***")
            print(f"    Improvement insufficient on this hardware/conditions")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("""
Twistronics-Oscillator Mapping:
┌─────────────────────────┬────────────────────────────────┐
│ Graphene                │ GPU Oscillators                │
├─────────────────────────┼────────────────────────────────┤
│ Magic angle (~1.1°)     │ Magic twist (~90°)             │
│ Moiré superlattice      │ Correlation interference       │
│ Flat bands              │ Sensitivity amplification      │
│ Layer stacking          │ r-value ordering               │
│ Lattice constant        │ r_base parameter               │
└─────────────────────────┴────────────────────────────────┘
""")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp19_magic_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_magic_configuration_test()
