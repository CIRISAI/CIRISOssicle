#!/usr/bin/env python3
"""
Experiment 13: Tetrahedral Geometry SNR Comparison

Compares 3-crystal (triangle) vs 4-crystal (tetrahedron) architectures:
- DOF: 3 vs 6 correlation pairs
- SNR: Expected √2 ≈ 1.41× improvement
- k_eff stability: More pairs = less variance
- Detection sensitivity: Attack detection comparison

Formal prediction (TetrahedralGeometry.lean):
  tetra_snr_improvement: √(6/3) = √2 > 1.4

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

from strain_sensor import StrainSensor, SensorConfig
from tetrahedral_sensor import TetrahedralSensor, TetraConfig, compute_k_eff, is_k_eff_safe


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

        elif self.workload_type == "crypto":
            data = cp.random.randint(0, 2**32, size=(size,), dtype=cp.uint32)
            while self.running:
                for _ in range(200):
                    data = data ^ (data << 13)
                    data = data ^ (data >> 17)
                    data = data ^ (data << 5)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.002)


def collect_triangle_data(sensor: StrainSensor, duration: float, window: int = 100):
    """Collect data from 3-crystal (triangle) sensor."""
    means_a, means_b, means_c = [], [], []
    correlations = []  # Mean correlation per window

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

            rho_ab = safe_corr(a_arr, b_arr)
            rho_bc = safe_corr(b_arr, c_arr)
            rho_ac = safe_corr(a_arr, c_arr)

            mean_rho = np.mean([rho_ab, rho_bc, rho_ac])
            correlations.append({
                'rho_ab': rho_ab, 'rho_bc': rho_bc, 'rho_ac': rho_ac,
                'mean': mean_rho
            })

    return correlations


def collect_tetra_data(sensor: TetrahedralSensor, duration: float):
    """Collect data from 4-crystal (tetrahedron) sensor."""
    correlations = []

    start = time.time()
    while time.time() - start < duration:
        reading = sensor.read()

        if reading.rho_ab != 0:  # Has valid correlations
            correlations.append({
                'rho_ab': reading.rho_ab,
                'rho_ac': reading.rho_ac,
                'rho_ad': reading.rho_ad,
                'rho_bc': reading.rho_bc,
                'rho_bd': reading.rho_bd,
                'rho_cd': reading.rho_cd,
                'mean': reading.mean_correlation,
                'std': reading.correlation_std
            })

    return correlations


def run_snr_comparison():
    """Run SNR comparison between triangle and tetrahedron sensors."""

    print("="*70)
    print("EXPERIMENT 13: TETRAHEDRAL GEOMETRY SNR COMPARISON")
    print("="*70)
    print()
    print("Formal prediction (TetrahedralGeometry.lean):")
    print("  SNR improvement = √(6/3) = √2 ≈ 1.41×")
    print()

    # Initialize sensors
    tri_sensor = StrainSensor(SensorConfig(n_iterations=5000))
    tetra_sensor = TetrahedralSensor(TetraConfig(n_iterations=5000))

    results = {
        'experiment': 'exp13_tetrahedral_snr',
        'timestamp': datetime.now().isoformat(),
        'prediction': {'snr_improvement': 1.414}
    }

    # =========================================================================
    # PHASE 1: Baseline comparison (idle)
    # =========================================================================
    print("[PHASE 1] BASELINE COMPARISON (IDLE)")
    print("-"*50)

    idle = WorkloadGenerator("idle")
    idle.start()
    time.sleep(2)

    print("Collecting triangle (3-crystal) baseline...")
    tri_baseline = collect_triangle_data(tri_sensor, duration=30.0)

    tetra_sensor.reset_history()
    print("Collecting tetrahedron (4-crystal) baseline...")
    tetra_baseline = collect_tetra_data(tetra_sensor, duration=30.0)

    idle.stop()

    # Compute statistics
    tri_means = [c['mean'] for c in tri_baseline]
    tetra_means = [c['mean'] for c in tetra_baseline]

    tri_sigma = np.std(tri_means)
    tetra_sigma = np.std(tetra_means)
    tri_mean = np.mean(tri_means)
    tetra_mean = np.mean(tetra_means)

    print(f"\nTriangle (3 correlations):")
    print(f"  Mean ρ: {tri_mean:.4f}")
    print(f"  σ(mean ρ): {tri_sigma:.4f}")
    print(f"  Samples: {len(tri_baseline)}")

    print(f"\nTetrahedron (6 correlations):")
    print(f"  Mean ρ: {tetra_mean:.4f}")
    print(f"  σ(mean ρ): {tetra_sigma:.4f}")
    print(f"  Samples: {len(tetra_baseline)}")

    # SNR = signal / noise. With same signal, SNR improvement = σ_tri / σ_tetra
    snr_improvement = tri_sigma / tetra_sigma if tetra_sigma > 0 else float('inf')
    print(f"\nSNR improvement (baseline): {snr_improvement:.3f}×")
    print(f"  Predicted: 1.414× (√2)")
    print(f"  Difference: {(snr_improvement - 1.414)/1.414 * 100:+.1f}%")

    results['baseline'] = {
        'triangle': {
            'mean_rho': float(tri_mean),
            'sigma': float(tri_sigma),
            'n_samples': len(tri_baseline)
        },
        'tetrahedron': {
            'mean_rho': float(tetra_mean),
            'sigma': float(tetra_sigma),
            'n_samples': len(tetra_baseline)
        },
        'snr_improvement': float(snr_improvement)
    }

    # =========================================================================
    # PHASE 2: Under attack comparison
    # =========================================================================
    print("\n" + "="*70)
    print("[PHASE 2] UNDER ATTACK COMPARISON (CRYPTO)")
    print("-"*50)

    crypto = WorkloadGenerator("crypto", intensity=0.8)
    crypto.start()
    time.sleep(2)

    tetra_sensor.reset_history()

    print("Collecting triangle under attack...")
    tri_attack = collect_triangle_data(tri_sensor, duration=30.0)

    tetra_sensor.reset_history()
    print("Collecting tetrahedron under attack...")
    tetra_attack = collect_tetra_data(tetra_sensor, duration=30.0)

    crypto.stop()

    # Compute statistics
    tri_attack_means = [c['mean'] for c in tri_attack]
    tetra_attack_means = [c['mean'] for c in tetra_attack]

    tri_attack_sigma = np.std(tri_attack_means)
    tetra_attack_sigma = np.std(tetra_attack_means)
    tri_attack_mean = np.mean(tri_attack_means)
    tetra_attack_mean = np.mean(tetra_attack_means)

    print(f"\nTriangle under attack:")
    print(f"  Mean ρ: {tri_attack_mean:.4f}")
    print(f"  Δρ: {tri_attack_mean - tri_mean:+.4f}")
    print(f"  σ: {tri_attack_sigma:.4f}")

    print(f"\nTetrahedron under attack:")
    print(f"  Mean ρ: {tetra_attack_mean:.4f}")
    print(f"  Δρ: {tetra_attack_mean - tetra_mean:+.4f}")
    print(f"  σ: {tetra_attack_sigma:.4f}")

    snr_improvement_attack = tri_attack_sigma / tetra_attack_sigma if tetra_attack_sigma > 0 else float('inf')
    print(f"\nSNR improvement (attack): {snr_improvement_attack:.3f}×")

    results['attack'] = {
        'triangle': {
            'mean_rho': float(tri_attack_mean),
            'delta_rho': float(tri_attack_mean - tri_mean),
            'sigma': float(tri_attack_sigma)
        },
        'tetrahedron': {
            'mean_rho': float(tetra_attack_mean),
            'delta_rho': float(tetra_attack_mean - tetra_mean),
            'sigma': float(tetra_attack_sigma)
        },
        'snr_improvement': float(snr_improvement_attack)
    }

    # =========================================================================
    # PHASE 3: Detection sensitivity comparison
    # =========================================================================
    print("\n" + "="*70)
    print("[PHASE 3] DETECTION SENSITIVITY")
    print("-"*50)

    # z-scores for detection
    tri_delta = tri_attack_mean - tri_mean
    tetra_delta = tetra_attack_mean - tetra_mean

    tri_z = abs(tri_delta) / tri_sigma if tri_sigma > 0 else 0
    tetra_z = abs(tetra_delta) / tetra_sigma if tetra_sigma > 0 else 0

    print(f"\nTriangle detection:")
    print(f"  |Δρ|: {abs(tri_delta):.4f}")
    print(f"  σ: {tri_sigma:.4f}")
    print(f"  z-score: {tri_z:.2f}σ")
    print(f"  Detectable (>3σ): {'YES' if tri_z > 3 else 'NO'}")

    print(f"\nTetrahedron detection:")
    print(f"  |Δρ|: {abs(tetra_delta):.4f}")
    print(f"  σ: {tetra_sigma:.4f}")
    print(f"  z-score: {tetra_z:.2f}σ")
    print(f"  Detectable (>3σ): {'YES' if tetra_z > 3 else 'NO'}")

    z_improvement = tetra_z / tri_z if tri_z > 0 else float('inf')
    print(f"\nz-score improvement: {z_improvement:.3f}×")
    print(f"  (Higher z = easier detection)")

    results['detection'] = {
        'triangle': {
            'delta_rho': float(tri_delta),
            'z_score': float(tri_z),
            'detectable': bool(tri_z > 3)
        },
        'tetrahedron': {
            'delta_rho': float(tetra_delta),
            'z_score': float(tetra_z),
            'detectable': bool(tetra_z > 3)
        },
        'z_improvement': float(z_improvement)
    }

    # =========================================================================
    # PHASE 4: k_eff stability comparison
    # =========================================================================
    print("\n" + "="*70)
    print("[PHASE 4] k_eff STABILITY")
    print("-"*50)

    # Compute k_eff for each sample
    tri_k_effs = []
    for c in tri_baseline:
        rho = c['mean']
        if is_k_eff_safe(rho, k=3, margin=0.1):
            tri_k_effs.append(compute_k_eff(rho, k=3))

    tetra_k_effs = []
    for c in tetra_baseline:
        rho = c['mean']
        if is_k_eff_safe(rho, k=6, margin=0.05):
            tetra_k_effs.append(compute_k_eff(rho, k=6))

    if tri_k_effs:
        tri_k_mean = np.mean(tri_k_effs)
        tri_k_std = np.std(tri_k_effs)
        print(f"\nTriangle k_eff (k=3):")
        print(f"  Mean: {tri_k_mean:.2f}")
        print(f"  Std: {tri_k_std:.2f}")
        print(f"  CV: {tri_k_std/tri_k_mean*100:.1f}%")
        print(f"  Valid samples: {len(tri_k_effs)}/{len(tri_baseline)}")
    else:
        print("\nTriangle: All samples near singularity!")
        tri_k_mean, tri_k_std = float('nan'), float('nan')

    if tetra_k_effs:
        tetra_k_mean = np.mean(tetra_k_effs)
        tetra_k_std = np.std(tetra_k_effs)
        print(f"\nTetrahedron k_eff (k=6):")
        print(f"  Mean: {tetra_k_mean:.2f}")
        print(f"  Std: {tetra_k_std:.2f}")
        print(f"  CV: {tetra_k_std/tetra_k_mean*100:.1f}%")
        print(f"  Valid samples: {len(tetra_k_effs)}/{len(tetra_baseline)}")
    else:
        print("\nTetrahedron: All samples near singularity!")
        tetra_k_mean, tetra_k_std = float('nan'), float('nan')

    results['k_eff'] = {
        'triangle': {
            'mean': float(tri_k_mean) if not np.isnan(tri_k_mean) else None,
            'std': float(tri_k_std) if not np.isnan(tri_k_std) else None,
            'valid_samples': len(tri_k_effs)
        },
        'tetrahedron': {
            'mean': float(tetra_k_mean) if not np.isnan(tetra_k_mean) else None,
            'std': float(tetra_k_std) if not np.isnan(tetra_k_std) else None,
            'valid_samples': len(tetra_k_effs)
        }
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"| Metric | Triangle | Tetrahedron | Improvement |")
    print(f"|--------|----------|-------------|-------------|")
    print(f"| DOF | 3 | 6 | 2.0× |")
    print(f"| σ (baseline) | {tri_sigma:.4f} | {tetra_sigma:.4f} | {snr_improvement:.2f}× |")
    print(f"| z-score | {tri_z:.2f}σ | {tetra_z:.2f}σ | {z_improvement:.2f}× |")

    # Verdict
    print()
    if snr_improvement > 1.2:
        print(f"*** TETRAHEDRAL GEOMETRY PROVIDES SNR IMPROVEMENT ***")
        print(f"    Measured: {snr_improvement:.2f}×")
        print(f"    Predicted: 1.41× (√2)")
        if snr_improvement > 1.3:
            print(f"    VERIFIED: Close to theoretical prediction!")
        else:
            print(f"    PARTIAL: Some improvement but below theoretical")
    else:
        print("Tetrahedral geometry did not provide expected SNR improvement.")
        print("Possible causes:")
        print("  - Correlations not independent (violates √n scaling)")
        print("  - Different r values create systematic differences")
        print("  - Need longer collection time")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp13_tetrahedral_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_snr_comparison()
