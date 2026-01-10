#!/usr/bin/env python3
"""
Analyze the original triple_rotation data to understand the correlation structure
and what the 6× effect actually looked like.

The ACCELEROMETER_THEORY.md claims:
- Stationary variance: σ²_ρ ≈ 0.02² = 0.0004
- Shaking variance:    σ²_ρ ≈ 0.05² = 0.0025
- F-ratio = 6.25×

Let's verify this by computing sliding window correlations on the existing data.
"""

import json
import numpy as np
from pathlib import Path

def load_triple_rotation_data(filepath):
    """Load data from a triple_rotation JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return (
        np.array(data['means_a']),
        np.array(data['means_b']),
        np.array(data['means_c']),
        data
    )

def compute_sliding_correlations(means_a, means_b, means_c, window_size=1000, step=500):
    """Compute correlations over sliding windows."""
    n = len(means_a)
    rho_ab, rho_bc, rho_ac = [], [], []

    for i in range(0, n - window_size + 1, step):
        wa = means_a[i:i + window_size]
        wb = means_b[i:i + window_size]
        wc = means_c[i:i + window_size]

        rho_ab.append(np.corrcoef(wa, wb)[0, 1])
        rho_bc.append(np.corrcoef(wb, wc)[0, 1])
        rho_ac.append(np.corrcoef(wa, wc)[0, 1])

    return np.array(rho_ab), np.array(rho_bc), np.array(rho_ac)

# Load the triple_rotation files from original experiment
data_dir = Path("/home/emoore/coherence_gradient_experiment/events")
files = sorted(data_dir.glob("triple_rotation_orientation_*.json"))

print("="*70)
print("ANALYSIS OF ORIGINAL TRIPLE ROTATION DATA")
print("="*70)
print()
print("Goal: Understand the baseline correlation variance (σ²_ρ)")
print("and compare to the claimed 6× effect during shaking.")
print()

all_variances = []

for f in files[:4]:  # Just the orientation files, not thermal
    print(f"File: {f.name}")
    means_a, means_b, means_c, meta = load_triple_rotation_data(f)

    print(f"  Samples: {len(means_a)}")
    print(f"  Duration: {meta.get('duration_sec', 'N/A'):.1f}s")

    # Overall correlations
    print(f"  Overall ρ(A,B): {meta.get('rho_ab', 'N/A'):+.4f}")
    print(f"  Overall ρ(B,C): {meta.get('rho_bc', 'N/A'):+.4f}")
    print(f"  Overall ρ(A,C): {meta.get('rho_ac', 'N/A'):+.4f}")

    # Compute sliding window correlations
    # ACCELEROMETER_THEORY.md used W=1000 (approx based on SE calculation)
    for window_size in [100, 500, 1000]:
        rho_ab, rho_bc, rho_ac = compute_sliding_correlations(
            means_a, means_b, means_c,
            window_size=window_size,
            step=window_size // 2
        )

        if len(rho_ab) > 2:
            var_ab = np.var(rho_ab)
            var_bc = np.var(rho_bc)
            var_ac = np.var(rho_ac)
            var_total = var_ab + var_bc + var_ac

            print(f"  Window={window_size}: σ²_ρ(A,B)={var_ab:.6f}, σ²_ρ(B,C)={var_bc:.6f}, σ²_ρ(A,C)={var_ac:.6f}, total={var_total:.6f}")
            all_variances.append({
                'file': f.name,
                'window': window_size,
                'var_ab': var_ab,
                'var_bc': var_bc,
                'var_ac': var_ac,
                'var_total': var_total,
                'n_windows': len(rho_ab)
            })

    print()

print("="*70)
print("SUMMARY")
print("="*70)
print()

# Compare to ACCELEROMETER_THEORY.md claims
print("From ACCELEROMETER_THEORY.md:")
print("  Stationary variance: σ²_ρ ≈ 0.02² = 0.0004")
print("  Shaking variance:    σ²_ρ ≈ 0.05² = 0.0025")
print("  F-ratio = 6.25×")
print()

if all_variances:
    # Find window=1000 results (closest to original)
    w1000 = [v for v in all_variances if v['window'] == 1000]
    if w1000:
        avg_var = np.mean([v['var_ab'] for v in w1000])
        print(f"Measured stationary variance (W=1000):")
        print(f"  σ²_ρ(A,B) average: {avg_var:.6f}")
        print(f"  σ_ρ = {np.sqrt(avg_var):.4f}")
        print()

        # Compare to 0.02 claim
        claimed_sigma = 0.02
        print(f"Comparison to claimed σ_ρ=0.02:")
        print(f"  Ratio: {np.sqrt(avg_var) / claimed_sigma:.2f}×")

print()

# Now analyze the exp1 results to see what we're measuring
print("="*70)
print("COMPARISON WITH CURRENT EXP1 MEASUREMENTS")
print("="*70)
print()

exp1_dir = Path("/home/emoore/strain_sensor_optimization/experiments/results")
exp1_files = sorted(exp1_dir.glob("exp1_*.json"))

if exp1_files:
    for f in exp1_files[-1:]:  # Most recent
        print(f"File: {f.name}")
        with open(f, 'r') as fp:
            data = json.load(fp)

        phases = data.get('phases', {})
        for phase_name in ['baseline_a', 'shake', 'baseline_b']:
            phase = phases.get(phase_name, {})
            if phase:
                print(f"  {phase_name}:")
                print(f"    σ²_ρ_total: {phase.get('rho_var_total', 'N/A')}")
                print(f"    n_samples: {phase.get('n_samples', 'N/A')}")

        print(f"  F-ratio: {data.get('F_ratio', 'N/A')}")
else:
    print("No exp1 results found")
