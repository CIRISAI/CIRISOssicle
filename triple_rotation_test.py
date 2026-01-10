#!/usr/bin/env python3
"""
Triple Crystal Rotation Test - Does physical rotation affect correlation matrix?

Author: Eric Moore
Date: 2026-01-08

Runs 3 TRNG crystals simultaneously and measures correlation matrix.
If physical rotation affects the TRNG, the correlation structure should change.

Protocol:
1. Collect baseline correlation matrix at orientation 0°
2. User physically rotates GPU
3. Collect correlation matrix at new orientation
4. Compare: Did ρ(A,B), ρ(B,C), ρ(A,C) change?
"""

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from scipy import stats
from triangulation_test import run_triple_crystal

def collect_correlation_matrix(duration_sec=30, label="orientation_0"):
    """Collect samples and compute correlation matrix."""
    print(f"\n{'='*60}")
    print(f"COLLECTING: {label}")
    print(f"{'='*60}")
    print(f"Duration: {duration_sec} seconds")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print()

    means_a, means_b, means_c = [], [], []
    start = time.time()
    sample_count = 0

    while time.time() - start < duration_sec:
        a, b, c = run_triple_crystal(seed=42)
        means_a.append(a.mean())
        means_b.append(b.mean())
        means_c.append(c.mean())
        sample_count += 1

        if sample_count % 100 == 0:
            elapsed = time.time() - start
            print(f"  {sample_count} samples, {elapsed:.1f}s elapsed")

    elapsed = time.time() - start

    means_a = np.array(means_a)
    means_b = np.array(means_b)
    means_c = np.array(means_c)

    # Compute correlation matrix
    rho_ab = np.corrcoef(means_a, means_b)[0, 1]
    rho_bc = np.corrcoef(means_b, means_c)[0, 1]
    rho_ac = np.corrcoef(means_a, means_c)[0, 1]

    results = {
        'label': label,
        'timestamp': datetime.now().isoformat(),
        'duration_sec': elapsed,
        'n_samples': sample_count,
        'samples_per_sec': sample_count / elapsed,
        'rho_ab': float(rho_ab),
        'rho_bc': float(rho_bc),
        'rho_ac': float(rho_ac),
        'means_a': means_a.tolist(),
        'means_b': means_b.tolist(),
        'means_c': means_c.tolist(),
        'stats': {
            'mean_a': float(np.mean(means_a)),
            'std_a': float(np.std(means_a)),
            'mean_b': float(np.mean(means_b)),
            'std_b': float(np.std(means_b)),
            'mean_c': float(np.mean(means_c)),
            'std_c': float(np.std(means_c)),
        }
    }

    print()
    print(f"RESULTS ({label}):")
    print(f"  Samples: {sample_count} ({sample_count/elapsed:.1f}/sec)")
    print()
    print(f"  CORRELATION MATRIX:")
    print(f"    ρ(A,B) [0°-60°]:   {rho_ab:+.4f}")
    print(f"    ρ(B,C) [60°-120°]: {rho_bc:+.4f}")
    print(f"    ρ(A,C) [0°-120°]:  {rho_ac:+.4f}")

    # Save to file
    outfile = Path(f"events/triple_rotation_{label}_{datetime.now().strftime('%H%M%S')}.json")
    outfile.parent.mkdir(exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {outfile}")

    return results


def compare_orientations(results_0, results_90):
    """Compare correlation matrices from two orientations."""
    print()
    print("="*60)
    print("CORRELATION MATRIX COMPARISON")
    print("="*60)

    # Extract correlations
    rho_ab_0 = results_0['rho_ab']
    rho_bc_0 = results_0['rho_bc']
    rho_ac_0 = results_0['rho_ac']

    rho_ab_90 = results_90['rho_ab']
    rho_bc_90 = results_90['rho_bc']
    rho_ac_90 = results_90['rho_ac']

    print(f"\nOrientation 0° (baseline):")
    print(f"  ρ(A,B): {rho_ab_0:+.4f}")
    print(f"  ρ(B,C): {rho_bc_0:+.4f}")
    print(f"  ρ(A,C): {rho_ac_0:+.4f}")

    print(f"\nOrientation 90° (rotated):")
    print(f"  ρ(A,B): {rho_ab_90:+.4f}")
    print(f"  ρ(B,C): {rho_bc_90:+.4f}")
    print(f"  ρ(A,C): {rho_ac_90:+.4f}")

    # Differences
    delta_ab = rho_ab_90 - rho_ab_0
    delta_bc = rho_bc_90 - rho_bc_0
    delta_ac = rho_ac_90 - rho_ac_0

    print(f"\nCHANGE IN CORRELATIONS (Δρ = rotated - baseline):")
    print(f"  Δρ(A,B): {delta_ab:+.4f}")
    print(f"  Δρ(B,C): {delta_bc:+.4f}")
    print(f"  Δρ(A,C): {delta_ac:+.4f}")

    # Fisher z-transform for significance testing
    def fisher_z(r):
        # Clamp to avoid inf
        r = np.clip(r, -0.999, 0.999)
        return 0.5 * np.log((1 + r) / (1 - r))

    n0 = results_0['n_samples']
    n90 = results_90['n_samples']

    # Combined SE for two independent correlations
    se = np.sqrt(1/(n0 - 3) + 1/(n90 - 3))

    z_ab = (fisher_z(rho_ab_90) - fisher_z(rho_ab_0)) / se
    z_bc = (fisher_z(rho_bc_90) - fisher_z(rho_bc_0)) / se
    z_ac = (fisher_z(rho_ac_90) - fisher_z(rho_ac_0)) / se

    p_ab = 2 * (1 - stats.norm.cdf(abs(z_ab)))
    p_bc = 2 * (1 - stats.norm.cdf(abs(z_bc)))
    p_ac = 2 * (1 - stats.norm.cdf(abs(z_ac)))

    print(f"\nSIGNIFICANCE TESTS (Fisher z-transform):")
    print(f"  ρ(A,B): z={z_ab:+.2f}, p={p_ab:.4f} {'*' if p_ab < 0.05 else ''}")
    print(f"  ρ(B,C): z={z_bc:+.2f}, p={p_bc:.4f} {'*' if p_bc < 0.05 else ''}")
    print(f"  ρ(A,C): z={z_ac:+.2f}, p={p_ac:.4f} {'*' if p_ac < 0.05 else ''}")

    # Overall test: did ANY correlation change significantly?
    any_significant = p_ab < 0.05 or p_bc < 0.05 or p_ac < 0.05

    # Bonferroni correction for 3 tests
    bonf_threshold = 0.05 / 3
    any_significant_bonf = p_ab < bonf_threshold or p_bc < bonf_threshold or p_ac < bonf_threshold

    print()
    print("="*60)
    print("INTERPRETATION")
    print("="*60)

    if any_significant_bonf:
        print("""
  ⚠ SIGNIFICANT CHANGE DETECTED (Bonferroni-corrected)!

  Physical rotation DOES affect the correlation matrix.
  The TRNG crystals show orientation-dependent coupling.

  This suggests:
  - Race condition entropy has directional sensitivity
  - GPU thread scheduling varies with physical orientation
  - Or: systematic drift during rotation (thermal, etc.)
        """)
    elif any_significant:
        print("""
  ⚠ MARGINAL CHANGE (significant before Bonferroni correction)

  Some correlations changed, but may be statistical fluctuation.
  Consider running longer tests to confirm.
        """)
    else:
        print("""
  ✓ NO SIGNIFICANT CHANGE

  The correlation matrix is stable under physical rotation.
  The TRNG entropy appears isotropic in physical space.

  This means:
  - Race condition entropy is independent of GPU orientation
  - The entropy source is purely computational (thread scheduling)
  - No evidence of external field coupling
        """)

    return {
        'delta_ab': delta_ab,
        'delta_bc': delta_bc,
        'delta_ac': delta_ac,
        'p_ab': p_ab,
        'p_bc': p_bc,
        'p_ac': p_ac,
        'significant': any_significant_bonf
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Triple Crystal Rotation Test")
        print()
        print("Usage:")
        print("  python triple_rotation_test.py baseline [duration]")
        print("    Collect baseline at orientation 0°")
        print()
        print("  python triple_rotation_test.py rotated [duration]")
        print("    Collect data at rotated orientation (90°)")
        print()
        print("  python triple_rotation_test.py compare")
        print("    Compare most recent baseline vs rotated results")
        print()
        print("Protocol:")
        print("  1. Run 'baseline' with GPU in normal position")
        print("  2. Physically rotate GPU 90° clockwise")
        print("  3. Run 'rotated' with GPU in new position")
        print("  4. Run 'compare' to analyze correlation matrix change")
        sys.exit(1)

    cmd = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if cmd == "baseline":
        collect_correlation_matrix(duration, "orientation_0")
        print()
        print("Next: Rotate the GPU 90° clockwise, then run:")
        print("  python triple_rotation_test.py rotated")

    elif cmd == "rotated":
        collect_correlation_matrix(duration, "orientation_90")
        print()
        print("Next: Compare results:")
        print("  python triple_rotation_test.py compare")

    elif cmd == "compare":
        import glob
        files_0 = sorted(glob.glob("events/triple_rotation_orientation_0_*.json"))
        files_90 = sorted(glob.glob("events/triple_rotation_orientation_90_*.json"))

        if files_0 and files_90:
            with open(files_0[-1]) as f:
                results_0 = json.load(f)
            with open(files_90[-1]) as f:
                results_90 = json.load(f)
            compare_orientations(results_0, results_90)
        else:
            print("Need both orientation_0 and orientation_90 results.")
            if not files_0:
                print("  Missing: baseline (run 'python triple_rotation_test.py baseline')")
            if not files_90:
                print("  Missing: rotated (run 'python triple_rotation_test.py rotated')")
    else:
        print(f"Unknown command: {cmd}")
        print("Use 'baseline', 'rotated', or 'compare'")
