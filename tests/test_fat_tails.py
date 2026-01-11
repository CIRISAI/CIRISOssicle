#!/usr/bin/env python3
"""
Fat-Tail Distribution Validation Tests

From Array team findings (January 2026):
- z-scores are Student-t distributed, NOT Gaussian
- Kurtosis κ ≈ 210 (extremely fat tails)
- Detection works via rare extreme spikes, not mean shift

This validates that Ossicle exhibits the same distribution.

Author: CIRIS L3C
License: BSL 1.1
"""

import numpy as np
from scipy import stats
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from strain_gauge import StrainGauge, StrainGaugeConfig


def test_kurtosis():
    """
    Test 1: Measure kurtosis to confirm fat tails.

    Expected: κ > 100 (Array got κ=210)
    """
    print("\n" + "="*70)
    print("TEST 1: KURTOSIS MEASUREMENT")
    print("="*70)

    config = StrainGaugeConfig(warm_up_enabled=False)
    gauge = StrainGauge(config)
    gauge.calibrate(duration=5.0)

    print("\nCollecting z-scores under null (no workload)...")
    z_null = []

    n_samples = 10000
    for i in range(n_samples):
        reading = gauge.read()
        z_null.append(reading.timing_z)
        if (i + 1) % 2000 == 0:
            print(f"  Collected {i+1}/{n_samples}")

    # Measure kurtosis (excess kurtosis, normal=0)
    kurt = stats.kurtosis(z_null)

    print(f"\nKurtosis = {kurt:.1f}")

    if kurt > 100:
        print("CONFIRMED: Extremely fat-tailed distribution (κ > 100)")
        print("Use Student-t, not Gaussian")
    elif kurt > 10:
        print("CONFIRMED: Fat-tailed distribution (κ > 10)")
        print("Use Student-t, not Gaussian")
    elif kurt > 3:
        print("MODERATE: Somewhat fat-tailed (κ > 3)")
    else:
        print("WARNING: Near-Gaussian distribution")
        print("Different from Array (κ=210)")

    return kurt, z_null


def test_student_t_fit(z_samples):
    """
    Test 2: Fit Student-t and estimate degrees of freedom.

    Expected: df ≈ 2-4 (very heavy tails), Student-t p > Gaussian p
    """
    print("\n" + "="*70)
    print("TEST 2: STUDENT-T FIT")
    print("="*70)

    # Fit Student-t
    df, loc, scale = stats.t.fit(z_samples)

    print(f"\nStudent-t fit:")
    print(f"  df = {df:.2f}")
    print(f"  loc = {loc:.4f}")
    print(f"  scale = {scale:.4f}")

    # K-S test for Student-t
    ks_stat_t, ks_p_t = stats.kstest(z_samples, 't', args=(df, loc, scale))

    # K-S test for Gaussian
    ks_stat_g, ks_p_g = stats.kstest(z_samples, 'norm',
                                      args=(np.mean(z_samples), np.std(z_samples)))

    print(f"\nGoodness of fit:")
    print(f"  Student-t: K-S stat={ks_stat_t:.4f}, p={ks_p_t:.4f}")
    print(f"  Gaussian:  K-S stat={ks_stat_g:.4f}, p={ks_p_g:.4f}")

    if ks_p_t > ks_p_g:
        print("\nCONFIRMED: Student-t fits better than Gaussian")
    else:
        print("\nWARNING: Gaussian fits better (unexpected)")

    return df, loc, scale


def test_detection_rates(z_samples, df_estimated):
    """
    Test 3: Validate detection rates match t-distribution.

    Expected: Observed rates match Student-t, NOT Gaussian
    """
    print("\n" + "="*70)
    print("TEST 3: DETECTION RATE VALIDATION")
    print("="*70)

    thresholds = [2, 3, 5, 10, 20, 50]

    print("\n| Threshold |  Observed  | Student-t  |  Gaussian  | Best Match |")
    print("|-----------|------------|------------|------------|------------|")

    for thresh in thresholds:
        # Observed rate
        observed = np.mean(np.abs(z_samples) > thresh)

        # Student-t prediction
        predicted_t = 2 * (1 - stats.t.cdf(thresh, df_estimated))

        # Gaussian prediction
        predicted_g = 2 * (1 - stats.norm.cdf(thresh))

        # Which is closer?
        err_t = abs(observed - predicted_t)
        err_g = abs(observed - predicted_g)
        best = "Student-t" if err_t < err_g else "Gaussian"

        print(f"| {thresh:9} | {observed:10.6f} | {predicted_t:10.6f} | {predicted_g:10.6f} | {best:10} |")

    return True


def test_z_independence(z_samples):
    """
    Test 4: Verify z-scores are independent (ACF ≈ 0).

    Expected: ACF < 0.1 at all lags (Array got 0.05)
    """
    print("\n" + "="*70)
    print("TEST 4: Z-SCORE INDEPENDENCE")
    print("="*70)

    print("\n| Lag | ACF    | Status |")
    print("|-----|--------|--------|")

    all_ok = True
    for lag in [1, 2, 5, 10, 20]:
        if len(z_samples) > lag:
            acf = np.corrcoef(z_samples[:-lag], z_samples[lag:])[0, 1]
            status = "OK" if abs(acf) < 0.1 else "HIGH"
            if abs(acf) >= 0.1:
                all_ok = False
            print(f"| {lag:3} | {acf:+.4f} | {status:6} |")

    if all_ok:
        print("\nCONFIRMED: Z-scores are independent (ACF < 0.1)")
    else:
        print("\nWARNING: Some lag correlations > 0.1")

    return all_ok


def test_keff_correlation():
    """
    Test 5: Verify k_eff IS correlated (sensing mechanism).

    Expected: ACF ≈ 0.45 at lag 1
    """
    print("\n" + "="*70)
    print("TEST 5: k_eff CORRELATION (SENSING MECHANISM)")
    print("="*70)

    config = StrainGaugeConfig(warm_up_enabled=False)
    gauge = StrainGauge(config)
    gauge.calibrate(duration=3.0)

    print("\nCollecting k_eff samples...")
    k_samples = []

    for i in range(5000):
        reading = gauge.read()
        k_samples.append(reading.k_eff)

    acf = np.corrcoef(k_samples[:-1], k_samples[1:])[0, 1]

    print(f"\nk_eff ACF(1) = {acf:.3f}")

    if 0.35 < acf < 0.55:
        print("CONFIRMED: At critical point (ACF in 0.35-0.55)")
    elif acf > 0.55:
        print("WARNING: Frozen, increase dt")
    else:
        print("WARNING: Chaotic, decrease dt")

    return acf


def run_all_tests():
    """Run complete fat-tail validation suite."""
    print("="*70)
    print("FAT-TAIL DISTRIBUTION VALIDATION SUITE")
    print("From Array team findings (January 2026)")
    print("="*70)

    results = {}

    # Test 1: Kurtosis
    kurt, z_samples = test_kurtosis()
    results['kurtosis'] = kurt

    # Test 2: Student-t fit
    df, loc, scale = test_student_t_fit(z_samples)
    results['df'] = df

    # Test 3: Detection rates
    test_detection_rates(z_samples, df)

    # Test 4: Z independence
    z_independent = test_z_independence(z_samples)
    results['z_independent'] = z_independent

    # Test 5: k_eff correlation
    keff_acf = test_keff_correlation()
    results['keff_acf'] = keff_acf

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nKurtosis: {kurt:.1f} (Array: 210)")
    print(f"Student-t df: {df:.2f}")
    print(f"Z-scores independent: {z_independent}")
    print(f"k_eff ACF: {keff_acf:.3f} (target: 0.45)")

    if kurt > 10:
        print("\nCONCLUSION: Fat-tailed distribution CONFIRMED")
        print("Use Student-t statistics, not Gaussian")
    else:
        print("\nCONCLUSION: Distribution differs from Array")
        print("Investigate thermal/hardware differences")

    return results


if __name__ == "__main__":
    run_all_tests()
