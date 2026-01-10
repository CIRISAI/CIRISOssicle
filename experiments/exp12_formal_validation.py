#!/usr/bin/env python3
"""
Experiment 12: Formal Verification Validation

Tests predictions from Lean 4 formal proofs:
1. Measure actual σ (noise floor) - compare to assumed 0.027
2. Verify k_eff INCREASES under attack (proven: attack_increases_k_eff)
3. Test that |Δρ_crypto| < 3σ (proven: crypto_not_single_sample_detectable)
4. Validate k_eff deviation detection (proven: crypto_detectable_via_k_eff)

Reference: RATCHET/formal/RATCHET/GPUTamper/RESEARCH_PROPOSAL.md

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


def compute_k_eff(rho: float, k: int = 3) -> float:
    """
    Compute effective degrees of freedom.

    k_eff = k / (1 + ρ(k-1))

    For negative ρ (GPU competitive regime):
    - More negative ρ → higher k_eff (divergence)
    - Attacks make ρ more negative → k_eff increases
    """
    denominator = 1 + rho * (k - 1)
    if abs(denominator) < 1e-10:
        return float('inf')
    return k / denominator


def compute_correlation_from_means(means_a: list, means_b: list, window: int = 100) -> float:
    """Compute correlation between two oscillator mean time series."""
    if len(means_a) < window:
        return 0.0
    a = np.array(means_a[-window:])
    b = np.array(means_b[-window:])
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


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

        elif self.workload_type == "transformer":
            q = cp.random.randn(size, 64, dtype=cp.float32)
            k = cp.random.randn(size, 64, dtype=cp.float32)
            v = cp.random.randn(size, 64, dtype=cp.float32)
            while self.running:
                attn = cp.matmul(q, k.T)
                attn = cp.exp(attn - cp.max(attn, axis=-1, keepdims=True))
                attn = attn / cp.sum(attn, axis=-1, keepdims=True)
                out = cp.matmul(attn, v)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.005)


def collect_correlations(sensor: StrainSensor, duration: float, window: int = 100) -> tuple:
    """
    Collect correlation samples over time.
    Returns (correlations_ab, correlations_bc, correlations_ac, means_a, means_b, means_c)
    """
    means_a, means_b, means_c = [], [], []
    correlations_ab, correlations_bc, correlations_ac = [], [], []

    start = time.time()
    while time.time() - start < duration:
        a, b, c = sensor.read_raw()
        means_a.append(a)
        means_b.append(b)
        means_c.append(c)

        if len(means_a) >= window and len(means_a) % (window // 4) == 0:
            rho_ab = compute_correlation_from_means(means_a, means_b, window)
            rho_bc = compute_correlation_from_means(means_b, means_c, window)
            rho_ac = compute_correlation_from_means(means_a, means_c, window)
            correlations_ab.append(rho_ab)
            correlations_bc.append(rho_bc)
            correlations_ac.append(rho_ac)

    return (correlations_ab, correlations_bc, correlations_ac,
            means_a, means_b, means_c)


def run_formal_validation():
    """Run validation experiments for formally proven predictions."""

    print("="*70)
    print("EXPERIMENT 12: FORMAL VERIFICATION VALIDATION")
    print("="*70)
    print()
    print("Testing predictions from Lean 4 formal proofs:")
    print("  1. Measure actual σ (compare to assumed 0.027)")
    print("  2. Verify k_eff INCREASES under attack")
    print("  3. Test |Δρ_crypto| < 3σ threshold")
    print("  4. Validate k_eff deviation detection")
    print()

    config = SensorConfig(n_iterations=5000)
    sensor = StrainSensor(config=config)

    results = {
        'experiment': 'exp12_formal_validation',
        'timestamp': datetime.now().isoformat()
    }

    # =========================================================================
    # TEST 1: Measure actual σ (noise floor)
    # =========================================================================
    print("[TEST 1] MEASURING NOISE FLOOR σ")
    print("-"*50)
    print("Collecting baseline correlations (idle workload, 30s)...")

    idle = WorkloadGenerator("idle")
    idle.start()
    time.sleep(2)  # Warm up

    corr_ab, corr_bc, corr_ac, _, _, _ = collect_correlations(sensor, duration=30.0)

    idle.stop()

    sigma_ab = np.std(corr_ab)
    sigma_bc = np.std(corr_bc)
    sigma_ac = np.std(corr_ac)
    sigma_mean = np.mean([sigma_ab, sigma_bc, sigma_ac])

    rho_mean_ab = np.mean(corr_ab)
    rho_mean_bc = np.mean(corr_bc)
    rho_mean_ac = np.mean(corr_ac)
    rho_baseline = np.mean([rho_mean_ab, rho_mean_bc, rho_mean_ac])

    print(f"\nBaseline measurements:")
    print(f"  ρ(A,B): {rho_mean_ab:.4f} ± {sigma_ab:.4f}")
    print(f"  ρ(B,C): {rho_mean_bc:.4f} ± {sigma_bc:.4f}")
    print(f"  ρ(A,C): {rho_mean_ac:.4f} ± {sigma_ac:.4f}")
    print(f"  Mean σ: {sigma_mean:.4f}")
    print(f"  Assumed σ: 0.027")
    print(f"  Ratio (measured/assumed): {sigma_mean/0.027:.2f}×")

    # Compute baseline k_eff
    k_eff_baseline = compute_k_eff(rho_baseline, k=3)
    print(f"\nBaseline k_eff: {k_eff_baseline:.2f}")
    print(f"  (Formula: k_eff = 3 / (1 + ρ×2) = 3 / {1 + rho_baseline*2:.3f})")

    results['test1_noise_floor'] = {
        'sigma_ab': float(sigma_ab),
        'sigma_bc': float(sigma_bc),
        'sigma_ac': float(sigma_ac),
        'sigma_mean': float(sigma_mean),
        'sigma_assumed': 0.027,
        'rho_baseline': float(rho_baseline),
        'k_eff_baseline': float(k_eff_baseline),
        'n_samples': len(corr_ab)
    }

    # =========================================================================
    # TEST 2: Verify k_eff INCREASES under crypto attack
    # =========================================================================
    print("\n" + "="*70)
    print("[TEST 2] VERIFY k_eff INCREASES UNDER ATTACK")
    print("-"*50)
    print("Formal prediction: attack_increases_k_eff (CorrelationFingerprint.lean:125)")
    print("Starting crypto mining workload...")

    crypto = WorkloadGenerator("crypto", intensity=0.8)
    crypto.start()
    time.sleep(2)

    corr_ab_c, corr_bc_c, corr_ac_c, _, _, _ = collect_correlations(sensor, duration=30.0)

    crypto.stop()

    rho_crypto_ab = np.mean(corr_ab_c)
    rho_crypto_bc = np.mean(corr_bc_c)
    rho_crypto_ac = np.mean(corr_ac_c)
    rho_crypto = np.mean([rho_crypto_ab, rho_crypto_bc, rho_crypto_ac])

    delta_rho = rho_crypto - rho_baseline
    k_eff_crypto = compute_k_eff(rho_crypto, k=3)
    delta_k_eff = k_eff_crypto - k_eff_baseline

    print(f"\nCrypto attack measurements:")
    print(f"  ρ(A,B): {rho_crypto_ab:.4f}")
    print(f"  ρ(B,C): {rho_crypto_bc:.4f}")
    print(f"  ρ(A,C): {rho_crypto_ac:.4f}")
    print(f"  Mean ρ: {rho_crypto:.4f}")
    print(f"  Δρ: {delta_rho:.4f}")
    print(f"\nk_eff under attack: {k_eff_crypto:.2f}")
    print(f"  Δk_eff: {delta_k_eff:+.2f}")

    k_eff_increased = k_eff_crypto > k_eff_baseline
    print(f"\n*** PREDICTION: k_eff increases under attack ***")
    print(f"    Result: k_eff_crypto ({k_eff_crypto:.2f}) {'>' if k_eff_increased else '<='} k_eff_baseline ({k_eff_baseline:.2f})")
    print(f"    VERIFIED: {'YES ✓' if k_eff_increased else 'NO ✗'}")

    results['test2_k_eff'] = {
        'rho_crypto': float(rho_crypto),
        'delta_rho': float(delta_rho),
        'k_eff_baseline': float(k_eff_baseline),
        'k_eff_crypto': float(k_eff_crypto),
        'delta_k_eff': float(delta_k_eff),
        'prediction_verified': bool(k_eff_increased)
    }

    # =========================================================================
    # TEST 3: Test single-sample 3σ detection threshold
    # =========================================================================
    print("\n" + "="*70)
    print("[TEST 3] SINGLE-SAMPLE 3σ DETECTION THRESHOLD")
    print("-"*50)
    print("Formal prediction: crypto_not_single_sample_detectable (CorrelationFingerprint.lean:173)")

    three_sigma = 3 * sigma_mean
    abs_delta_rho = abs(delta_rho)

    print(f"\n  |Δρ_crypto|: {abs_delta_rho:.4f}")
    print(f"  3σ threshold: {three_sigma:.4f}")
    print(f"  |Δρ| < 3σ: {abs_delta_rho < three_sigma}")

    single_sample_fails = abs_delta_rho < three_sigma
    print(f"\n*** PREDICTION: Single-sample 3σ detection fails ***")
    print(f"    Result: |Δρ| = {abs_delta_rho:.4f} {'<' if single_sample_fails else '>='} 3σ = {three_sigma:.4f}")
    print(f"    VERIFIED: {'YES ✓' if single_sample_fails else 'NO ✗'}")

    # Z-score for actual detection
    z_score = abs_delta_rho / sigma_mean if sigma_mean > 0 else 0
    print(f"\n  Actual z-score: {z_score:.2f}σ")
    print(f"  Samples needed for 3σ detection: ~{(3/z_score)**2 if z_score > 0 else 'inf':.0f}")

    results['test3_threshold'] = {
        'delta_rho': float(delta_rho),
        'abs_delta_rho': float(abs_delta_rho),
        'sigma': float(sigma_mean),
        'three_sigma': float(three_sigma),
        'z_score': float(z_score),
        'single_sample_fails': bool(single_sample_fails)
    }

    # =========================================================================
    # TEST 4: k_eff deviation detection
    # =========================================================================
    print("\n" + "="*70)
    print("[TEST 4] k_eff DEVIATION DETECTION")
    print("-"*50)
    print("Formal prediction: crypto_detectable_via_k_eff (CorrelationFingerprint.lean:185)")

    # Compute k_eff for each individual correlation sample
    k_eff_baseline_samples = [compute_k_eff(r, k=3) for r in corr_ab]
    k_eff_crypto_samples = [compute_k_eff(r, k=3) for r in corr_ab_c]

    k_eff_baseline_mean = np.mean(k_eff_baseline_samples)
    k_eff_baseline_std = np.std(k_eff_baseline_samples)
    k_eff_crypto_mean = np.mean(k_eff_crypto_samples)

    k_eff_z = (k_eff_crypto_mean - k_eff_baseline_mean) / (k_eff_baseline_std + 1e-10)

    print(f"\nk_eff statistics:")
    print(f"  Baseline: {k_eff_baseline_mean:.2f} ± {k_eff_baseline_std:.2f}")
    print(f"  Crypto:   {k_eff_crypto_mean:.2f}")
    print(f"  z-score:  {k_eff_z:.2f}σ")

    k_eff_detectable = abs(k_eff_z) > 3.0
    print(f"\n*** PREDICTION: k_eff deviation is detectable ***")
    print(f"    Result: |z| = {abs(k_eff_z):.2f} {'>' if k_eff_detectable else '<='} 3.0")
    print(f"    VERIFIED: {'YES ✓' if k_eff_detectable else 'NO ✗ (may need more samples)'}")

    results['test4_k_eff_detection'] = {
        'k_eff_baseline_mean': float(k_eff_baseline_mean),
        'k_eff_baseline_std': float(k_eff_baseline_std),
        'k_eff_crypto_mean': float(k_eff_crypto_mean),
        'k_eff_z_score': float(k_eff_z),
        'k_eff_detectable': bool(k_eff_detectable)
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FORMAL VALIDATION SUMMARY")
    print("="*70)
    print()
    print(f"| Test | Prediction | Result |")
    print(f"|------|------------|--------|")
    print(f"| Noise floor σ | 0.027 | {sigma_mean:.4f} ({sigma_mean/0.027:.1f}×) |")
    print(f"| k_eff increases | True | {'✓' if k_eff_increased else '✗'} |")
    print(f"| Single-sample fails | True | {'✓' if single_sample_fails else '✗'} |")
    print(f"| k_eff detection | True | {'✓' if k_eff_detectable else '✗'} |")
    print()

    # Implications
    print("IMPLICATIONS:")
    if sigma_mean > 0.027 * 1.5:
        print(f"  - σ is {sigma_mean/0.027:.1f}× higher than assumed")
        print(f"    → Detection thresholds need recalibration")
    if k_eff_increased:
        print(f"  - k_eff framework validated for negative-ρ regime")
        print(f"    → Attacks increase divergence (k_eff ↑)")
    if single_sample_fails:
        print(f"  - Single-sample detection insufficient (z={z_score:.2f}σ)")
        print(f"    → Transformer/CUSUM approach validated")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp12_formal_validation_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_formal_validation()
