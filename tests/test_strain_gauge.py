#!/usr/bin/env python3
"""
Strain Gauge Sensitivity and Discrimination Tests

Scientific validation of strain_gauge.py claims:
1. Workload detection sensitivity (can we detect workloads?)
2. Workload discrimination (can we tell them apart?)

All tests follow null hypothesis methodology with p-values.

Author: CIRIS L3C
License: BSL 1.1
"""

import numpy as np
import cupy as cp
import time
import threading
from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from strain_gauge import StrainGauge, StrainGaugeConfig


@dataclass
class TestResult:
    """Result of a hypothesis test."""
    claim: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    effect_size: float
    reject_null: bool
    alpha: float = 0.05
    notes: str = ""

    def __str__(self):
        verdict = "REJECT H0 (claim supported)" if self.reject_null else "FAIL TO REJECT H0"
        return f"""
{'='*70}
CLAIM: {self.claim}
{'='*70}
H0 (null):        {self.null_hypothesis}
H1 (alternative): {self.alternative_hypothesis}

Test statistic: {self.test_statistic:.4f}
P-value:        {self.p_value:.6f}
Effect size:    {self.effect_size:.4f}
Alpha:          {self.alpha}

VERDICT: {verdict}
{self.notes}
{'='*70}
"""


class CryptoWorkload:
    """Simulated crypto mining workload."""

    KERNEL = r'''
    extern "C" __global__ void hash_compute(
        unsigned int* data, unsigned int* nonce, int n, int rounds
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        unsigned int h = nonce[idx % 256];
        unsigned int d = data[idx];
        for (int r = 0; r < rounds; r++) {
            h = ((h << 7) | (h >> 25)) ^ d;
            h = h + ((h << 3) ^ (h >> 5));
            d = d ^ h;
            d = ((d << 11) | (d >> 21)) + h;
            h = h ^ ((d << 13) | (d >> 19));
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
        self.module = cp.RawModule(code=self.KERNEL)
        self.kernel = self.module.get_function('hash_compute')
        self.size = int(1024 * 1024 * intensity)
        self.rounds = int(100 * intensity)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # Let it stabilize

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        time.sleep(0.2)

    def _run(self):
        data = cp.random.randint(0, 2**31, self.size, dtype=cp.uint32)
        nonce = cp.arange(256, dtype=cp.uint32)
        block_size = 256
        grid_size = (self.size + block_size - 1) // block_size
        while self.running:
            self.kernel((grid_size,), (block_size,),
                       (data, nonce, cp.int32(self.size), cp.int32(self.rounds)))
            cp.cuda.Stream.null.synchronize()
            nonce = nonce + 256


class MemoryWorkload:
    """Memory bandwidth workload."""

    def __init__(self, intensity: float = 0.7):
        self.intensity = intensity
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.5)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        time.sleep(0.2)

    def _run(self):
        size = int(2048 * self.intensity)
        src = cp.random.randn(size * size, dtype=cp.float32)
        dst = cp.zeros(size * size, dtype=cp.float32)
        while self.running:
            dst[:] = src
            src[:] = dst
            cp.cuda.Stream.null.synchronize()


class ComputeWorkload:
    """Pure compute (matrix multiply) workload."""

    def __init__(self, intensity: float = 0.7):
        self.intensity = intensity
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.5)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        time.sleep(0.2)

    def _run(self):
        size = int(1024 * self.intensity)
        a = cp.random.randn(size, size, dtype=cp.float32)
        b = cp.random.randn(size, size, dtype=cp.float32)
        while self.running:
            c = cp.matmul(a, b)
            cp.cuda.Stream.null.synchronize()
            a = c


def collect_readings(gauge: StrainGauge, duration: float) -> List[dict]:
    """Collect strain readings for specified duration."""
    readings = []
    start = time.time()

    while time.time() - start < duration:
        reading = gauge.read()
        readings.append({
            'timing_us': reading.timing_mean_us,
            'timing_std_us': reading.timing_std_us,
            'timing_z': reading.timing_z,
            'k_eff': reading.k_eff,
            'acf': reading.acf,
            'detected': reading.detected,
            'timestamp': reading.timestamp
        })

    return readings


def test_workload_sensitivity():
    """
    TEST 1: Workload Detection Sensitivity

    H0: Timing/k_eff under workload = Timing/k_eff at baseline
    H1: Workload creates detectable change

    Uses the new strain gauge with dt=0.025 critical point.
    """
    print("\n" + "="*70)
    print("TEST 1: WORKLOAD DETECTION SENSITIVITY")
    print("="*70)
    print("\nUsing strain_gauge.py with dt=0.025 (critical point)")

    config = StrainGaugeConfig(
        warm_up_duration=10.0,  # Shorter for testing
        warm_up_enabled=True
    )
    gauge = StrainGauge(config)

    n_trials = 8
    trial_duration = 5.0
    intensities = [0.3, 0.5, 0.7, 0.9]

    # Phase 1: Establish baseline (with warm-up and calibration)
    print("\nPhase 1: Calibrating and establishing baseline...")
    gauge.calibrate(duration=10.0)

    baseline_timings = []
    baseline_k_effs = []

    for i in range(n_trials):
        gauge.lorenz.reset()
        readings = collect_readings(gauge, trial_duration)

        mean_timing = np.mean([r['timing_us'] for r in readings])
        mean_k_eff = np.mean([r['k_eff'] for r in readings])

        baseline_timings.append(mean_timing)
        baseline_k_effs.append(mean_k_eff)

        print(f"  Baseline {i+1}/{n_trials}: timing={mean_timing:.1f}μs, k_eff={mean_k_eff:.4f}")

    baseline_timing_mean = np.mean(baseline_timings)
    baseline_timing_std = np.std(baseline_timings)
    baseline_k_eff_mean = np.mean(baseline_k_effs)
    baseline_k_eff_std = np.std(baseline_k_effs)

    print(f"\nBaseline timing: {baseline_timing_mean:.1f} ± {baseline_timing_std:.1f} μs")
    print(f"Baseline k_eff:  {baseline_k_eff_mean:.4f} ± {baseline_k_eff_std:.4f}")

    # Phase 2: Test crypto workload at various intensities
    print("\n" + "-"*70)
    print("Phase 2: Testing crypto workload at various intensities")
    print("-"*70)

    results = []

    for intensity in intensities:
        print(f"\nIntensity: {intensity*100:.0f}%")

        workload = CryptoWorkload(intensity=intensity)
        timing_z_scores = []
        k_eff_z_scores = []

        for i in range(n_trials):
            gauge.lorenz.reset()

            workload.start()
            readings = collect_readings(gauge, trial_duration)
            workload.stop()

            mean_timing = np.mean([r['timing_us'] for r in readings])
            mean_k_eff = np.mean([r['k_eff'] for r in readings])

            timing_z = abs(mean_timing - baseline_timing_mean) / baseline_timing_std if baseline_timing_std > 0 else 0
            k_eff_z = abs(mean_k_eff - baseline_k_eff_mean) / baseline_k_eff_std if baseline_k_eff_std > 0 else 0

            timing_z_scores.append(timing_z)
            k_eff_z_scores.append(k_eff_z)

            print(f"  Trial {i+1}: timing_z={timing_z:.2f}, k_eff_z={k_eff_z:.2f}")

        results.append({
            'intensity': intensity,
            'timing_z_mean': np.mean(timing_z_scores),
            'timing_z_max': np.max(timing_z_scores),
            'k_eff_z_mean': np.mean(k_eff_z_scores),
            'k_eff_z_max': np.max(k_eff_z_scores),
            'timing_z_scores': timing_z_scores,
            'k_eff_z_scores': k_eff_z_scores
        })

    # Summary
    print("\n" + "="*70)
    print("SENSITIVITY SUMMARY")
    print("="*70)
    print()
    print("| Intensity | Timing z (mean) | Timing z (max) | k_eff z (mean) | k_eff z (max) |")
    print("|-----------|-----------------|----------------|----------------|---------------|")

    for r in results:
        print(f"| {r['intensity']*100:6.0f}%   | {r['timing_z_mean']:15.2f} | {r['timing_z_max']:14.2f} | {r['k_eff_z_mean']:14.2f} | {r['k_eff_z_max']:13.2f} |")

    # Statistical test: Are workload z-scores > baseline (which would be ~1)?
    all_timing_z = []
    for r in results:
        all_timing_z.extend(r['timing_z_scores'])

    # One-sample t-test: Is mean z-score > 2 (detection threshold)?
    t_stat, p_value = stats.ttest_1samp(all_timing_z, 2.0)
    mean_z = np.mean(all_timing_z)

    # Detection rate
    detections = sum(1 for z in all_timing_z if z > 2.0)
    detection_rate = detections / len(all_timing_z)

    result = TestResult(
        claim="Workload detection via timing",
        null_hypothesis="Mean timing z-score ≤ 2.0 (not detectable)",
        alternative_hypothesis="Mean timing z-score > 2.0 (detectable)",
        test_statistic=t_stat,
        p_value=p_value / 2,  # One-tailed
        effect_size=mean_z,
        reject_null=mean_z > 2.0 and p_value < 0.05,
        notes=f"Mean z-score across all trials: {mean_z:.2f}\n"
              f"Detection rate (z > 2.0): {detection_rate*100:.0f}% ({detections}/{len(all_timing_z)})\n"
              f"Best intensity: {max(results, key=lambda x: x['timing_z_mean'])['intensity']*100:.0f}%"
    )

    print(result)
    return results


def test_workload_discrimination():
    """
    TEST 2: Workload Type Discrimination

    H0: Different workload types produce indistinguishable signals
    H1: Different workload types produce distinguishable signals

    Tests: crypto vs memory vs compute
    """
    print("\n" + "="*70)
    print("TEST 2: WORKLOAD TYPE DISCRIMINATION")
    print("="*70)

    config = StrainGaugeConfig(
        warm_up_duration=10.0,
        warm_up_enabled=True
    )
    gauge = StrainGauge(config)

    n_trials = 8
    trial_duration = 5.0
    intensity = 0.7

    # Calibrate (includes warm-up)
    print("\nCalibrating...")
    gauge.calibrate(duration=10.0)

    # Collect baseline
    print("\nEstablishing baseline...")
    baseline_readings = []
    for i in range(4):
        gauge.lorenz.reset()
        readings = collect_readings(gauge, trial_duration)
        baseline_readings.extend(readings)

    baseline_timing = np.mean([r['timing_us'] for r in baseline_readings])
    baseline_k_eff = np.mean([r['k_eff'] for r in baseline_readings])

    print(f"Baseline: timing={baseline_timing:.1f}μs, k_eff={baseline_k_eff:.4f}")

    workloads = {
        'crypto': CryptoWorkload(intensity=intensity),
        'memory': MemoryWorkload(intensity=intensity),
        'compute': ComputeWorkload(intensity=intensity)
    }

    workload_data = {}

    for name, workload in workloads.items():
        print(f"\nTesting {name} workload...")

        timing_deltas = []
        k_eff_deltas = []
        timing_stds = []

        for i in range(n_trials):
            gauge.lorenz.reset()

            workload.start()
            readings = collect_readings(gauge, trial_duration)
            workload.stop()

            timings = [r['timing_us'] for r in readings]
            k_effs = [r['k_eff'] for r in readings]

            timing_delta = np.mean(timings) - baseline_timing
            k_eff_delta = np.mean(k_effs) - baseline_k_eff
            timing_std = np.std(timings)

            timing_deltas.append(timing_delta)
            k_eff_deltas.append(k_eff_delta)
            timing_stds.append(timing_std)

            print(f"  Trial {i+1}: Δtiming={timing_delta:+.2f}μs, Δk_eff={k_eff_delta:+.4f}, σ_timing={timing_std:.2f}μs")

        workload_data[name] = {
            'timing_deltas': timing_deltas,
            'k_eff_deltas': k_eff_deltas,
            'timing_stds': timing_stds
        }

    # Statistical tests for discrimination
    print("\n" + "-"*70)
    print("DISCRIMINATION ANALYSIS")
    print("-"*70)

    pairs = [('crypto', 'memory'), ('crypto', 'compute'), ('memory', 'compute')]

    print("\nTiming delta comparison:")
    for w1, w2 in pairs:
        t_stat, p_val = stats.ttest_ind(
            workload_data[w1]['timing_deltas'],
            workload_data[w2]['timing_deltas']
        )
        mean1 = np.mean(workload_data[w1]['timing_deltas'])
        mean2 = np.mean(workload_data[w2]['timing_deltas'])
        significant = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {w1} vs {w2}: Δ={mean1:.2f}μs vs {mean2:.2f}μs, p={p_val:.4f} {significant}")

    print("\nTiming variance comparison:")
    for w1, w2 in pairs:
        stat, p_val = stats.levene(
            workload_data[w1]['timing_stds'],
            workload_data[w2]['timing_stds']
        )
        mean1 = np.mean(workload_data[w1]['timing_stds'])
        mean2 = np.mean(workload_data[w2]['timing_stds'])
        significant = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {w1} vs {w2}: σ={mean1:.2f}μs vs {mean2:.2f}μs, p={p_val:.4f} {significant}")

    # ANOVA test for overall discrimination
    f_stat_timing, p_anova_timing = stats.f_oneway(
        workload_data['crypto']['timing_deltas'],
        workload_data['memory']['timing_deltas'],
        workload_data['compute']['timing_deltas']
    )

    f_stat_std, p_anova_std = stats.f_oneway(
        workload_data['crypto']['timing_stds'],
        workload_data['memory']['timing_stds'],
        workload_data['compute']['timing_stds']
    )

    # Summary
    print("\n" + "="*70)
    print("DISCRIMINATION SUMMARY")
    print("="*70)

    print("\nWorkload signatures (mean ± std):")
    print("| Workload | Δ Timing (μs)    | σ Timing (μs)    | Δ k_eff          |")
    print("|----------|------------------|------------------|------------------|")

    for name in ['crypto', 'memory', 'compute']:
        d = workload_data[name]
        print(f"| {name:8s} | {np.mean(d['timing_deltas']):+7.2f} ± {np.std(d['timing_deltas']):5.2f} | "
              f"{np.mean(d['timing_stds']):7.2f} ± {np.std(d['timing_stds']):5.2f} | "
              f"{np.mean(d['k_eff_deltas']):+7.4f} ± {np.std(d['k_eff_deltas']):5.4f} |")

    result = TestResult(
        claim="Workload type discrimination",
        null_hypothesis="All workload types produce same timing signature",
        alternative_hypothesis="Workload types produce different signatures",
        test_statistic=f_stat_timing,
        p_value=p_anova_timing,
        effect_size=f_stat_timing,
        reject_null=p_anova_timing < 0.05,
        notes=f"ANOVA (timing delta): F={f_stat_timing:.2f}, p={p_anova_timing:.4f}\n"
              f"ANOVA (timing variance): F={f_stat_std:.2f}, p={p_anova_std:.4f}\n"
              f"Discrimination possible: {p_anova_timing < 0.05 or p_anova_std < 0.05}"
    )

    print(result)
    return workload_data


def test_acf_at_criticality():
    """
    TEST 3: ACF Monitoring at Critical Point

    Verify that dt=0.025 produces ACF ~0.5 (criticality)
    """
    print("\n" + "="*70)
    print("TEST 3: ACF AT CRITICAL POINT")
    print("="*70)

    config = StrainGaugeConfig(warm_up_enabled=False)
    gauge = StrainGauge(config)

    print(f"\nConfiguration: dt={config.dt}")
    print("Calibrating and collecting samples to measure ACF...")

    gauge.calibrate(duration=5.0)
    readings = collect_readings(gauge, 10.0)

    acf_values = [r['acf'] for r in readings]
    mean_acf = np.mean(acf_values)
    std_acf = np.std(acf_values)

    print(f"\nACF: {mean_acf:.3f} ± {std_acf:.3f}")
    print(f"Target: 0.5 (criticality)")
    print(f"State: {gauge.lorenz.get_system_state()}")

    # Test if ACF is in critical range (0.3 - 0.7)
    in_critical_range = 0.3 <= mean_acf <= 0.7

    result = TestResult(
        claim="System operates at criticality (ACF ~0.5)",
        null_hypothesis="ACF outside critical range (0.3-0.7)",
        alternative_hypothesis="ACF in critical range (0.3-0.7)",
        test_statistic=mean_acf,
        p_value=0.0 if in_critical_range else 1.0,
        effect_size=abs(mean_acf - 0.5),
        reject_null=in_critical_range,
        notes=f"ACF = {mean_acf:.3f}\n"
              f"Target range: 0.3-0.7\n"
              f"Distance from optimal (0.5): {abs(mean_acf - 0.5):.3f}"
    )

    print(result)
    return mean_acf


def run_all_tests():
    """Run complete test suite."""
    print("="*70)
    print("STRAIN GAUGE VALIDATION SUITE")
    print("Based on RATCHET Experiments 68-116")
    print("="*70)

    results = {}

    # Test 1: Sensitivity
    results['sensitivity'] = test_workload_sensitivity()

    # Test 2: Discrimination
    results['discrimination'] = test_workload_discrimination()

    # Test 3: ACF criticality
    results['acf'] = test_acf_at_criticality()

    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)

    return results


if __name__ == "__main__":
    run_all_tests()
