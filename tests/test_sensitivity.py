#!/usr/bin/env python3
"""
Empirical Sensitivity Characterization

Test: What is the minimum detectable workload intensity?

Methodology:
1. Establish baseline (no workload)
2. Test detection at various intensities (10%, 30%, 50%, 70%, 90%)
3. For each intensity, run multiple trials
4. Report detection rate and mean z-score

Author: CIRIS L3C
"""

import numpy as np
import cupy as cp
import time
import threading
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ossicle import OssicleKernel, OssicleConfig


class CryptoWorkload:
    """Simulated crypto mining at variable intensity."""

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

    def __init__(self, intensity: float = 0.5):
        self.intensity = intensity
        self.running = False
        self.thread = None
        self.module = cp.RawModule(code=self.KERNEL)
        self.kernel = self.module.get_function('hash_compute')
        # Scale size and rounds with intensity
        self.size = int(512 * 1024 * intensity)  # Up to 512K elements
        self.rounds = int(50 + 150 * intensity)   # 50-200 rounds

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.3)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run(self):
        if self.size < 256:
            return
        data = cp.random.randint(0, 2**31, self.size, dtype=cp.uint32)
        nonce = cp.arange(256, dtype=cp.uint32)
        block_size = 256
        grid_size = (self.size + block_size - 1) // block_size
        while self.running:
            self.kernel((grid_size,), (block_size,),
                       (data, nonce, cp.int32(self.size), cp.int32(self.rounds)))
            cp.cuda.Stream.null.synchronize()
            nonce = nonce + 256


def collect_samples(sensor, duration):
    """Collect samples and compute mean correlation."""
    ha, hb, hc = [], [], []
    start = time.time()
    while time.time() - start < duration:
        a, b, c = sensor.step()
        ha.append(a)
        hb.append(b)
        hc.append(c)

    arr_a, arr_b, arr_c = np.array(ha), np.array(hb), np.array(hc)

    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        r = np.corrcoef(x, y)[0, 1]
        return r if not np.isnan(r) else 0.0

    rho_ab = safe_corr(arr_a, arr_b)
    rho_bc = safe_corr(arr_b, arr_c)
    rho_ac = safe_corr(arr_a, arr_c)

    return (rho_ab + rho_bc + rho_ac) / 3


def run_sensitivity_test():
    print("=" * 70)
    print("EMPIRICAL SENSITIVITY CHARACTERIZATION")
    print("=" * 70)
    print()

    config = OssicleConfig()
    sensor = OssicleKernel(config)

    trial_duration = 5.0
    n_baseline_trials = 10
    n_workload_trials = 8

    # Phase 1: Establish baseline
    print("Phase 1: Establishing baseline...")
    baseline_corrs = []

    for i in range(n_baseline_trials):
        sensor.reset()
        corr = collect_samples(sensor, trial_duration)
        baseline_corrs.append(corr)
        print(f"  Baseline {i+1}/{n_baseline_trials}: {corr:+.6f}")

    baseline_mean = np.mean(baseline_corrs)
    baseline_std = np.std(baseline_corrs)

    print(f"\nBaseline: {baseline_mean:.6f} ± {baseline_std:.6f}")
    print(f"2σ threshold: {2*baseline_std:.6f}")
    print(f"3σ threshold: {3*baseline_std:.6f}")

    # Phase 2: Test at various intensities
    intensities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("\n" + "=" * 70)
    print("Phase 2: Testing workload intensities")
    print("=" * 70)

    results = []

    for intensity in intensities:
        print(f"\nTesting intensity: {intensity*100:.0f}%")

        workload = CryptoWorkload(intensity=intensity)
        z_scores = []
        detections_2sigma = 0
        detections_3sigma = 0

        for i in range(n_workload_trials):
            sensor.reset()
            workload.start()
            corr = collect_samples(sensor, trial_duration)
            workload.stop()

            z = abs(corr - baseline_mean) / baseline_std if baseline_std > 0 else 0
            z_scores.append(z)

            if z > 2:
                detections_2sigma += 1
            if z > 3:
                detections_3sigma += 1

            print(f"  Trial {i+1}: corr={corr:+.6f}, z={z:.2f}σ")
            time.sleep(0.3)

        mean_z = np.mean(z_scores)
        detection_rate_2s = detections_2sigma / n_workload_trials
        detection_rate_3s = detections_3sigma / n_workload_trials

        results.append({
            'intensity': intensity,
            'mean_z': mean_z,
            'max_z': max(z_scores),
            'detection_rate_2sigma': detection_rate_2s,
            'detection_rate_3sigma': detection_rate_3s
        })

    # Summary
    print("\n" + "=" * 70)
    print("SENSITIVITY SUMMARY")
    print("=" * 70)
    print()
    print("| Intensity | Mean z | Max z  | Det@2σ | Det@3σ |")
    print("|-----------|--------|--------|--------|--------|")

    for r in results:
        print(f"| {r['intensity']*100:6.0f}%   | {r['mean_z']:6.2f} | {r['max_z']:6.2f} | {r['detection_rate_2sigma']*100:5.0f}%  | {r['detection_rate_3sigma']*100:5.0f}%  |")

    # Find minimum detectable intensity
    print("\n" + "-" * 70)

    min_2sigma = None
    min_3sigma = None

    for r in results:
        if r['detection_rate_2sigma'] >= 0.5 and min_2sigma is None:
            min_2sigma = r['intensity']
        if r['detection_rate_3sigma'] >= 0.5 and min_3sigma is None:
            min_3sigma = r['intensity']

    if min_2sigma:
        print(f"Minimum detectable at 2σ (50% rate): {min_2sigma*100:.0f}%")
    else:
        print("2σ detection: Not reliably achieved at any intensity")

    if min_3sigma:
        print(f"Minimum detectable at 3σ (50% rate): {min_3sigma*100:.0f}%")
    else:
        print("3σ detection: Not reliably achieved at any intensity")

    # Best detection
    best = max(results, key=lambda x: x['mean_z'])
    print(f"\nBest detection: {best['intensity']*100:.0f}% intensity → z={best['mean_z']:.2f}σ mean")

    return results


if __name__ == "__main__":
    run_sensitivity_test()
