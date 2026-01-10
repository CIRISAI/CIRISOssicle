#!/usr/bin/env python3
"""
Experiment 8: Workload Fingerprinting and Tamper Detection

Create correlation "fingerprints" for different GPU workloads and detect
when unauthorized workloads run concurrently.

Workload types (synthetic, mimic real applications):
1. "idle" - No load (baseline)
2. "transformer" - Attention-like: matmul + softmax + matmul
3. "convolution" - Strided memory access patterns
4. "training" - Forward + backward (gradient compute)
5. "mining" - Hash-like repeated compute

Goal: Can we detect if someone runs "mining" alongside "transformer"?

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
from strain_sensor import StrainSensor, SensorConfig


class WorkloadGenerator:
    """Generate different types of GPU workloads."""

    def __init__(self):
        self.running = False
        self.thread = None
        self.workload_type = None

    def start(self, workload_type: str, intensity: float = 0.7):
        """Start a workload in background."""
        self.workload_type = workload_type
        self.intensity = intensity
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        """Stop the workload."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)

    def _run(self):
        """Run the workload."""
        size = int(1024 + 3072 * self.intensity)

        if self.workload_type == "idle":
            while self.running:
                time.sleep(0.1)

        elif self.workload_type == "transformer":
            # Attention-like: Q @ K.T -> softmax -> @ V
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

        elif self.workload_type == "convolution":
            # Simulated strided memory access
            data = cp.random.randn(size * size, dtype=cp.float32)
            kernel = cp.random.randn(64, dtype=cp.float32)

            while self.running:
                # Strided access pattern
                for stride in [1, 4, 16, 64]:
                    view = data[::stride][:len(kernel)]
                    result = cp.sum(view * kernel[:len(view)])
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.005)

        elif self.workload_type == "training":
            # Forward + backward simulation
            w = cp.random.randn(size, size // 4, dtype=cp.float32)
            x = cp.random.randn(size // 4, 64, dtype=cp.float32)

            while self.running:
                # Forward
                y = cp.matmul(w, x)
                y = cp.maximum(y, 0)  # ReLU
                # Backward (gradient)
                grad = cp.ones_like(y)
                grad = grad * (y > 0).astype(cp.float32)
                grad_w = cp.matmul(grad, x.T)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.005)

        elif self.workload_type == "mining":
            # Hash-like repeated compute (many iterations, same data)
            data = cp.random.randint(0, 2**32, size=(size,), dtype=cp.uint32)

            while self.running:
                for _ in range(100):
                    # XOR and shift operations (hash-like)
                    data = data ^ (data << 13)
                    data = data ^ (data >> 17)
                    data = data ^ (data << 5)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.005)

        elif self.workload_type == "memory_stress":
            # Pure memory bandwidth
            a = cp.random.randn(size * size * 4, dtype=cp.float32)
            b = cp.zeros_like(a)

            while self.running:
                cp.copyto(b, a)
                cp.cuda.Stream.null.synchronize()
                time.sleep(0.005)


def collect_fingerprint(sensor, duration: float = 30.0, window_size: int = 500,
                        label: str = ""):
    """Collect correlation fingerprint for current GPU state."""
    means_a, means_b, means_c = [], [], []
    correlations = []
    timestamps = []

    start = time.time()
    while time.time() - start < duration:
        mean_a, mean_b, mean_c = sensor.read_raw()
        means_a.append(mean_a)
        means_b.append(mean_b)
        means_c.append(mean_c)
        timestamps.append(time.time() - start)

        n = len(means_a)
        if n >= window_size and n % (window_size // 4) == 0:
            wa = np.array(means_a[-window_size:])
            wb = np.array(means_b[-window_size:])
            wc = np.array(means_c[-window_size:])

            rho_ab = np.corrcoef(wa, wb)[0, 1]
            rho_bc = np.corrcoef(wb, wc)[0, 1]
            rho_ac = np.corrcoef(wa, wc)[0, 1]

            correlations.append({
                'time': timestamps[-1],
                'rho_ab': float(rho_ab),
                'rho_bc': float(rho_bc),
                'rho_ac': float(rho_ac),
            })

    # Compute fingerprint statistics
    if correlations:
        rho_ab = np.array([c['rho_ab'] for c in correlations])
        rho_bc = np.array([c['rho_bc'] for c in correlations])
        rho_ac = np.array([c['rho_ac'] for c in correlations])

        fingerprint = {
            'label': label,
            'n_samples': len(means_a),
            'n_windows': len(correlations),
            'rho_ab_mean': float(np.mean(rho_ab)),
            'rho_ab_std': float(np.std(rho_ab)),
            'rho_bc_mean': float(np.mean(rho_bc)),
            'rho_bc_std': float(np.std(rho_bc)),
            'rho_ac_mean': float(np.mean(rho_ac)),
            'rho_ac_std': float(np.std(rho_ac)),
            # Combined fingerprint vector
            'fingerprint_vector': [
                float(np.mean(rho_ab)),
                float(np.mean(rho_bc)),
                float(np.mean(rho_ac)),
                float(np.std(rho_ab)),
                float(np.std(rho_bc)),
                float(np.std(rho_ac)),
            ],
            'raw_correlations': correlations,
        }
    else:
        fingerprint = {'label': label, 'error': 'No data collected'}

    return fingerprint


def compare_fingerprints(fp1: dict, fp2: dict):
    """Compare two fingerprints, return similarity metrics."""
    if 'error' in fp1 or 'error' in fp2:
        return {'error': 'Invalid fingerprint'}

    v1 = np.array(fp1['fingerprint_vector'])
    v2 = np.array(fp2['fingerprint_vector'])

    # Euclidean distance
    distance = np.linalg.norm(v1 - v2)

    # Cosine similarity
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

    # Mean correlation difference
    mean_diff = abs(fp1['rho_ab_mean'] - fp2['rho_ab_mean'])

    # Statistical test on raw correlations
    rho1 = [c['rho_ab'] for c in fp1['raw_correlations']]
    rho2 = [c['rho_ab'] for c in fp2['raw_correlations']]
    t_stat, p_value = stats.ttest_ind(rho1, rho2)

    return {
        'euclidean_distance': float(distance),
        'cosine_similarity': float(cosine_sim),
        'mean_rho_diff': float(mean_diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significantly_different': p_value < 0.01,
    }


def run_fingerprint_experiment():
    """Run workload fingerprinting experiment."""
    print("="*70)
    print("EXPERIMENT 8: WORKLOAD FINGERPRINTING")
    print("="*70)
    print()
    print("Creating correlation fingerprints for different workloads.")
    print("Goal: Detect if unauthorized workloads run alongside known workloads.")
    print()

    config = SensorConfig(n_iterations=5000)
    sensor = StrainSensor(config=config)
    workload = WorkloadGenerator()

    fingerprints = {}

    # Collect fingerprints for each workload type
    workload_types = ["idle", "transformer", "training", "mining", "memory_stress"]

    print("[PHASE 1] COLLECTING WORKLOAD FINGERPRINTS")
    print("-"*50)

    for wtype in workload_types:
        print(f"\n{wtype.upper()}:")
        workload.start(wtype, intensity=0.7)
        time.sleep(2)  # Let workload stabilize

        fp = collect_fingerprint(sensor, duration=25.0, label=wtype)
        fingerprints[wtype] = fp

        workload.stop()
        time.sleep(1)

        if 'error' not in fp:
            print(f"  ρ_AB = {fp['rho_ab_mean']:+.4f} ± {fp['rho_ab_std']:.4f}")
            print(f"  ρ_BC = {fp['rho_bc_mean']:+.4f} ± {fp['rho_bc_std']:.4f}")
            print(f"  ρ_AC = {fp['rho_ac_mean']:+.4f} ± {fp['rho_ac_std']:.4f}")

    # Compare fingerprints
    print("\n" + "="*70)
    print("[PHASE 2] FINGERPRINT COMPARISON")
    print("-"*50)

    print(f"\n{'Comparison':<30} {'Distance':<10} {'Cosine':<10} {'p-value':<12} {'Different?':<10}")
    print("-"*75)

    comparisons = []
    for i, w1 in enumerate(workload_types):
        for w2 in workload_types[i+1:]:
            comp = compare_fingerprints(fingerprints[w1], fingerprints[w2])
            comp['pair'] = f"{w1} vs {w2}"
            comparisons.append(comp)

            diff = "YES" if comp.get('significantly_different') else "no"
            print(f"{w1} vs {w2:<20} {comp.get('euclidean_distance', 0):<10.4f} "
                  f"{comp.get('cosine_similarity', 0):<10.4f} "
                  f"{comp.get('p_value', 1):<12.6f} {diff:<10}")

    # Tamper detection test
    print("\n" + "="*70)
    print("[PHASE 3] TAMPER DETECTION TEST")
    print("-"*50)
    print("\nRunning 'transformer' with and without concurrent 'mining'...")

    # Baseline: transformer alone
    print("\n1. Transformer ALONE:")
    workload.start("transformer", intensity=0.7)
    time.sleep(2)
    fp_clean = collect_fingerprint(sensor, duration=20.0, label="transformer_clean")
    workload.stop()
    time.sleep(1)
    print(f"   ρ_AB = {fp_clean['rho_ab_mean']:+.4f} ± {fp_clean['rho_ab_std']:.4f}")

    # Tampered: transformer + mining
    print("\n2. Transformer + MINING (tampered):")
    workload.start("transformer", intensity=0.7)
    time.sleep(1)

    # Start second workload (mining) in another thread
    mining_load = WorkloadGenerator()
    mining_load.start("mining", intensity=0.5)
    time.sleep(2)

    fp_tampered = collect_fingerprint(sensor, duration=20.0, label="transformer_tampered")

    mining_load.stop()
    workload.stop()
    time.sleep(1)
    print(f"   ρ_AB = {fp_tampered['rho_ab_mean']:+.4f} ± {fp_tampered['rho_ab_std']:.4f}")

    # Compare clean vs tampered
    tamper_result = compare_fingerprints(fp_clean, fp_tampered)

    print("\n" + "="*70)
    print("TAMPER DETECTION RESULT")
    print("="*70)
    print()
    print(f"  Clean fingerprint:    ρ_AB = {fp_clean['rho_ab_mean']:+.4f}")
    print(f"  Tampered fingerprint: ρ_AB = {fp_tampered['rho_ab_mean']:+.4f}")
    print(f"  Difference: {tamper_result['mean_rho_diff']:.4f}")
    print(f"  p-value: {tamper_result['p_value']:.6f}")
    print()

    if tamper_result['significantly_different']:
        print("  *** TAMPER DETECTED ***")
        print("  The presence of concurrent mining workload was detected!")
    else:
        print("  Tamper NOT detected (fingerprints too similar)")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp8_fingerprint_{timestamp}.json"

    # Remove raw correlations for smaller output
    for fp in fingerprints.values():
        if 'raw_correlations' in fp:
            del fp['raw_correlations']
    if 'raw_correlations' in fp_clean:
        del fp_clean['raw_correlations']
    if 'raw_correlations' in fp_tampered:
        del fp_tampered['raw_correlations']

    output = {
        'experiment': 'exp8_workload_fingerprint',
        'timestamp': datetime.now().isoformat(),
        'fingerprints': fingerprints,
        'comparisons': comparisons,
        'tamper_test': {
            'clean': fp_clean,
            'tampered': fp_tampered,
            'result': tamper_result,
            'tamper_detected': bool(tamper_result.get('significantly_different', False)),
        },
    }

    # Ensure all bools are Python native
    for comp in output['comparisons']:
        if 'significantly_different' in comp:
            comp['significantly_different'] = bool(comp['significantly_different'])

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == "__main__":
    run_fingerprint_experiment()
