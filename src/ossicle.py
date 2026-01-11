#!/usr/bin/env python3
"""
CIRISOssicle - GPU Tamper Detection Sensor

A 0.75KB GPU sensor that detects unauthorized workloads through
correlation fingerprinting of chaotic oscillators.

VALIDATED FEATURES (January 2026, RTX 4090):
1. Local tamper detection - correlation mean shifts under workload (p=0.007)
2. Reset strategy - 7x sensitivity improvement with periodic resets (p=0.032)
3. Bounded noise floor - σ ≈ 0.003

See VALIDATION_RESULTS.md for full null hypothesis test methodology.

Author: CIRIS L3C (Eric Moore)
License: BSL 1.1
"""

import numpy as np
import cupy as cp
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


# Validated noise floor from testing
VALIDATED_NOISE_FLOOR = 0.003


@dataclass
class OssicleConfig:
    """Configuration for the ossicle sensor."""
    n_cells: int = 64
    n_iterations: int = 500
    n_oscillators: int = 3
    twist_deg: float = 1.1      # Empirically optimal angle
    r_base: float = 3.70        # Base bifurcation parameter
    spacing: float = 0.03       # Spacing between oscillator r values
    coupling: float = 0.05      # Coupling strength (epsilon)

    # Detection thresholds (based on validated noise floor)
    z_threshold: float = 2.0    # 2σ detection
    z_strong: float = 3.0       # 3σ strong detection

    @property
    def memory_bytes(self) -> int:
        return self.n_oscillators * self.n_cells * 4  # float32

    @property
    def memory_kb(self) -> float:
        return self.memory_bytes / 1024


@dataclass
class Detection:
    """Detection result from the sensor."""
    timestamp: float
    mean_correlation: float
    z_score: float
    detected: bool
    strong: bool

    # Raw oscillator means
    mean_a: float
    mean_b: float
    mean_c: float

    # Individual correlations
    rho_ab: float
    rho_bc: float
    rho_ac: float


class OssicleKernel:
    """
    Core CUDA kernel for the ossicle sensor.

    Uses 3 coupled chaotic oscillators with 1.1 degree twist angle.
    Memory footprint: 0.75 KB (3 oscillators × 64 cells × 4 bytes).
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

    def __init__(self, config: OssicleConfig = None):
        self.config = config or OssicleConfig()
        cfg = self.config

        self.r_a = cfg.r_base
        self.r_b = cfg.r_base + cfg.spacing
        self.r_c = cfg.r_base + 2 * cfg.spacing

        self.twist_ab = np.radians(cfg.twist_deg)
        self.twist_bc = np.radians(cfg.twist_deg)

        # Compile kernel
        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('ossicle_step')

        # Initialize states
        self.reset()

        # Execution config
        self.block_size = 64
        self.grid_size = (cfg.n_cells + self.block_size - 1) // self.block_size

    def reset(self):
        """
        Reset oscillator states.

        IMPORTANT: Validated to improve sensitivity by 7x (p=0.032).
        Call this between measurement trials for best detection.
        """
        n = self.config.n_cells
        self.state_a = cp.random.uniform(0.1, 0.9, n, dtype=cp.float32)
        self.state_b = cp.random.uniform(0.1, 0.9, n, dtype=cp.float32)
        self.state_c = cp.random.uniform(0.1, 0.9, n, dtype=cp.float32)

    def step(self) -> Tuple[float, float, float]:
        """Execute one step and return oscillator means."""
        cfg = self.config

        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.state_a, self.state_b, self.state_c,
             cp.float32(self.r_a), cp.float32(self.r_b), cp.float32(self.r_c),
             cp.float32(self.twist_ab), cp.float32(self.twist_bc),
             cp.float32(cfg.coupling), cp.int32(cfg.n_cells),
             cp.int32(cfg.n_iterations))
        )
        cp.cuda.Stream.null.synchronize()

        return (
            float(cp.mean(self.state_a)),
            float(cp.mean(self.state_b)),
            float(cp.mean(self.state_c))
        )


class OssicleDetector:
    """
    Tamper detection using the ossicle sensor.

    Validated methodology:
    1. Reset oscillator states before each measurement trial
    2. Collect samples for several seconds
    3. Compute mean correlation across oscillator pairs
    4. Compare to baseline using z-score
    5. Detect if |z| > threshold

    Usage:
        detector = OssicleDetector()
        baseline = detector.calibrate(duration=30)

        # Later, to detect tampering:
        result = detector.measure(duration=5)
        if result.detected:
            print(f"TAMPER DETECTED: z={result.z_score:.2f}")
    """

    def __init__(self, config: OssicleConfig = None):
        self.config = config or OssicleConfig()
        self.kernel = OssicleKernel(self.config)

        # Baseline statistics
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None

    def _collect_samples(self, duration: float) -> Tuple[List[float], List[float], List[float]]:
        """Collect oscillator samples for specified duration."""
        history_a, history_b, history_c = [], [], []
        start = time.time()

        while time.time() - start < duration:
            a, b, c = self.kernel.step()
            history_a.append(a)
            history_b.append(b)
            history_c.append(c)

        return history_a, history_b, history_c

    def _compute_correlations(self, ha: List[float], hb: List[float], hc: List[float]) -> Tuple[float, float, float]:
        """Compute pairwise correlations between oscillators."""
        arr_a = np.array(ha)
        arr_b = np.array(hb)
        arr_c = np.array(hc)

        def safe_corr(x, y):
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                return 0.0
            r = np.corrcoef(x, y)[0, 1]
            return r if not np.isnan(r) else 0.0

        return (
            safe_corr(arr_a, arr_b),
            safe_corr(arr_b, arr_c),
            safe_corr(arr_a, arr_c)
        )

    def calibrate(self, duration: float = 30.0, n_trials: int = 10) -> Dict:
        """
        Calibrate baseline with multiple trials.

        IMPORTANT: Run with NO concurrent workloads.

        Args:
            duration: Duration per trial in seconds
            n_trials: Number of trials (more = better statistics)

        Returns:
            Dict with baseline_mean, baseline_std, n_trials
        """
        print(f"Calibrating baseline ({n_trials} trials, {duration}s each)...")
        print("Keep system IDLE during calibration.")

        correlations = []

        for i in range(n_trials):
            # Reset before each trial (validated 7x improvement)
            self.kernel.reset()

            # Collect samples
            ha, hb, hc = self._collect_samples(duration)

            # Compute mean correlation
            rho_ab, rho_bc, rho_ac = self._compute_correlations(ha, hb, hc)
            mean_corr = (rho_ab + rho_bc + rho_ac) / 3
            correlations.append(mean_corr)

            print(f"  Trial {i+1}/{n_trials}: corr = {mean_corr:.6f}")

        self.baseline_mean = float(np.mean(correlations))
        self.baseline_std = float(np.std(correlations))

        # Use validated noise floor if measured std is too small
        if self.baseline_std < VALIDATED_NOISE_FLOOR / 2:
            self.baseline_std = VALIDATED_NOISE_FLOOR

        print(f"\nCalibration complete:")
        print(f"  Mean: {self.baseline_mean:.6f}")
        print(f"  Std:  {self.baseline_std:.6f}")
        print(f"  2σ threshold: ±{2*self.baseline_std:.6f}")
        print(f"  3σ threshold: ±{3*self.baseline_std:.6f}")

        return {
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'n_trials': n_trials
        }

    def set_baseline(self, mean: float, std: float):
        """Set baseline statistics directly."""
        self.baseline_mean = mean
        self.baseline_std = std if std > 0 else VALIDATED_NOISE_FLOOR

    def measure(self, duration: float = 5.0) -> Detection:
        """
        Take a measurement and check for tampering.

        IMPORTANT: Resets oscillator state before measuring (validated 7x improvement).

        Args:
            duration: Measurement duration in seconds

        Returns:
            Detection result with z-score and detection flags
        """
        if self.baseline_mean is None:
            raise RuntimeError("No baseline. Call calibrate() first.")

        # Reset before measurement (validated 7x improvement)
        self.kernel.reset()

        # Collect samples
        ha, hb, hc = self._collect_samples(duration)

        # Compute correlations
        rho_ab, rho_bc, rho_ac = self._compute_correlations(ha, hb, hc)
        mean_corr = (rho_ab + rho_bc + rho_ac) / 3

        # Compute z-score
        z_score = abs(mean_corr - self.baseline_mean) / self.baseline_std

        # Detection
        detected = z_score > self.config.z_threshold
        strong = z_score > self.config.z_strong

        return Detection(
            timestamp=time.time(),
            mean_correlation=mean_corr,
            z_score=z_score,
            detected=detected,
            strong=strong,
            mean_a=np.mean(ha),
            mean_b=np.mean(hb),
            mean_c=np.mean(hc),
            rho_ab=rho_ab,
            rho_bc=rho_bc,
            rho_ac=rho_ac
        )

    def monitor(self, interval: float = 5.0, callback=None):
        """
        Continuous monitoring mode.

        Args:
            interval: Measurement interval in seconds
            callback: Optional callback(detection) for each measurement
        """
        if self.baseline_mean is None:
            raise RuntimeError("No baseline. Call calibrate() first.")

        print(f"Monitoring (interval={interval}s, threshold={self.config.z_threshold}σ)...")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                result = self.measure(interval)

                status = "DETECTED!" if result.detected else "OK"
                if result.strong:
                    status = "STRONG DETECTION!"

                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"corr={result.mean_correlation:+.6f} "
                      f"z={result.z_score:.2f}σ "
                      f"{status}")

                if callback:
                    callback(result)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


def demo():
    """Quick demonstration of the ossicle detector."""
    print("=" * 70)
    print("CIRISOssicle TAMPER DETECTOR DEMO")
    print("=" * 70)
    print()

    config = OssicleConfig()
    print(f"Configuration:")
    print(f"  Memory footprint: {config.memory_kb:.2f} KB")
    print(f"  Oscillators: {config.n_oscillators}")
    print(f"  Cells per oscillator: {config.n_cells}")
    print(f"  Twist angle: {config.twist_deg}°")
    print()

    detector = OssicleDetector(config)

    # Calibrate (short for demo)
    detector.calibrate(duration=3.0, n_trials=5)
    print()

    # Take a few measurements
    print("Taking measurements...")
    for i in range(3):
        result = detector.measure(duration=3.0)
        status = "DETECTED" if result.detected else "clean"
        print(f"  Measurement {i+1}: z={result.z_score:.2f}σ ({status})")

    print()
    print("Demo complete.")


if __name__ == "__main__":
    demo()
