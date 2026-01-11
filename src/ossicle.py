#!/usr/bin/env python3
"""
CIRISOssicle - GPU Tamper Detection Sensor

A 0.75KB GPU sensor that detects unauthorized workloads through
correlation fingerprinting of chaotic oscillators.

VALIDATED FEATURES (January 2026):
1. Local tamper detection - correlation mean shifts under workload (p=0.007)
2. Reset strategy - 7x sensitivity improvement with periodic resets (p=0.032)
3. Bounded noise floor - σ ≈ 0.003

NOT VALIDATED (removed from claims):
- Workload type classification (crypto vs memory indistinguishable, p=0.49)
- Startup transient (no variance difference detected, p=0.14)

See VALIDATION_RESULTS.md for full null hypothesis test methodology.

Author: CIRIS L3C (Eric Moore)
License: BSL 1.1
"""

import numpy as np
import cupy as cp
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import json
from pathlib import Path


class WorkloadType(Enum):
    """Classification of detected workload types."""
    BASELINE = "baseline"
    NEGENTROPIC = "negentropic"  # Coherent: matrix ops, inference (4x response)
    ENTROPIC = "entropic"        # Incoherent: random memory, crypto (1x response)
    UNKNOWN = "unknown"


@dataclass
class OssicleConfig:
    """Configuration for the ossicle sensor."""
    n_cells: int = 64
    n_iterations: int = 500
    n_oscillators: int = 3
    twist_deg: float = 1.1          # Magic angle (empirically optimal)
    r_base: float = 3.70
    spacing: float = 0.03
    coupling: float = 0.05          # Epsilon - EM coupling strength

    # Timing parameters
    startup_transient_sec: float = 2.0   # Discard samples during warmup
    reset_interval_sec: float = 20.0     # Reset to maintain sensitivity

    # Classification thresholds
    negentropic_threshold: float = 0.15  # High spectral power = negentropic
    entropic_threshold: float = 0.05     # Low spectral power = entropic

    # Detection thresholds
    z_threshold: float = 2.0             # Detection threshold (sigma)
    z_strong: float = 3.0                # Strong detection threshold

    @property
    def memory_bytes(self) -> int:
        return self.n_oscillators * self.n_cells * 4  # float32

    @property
    def memory_kb(self) -> float:
        return self.memory_bytes / 1024


@dataclass
class OssicleReading:
    """Single sensor reading with all computed metrics."""
    timestamp: float
    mean_a: float
    mean_b: float
    mean_c: float

    # Correlations
    rho_ab: float = 0.0
    rho_bc: float = 0.0
    rho_ac: float = 0.0

    # Detection metrics
    z_score: float = 0.0
    detected: bool = False
    strong_detection: bool = False

    # Classification
    workload_type: WorkloadType = WorkloadType.UNKNOWN
    spectral_power: float = 0.0  # 0.1-0.5 Hz band power


class OssicleKernel:
    """
    Core CUDA kernel for the ossicle sensor.

    Uses 3 coupled chaotic oscillators with 1.1 degree twist angle
    to create moire interference patterns sensitive to EM coupling changes.
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
        self._init_states()

        # Execution config
        self.block_size = 64
        self.grid_size = (cfg.n_cells + self.block_size - 1) // self.block_size

    def _init_states(self):
        """Initialize oscillator states with random values."""
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

    def reset(self):
        """Reset oscillator states to maintain sensitivity."""
        self._init_states()


class OssicleDetector:
    """
    Full-featured tamper detection using the ossicle sensor.

    Features:
    - Startup transient handling
    - Periodic reset for continuous sensitivity
    - Workload classification using 4:1 asymmetry
    - Spectral analysis for environmental sensing
    """

    def __init__(self, config: OssicleConfig = None):
        self.config = config or OssicleConfig()
        self.kernel = OssicleKernel(self.config)

        # Timing
        self.start_time: Optional[float] = None
        self.last_reset_time: Optional[float] = None
        self.in_transient: bool = True

        # History for correlation/spectral analysis
        self.history_a: List[float] = []
        self.history_b: List[float] = []
        self.history_c: List[float] = []
        self.timestamps: List[float] = []

        # Baseline statistics
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.is_calibrated: bool = False

    def start(self):
        """Start the detector, initializing timing."""
        self.start_time = time.time()
        self.last_reset_time = self.start_time
        self.in_transient = True
        self.kernel.reset()
        self._clear_history()

    def _clear_history(self):
        """Clear history buffers."""
        self.history_a.clear()
        self.history_b.clear()
        self.history_c.clear()
        self.timestamps.clear()

    def _check_reset(self) -> bool:
        """Check if reset is needed and perform it."""
        now = time.time()

        # Check if still in startup transient
        if self.start_time and now - self.start_time < self.config.startup_transient_sec:
            self.in_transient = True
            return False
        else:
            self.in_transient = False

        # Check if periodic reset is needed
        if self.last_reset_time and now - self.last_reset_time > self.config.reset_interval_sec:
            self.kernel.reset()
            self.last_reset_time = now
            self._clear_history()
            return True

        return False

    def step(self) -> OssicleReading:
        """Take a single reading with all processing."""
        now = time.time()

        # Initialize if needed
        if self.start_time is None:
            self.start()

        # Check for reset
        did_reset = self._check_reset()

        # Get raw reading
        a, b, c = self.kernel.step()

        # Store in history
        self.history_a.append(a)
        self.history_b.append(b)
        self.history_c.append(c)
        self.timestamps.append(now)

        # Create reading
        reading = OssicleReading(
            timestamp=now,
            mean_a=a,
            mean_b=b,
            mean_c=c
        )

        # Skip processing during transient
        if self.in_transient:
            return reading

        # Compute correlations if we have enough data
        if len(self.history_a) >= 50:
            reading.rho_ab, reading.rho_bc, reading.rho_ac = self._compute_correlations()

        # Compute spectral power for classification
        if len(self.history_a) >= 100:
            reading.spectral_power = self._get_band_power(0.1, 0.5)
            reading.workload_type = self._classify_workload(reading.spectral_power)

        # Compute z-score if calibrated
        if self.is_calibrated and self.baseline_std > 0:
            mean_corr = (reading.rho_ab + reading.rho_bc + reading.rho_ac) / 3
            reading.z_score = abs(mean_corr - self.baseline_mean) / self.baseline_std
            reading.detected = reading.z_score > self.config.z_threshold
            reading.strong_detection = reading.z_score > self.config.z_strong

        return reading

    def _compute_correlations(self) -> Tuple[float, float, float]:
        """Compute correlation coefficients between oscillators."""
        arr_a = np.array(self.history_a)
        arr_b = np.array(self.history_b)
        arr_c = np.array(self.history_c)

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

    def _get_band_power(self, low_hz: float, high_hz: float) -> float:
        """Get spectral power in the specified frequency band."""
        if len(self.history_a) < 100:
            return 0.0

        # Estimate sample rate from timestamps
        if len(self.timestamps) < 2:
            return 0.0

        dt = np.mean(np.diff(self.timestamps))
        if dt <= 0:
            return 0.0

        fs = 1.0 / dt

        # Combine oscillator signals
        signal = np.array(self.history_a) - np.mean(self.history_a)

        # FFT
        n = len(signal)
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        power = np.abs(fft) ** 2

        # Band power
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(band_mask):
            return 0.0

        band_power = np.sum(power[band_mask]) / np.sum(power)
        return float(band_power)

    def _classify_workload(self, spectral_power: float) -> WorkloadType:
        """
        Classify workload based on spectral power.

        Negentropic (coherent) workloads show 4x stronger response in 0.1-0.5 Hz band.
        Entropic (incoherent) workloads show weaker response.
        """
        if spectral_power > self.config.negentropic_threshold:
            return WorkloadType.NEGENTROPIC
        elif spectral_power < self.config.entropic_threshold:
            return WorkloadType.ENTROPIC
        else:
            return WorkloadType.BASELINE

    def calibrate(self, duration: float = 30.0) -> Dict:
        """
        Calibrate baseline statistics.

        Run this with no concurrent workloads to establish baseline.
        """
        print(f"Calibrating for {duration}s (keep system idle)...")

        self.start()

        # Wait for transient
        print(f"  Waiting {self.config.startup_transient_sec}s for startup transient...")
        time.sleep(self.config.startup_transient_sec + 0.5)

        # Collect samples
        correlations = []
        start = time.time()
        sample_count = 0

        while time.time() - start < duration:
            reading = self.step()
            if not self.in_transient and len(self.history_a) >= 50:
                mean_corr = (reading.rho_ab + reading.rho_bc + reading.rho_ac) / 3
                correlations.append(mean_corr)
                sample_count += 1

            if sample_count > 0 and sample_count % 100 == 0:
                print(f"  Collected {sample_count} samples...")

        if len(correlations) < 10:
            raise RuntimeError("Not enough samples for calibration")

        self.baseline_mean = float(np.mean(correlations))
        self.baseline_std = float(np.std(correlations))
        self.is_calibrated = True

        print(f"Calibration complete:")
        print(f"  Samples: {len(correlations)}")
        print(f"  Mean correlation: {self.baseline_mean:.6f}")
        print(f"  Std correlation: {self.baseline_std:.6f}")

        return {
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'n_samples': len(correlations)
        }

    def set_baseline(self, mean: float, std: float):
        """Set baseline statistics directly."""
        self.baseline_mean = mean
        self.baseline_std = std
        self.is_calibrated = True


class MultiEpsilonArray:
    """
    Array of ossicle sensors with different coupling strengths.

    Different epsilon values provide different sensitivity/response tradeoffs:
    - Low epsilon (0.03): High sensitivity, slower response
    - Medium epsilon (0.05): Balanced (default)
    - High epsilon (0.10): Low sensitivity, faster response

    Using multiple sensors enables better workload discrimination.
    """

    DEFAULT_COUPLINGS = [0.03, 0.05, 0.10]

    def __init__(self, couplings: List[float] = None, base_config: OssicleConfig = None):
        self.couplings = couplings or self.DEFAULT_COUPLINGS
        self.detectors: List[OssicleDetector] = []

        base = base_config or OssicleConfig()

        for coupling in self.couplings:
            config = OssicleConfig(
                n_cells=base.n_cells,
                n_iterations=base.n_iterations,
                twist_deg=base.twist_deg,
                r_base=base.r_base,
                spacing=base.spacing,
                coupling=coupling,
                startup_transient_sec=base.startup_transient_sec,
                reset_interval_sec=base.reset_interval_sec
            )
            self.detectors.append(OssicleDetector(config))

    def start(self):
        """Start all detectors."""
        for det in self.detectors:
            det.start()

    def step(self) -> List[OssicleReading]:
        """Take readings from all detectors."""
        return [det.step() for det in self.detectors]

    def calibrate(self, duration: float = 30.0) -> List[Dict]:
        """Calibrate all detectors."""
        results = []
        for i, det in enumerate(self.detectors):
            print(f"\nCalibrating sensor {i+1}/{len(self.detectors)} (epsilon={self.couplings[i]})...")
            results.append(det.calibrate(duration))
        return results

    def get_consensus_detection(self, readings: List[OssicleReading]) -> Tuple[bool, float]:
        """
        Get consensus detection across all sensors.

        Returns (detected, max_z_score).
        """
        detected = any(r.detected for r in readings)
        max_z = max((r.z_score for r in readings), default=0.0)
        return detected, max_z

    def get_workload_consensus(self, readings: List[OssicleReading]) -> WorkloadType:
        """
        Get consensus workload classification.

        Uses voting across sensors.
        """
        types = [r.workload_type for r in readings if r.workload_type != WorkloadType.UNKNOWN]
        if not types:
            return WorkloadType.UNKNOWN

        # Simple majority vote
        from collections import Counter
        counts = Counter(types)
        return counts.most_common(1)[0][0]


def quick_test(duration: float = 30.0):
    """Quick test of the improved ossicle sensor."""
    print("=" * 70)
    print("CIRISOssicle IMPROVED SENSOR TEST")
    print("=" * 70)
    print()

    config = OssicleConfig()
    print(f"Configuration:")
    print(f"  Memory: {config.memory_kb:.2f} KB")
    print(f"  Cells: {config.n_cells}")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Twist angle: {config.twist_deg} deg")
    print(f"  Coupling (epsilon): {config.coupling}")
    print(f"  Startup transient: {config.startup_transient_sec}s")
    print(f"  Reset interval: {config.reset_interval_sec}s")
    print()

    detector = OssicleDetector(config)
    detector.start()

    print(f"Running for {duration}s...")
    print()

    readings = []
    start = time.time()

    while time.time() - start < duration:
        reading = detector.step()
        readings.append(reading)

        if not detector.in_transient and len(detector.history_a) >= 100:
            status = "TRANSIENT" if detector.in_transient else "ACTIVE"
            wl_type = reading.workload_type.value
            print(f"  [{status}] rho_ab={reading.rho_ab:+.4f} "
                  f"spectral={reading.spectral_power:.4f} "
                  f"type={wl_type}")

    print()
    print("Test complete.")
    print(f"  Total readings: {len(readings)}")

    active_readings = [r for r in readings if r.spectral_power > 0]
    if active_readings:
        print(f"  Active readings: {len(active_readings)}")
        avg_spectral = np.mean([r.spectral_power for r in active_readings])
        print(f"  Avg spectral power: {avg_spectral:.4f}")


if __name__ == "__main__":
    quick_test()
