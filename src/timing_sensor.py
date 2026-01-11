#!/usr/bin/env python3
"""
TimingSensor - Pure Timing-Based GPU Sensor (SUPERSEDED)

================================================================================
NOTE: For production, use strain_gauge.py instead
================================================================================

strain_gauge.py provides:
- Lorenz oscillator at dt=0.025 (critical point, max sensitivity)
- ACF monitoring to confirm criticality
- Dual output: TRNG (4 LSBs) + Strain sensing (k_eff dynamics)

This module (timing_sensor.py) provides timing-only detection without
the Lorenz oscillator. It's simpler but less sensitive than the full
strain gauge implementation.

Based on RATCHET Experiments 68-116 (January 2026):
- dt = 0.025 is the critical operating point
- Raw timing LSBs (4 bits) optimal for TRNG
- Lorenz dynamics at criticality for strain sensing

See: src/strain_gauge.py for production implementation.

Author: CIRIS L3C
License: BSL 1.1
"""

import numpy as np
import cupy as cp
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque


@dataclass
class TimingConfig:
    """Configuration for timing sensor."""
    # Kernel config (minimal work, just enough to measure timing)
    kernel_size: int = 64          # Threads to launch
    kernel_iterations: int = 10    # Minimal iterations

    # Detection thresholds
    z_threshold: float = 2.0       # 2σ detection
    variance_threshold: float = 2.0  # 2x variance increase

    # TRNG settings
    use_von_neumann: bool = True   # Apply debiasing

    # Warm-up settings (from Array characterization)
    warm_up_duration: float = 30.0  # Seconds to warm up before calibration
    warm_up_enabled: bool = True    # Whether to run warm-up

    @property
    def memory_bytes(self) -> int:
        return self.kernel_size * 4  # Single float32 array


@dataclass
class TimingReading:
    """Single timing measurement."""
    timestamp: float
    timing_ns: int
    timing_lsb: int  # Lower 8 bits


@dataclass
class StrainReading:
    """Strain gauge reading from timing variance."""
    timestamp: float
    timing_mean_us: float
    timing_std_us: float
    timing_z: float
    variance_ratio: float
    detected: bool
    n_samples: int


class TimingSensor:
    """
    Pure timing-based GPU sensor.

    Uses minimal kernel execution to extract timing jitter.
    No chaotic oscillators - timing IS the signal.
    """

    # Minimal kernel - just enough work to get consistent timing
    MINIMAL_KERNEL = r'''
    extern "C" __global__ void timing_kernel(float* data, int n, int iterations) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float x = data[idx];
        for (int i = 0; i < iterations; i++) {
            x = x * 0.99f + 0.01f;  // Minimal computation
        }
        data[idx] = x;
    }
    '''

    def __init__(self, config: TimingConfig = None):
        self.config = config or TimingConfig()
        cfg = self.config

        # Compile minimal kernel
        self.module = cp.RawModule(code=self.MINIMAL_KERNEL)
        self.kernel = self.module.get_function('timing_kernel')

        # Allocate minimal state
        self.data = cp.random.randn(cfg.kernel_size, dtype=cp.float32)

        # Execution config
        self.block_size = min(64, cfg.kernel_size)
        self.grid_size = (cfg.kernel_size + self.block_size - 1) // self.block_size

    def read_timing(self) -> TimingReading:
        """Read single timing measurement."""
        cfg = self.config

        start = time.perf_counter_ns()

        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.data, cp.int32(cfg.kernel_size), cp.int32(cfg.kernel_iterations))
        )
        cp.cuda.Stream.null.synchronize()

        end = time.perf_counter_ns()
        timing_ns = end - start

        return TimingReading(
            timestamp=time.time(),
            timing_ns=timing_ns,
            timing_lsb=timing_ns & 0xFF
        )

    def read_batch(self, n_samples: int) -> List[TimingReading]:
        """Read multiple timing measurements."""
        return [self.read_timing() for _ in range(n_samples)]


class TimingTRNG:
    """
    True Random Number Generator from GPU timing.

    Extracts entropy from kernel execution timing jitter.
    Applies von Neumann debiasing for uniform output.

    Validated: 7.99 bits/byte raw entropy.
    """

    def __init__(self, config: TimingConfig = None):
        self.config = config or TimingConfig()
        self.sensor = TimingSensor(self.config)

        # Buffer for von Neumann debiasing
        self._bit_buffer = []

    def _von_neumann_debias(self, bits: List[int]) -> List[int]:
        """
        Apply von Neumann debiasing to bit stream.

        Compares pairs: 01→0, 10→1, 00/11→discard
        Removes bias but reduces throughput ~50%.
        """
        output = []
        for i in range(0, len(bits) - 1, 2):
            b1, b2 = bits[i], bits[i + 1]
            if b1 == 0 and b2 == 1:
                output.append(0)
            elif b1 == 1 and b2 == 0:
                output.append(1)
            # 00 and 11 are discarded
        return output

    def _extract_bits(self, timing_ns: int, n_bits: int = 8) -> List[int]:
        """Extract n LSBs from timing value."""
        return [(timing_ns >> i) & 1 for i in range(n_bits)]

    def generate_bytes(self, n_bytes: int) -> bytes:
        """
        Generate random bytes from timing entropy.

        Uses von Neumann debiasing if configured.
        """
        bits_needed = n_bytes * 8

        # Collect raw bits (oversample for von Neumann loss)
        raw_bits = []
        samples_needed = bits_needed * 4 if self.config.use_von_neumann else bits_needed

        while len(raw_bits) < samples_needed:
            reading = self.sensor.read_timing()
            raw_bits.extend(self._extract_bits(reading.timing_ns, 8))

        # Apply debiasing if configured
        if self.config.use_von_neumann:
            bits = self._von_neumann_debias(raw_bits)
        else:
            bits = raw_bits

        # Convert bits to bytes
        output = []
        for i in range(0, min(len(bits), bits_needed), 8):
            if i + 8 <= len(bits):
                byte_val = sum(bits[i + j] << j for j in range(8))
                output.append(byte_val)

        # Pad if needed (shouldn't happen with proper oversampling)
        while len(output) < n_bytes:
            reading = self.sensor.read_timing()
            output.append(reading.timing_lsb)

        return bytes(output[:n_bytes])

    def generate_int(self, min_val: int = 0, max_val: int = 255) -> int:
        """Generate random integer in range."""
        range_size = max_val - min_val + 1
        bytes_needed = (range_size.bit_length() + 7) // 8
        raw = int.from_bytes(self.generate_bytes(bytes_needed), 'little')
        return min_val + (raw % range_size)


class TimingStrainGauge:
    """
    Strain gauge based on kernel timing variance.

    Detects workloads/tampering by monitoring timing distribution shifts.

    Validated: z=8.56 at 90% GPU load (excellent sensitivity).
    """

    def __init__(self, config: TimingConfig = None):
        self.config = config or TimingConfig()
        self.sensor = TimingSensor(self.config)

        # Baseline statistics
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None

    def warm_up(self, duration: float = None):
        """
        Run GPU warm-up before calibration.

        From Array characterization: noise decreases as GPU warms.
        Calibrating after warm-up gives more stable baseline.
        """
        duration = duration or self.config.warm_up_duration
        print(f"Warming up GPU ({duration}s)...")

        start = time.time()
        while time.time() - start < duration:
            _ = self.sensor.read_timing()

        print("Warm-up complete.")

    def calibrate(self, duration: float = 10.0, skip_warmup: bool = False) -> dict:
        """
        Calibrate baseline timing statistics.

        Args:
            duration: Calibration duration in seconds
            skip_warmup: Skip warm-up phase (not recommended)

        Run with system IDLE for accurate baseline.
        """
        # Warm-up phase (from Array characterization)
        if self.config.warm_up_enabled and not skip_warmup:
            self.warm_up()

        print(f"Calibrating timing baseline ({duration}s)...")
        print("Keep system IDLE during calibration.")

        timings = []
        start = time.time()

        while time.time() - start < duration:
            reading = self.sensor.read_timing()
            timings.append(reading.timing_ns)

        self.baseline_mean = float(np.mean(timings))
        self.baseline_std = float(np.std(timings))

        # Ensure non-zero std
        if self.baseline_std < 1:
            self.baseline_std = 1.0

        print(f"Baseline: {self.baseline_mean/1000:.1f} ± {self.baseline_std/1000:.1f} μs")
        print(f"Samples: {len(timings)}")

        return {
            'mean_ns': self.baseline_mean,
            'std_ns': self.baseline_std,
            'mean_us': self.baseline_mean / 1000,
            'std_us': self.baseline_std / 1000,
            'n_samples': len(timings)
        }

    def set_baseline(self, mean_ns: float, std_ns: float):
        """Set baseline directly."""
        self.baseline_mean = mean_ns
        self.baseline_std = std_ns if std_ns > 0 else 1.0

    def measure(self, duration: float = 5.0) -> StrainReading:
        """
        Measure timing strain.

        Returns detection result based on timing variance shift.
        """
        if self.baseline_mean is None:
            raise RuntimeError("Call calibrate() first")

        timings = []
        start = time.time()

        while time.time() - start < duration:
            reading = self.sensor.read_timing()
            timings.append(reading.timing_ns)

        timing_mean = np.mean(timings)
        timing_std = np.std(timings)

        # Z-score for mean shift
        timing_z = abs(timing_mean - self.baseline_mean) / self.baseline_std

        # Variance ratio
        variance_ratio = timing_std / self.baseline_std if self.baseline_std > 0 else 1.0

        # Detection
        detected = (timing_z > self.config.z_threshold or
                    variance_ratio > self.config.variance_threshold)

        return StrainReading(
            timestamp=time.time(),
            timing_mean_us=float(timing_mean / 1000),
            timing_std_us=float(timing_std / 1000),
            timing_z=float(timing_z),
            variance_ratio=float(variance_ratio),
            detected=detected,
            n_samples=len(timings)
        )

    def monitor(self, interval: float = 5.0, callback=None):
        """Continuous monitoring mode."""
        if self.baseline_mean is None:
            raise RuntimeError("Call calibrate() first")

        print(f"Monitoring (interval={interval}s)...")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                result = self.measure(interval)

                status = "DETECTED!" if result.detected else "OK"
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"timing={result.timing_mean_us:.1f}μs "
                      f"z={result.timing_z:.2f} "
                      f"var={result.variance_ratio:.1f}x "
                      f"{status}")

                if callback:
                    callback(result)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


def demo():
    """Quick demonstration of timing sensor."""
    print("=" * 60)
    print("TIMING SENSOR DEMO")
    print("=" * 60)
    print()

    # Disable warm-up for quick demo
    config = TimingConfig(warm_up_enabled=False)
    print(f"Memory footprint: {config.memory_bytes} bytes")
    print()

    # TRNG demo
    print("--- TRNG Demo ---")
    trng = TimingTRNG(config)
    random_bytes = trng.generate_bytes(16)
    print(f"Random bytes: {random_bytes.hex()}")
    print(f"Random int [0-100]: {trng.generate_int(0, 100)}")
    print()

    # Strain gauge demo
    print("--- Strain Gauge Demo ---")
    gauge = TimingStrainGauge(config)
    gauge.calibrate(duration=5.0)  # skip_warmup implicit due to config
    print()

    print("Taking measurement...")
    result = gauge.measure(duration=3.0)
    print(f"Timing: {result.timing_mean_us:.1f} ± {result.timing_std_us:.1f} μs")
    print(f"Z-score: {result.timing_z:.2f}")
    print(f"Variance ratio: {result.variance_ratio:.1f}x")
    print(f"Detected: {'YES' if result.detected else 'NO'}")


if __name__ == "__main__":
    demo()
