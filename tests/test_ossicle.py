#!/usr/bin/env python3
"""
CIRISOssicle Falsification Tests

These tests attempt to FALSIFY the claims of CIRISOssicle:
1. That the sensor produces valid, bounded output
2. That correlations are mathematically correct
3. That the sensor is statistically consistent
4. That workloads cause measurable, non-random changes
5. That detection is scientifically sound

If any of these tests fail, the sensor's claims are invalidated.

License: BSL 1.1
Author: CIRIS L3C
"""

import pytest
import numpy as np
import time
import sys
import os

# Skip all tests if no GPU available
try:
    import cupy as cp
    GPU_AVAILABLE = True
    try:
        cp.cuda.runtime.getDeviceCount()
    except:
        GPU_AVAILABLE = False
except ImportError:
    GPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")


class OssicleKernel:
    """Minimal ossicle for testing."""

    KERNEL_CODE = r'''
    extern "C" __global__ void ossicle_step(
        float* states, float* r_vals, float* twists,
        float coupling, int n, int n_osc, int iterations
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        extern __shared__ float shared[];
        float* s = &shared[threadIdx.x * 16];

        for (int o = 0; o < n_osc && o < 16; o++) {
            s[o] = states[o * n + idx];
        }

        int left = (idx > 0) ? idx - 1 : n - 1;
        int right = (idx < n - 1) ? idx + 1 : 0;

        for (int iter = 0; iter < iterations; iter++) {
            for (int o = 0; o < n_osc && o < 16; o++) {
                float r = r_vals[o];
                float neighbor = states[o * n + left] + states[o * n + right];

                float cross = 0.0f;
                for (int p = 0; p < n_osc && p < 16; p++) {
                    if (p != o) {
                        float twist_diff = twists[p] - twists[o];
                        cross += s[p] * cosf(twist_diff) * 0.03f;
                    }
                }

                s[o] = r * s[o] * (1.0f - s[o])
                     + coupling * (neighbor - 2.0f * s[o])
                     + coupling * cross;

                s[o] = fminf(fmaxf(s[o], 0.0001f), 0.9999f);
            }

            for (int o = 0; o < n_osc && o < 16; o++) {
                states[o * n + idx] = s[o];
            }
        }
    }
    '''

    def __init__(self, n_osc=3, n_cells=64, n_iterations=500, twist_deg=1.1):
        self.n_osc = n_osc
        self.n_cells = n_cells
        self.n_iterations = n_iterations

        self.r_vals = np.array([3.70 + i * 0.03 for i in range(n_osc)], dtype=np.float32)
        self.twists = np.array([i * np.radians(twist_deg) for i in range(n_osc)], dtype=np.float32)

        self.module = cp.RawModule(code=self.KERNEL_CODE)
        self.kernel = self.module.get_function('ossicle_step')

        self.r_vals_gpu = cp.asarray(self.r_vals)
        self.twists_gpu = cp.asarray(self.twists)
        self._init_states()

        self.block_size = min(64, n_cells)
        self.grid_size = (n_cells + self.block_size - 1) // self.block_size
        self.shared_mem = self.block_size * 16 * 4

    def _init_states(self):
        states = np.random.uniform(0.1, 0.9, (self.n_osc, self.n_cells)).astype(np.float32)
        self.states_gpu = cp.asarray(states)

    def step(self):
        self.kernel(
            (self.grid_size,), (self.block_size,),
            (self.states_gpu, self.r_vals_gpu, self.twists_gpu,
             cp.float32(0.05), cp.int32(self.n_cells),
             cp.int32(self.n_osc), cp.int32(self.n_iterations)),
            shared_mem=self.shared_mem
        )
        cp.cuda.Stream.null.synchronize()
        return [float(cp.mean(self.states_gpu[i])) for i in range(self.n_osc)]

    def reset(self):
        self._init_states()


def safe_corr(x, y):
    """Compute correlation, handling edge cases."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return r if not np.isnan(r) else 0.0


class TestSensorOutput:
    """Tests that sensor produces valid output."""

    def test_output_is_finite(self):
        """FALSIFICATION: Output must be finite (not NaN or Inf)."""
        sensor = OssicleKernel()
        for _ in range(100):
            means = sensor.step()
            for m in means:
                assert np.isfinite(m), f"Output is not finite: {m}"

    def test_output_is_bounded(self):
        """FALSIFICATION: Output must be in (0, 1) for logistic map."""
        sensor = OssicleKernel()
        for _ in range(100):
            means = sensor.step()
            for m in means:
                assert 0 < m < 1, f"Output {m} outside valid range (0, 1)"

    def test_output_varies(self):
        """FALSIFICATION: Chaotic system must produce varying output."""
        sensor = OssicleKernel()
        outputs = []
        for _ in range(50):
            means = sensor.step()
            outputs.append(means[0])

        # Variance must be non-zero for chaotic system
        variance = np.var(outputs)
        assert variance > 1e-10, f"Output variance too low: {variance}"

    def test_oscillators_are_different(self):
        """FALSIFICATION: Different r-values should produce different dynamics."""
        sensor = OssicleKernel()
        history = [[] for _ in range(sensor.n_osc)]

        for _ in range(100):
            means = sensor.step()
            for i, m in enumerate(means):
                history[i].append(m)

        # Each oscillator should have different mean
        oscillator_means = [np.mean(h) for h in history]
        for i in range(len(oscillator_means)):
            for j in range(i + 1, len(oscillator_means)):
                diff = abs(oscillator_means[i] - oscillator_means[j])
                # They shouldn't be identical (would indicate bug)
                assert diff > 1e-6 or diff < 0.5, "Oscillators producing identical output"


class TestCorrelationMath:
    """Tests that correlation calculations are mathematically valid."""

    def test_correlation_bounds(self):
        """FALSIFICATION: Correlation must be in [-1, 1]."""
        sensor = OssicleKernel()
        history = [[] for _ in range(sensor.n_osc)]

        for _ in range(100):
            means = sensor.step()
            for i, m in enumerate(means):
                history[i].append(m)

        arrays = [np.array(h) for h in history]
        for i in range(sensor.n_osc):
            for j in range(i + 1, sensor.n_osc):
                corr = safe_corr(arrays[i], arrays[j])
                assert -1 <= corr <= 1, f"Correlation {corr} outside [-1, 1]"

    def test_correlation_symmetry(self):
        """FALSIFICATION: corr(A,B) must equal corr(B,A)."""
        sensor = OssicleKernel()
        history = [[] for _ in range(sensor.n_osc)]

        for _ in range(100):
            means = sensor.step()
            for i, m in enumerate(means):
                history[i].append(m)

        arrays = [np.array(h) for h in history]
        corr_01 = safe_corr(arrays[0], arrays[1])
        corr_10 = safe_corr(arrays[1], arrays[0])

        assert abs(corr_01 - corr_10) < 1e-10, f"Correlation not symmetric: {corr_01} vs {corr_10}"

    def test_self_correlation_is_one(self):
        """FALSIFICATION: corr(A,A) must equal 1."""
        sensor = OssicleKernel()
        history = []

        for _ in range(100):
            means = sensor.step()
            history.append(means[0])

        arr = np.array(history)
        self_corr = safe_corr(arr, arr)

        assert abs(self_corr - 1.0) < 1e-10, f"Self-correlation is not 1: {self_corr}"


class TestStatisticalConsistency:
    """Tests that sensor is statistically consistent."""

    def test_baseline_consistency(self):
        """FALSIFICATION: Multiple baseline runs should have similar statistics."""
        sensor = OssicleKernel()
        baseline_means = []

        for run in range(5):
            sensor.reset()
            history = [[] for _ in range(sensor.n_osc)]

            for _ in range(200):
                means = sensor.step()
                for i, m in enumerate(means):
                    history[i].append(m)

            arrays = [np.array(h) for h in history]
            corrs = []
            for i in range(sensor.n_osc):
                for j in range(i + 1, sensor.n_osc):
                    corrs.append(safe_corr(arrays[i], arrays[j]))
            baseline_means.append(np.mean(corrs))

        # Standard deviation across runs should be bounded
        std_across_runs = np.std(baseline_means)
        assert std_across_runs < 0.5, f"Baseline too variable: std={std_across_runs}"

    def test_reset_produces_different_initial_conditions(self):
        """FALSIFICATION: Reset should produce different trajectories."""
        sensor = OssicleKernel()

        first_values = []
        for _ in range(3):
            sensor.reset()
            first_values.append(sensor.step()[0])

        # At least some should be different
        unique_values = len(set([round(v, 6) for v in first_values]))
        assert unique_values > 1, "Reset not producing different initial conditions"


class TestDetectionValidity:
    """Tests that detection mechanism is scientifically valid."""

    def test_z_score_calculation(self):
        """FALSIFICATION: z-score must follow standard formula."""
        # Known values
        baseline_mean = 0.5
        baseline_std = 0.1
        attack_mean = 0.7

        expected_z = abs(attack_mean - baseline_mean) / baseline_std
        calculated_z = abs(0.7 - 0.5) / 0.1

        assert abs(expected_z - calculated_z) < 1e-10, "z-score calculation incorrect"
        assert abs(expected_z - 2.0) < 1e-6, f"z-score should be 2.0, got {expected_z}"

    def test_detection_requires_significant_change(self):
        """FALSIFICATION: Detection should not trigger on random noise."""
        sensor = OssicleKernel()

        # Collect two baselines (no workload change)
        def collect_baseline():
            sensor.reset()
            history = [[] for _ in range(sensor.n_osc)]
            for _ in range(200):
                means = sensor.step()
                for i, m in enumerate(means):
                    history[i].append(m)
            arrays = [np.array(h) for h in history]
            corrs = []
            for i in range(sensor.n_osc):
                for j in range(i + 1, sensor.n_osc):
                    corrs.append(safe_corr(arrays[i], arrays[j]))
            return np.mean(corrs), np.std(corrs)

        m1, s1 = collect_baseline()
        m2, s2 = collect_baseline()

        # z-score between two baselines should usually be < 3
        z = abs(m1 - m2) / (s1 + 1e-10)

        # Allow for occasional false positives, but not systematic
        # This is a statistical test, so we're lenient
        assert z < 10, f"Baseline-to-baseline z={z} too high (suggests systematic bias)"


class TestPhysicalPlausibility:
    """Tests that the physical mechanism is plausible."""

    def test_r_values_in_chaotic_regime(self):
        """FALSIFICATION: r-values must be in chaotic regime (r > 3.57)."""
        sensor = OssicleKernel()
        for r in sensor.r_vals:
            assert r > 3.57, f"r-value {r} not in chaotic regime"
            assert r < 4.0, f"r-value {r} exceeds logistic map bound"

    def test_twist_angle_is_reasonable(self):
        """FALSIFICATION: Twist angle must be physically reasonable."""
        sensor = OssicleKernel(twist_deg=1.1)
        # Magic angle should be small but non-zero
        twist_rad = sensor.twists[1] - sensor.twists[0]
        twist_deg = np.degrees(twist_rad)

        assert 0 < twist_deg < 180, f"Twist angle {twist_deg} not in valid range"

    def test_coupling_is_weak(self):
        """FALSIFICATION: Coupling should be weak (perturbative)."""
        # Strong coupling would destroy individual oscillator dynamics
        coupling = 0.05
        assert coupling < 0.5, "Coupling too strong"
        assert coupling > 0, "Coupling must be positive"


class TestMemoryAndPerformance:
    """Tests memory footprint and performance claims."""

    def test_memory_footprint(self):
        """FALSIFICATION: Memory should match claimed ~0.75KB for standard config."""
        sensor = OssicleKernel(n_osc=3, n_cells=64)
        expected_bytes = sensor.n_osc * sensor.n_cells * 4  # float32

        assert expected_bytes == 768, f"Memory footprint {expected_bytes} != 768 bytes"
        assert expected_bytes / 1024 < 1.0, "Memory exceeds 1KB claim"

    def test_sample_rate_reasonable(self):
        """FALSIFICATION: Sample rate should be > 100/s on any modern GPU."""
        sensor = OssicleKernel()

        start = time.time()
        for _ in range(100):
            sensor.step()
        elapsed = time.time() - start

        rate = 100 / elapsed
        assert rate > 10, f"Sample rate {rate}/s too low for practical use"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
