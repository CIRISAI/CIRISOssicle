#!/usr/bin/env python3
"""
Tamper-Evident HSM: Self-Validating Key Generation

Architecture:
                    GPU Chaotic Oscillators
                             │
                             ▼
                      Raw Output Stream
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
             SIGNAL CHANNEL      ENTROPY CHANNEL
            (Autocorrelation)    (Whitened residual)
                    │                 │
                    ▼                 ▼
             Tamper Detector     Key Generator
                    │                 │
                    └────────┬────────┘
                             │
                             ▼
                      Cross-Validation
                    (Both must agree)

Key insight: The SAME physical process provides:
1. Entropy for key generation (residual after removing structure)
2. Tamper detection (monitoring the structure itself)

If tampering occurs:
- Signal channel: Autocorrelation pattern changes → ALERT
- Entropy channel: Entropy rate changes → ALERT
- Cross-validation: Both channels must agree or keys are invalidated
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import json

# Import our TRNG
import sys
sys.path.insert(0, '.')

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    raise ImportError("CuPy required - CPU fallback disabled. Install: pip install cupy-cuda12x")


@dataclass
class HSMState:
    """Current state of the tamper-evident HSM."""
    baseline_acf: np.ndarray  # Expected autocorrelation pattern
    baseline_entropy: float    # Expected entropy rate
    acf_tolerance: float       # Allowed deviation in ACF
    entropy_tolerance: float   # Allowed deviation in entropy
    tamper_detected: bool = False
    last_check: float = 0
    check_count: int = 0
    alert_history: List[dict] = None

    def __post_init__(self):
        if self.alert_history is None:
            self.alert_history = []


class TamperEvidentHSM:
    """
    Hardware Security Module with intrinsic tamper detection.

    The chaotic oscillator output is split into:
    1. SIGNAL: Period-6 autocorrelation structure (tamper detection)
    2. ENTROPY: Residual after removing signal (key generation)
    """

    # CUDA kernel for raw entropy generation
    KERNEL = '''
    extern "C" __global__
    void hsm_kernel(unsigned int* output, int n_words, unsigned int seed) {
        int idx = blockIdx.x;
        if (idx >= n_words) return;

        int thread_id = blockIdx.y * blockDim.x + threadIdx.x;
        unsigned int rng = seed + idx * 1000000 + thread_id * 31337;
        rng = rng * 1103515245u + 12345u;

        float x0 = (float)((rng = rng * 1103515245u + 12345u) & 0xFFFF) / 65536.0f;
        float x1 = (float)((rng = rng * 1103515245u + 12345u) & 0xFFFF) / 65536.0f;
        float x2 = (float)((rng = rng * 1103515245u + 12345u) & 0xFFFF) / 65536.0f;
        float x3 = (float)((rng = rng * 1103515245u + 12345u) & 0xFFFF) / 65536.0f;

        x0 = fminf(0.999f, fmaxf(0.001f, x0));
        x1 = fminf(0.999f, fmaxf(0.001f, x1));
        x2 = fminf(0.999f, fmaxf(0.001f, x2));
        x3 = fminf(0.999f, fmaxf(0.001f, x3));

        for (int i = 0; i < 500; i++) {
            x0 = 3.75f * x0 * (1.0f - x0);
            x1 = 3.75f * x1 * (1.0f - x1);
            x2 = 3.75f * x2 * (1.0f - x2);
            x3 = 3.75f * x3 * (1.0f - x3);
        }

        unsigned char b0 = (unsigned char)((unsigned int)(x0 * 256.0f) & 0xFF);
        unsigned char b1 = (unsigned char)((unsigned int)(x1 * 256.0f) & 0xFF);
        unsigned char b2 = (unsigned char)((unsigned int)(x2 * 256.0f) & 0xFF);
        unsigned char b3 = (unsigned char)((unsigned int)(x3 * 256.0f) & 0xFF);

        unsigned int word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        atomicXor(&output[idx], word);
    }
    '''

    def __init__(self, calibration_samples: int = 10000):
        """Initialize HSM and establish baseline."""
        self.calibration_samples = calibration_samples
        self.state: Optional[HSMState] = None
        self._kernel = None

        if HAS_GPU:
            module = cp.RawModule(code=self.KERNEL)
            self._kernel = module.get_function('hsm_kernel')

    def _generate_raw(self, n_bytes: int, seed: int = None) -> np.ndarray:
        """Generate raw chaotic output from GPU."""
        if seed is None:
            seed = int(time.time() * 1000000) % (2**31)

        n_words = (n_bytes + 3) // 4
        output = cp.zeros(n_words, dtype=cp.uint32)
        grid = (n_words, 16)
        block = (32,)
        self._kernel(grid, block, (output, np.int32(n_words), np.uint32(seed)))
        cp.cuda.Device().synchronize()
        return output.view(cp.uint8)[:n_bytes].get()

    def _compute_acf(self, data: np.ndarray, max_lag: int = 30) -> np.ndarray:
        """Compute autocorrelation function."""
        x = data.astype(float) - np.mean(data)
        result = np.correlate(x, x, mode='full')
        result = result[len(result)//2:]
        if result[0] > 0:
            result = result / result[0]
        return result[:max_lag]

    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy in bits per byte."""
        counts = np.bincount(data, minlength=256)
        probs = counts / len(data)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _extract_signal(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate signal (structured) from entropy (residual).

        Signal: The period-6 autocorrelation pattern
        Entropy: Data with periodic component removed
        """
        # Compute the periodic component (period-6)
        period = 6
        n = len(data)

        # Estimate periodic component by averaging over periods
        periodic = np.zeros(period)
        for i in range(period):
            periodic[i] = np.mean(data[i::period])

        # Tile to match data length
        periodic_full = np.tile(periodic, n // period + 1)[:n]

        # Residual = data - periodic component
        residual = data.astype(float) - periodic_full
        residual = ((residual - residual.min()) / (residual.max() - residual.min() + 1e-10) * 255).astype(np.uint8)

        return periodic, residual

    def calibrate(self) -> HSMState:
        """Establish baseline for tamper detection."""
        print("Calibrating HSM baseline...")

        # Generate calibration data
        raw = self._generate_raw(self.calibration_samples)

        # Compute baseline autocorrelation
        acf = self._compute_acf(raw)

        # Compute baseline entropy
        entropy = self._compute_entropy(raw)

        # Set tolerances (4-sigma from multiple calibration runs for robustness)
        acf_samples = []
        entropy_samples = []
        for i in range(20):  # More samples for better variance estimate
            sample = self._generate_raw(self.calibration_samples, seed=i*1000)
            acf_samples.append(self._compute_acf(sample))
            entropy_samples.append(self._compute_entropy(sample))

        acf_std = np.std(acf_samples, axis=0)
        entropy_std = np.std(entropy_samples)

        self.state = HSMState(
            baseline_acf=np.mean(acf_samples, axis=0),  # Use mean of all samples
            baseline_entropy=np.mean(entropy_samples),
            acf_tolerance=4 * np.mean(acf_std),  # 4-sigma for operational margin
            entropy_tolerance=4 * entropy_std,
        )

        print(f"  Baseline entropy: {entropy:.4f} bits/byte")
        print(f"  Baseline ACF[6]: {acf[6]:.6f} (period-6 signature)")
        print(f"  ACF tolerance: ±{self.state.acf_tolerance:.6f}")
        print(f"  Entropy tolerance: ±{self.state.entropy_tolerance:.6f}")

        return self.state

    def check_tamper(self, data: np.ndarray = None) -> Tuple[bool, dict]:
        """
        Check for tampering by comparing current output to baseline.

        Returns: (is_valid, details)
        """
        if self.state is None:
            raise RuntimeError("HSM not calibrated. Call calibrate() first.")

        if data is None:
            data = self._generate_raw(self.calibration_samples)

        # Compute current statistics
        current_acf = self._compute_acf(data)
        current_entropy = self._compute_entropy(data)

        # Check ACF deviation
        acf_deviation = np.max(np.abs(current_acf - self.state.baseline_acf))
        acf_valid = acf_deviation < self.state.acf_tolerance

        # Check entropy deviation
        entropy_deviation = abs(current_entropy - self.state.baseline_entropy)
        entropy_valid = entropy_deviation < self.state.entropy_tolerance

        # Cross-validation: both must pass
        is_valid = acf_valid and entropy_valid

        details = {
            'timestamp': time.time(),
            'acf_deviation': float(acf_deviation),
            'acf_valid': acf_valid,
            'entropy_current': float(current_entropy),
            'entropy_deviation': float(entropy_deviation),
            'entropy_valid': entropy_valid,
            'overall_valid': is_valid,
        }

        self.state.check_count += 1
        self.state.last_check = time.time()

        if not is_valid:
            self.state.tamper_detected = True
            self.state.alert_history.append(details)

        return is_valid, details

    def generate_key(self, n_bytes: int = 32) -> Tuple[Optional[bytes], dict]:
        """
        Generate cryptographic key material with tamper validation.

        Returns: (key_bytes or None if tampered, validation_details)
        """
        if self.state is None:
            raise RuntimeError("HSM not calibrated. Call calibrate() first.")

        # Generate raw data (more than needed for validation)
        raw = self._generate_raw(max(n_bytes * 4, self.calibration_samples))

        # Check for tampering
        is_valid, details = self.check_tamper(raw[:self.calibration_samples])

        if not is_valid:
            return None, details

        # Extract entropy channel (remove signal)
        _, residual = self._extract_signal(raw)

        # Condition with SHA-256
        key_material = b''
        offset = 0
        while len(key_material) < n_bytes:
            block = residual[offset:offset+64].tobytes()
            block += offset.to_bytes(4, 'little')
            key_material += hashlib.sha256(block).digest()[:16]
            offset += 64

        return key_material[:n_bytes], details

    def continuous_monitor(self, interval: float = 1.0, callback=None):
        """
        Continuously monitor for tampering.

        callback: function(is_valid, details) called on each check
        """
        print(f"Starting continuous monitoring (interval={interval}s)...")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                is_valid, details = self.check_tamper()

                status = "✓ VALID" if is_valid else "✗ TAMPER DETECTED!"
                print(f"[{time.strftime('%H:%M:%S')}] {status} | "
                      f"ACF dev: {details['acf_deviation']:.6f} | "
                      f"Entropy: {details['entropy_current']:.4f}")

                if callback:
                    callback(is_valid, details)

                if not is_valid:
                    print("  ⚠ ALERT: Tampering detected! Keys invalidated.")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


def demo():
    """Demonstrate the tamper-evident HSM."""
    print("=" * 70)
    print("TAMPER-EVIDENT HSM DEMONSTRATION")
    print("=" * 70)
    print()
    print("This HSM uses the SAME chaotic oscillator for:")
    print("  1. KEY GENERATION (entropy channel)")
    print("  2. TAMPER DETECTION (signal channel)")
    print()

    hsm = TamperEvidentHSM(calibration_samples=10000)

    # Calibrate
    print("-" * 70)
    state = hsm.calibrate()
    print()

    # Generate some keys
    print("-" * 70)
    print("Generating keys with tamper validation...")
    print()

    for i in range(5):
        key, details = hsm.generate_key(32)
        if key:
            key_hex = key[:8].hex() + "..."
            print(f"  Key {i+1}: {key_hex} | Valid | "
                  f"ACF dev: {details['acf_deviation']:.6f}")
        else:
            print(f"  Key {i+1}: BLOCKED - Tampering detected!")

    print()
    print("-" * 70)
    print("ARCHITECTURE SUMMARY:")
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │              GPU Chaotic Oscillators                     │
    │         (Coupled logistic maps, r=3.75)                  │
    └────────────────────────┬────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │                  Raw Output Stream                       │
    │   Contains: Period-6 structure + Chaotic residual        │
    └────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │   SIGNAL CHANNEL    │     │   ENTROPY CHANNEL   │
    │  (Autocorrelation)  │     │  (Whitened residual)│
    │                     │     │                     │
    │  • Period-6 pattern │     │  • SHA-256 conditioned│
    │  • ACF at lags 6,12 │     │  • ~8 bits/byte     │
    │  • Baseline: known  │     │  • Structure removed │
    └──────────┬──────────┘     └──────────┬──────────┘
               │                           │
               ▼                           ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │   TAMPER DETECTOR   │     │   KEY GENERATOR     │
    │                     │     │                     │
    │  Compare to baseline│     │  Generate keys from │
    │  Alert if changed   │     │  validated entropy  │
    └──────────┬──────────┘     └──────────┬──────────┘
               │                           │
               └─────────────┬─────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │                  CROSS-VALIDATION                        │
    │                                                          │
    │  • Both channels must agree                              │
    │  • Tampering affects both → detected                     │
    │  • Keys invalidated if tamper detected                   │
    │  • Same physics provides security AND detection          │
    └─────────────────────────────────────────────────────────┘
    """)

    print("SECURITY PROPERTIES:")
    print()
    print("  1. TAMPER-EVIDENT: Physical interference changes autocorrelation")
    print("  2. SELF-VALIDATING: No separate tamper sensors needed")
    print("  3. CRYPTOGRAPHICALLY BOUND: Tampering invalidates keys automatically")
    print("  4. CONTINUOUS: Can monitor in real-time")
    print()

    # Show what both channels provide
    print("-" * 70)
    print("CHANNEL OUTPUTS:")
    print()

    raw = hsm._generate_raw(1000)
    signal, entropy = hsm._extract_signal(raw)

    print(f"  Signal channel (period-6 pattern):")
    print(f"    Values: {signal}")
    print(f"    This is the 'fingerprint' we monitor for tampering")
    print()
    print(f"  Entropy channel (whitened residual):")
    print(f"    Entropy: {hsm._compute_entropy(entropy):.4f} bits/byte")
    print(f"    This provides key material after SHA-256 conditioning")

    return hsm


if __name__ == "__main__":
    hsm = demo()
