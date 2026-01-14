# CIRISOssicle

**GPU workload detection via timing-based strain gauge.**

Software-only detection of unauthorized workloads (crypto mining, resource hijacking) - no external hardware required.

```
┌────────────────────────────────────────────────────────────────┐
│                      CIRISOssicle                              │
│                                                                │
│   Kernel ──► Timing ──┬──► WORKLOAD [VALIDATED]                │
│              4000 Hz  │    Mean shift >20%, latency 2.5 ms     │
│                       │                                        │
│                       ├──► EMI/VDF/THERMAL [THEORIZED]         │
│                       │    Spectrum analysis, not yet proven   │
│                       │                                        │
│                       └──► TRNG (7.99 bits/byte)               │
│                                                                │
│   Workload detection: O1-O7 Validated January 2026             │
└────────────────────────────────────────────────────────────────┘
```

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.

## What's Validated vs Theorized

| Capability | Status | Evidence |
|------------|--------|----------|
| **Workload Detection** | VALIDATED | O1-O7 experiments, 100% TP, 0% FP |
| EMI Detection | Theorized | Spectrum analysis implemented, not tested |
| VDF Detection | Theorized | Band 20-100 Hz hypothesized |
| Thermal Detection | Theorized | ACF drift observed, not characterized |

## Workload Detection Results (O1-O7 Validated)

Detection uses **mean shift** at 4000 Hz sample rate:

| Workload | Mean Shift | CV | Detection |
|----------|------------|-----|-----------|
| 1% | +192% | - | 100% TP |
| 30% | +188% | 13.6% | 100% TP |
| 50% | +248% | 13.6% | 100% TP |
| 70% | +317% | 3.4% | 100% TP |
| 90% | +380% | - | 100% TP |

**Key Specs:**

| Metric | Value | Source |
|--------|-------|--------|
| Sample rate | **4000 Hz** | O2 |
| Detection latency | **2.5 ms** | O4 |
| Detection floor | **1% workload** | O5 |
| Mean shift at 50% | +248% | O5 |
| CV at 70% load | 3.4% | O7 |
| Avoid zone | 1900-2100 Hz | O2b |

## Quick Start

```bash
pip install cupy-cuda12x numpy scipy

# Production strain gauge
python src/strain_gauge.py

# Run validation experiments
python experiments/o1_replicate_b1e.py
python experiments/o3_o5_4khz_validation.py
python experiments/o4_o7_latency_cv.py
```

## Usage

```python
from src.strain_gauge import StrainGauge

# Workload detection
gauge = StrainGauge()
gauge.calibrate(duration=10.0)  # Includes warm-up
reading = gauge.read()
if reading.detected:
    print(f"WORKLOAD DETECTED: mean_shift={reading.mean_shift_pct:.1f}%")

# Check system health (ACF should be ~0.45)
print(f"State: {reading.system_state}, ACF: {reading.acf:.2f}")

# TRNG
trng_result = gauge.generate_trng(16)
print(f"Random: {trng_result.bytes.hex()}")
```

## How It Works

Detection is based on **mean shift monitoring** at 4000 Hz:

1. Run minimal GPU kernel, measure timing (nanoseconds)
2. Sample at **4000 Hz** for optimal signal-to-noise
3. Detect workloads via **mean_shift > 20%**
4. GPU contention causes timing to **triple** (~7μs → ~21μs)
5. Detection latency: **2.5 ms** (10ms window)

Key insights:
- 4000 Hz has lowest variance (±14% vs ±41% at 1000 Hz)
- Avoid 1900-2100 Hz (interference dip at ~2050 Hz)
- Detection floor is 1% workload (+192% mean shift)

## Validation Summary (O1-O7)

| Exp | Test | Result | Status |
|-----|------|--------|--------|
| O1 | Mean shift at 1790 Hz | +153% | PASS |
| O2 | Optimal sample rate | 4000 Hz | PASS |
| O2b | Interference dip | ~2050 Hz | DOCUMENTED |
| O3 | Workload band at 4000 Hz | 7.1% | PASS |
| O4 | Detection latency | 2.5 ms | PASS |
| O5 | Detection floor | 1% | PASS |
| O7 | CV at 70% load | 3.4% | PASS |

## Distribution Finding

Z-scores are **Student-t distributed** (NOT Gaussian):
- Kurtosis κ = 230
- Degrees of freedom ≈ 1.3
- Individual z-scores unreliable, use mean shift

## Prior Art

| Work | Year | Relevance |
|------|------|-----------|
| [US9459834B2](https://patents.google.com/patent/US9459834) | 2011 | GPU TRNG via timing (expired 2024) |
| [GPUs and chaos: TRNG](https://link.springer.com/article/10.1007/s11071-015-2287-7) | 2015 | Chaos + timing for RNG |
| [ShadowScope](https://arxiv.org/abs/2509.00300) | 2025 | GPU monitoring via side channels |
| [GPU Cryptojacking Detection](https://dl.acm.org/doi/10.1145/3577923.3583655) | 2023 | ML-based detection |

**What's different here:** Lightweight implementation (256 bytes), no CUPTI/nvidia-smi dependency, 2.5ms detection latency, 1% detection floor.

## Limitations

- CV is higher at lower intensities (13.6% at 30-50% vs 3.4% at 70%)
- Avoid 1900-2100 Hz sample rates (GPU-specific interference)
- Validated on RTX 4090 only (other GPUs may differ)
- CPU-only attacks not detected
- Timing-aware evasion possible for sophisticated attackers

## Related Projects

| Project | Description |
|---------|-------------|
| **RATCHET** | GPU Lorenz oscillator research (same findings) |
| **CIRISArray** | Multi-sensor detection |

## Requirements

- NVIDIA GPU with CUDA
- CuPy (`pip install cupy-cuda12x`)
- NumPy, SciPy

## Why "Ossicle"?

The ossicles are tiny bones in the middle ear that amplify vibrations. The name reflects the original goal of maximum sensitivity from minimum size.

## Citation

```bibtex
@software{cirisossicle2026,
  title = {CIRISOssicle: GPU Workload Detection via Mean Shift},
  author = {CIRIS L3C},
  year = {2026},
  license = {BSL-1.1},
  note = {O1-O7 validated: 4000 Hz, 2.5ms latency, 1% floor}
}
```
