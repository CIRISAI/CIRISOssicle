# CIRISOssicle

**GPU workload detection via kernel timing variance.**

Software-only detection of unauthorized workloads (crypto mining, resource hijacking) - no external hardware required.

```
┌────────────────────────────────────────────────────────────────┐
│                      CIRISOssicle                              │
│                                                                │
│   Kernel ──► Timing (ns) ──┬──► Variance ──► DETECTION        │
│                            │    (z=8.56 @ 90% load)            │
│                            │                                   │
│                            └──► LSBs ──► TRNG                  │
│                                 (7.99 bits/byte)               │
│                                                                │
│   Memory: 256 bytes    Sample rate: ~650k/s                   │
└────────────────────────────────────────────────────────────────┘
```

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.

## Detection Results

| Workload | Timing z-score | Detection |
|----------|----------------|-----------|
| 30% load | 0.69 | - |
| 50% load | 2.02 | YES |
| 70% load | 4.32 | **STRONG** |
| 90% load | **8.56** | **STRONG** |

## Quick Start

```bash
pip install cupy-cuda12x numpy scipy

# Recommended: Pure timing-based detection
python src/timing_sensor.py

# Legacy: Chaotic oscillator (works but unnecessary complexity)
python experiments/exp26_ossicle_crypto.py
```

## Usage

```python
from src.timing_sensor import TimingStrainGauge, TimingTRNG

# Workload detection
gauge = TimingStrainGauge()
gauge.calibrate(duration=10.0)  # Run with system idle
result = gauge.measure(duration=5.0)
if result.detected:
    print(f"WORKLOAD DETECTED: z={result.timing_z:.2f}")

# TRNG (von Neumann debiased)
trng = TimingTRNG()
random_bytes = trng.generate_bytes(16)
```

## How It Works

Detection is based on **kernel execution timing variance**:

1. Run minimal GPU kernel repeatedly
2. Measure execution time (nanosecond precision)
3. Under load, timing variance increases due to scheduling contention
4. Detect via z-score on timing distribution shift

This is a well-established technique. See Prior Art section.

## Key Specs

| Sensor | Memory | Sample Rate | Detection (90% load) |
|--------|--------|-------------|---------------------|
| **TimingSensor** | 256 B | ~650k/s | z=8.56 |
| OssicleKernel (legacy) | 768 B | ~2k/s | z=3.50 |

## Prior Art

This project builds on established research:

| Work | Year | Relevance |
|------|------|-----------|
| [US9459834B2](https://patents.google.com/patent/US9459834) | 2011 | GPU TRNG via timing (expired 2024) |
| [GPUs and chaos: TRNG](https://link.springer.com/article/10.1007/s11071-015-2287-7) | 2015 | Chaos + timing for RNG |
| [ShadowScope](https://arxiv.org/abs/2509.00300) | 2025 | GPU monitoring via side channels |
| [GPU Cryptojacking Detection](https://dl.acm.org/doi/10.1145/3577923.3583655) | 2023 | ML-based detection |

**What's different here:** Lightweight implementation (256 bytes), no CUPTI/nvidia-smi dependency, dual-purpose (TRNG + detection).

## Research History

This project originally hypothesized that chaotic oscillators could detect PDN (Power Delivery Network) voltage changes through "correlation fingerprinting." Experiments revealed:

| Original Claim | Experimental Result |
|----------------|---------------------|
| PDN voltage coupling | NOT VALIDATED - cross-GPU test showed 100% algorithmic |
| "Magic angle" (1.1°) matters | IRRELEVANT - detection identical at 0°, 45°, 90° |
| Correlations encode physical state | FALSE - correlations are purely mathematical |
| Timing is the signal | **VALIDATED** - 3.2x more sensitive than correlations |

The detection capability is real, but the mechanism is simpler than originally thought: it's just timing variance from GPU scheduling contention.

## Limitations

Does NOT detect:
- Workloads using <50% GPU (below reliable threshold)
- CPU-only attacks
- Attacks when sensor isn't running
- Timing-aware evasion by sophisticated attackers

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

The ossicles are tiny bones in the middle ear that amplify vibrations. The name reflects the original goal of maximum sensitivity from minimum size. While the physics hypothesis didn't pan out, the principle of lightweight detection remains.

## Citation

```bibtex
@software{cirisossicle2026,
  title = {CIRISOssicle: GPU Workload Detection via Timing},
  author = {CIRIS L3C},
  year = {2026},
  license = {BSL-1.1},
  note = {Based on prior art in GPU timing side-channels}
}
```
