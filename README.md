# CIRISOssicle

**GPU workload detection via timing-based strain gauge.**

Software-only detection of unauthorized workloads (crypto mining, resource hijacking) - no external hardware required.

```
┌────────────────────────────────────────────────────────────────┐
│                      CIRISOssicle                              │
│                                                                │
│   Kernel ──► Timing ──┬──► Lorenz (dt=0.025) ──► DETECTION    │
│                       │    z=534-1652, 100% rate               │
│                       │    Workload discrimination: p<0.0001   │
│                       │                                        │
│                       └──► LSBs ──► TRNG                       │
│                            7.99 bits/byte                      │
│                                                                │
│   ACF=0.45 (critical point)   Validated January 2026           │
└────────────────────────────────────────────────────────────────┘
```

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.

## Detection Results (Validated January 2026)

| Workload | Timing z-score | Detection |
|----------|----------------|-----------|
| crypto 30% | 534 | **YES** |
| crypto 50% | 682 | **YES** |
| crypto 70% | 745 | **YES** |
| crypto 90% | 1652 | **YES** |

### Workload Discrimination

| Workload | Δ Timing (μs) | Distinguishable |
|----------|---------------|-----------------|
| crypto | +25.84 ± 1.48 | p < 0.0001 |
| memory | +11.00 ± 0.32 | p < 0.0001 |
| compute | +51.13 ± 1.06 | p < 0.0001 |

## Quick Start

```bash
pip install cupy-cuda12x numpy scipy

# Production strain gauge
python src/strain_gauge.py

# Run validation tests
python tests/test_strain_gauge.py
```

## Usage

```python
from src.strain_gauge import StrainGauge

# Workload detection
gauge = StrainGauge()
gauge.calibrate(duration=10.0)  # Includes warm-up
reading = gauge.read()
if reading.detected:
    print(f"WORKLOAD DETECTED: z={reading.timing_z:.2f}")

# Check system health (ACF should be ~0.5)
print(f"State: {reading.system_state}, ACF: {reading.acf:.2f}")

# TRNG
trng_result = gauge.generate_trng(16)
print(f"Random: {trng_result.bytes.hex()}")
```

## How It Works

Detection is based on **Lorenz oscillator at critical point (dt=0.025)**:

1. Run minimal GPU kernel, measure timing (nanoseconds)
2. Feed timing perturbations into Lorenz oscillator
3. Monitor ACF to ensure criticality (ACF ~0.5)
4. Detect workloads via timing distribution shift (z-scores)
5. Discriminate workload types by timing signature

Key insight from RATCHET: dt=0.025 is the critical operating point where sensitivity is maximized.

## Key Specs

| Metric | Value |
|--------|-------|
| Detection z-score | 534-1652 (100% rate) |
| Workload discrimination | p < 0.0001 |
| ACF at criticality | 0.45 |
| TRNG entropy | 7.99 bits/byte |

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

This project evolved through several phases:

| Phase | Claim | Result |
|-------|-------|--------|
| Original | PDN voltage coupling via correlations | NOT VALIDATED (Exp 28) |
| Original | "Magic angle" (1.1°) matters | IRRELEVANT (Exp 30) |
| Revised | Timing variance detection | VALIDATED |
| **Current** | **Lorenz at dt=0.025 critical point** | **VALIDATED (z=534-1652)** |
| **Current** | **Workload type discrimination** | **VALIDATED (p<0.0001)** |

The dt=0.025 critical point finding from RATCHET Experiments 68-116 dramatically improved sensitivity.

## Limitations

Does NOT detect:
- Workloads using <30% GPU (below tested threshold)
- CPU-only attacks
- Attacks when sensor isn't running
- Timing-aware evasion by sophisticated attackers

Note: Detection validated down to 30% GPU load with z=534.

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
