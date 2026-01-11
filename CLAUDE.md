# CIRISOssicle - GPU Tamper Detection

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.
Commercial license required for larger organizations.
BSL text: https://mariadb.com/bsl11/

## What is CIRISOssicle?

A **GPU workload detector** that identifies unauthorized workloads (crypto mining, resource hijacking) using kernel execution timing. Named after the inner ear ossicles - tiny bones that are incredibly sensitive to vibration.

> **Research Update (January 2026):** Mechanism reframed based on experimental evidence. Detection works via **timing**, not PDN voltage coupling as originally hypothesized.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOW DETECTION ACTUALLY WORKS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Kernel ──► Timing (ns) ──┬──► Variance Shift ──► DETECTION   │
│                            │    (z=8.56 @ 90% load)             │
│                            │                                    │
│                            └──► LSB Extract ──► TRNG            │
│                                 (7.99 bits/byte)                │
│                                                                 │
│   NOT: PDN voltage → chaotic divergence → correlation shift    │
│   YES: Workload → timing contention → variance increase        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Capability Summary (Validated January 2026)

| Capability | Status | Evidence |
|------------|--------|----------|
| Workload detection (timing) | **VALIDATED** | z=8.56 at 90% load |
| TRNG from timing LSBs | **VALIDATED** | 7.99 bits/byte entropy |
| Cross-GPU coherence | **100% ALGORITHMIC** | Identical on RTX 4090 & Jetson |
| Twist angle relevance | **IRRELEVANT** | Detection identical at 0°, 45°, 90° |
| PDN voltage coupling | **NOT VALIDATED** | No evidence of chaotic amplification |

## Key Experimental Findings

### Exp 28: Cross-GPU Coherence Test
```
RTX 4090 vs Jetson Orin with identical seeds:
  seed=42:   -0.020063 vs -0.020063  EXACT MATCH
  seed=123:  -0.011182 vs -0.011182  EXACT MATCH
  ...

RESULT: Correlation structure is 100% ALGORITHMIC
        Not from hardware/PDN differences
```

### Exp 29: Timing vs Correlation Comparison
```
                Timing z    Correlation z
idle            0.13        2.06
load_30%        0.69        0.30
load_50%        2.02        0.87
load_70%        4.32        0.16
load_90%        8.56        3.50

RESULT: Timing is 3.2x MORE SENSITIVE than correlations
```

### Exp 30: Twist Angle Test
```
Timing delta under 70% load:
  0.0°  → +23.3 μs
  1.1°  → +27.0 μs  (the "magic angle")
  45.0° → +27.6 μs
  90.0° → +27.2 μs

RESULT: Twist angle is IRRELEVANT (std = 1.42 μs)
```

## Recommended Architecture

Based on experimental evidence, use `TimingSensor` (cleaner, simpler):

```python
from src.timing_sensor import TimingStrainGauge, TimingTRNG

# Strain gauge for workload detection (256 bytes, ~650k samples/sec)
gauge = TimingStrainGauge()
gauge.calibrate(duration=10.0)
result = gauge.measure(duration=5.0)
if result.detected:
    print(f"WORKLOAD DETECTED: z={result.timing_z:.2f}")

# TRNG with von Neumann debiasing
trng = TimingTRNG()
random_bytes = trng.generate_bytes(16)
```

The original `OssicleKernel` with chaotic oscillators still works but adds unnecessary complexity.

## Key Specs

| Sensor | Memory | Sample Rate | Detection (90% load) |
|--------|--------|-------------|---------------------|
| TimingSensor | **256 B** | ~650k/s | z=8.56 |
| OssicleKernel | 768 B | ~2k/s | z=3.50 |

## Quick Start

```bash
# NEW: Pure timing-based detection (recommended)
python src/timing_sensor.py

# Legacy: Chaotic oscillator detection
python experiments/exp26_ossicle_crypto.py

# Cross-GPU coherence test
python experiments/exp28_cross_gpu_coherence.py

# Timing vs correlation comparison
python experiments/exp29_timing_ossicle.py

# Twist angle relevance test
python experiments/exp30_twist_angle_test.py
```

## Key Files

| File | Purpose |
|------|---------|
| `src/timing_sensor.py` | **Pure timing-based sensor (recommended)** |
| `src/ossicle.py` | Original chaotic oscillator sensor |
| `experiments/exp28_cross_gpu_coherence.py` | Cross-GPU test |
| `experiments/exp29_timing_ossicle.py` | Timing vs correlation comparison |
| `experiments/exp30_twist_angle_test.py` | Twist angle relevance test |

## What We Learned

### Original Hypothesis (INVALIDATED)
- PDN voltage noise couples to chaotic oscillators
- Twist angle creates interference pattern
- Correlations encode physical state

### Actual Mechanism (VALIDATED)
- Workloads cause kernel timing contention
- Timing variance increases under load
- Chaotic oscillator acts as timing-sensitive sampler
- Twist angle is irrelevant

### Why Detection Still Works
The oscillator provides a consistent workload to time. Any GPU kernel would work - the chaos is incidental, not causal.

## Lessons from Array Characterization

Applied from CIRISArray spatial mapping experiments:

| Finding | Implication | Implementation |
|---------|-------------|----------------|
| Warm-up effect | Noise decreases as GPU warms | 30s warm-up before calibration |
| Fast timescale | ~2ms response for scheduling | Default detection window |
| Slow timescale | ~100ms for thermal effects | Optional long-window mode |
| Baseline drift | Track and compensate | Re-calibrate periodically |

```python
# Production usage with warm-up
config = TimingConfig(warm_up_enabled=True, warm_up_duration=30.0)
gauge = TimingStrainGauge(config)
gauge.calibrate(duration=10.0)  # Includes warm-up

# Quick testing (skip warm-up)
config = TimingConfig(warm_up_enabled=False)
gauge = TimingStrainGauge(config)
gauge.calibrate(duration=5.0, skip_warmup=True)
```

## Prior Art

This project builds on established research:

| Work | Year | Description |
|------|------|-------------|
| [US9459834B2](https://patents.google.com/patent/US9459834) | 2011 | GPU TRNG via timing/temperature (expired 2024) |
| [GPUs and chaos: TRNG](https://link.springer.com/article/10.1007/s11071-015-2287-7) | 2015 | Chaos + GPU timing for RNG (447 Mbit/s) |
| [ShadowScope](https://arxiv.org/abs/2509.00300) | 2025 | GPU monitoring via CUPTI side channels |
| [GPU Cryptojacking Detection](https://dl.acm.org/doi/10.1145/3577923.3583655) | 2023 | ML-based GPU workload detection |

**What's different here:** Lightweight (256 bytes), no CUPTI/nvidia-smi, dual-purpose (TRNG + detection).

## Related Projects

| Project | Description |
|---------|-------------|
| **RATCHET** | GPU Lorenz oscillator research (same timing findings) |
| **CIRISArray** | Multi-sensor spatial detection |

## Honest Science

This documentation was updated when experimental evidence contradicted the original PDN coupling hypothesis. The detection capability is real and validated, but the mechanism explanation has been corrected.

> "The first principle is that you must not fool yourself - and you are the easiest person to fool." - Richard Feynman
