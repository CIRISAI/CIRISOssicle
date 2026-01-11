# CIRISOssicle Experiments

## Overview

This directory contains experiments that led to the development and characterization of CIRISOssicle - a GPU workload detector.

## Key Findings (January 2026)

| Experiment | Finding |
|------------|---------|
| **exp28** | Correlations are 100% ALGORITHMIC (not physical) |
| **exp29** | Timing is 3.2x more sensitive than correlations |
| **exp30** | Twist angle is IRRELEVANT to detection |

These findings led to the recommendation to use `TimingSensor` instead of the chaotic oscillator approach.

## Current Experiments

### Validation (Recommended)

| Experiment | Purpose | Key Finding |
|------------|---------|-------------|
| **exp28_cross_gpu_coherence.py** | Test algorithmic vs physical | 100% algorithmic |
| **exp29_timing_ossicle.py** | Compare timing vs correlations | Timing 3.2x better |
| **exp30_twist_angle_test.py** | Test if angle matters | Angle irrelevant |

### Detection (Working but Superseded)

| Experiment | Purpose | Notes |
|------------|---------|-------|
| exp26_ossicle_crypto.py | Crypto mining detection | Works, but use TimingSensor |
| exp24_minimum_antenna.py | Minimum viable sensor | Superseded by TimingSensor |

### Legacy (Historical Interest Only)

| Experiment | Original Purpose | Status |
|------------|------------------|--------|
| exp25_entropy_strain.py | "Entropy" measurement | INVALIDATED - no physical basis |
| exp22_7osc_prime.py | "Magic angle" discovery | INVALIDATED - angle irrelevant |
| exp21_8osc_magic_angle.py | 8-oscillator testing | INVALIDATED |
| exp19_magic_configuration.py | Quadrature testing | INVALIDATED |

## Quick Start

```bash
# Recommended: Use the timing sensor directly
python ../src/timing_sensor.py

# Validation experiments
python exp28_cross_gpu_coherence.py
python exp29_timing_ossicle.py
python exp30_twist_angle_test.py

# Legacy (still works, just unnecessary complexity)
python exp26_ossicle_crypto.py
```

## Recommended Configuration

```python
# NEW: Pure timing-based detection (recommended)
from src.timing_sensor import TimingStrainGauge

gauge = TimingStrainGauge()  # 256 bytes, ~650k samples/sec
gauge.calibrate(duration=10.0)
result = gauge.measure(duration=5.0)

# LEGACY: Chaotic oscillator (works but unnecessary)
from src.ossicle import OssicleKernel

kernel = OssicleKernel()  # 768 bytes, ~2k samples/sec
```

## Research History

The project went through several phases:

1. **exp1-10**: Initial development, assumed PDN voltage coupling
2. **exp11-22**: "Magic angle" and "entropy" hypotheses
3. **exp24-26**: Minimum sensor, crypto detection (empirically validated)
4. **exp28-30**: Mechanism investigation - discovered timing is the real signal

The final experiments (28-30) revealed that the original physics hypothesis was incorrect, but detection still works via timing variance.

## Results Directory

Experiment results are saved to `results/` as JSON files with timestamps.
