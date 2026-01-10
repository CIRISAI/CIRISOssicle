# CIRISOssicle Experiments

## Overview

This directory contains experiments that led to the development of **CIRISOssicle** - a 0.75KB GPU tamper detection sensor.

## Key Experiments

### CIRISOssicle (Current)

| Experiment | Purpose | Key Finding |
|------------|---------|-------------|
| **exp26_ossicle_crypto.py** | Crypto mining detection | z=3.59 at 90% crypto |
| **exp25_entropy_strain.py** | Moire pattern entropy | Entropic vs negentropic strain |
| **exp24_minimum_antenna.py** | Minimum viable sensor | 0.75KB achieves z=4.25 |

### Magic Angle Discovery

| Experiment | Purpose | Key Finding |
|------------|---------|-------------|
| exp22_7osc_prime.py | 7-oscillator (DOF=21) | **1.1 deg magic angle** |
| exp21_8osc_magic_angle.py | 8-oscillator testing | 0.55 deg optimal |
| exp19_magic_configuration.py | Quadrature (90 deg) | 21x amplification |

### Foundation

| Experiment | Purpose | Key Finding |
|------------|---------|-------------|
| exp10_llm_tamper.py | LLM-scale detection | Works at inference scale |
| exp9_tamper_detector.py | Real-time tamper detection | 0.1s detection time |
| exp8_workload_fingerprint.py | Workload classification | Different fingerprints |

## Quick Start

```bash
# Run the main crypto detection experiment
python exp26_ossicle_crypto.py

# Characterize minimum viable sensor
python exp24_minimum_antenna.py

# Measure entropy strain
python exp25_entropy_strain.py
```

## Experiment History

1. **exp1-4**: Initial sensor characterization and parameter sweeps
2. **exp5-6**: PDN voltage noise discovery, spatial distribution
3. **exp7-8**: Noise floor characterization, workload fingerprinting
4. **exp9-10**: Tamper detection demonstrated
5. **exp11-14**: Formal model validation, tetrahedral sensor
6. **exp15-18**: Probe characterization, relative detection
7. **exp19-20**: Magic angle (90 deg quadrature) discovery
8. **exp21-22**: DOF scaling, 1.1 deg magic angle at DOF=21
9. **exp23**: Fractal/hierarchical oscillator arrays
10. **exp24**: Minimum viable sensor (ossicle hypothesis confirmed)
11. **exp25**: Entropic/negentropic strain via moire patterns
12. **exp26**: Crypto mining detection with ossicle

## CIRISOssicle Configuration

```python
OssicleKernel(
    n_cells=64,
    n_iterations=500,
    n_oscillators=3,  # DOF = 3
    twist_deg=1.1,    # Magic angle
    r_base=3.70,
    spacing=0.03,
    coupling=0.05
)
```

**Memory: 0.75 KB** (3 oscillators × 64 cells × 4 bytes)

## Results Directory

Experiment results are saved to `results/` as JSON files with timestamps.
