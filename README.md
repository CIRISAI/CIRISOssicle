# CIRISOssicle

**A 0.75KB GPU sensor that detects crypto mining and unauthorized workloads.**

Software-only intrusion detection using chaotic oscillator correlation fingerprinting - no external hardware required.

> **Note:** This is experimental, research-grade software. The detection mechanism and optimal parameters were discovered empirically. See "Research Status" below.

```
┌────────────────────────────────────────────────────────────────┐
│                      CIRISOssicle                              │
│                                                                │
│   3 oscillators    64 cells    500 iterations    0.75 KB      │
│                                                                │
│      A ────┬──── rho_AB ────┬──── B ────┬──── rho_BC ────┬──── C
│            │                │           │                │
│            └────────────────┴───────────┴────────────────┘
│                        Moire Pattern                           │
│                                                                │
│   Twist: 1.1 deg (empirically optimal)                         │
│   Like inner ear bones: tiny but incredibly sensitive          │
└────────────────────────────────────────────────────────────────┘
```

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.
Commercial license required for larger organizations.

## Detection Results

| Workload | Intensity | z-score | Detection |
|----------|-----------|---------|-----------|
| Crypto mining | 30% | 2.78 | YES |
| Crypto mining | 90% | **3.59** | **STRONG** |
| Memory bandwidth | 50% | **3.09** | **STRONG** |

**Minimum detectable crypto mining: 30% GPU utilization**

## Key Specifications

| Parameter | Value |
|-----------|-------|
| Memory footprint | **0.75 KB** |
| Oscillators | 3 (DOF = 3) |
| Cells per oscillator | 64 |
| Iterations per sample | 500 |
| Twist angle | 1.1 degrees |
| Sample rate | ~2000 samples/sec |
| Detection time | < 0.1 seconds |

## Quick Start

```bash
# Install dependencies
pip install cupy-cuda12x numpy scipy

# Run crypto mining detection
python experiments/exp26_ossicle_crypto.py

# Run entropy strain measurement
python experiments/exp25_entropy_strain.py

# Characterize minimum viable antenna
python experiments/exp24_minimum_antenna.py
```

## Capability Summary (Validated January 2026)

| Capability | Status | Evidence |
|------------|--------|----------|
| Local tamper detection | **VALIDATED** | p=0.007, mean shift detected |
| Reset improves sensitivity | **VALIDATED** | p=0.032, 7x improvement |
| Bounded noise floor | **VALIDATED** | σ=0.003 |
| Workload type classification | NOT VALIDATED | p=0.49, indistinguishable |
| Startup transient | NOT DETECTED | p=0.14 |

See `VALIDATION_RESULTS.md` for full null hypothesis test results.

## How It Works

### The Ossicle Analogy

Like the tiny bones in your inner ear (malleus, incus, stapes) that amplify sound vibrations, CIRISOssicle uses **minimal size for maximum sensitivity**.

1. **Three coupled chaotic oscillators** run on the GPU
2. Each oscillator is a 1D cellular automaton with logistic map dynamics
3. **1.1 degree twist** between oscillators (empirically discovered optimal)
4. Correlations between oscillators form interference patterns
5. **Power draw modulates EM coupling** - workloads affect coupling to ambient field
6. Unauthorized workloads create detectable power-draw signatures

**Key insight:** The oscillator doesn't detect workloads directly—it detects the workload's effect on local power draw, which modulates coupling to the ambient EM field. This physical side-channel is hard to spoof.

### Twist Angle

The 1.1 degree twist angle was discovered empirically to be optimal on RTX 4090. The numerical coincidence with graphene's "magic angle" is noted but **not proven to be causally related** - it may be coincidental.

- Creates interference patterns in correlation space
- Amplifies sensitivity to PDN perturbations
- Optimal angle varies by platform (~0.5 deg on Jetson Orin)

### Physical Mechanism

```
Unauthorized GPU workload (crypto mining)
    → Power draw pattern changes
    → Modulates coupling to ambient EM field
    → Oscillator correlation structure shifts
    → z-score > 2.0
    → DETECTED!
```

This is a **physical side-channel** that:
- Can't be spoofed by software-only attacks
- Is hard to mask without affecting the malicious workload
- Works regardless of what the unauthorized code is doing logically

## Related Projects

- **CIRISArray** - Research implementation for searching correlations across multiple ossicles. Enables spatial mapping and multi-sensor detection.

## Use Cases

- **Cloud GPU security** - Detect if your rented GPU is being hijacked for mining
- **Tamper-evident AI inference** - Verify no unauthorized code runs alongside your model
- **GPU integrity monitoring** - Continuous interference detection
- **Shared compute environments** - Detect resource theft in multi-tenant systems

## Limitations

CIRISOssicle does NOT detect:
- Workloads using <30% GPU (below detection threshold)
- CPU-only attacks (no GPU PDN impact)
- Attacks when sensor isn't running
- Power-matched evasion by sophisticated attackers

See `THREAT_MODEL.md` for complete security analysis.

## Experiments

| Experiment | Purpose |
|------------|---------|
| exp26_ossicle_crypto.py | Crypto mining detection |
| exp25_entropy_strain.py | Entropic/negentropic strain measurement |
| exp24_minimum_antenna.py | Minimum viable sensor characterization |
| exp22_7osc_prime.py | 7-oscillator (DOF=21) magic angle discovery |
| exp19_magic_configuration.py | Quadrature (90 deg) twist testing |

## Research Highlights

### Discovery Path

1. **exp9-10**: Tamper detection demonstrated with larger sensors
2. **exp19-20**: Magic angle (90 deg quadrature) improves detection 21x
3. **exp21-22**: At DOF=21 (7 oscillators), optimal twist = 1.1 deg (graphene magic angle!)
4. **exp24**: Smaller sensors are MORE sensitive - ossicle hypothesis confirmed
5. **exp25**: Sensor measures entropic strain through moire pattern deformation
6. **exp26**: Crypto mining detectable at 30% GPU utilization

### Key Findings

- **Smaller is better**: 0.75KB sensor outperforms larger configurations
- **1.1 degree optimal twist**: Empirically discovered (correlation with graphene magic angle is unproven)
- **Interference patterns**: Correlation patterns show workload-dependent shifts
- **Entropic strain**: Can distinguish ordered vs disordered workloads

## Platform-Specific Configurations

| Platform | Cells | Iterations | Twist | Memory | z-score |
|----------|-------|------------|-------|--------|---------|
| **RTX 4090** (4nm) | 64 | 500 | 1.1 deg | 0.75 KB | 3.59 |
| **Jetson Orin** (8nm) | 256 | 2000 | 0.5 deg | 3 KB | 3.55 |

The optimal twist angle appears to vary by platform (~1.1 deg for 4nm, ~0.5 deg for 8nm). The relationship to process node is observed but not explained.

## Requirements

- NVIDIA GPU with CUDA support
- CuPy (`pip install cupy-cuda12x`)
- NumPy, SciPy

**Tested on:** RTX 4090 (16GB), Jetson Orin (8GB)

## Documentation

| File | Contents |
|------|----------|
| `CLAUDE.md` | Project overview and quick reference |
| `PHYSICS_SUMMARY.md` | Physical mechanism details |
| `FORMAL_MODEL_UPDATE.md` | Mathematical model and theorems |
| `THREAT_MODEL.md` | What we detect/don't detect, security properties |
| `PRIOR_ART.md` | Timestamped innovation claims |
| `HYPOTHESES.md` | Research hypotheses |

## Why "Ossicle"?

The ossicles are the three smallest bones in the human body, located in the middle ear:
- **Malleus** (hammer)
- **Incus** (anvil)
- **Stapes** (stirrup)

Despite their tiny size (~3mm), they amplify sound vibrations 20x before transmitting to the cochlea. CIRISOssicle follows the same principle: **minimum size, maximum sensitivity**.

Our 0.75KB sensor (3 oscillators x 64 cells x 4 bytes) achieves z-scores > 3.5 for crypto mining detection - better than sensors 10x larger.

## Research Status

**This is experimental, research-grade software.**

| Aspect | Status |
|--------|--------|
| Detection mechanism | Empirically validated |
| Optimal parameters | Discovered through experimentation |
| 1.1° twist angle | Observed correlation with graphene magic angle, **causation unproven** |
| Platform scaling | Observed pattern, mechanism not established |
| PDN coupling theory | Hypothesized, not directly measured |

The software works (detects crypto mining with z > 3), but the underlying physics is not fully understood. Use accordingly.

## Citation

```bibtex
@software{cirisossicle2026,
  title = {CIRISOssicle: Sub-Kilobyte GPU Tamper Detection},
  author = {CIRIS L3C},
  year = {2026},
  license = {BSL-1.1}
}
```
