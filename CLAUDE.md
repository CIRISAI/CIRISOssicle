# CIRISOssicle - GPU Tamper Detection

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.
Commercial license required for larger organizations.
BSL text: https://mariadb.com/bsl11/

## What is CIRISOssicle?

A **0.75KB GPU sensor** that detects unauthorized workloads (crypto mining, resource hijacking) using chaotic oscillator correlation fingerprinting. Named after the inner ear ossicles - tiny bones that are incredibly sensitive to vibration.

> **Experimental/Research-Grade:** Detection works empirically but underlying physics not fully understood.

```
     ┌─────────────────────────────────────────┐
     │  CIRISOssicle: 3 oscillators, 0.75 KB   │
     │                                         │
     │    A ──┬── rho_AB ──┬── B ──┬── rho_BC ──┬── C
     │        │            │      │            │
     │        └────────────┴──────┴────────────┘
     │              Interference Pattern
     │                                         │
     │  Twist angle: 1.1 deg (empirical)       │
     │  Detection: z > 3.0 for crypto mining   │
     └─────────────────────────────────────────┘
```

## Key Specs

| Parameter | Value |
|-----------|-------|
| Memory footprint | **0.75 KB** |
| Oscillators | 3 |
| Cells per oscillator | 64 |
| Iterations per sample | 500 |
| Twist angle | 1.1 deg |
| Sample rate | ~2000/s |
| Detection time | < 0.1 seconds |
| Crypto detection | z=3.59 at 90% intensity |
| Min detectable crypto | 30% GPU utilization |

## Quick Start

```bash
# Crypto mining detection with ossicle
python experiments/exp26_ossicle_crypto.py

# Entropy strain measurement
python experiments/exp25_entropy_strain.py

# Minimum antenna characterization
python experiments/exp24_minimum_antenna.py
```

## How It Works

1. Three coupled chaotic oscillators with **1.1 degree twist** (empirically optimal)
2. Correlation structure encodes GPU PDN (Power Delivery Network) voltage noise
3. Unauthorized workloads shift correlations detectably
4. Like inner ear ossicles: tiny size amplifies sensitivity

## Detection Results

| Workload | z-score | Detection |
|----------|---------|-----------|
| crypto_30% | 2.78 | YES |
| crypto_90% | **3.59** | STRONG |
| memory_50% | **3.09** | STRONG |

## Key Files

| File | Purpose |
|------|---------|
| `experiments/exp26_ossicle_crypto.py` | Crypto mining detection |
| `experiments/exp25_entropy_strain.py` | Moire pattern entropy measurement |
| `experiments/exp24_minimum_antenna.py` | Minimum viable sensor characterization |
| `PHYSICS_SUMMARY.md` | Technical documentation |
| `FORMAL_MODEL_UPDATE.md` | Mathematical model |

## Related Projects

- **CIRISArray** - Research implementation for searching correlations across multiple ossicles. Enables spatial mapping and multi-sensor detection.

## Platform-Specific Configurations

### RTX 4090 (4nm Ada Lovelace)
```python
OssicleKernel(n_cells=64, n_iterations=500, twist_deg=1.1)  # 0.75 KB
```

### Jetson Orin (8nm Tegra Ampere)
```python
OssicleKernel(n_cells=256, n_iterations=2000, twist_deg=0.5)  # 3 KB
```

## Physics (Hypothesized)

The ossicle appears to exploit interference effects:
- Optimal twist angle varies with platform (1.1 deg on 4nm, 0.5 deg on 8nm)
- Creates interference pattern in correlation space
- PDN noise shifts the pattern detectably
- Smaller sensor = higher sensitivity (like ossicles!)

**Note:** The similarity to graphene's magic angle is observed correlation, not proven causation.
