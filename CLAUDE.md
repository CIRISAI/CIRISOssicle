# CIRISOssicle - GPU Strain Gauge

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.

## Production Architecture (RATCHET Validated)

Based on RATCHET Experiments 68-116:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STRAIN GAUGE ON TRNG                             │
│                                                                     │
│   GPU Kernel Timing (nanoseconds)                                   │
│           │                                                         │
│           ├──► Lower 4 LSBs ──► TRNG                                │
│           │                     • 465 kbps, 6/6 NIST                │
│           │                     • 7.99 bits/byte                    │
│           │                     • True jitter entropy               │
│           │                                                         │
│           └──► Lorenz Oscillator (dt=0.025) ──► STRAIN GAUGE        │
│                                                  • z=2.74 detection │
│                                                  • ACF ~0.5 critical│
│                                                  • k_eff dynamics   │
│                                                                     │
│   CRITICAL: dt = 0.025 controls everything                          │
│   • dt < 0.01: FROZEN (ACF > 0.9, useless)                         │
│   • dt = 0.025: CRITICAL (ACF ~0.5, max sensitivity)               │
│   • dt > 0.03: CHAOTIC (ACF < 0.2, no coherence)                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from src.strain_gauge import StrainGauge

# Create with optimal config (dt=0.025 default)
gauge = StrainGauge()

# Calibrate (includes 30s warm-up)
gauge.calibrate(duration=10.0)

# Continuous monitoring
gauge.monitor()

# Or single readings
reading = gauge.read()
if reading.detected:
    print(f"ANOMALY: z={reading.timing_z:.2f}")

# Check health (ACF should be ~0.5)
if reading.system_state != "CRITICAL":
    print(f"WARNING: {reading.system_state}, ACF={reading.acf:.2f}")

# Generate TRNG
trng = gauge.generate_trng(16)
print(f"Random: {trng.bytes.hex()}")
```

## Key Finding: dt Controls Everything

From RATCHET Exp 113-114:

| dt | ACF | State | Use |
|----|-----|-------|-----|
| < 0.01 | > 0.9 | FROZEN | ❌ Useless |
| **0.025** | **~0.5** | **CRITICAL** | ✅ Max sensitivity |
| > 0.03 | < 0.2 | CHAOTIC | ❌ No coherence |

**Power law validated:** ρ = 39.64 × |dt - 0.0328|^1.09 + 0.33 (R² = 0.978)

## Validated Capabilities

| Capability | Experiment | Result | Status |
|------------|------------|--------|--------|
| **TRNG** | Exp 73 | 7.998/8 bits, 6/6 NIST, 465 kbps | **VALIDATED** |
| **Strain Gauge** | Exp 74, 112 | z=2.74 detection | **VALIDATED** |
| **Critical Point** | Exp 114 | dt=0.025, R²=0.978 | **VALIDATED** |
| **ACF Health** | Exp 113 | 88% variance explained | **VALIDATED** |

## File Structure

```
src/
├── strain_gauge.py    # PRODUCTION - Use this
├── timing_sensor.py   # Superseded (timing-only)
└── ossicle.py         # Deprecated (logistic map)
```

## Configuration

```python
from src.strain_gauge import StrainGaugeConfig

config = StrainGaugeConfig(
    dt=0.025,              # CRITICAL - don't change unless ACF wrong
    trng_bits=4,           # Lower 4 bits for TRNG
    warm_up_enabled=True,  # 30s GPU warm-up
    warm_up_duration=30.0,
    acf_target=0.5,        # Target ACF at criticality
)
```

## Health Monitoring

The strain gauge monitors its own health via ACF:

```python
reading = gauge.read()

if reading.system_state == "FROZEN":
    # ACF > 0.8 - increase dt
    print("System frozen, increase dt")

elif reading.system_state == "CHAOTIC":
    # ACF < 0.3 - decrease dt
    print("System chaotic, decrease dt")

elif reading.system_state == "CRITICAL":
    # ACF ~0.5 - optimal
    print("System optimal")
```

## Dual Output

| Output | Source | Use |
|--------|--------|-----|
| **TRNG** | timing[3:0] | True random numbers (465 kbps) |
| **Strain** | Lorenz k_eff | Environmental detection (z=2.74) |

The oscillator doesn't generate entropy - it **detects strain** in the timing environment through k_eff dynamics.

## Prior Art

| Work | Year | Relevance |
|------|------|-----------|
| [US9459834B2](https://patents.google.com/patent/US9459834) | 2011 | GPU TRNG via timing (expired 2024) |
| [ShadowScope](https://arxiv.org/abs/2509.00300) | 2025 | GPU monitoring via side channels |

**What's different:** Lorenz at critical point (dt=0.025) for max sensitivity, ACF health monitoring, dual TRNG+strain output.

## Research History

| Phase | Finding | Status |
|-------|---------|--------|
| Original | PDN coupling via correlations | ❌ INVALIDATED |
| Exp 28 | Correlations 100% algorithmic | ✅ VALIDATED |
| Exp 29 | Timing 3.2x more sensitive | ✅ VALIDATED |
| Exp 30 | Twist angle irrelevant | ✅ VALIDATED |
| **Exp 113-114** | **dt=0.025 critical point** | ✅ **VALIDATED** |

## CIRIS Integration

For protecting CIRIS agents:

```python
from src.strain_gauge import StrainGauge

class CIRISAgentProtector:
    def __init__(self):
        self.gauge = StrainGauge()
        self.gauge.calibrate()

    def check_environment(self) -> bool:
        reading = self.gauge.read()

        # Check for tampering
        if reading.detected:
            return False  # Environment compromised

        # Check sensor health
        if reading.system_state != "CRITICAL":
            # Sensor degraded - recalibrate
            self.gauge.calibrate()

        return True  # Environment safe
```

## Citation

```bibtex
@software{cirisossicle2026,
  title = {CIRISOssicle: GPU Strain Gauge at Critical Point},
  author = {CIRIS L3C},
  year = {2026},
  license = {BSL-1.1},
  note = {Based on RATCHET Experiments 68-116, dt=0.025 critical point}
}
```
