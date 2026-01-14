# CIRISOssicle - GPU Strain Gauge

**License:** Business Source License 1.1 (BSL 1.1)
**Licensor:** CIRIS L3C (Eric Moore)
**Change Date:** January 1, 2030
**Change License:** AGPL 3.0

Free for individuals, DIY, academics, nonprofits, and orgs <$1M revenue.

## Production Architecture (O1-O7 Validated January 2026)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-MODAL STRAIN GAUGE                         │
│                                                                     │
│   GPU Kernel Timing ──┬──► WORKLOAD [VALIDATED]                     │
│                       │    • 4000 Hz, mean shift >20%               │
│                       │    • +248% at 50% load, +380% at 90%        │
│                       │    • Latency: 2.5 ms, Floor: 1%             │
│                       │                                             │
│                       ├──► EMI [THEORIZED]                          │
│                       │    • 500 Hz sample rate, --emi mode         │
│                       │    • 60 Hz harmonics, subharmonics, VRM     │
│                       │                                             │
│                       ├──► VDF [THEORIZED]                          │
│                       │    • Voltage/frequency scaling              │
│                       │    • Band 20-100 Hz in spectrum             │
│                       │                                             │
│                       ├──► THERMAL [THEORIZED]                      │
│                       │    • Drift compensation via ACF feedback    │
│                       │                                             │
│                       └──► TRNG (lower 4 LSBs, 7.99 bits/byte)      │
│                                                                     │
│   AVOID: 1900-2100 Hz sample rates (interference dip at ~2050 Hz)   │
└─────────────────────────────────────────────────────────────────────┘
```

**Status:**
- **WORKLOAD**: O1-O7 validated, production ready
- **EMI/VDF/THERMAL**: Theorized capabilities, not yet experimentally validated

## Key Specs (O1-O7 Validated)

| Metric | Value | Source |
|--------|-------|--------|
| Sample rate | **4000 Hz** | O2 |
| Detection latency | **2.5 ms** | O4 |
| Detection floor | **1% workload** | O5 |
| Mean shift at 50% | +248% | O5 |
| Mean shift at 90% | +380% | O5 |
| CV at 70% load | 3.4% | O7 |
| Avoid zone | 1900-2100 Hz | O2b |

## Quick Start

```python
from src.strain_gauge import StrainGauge

gauge = StrainGauge()
gauge.calibrate(duration=10.0)

reading = gauge.read()
if reading.detected:
    print(f"WORKLOAD: mean_shift={reading.mean_shift_pct:.1f}%")
```

### EMI Mode (Experimental)

**Note:** EMI detection is theorized but not yet validated. The mode exists for experimentation.

```bash
# Run EMI spectrum analysis (30s capture at 500 Hz)
python src/strain_gauge.py --emi

# Custom duration and sample rate
python src/strain_gauge.py --emi --emi-duration 60 --emi-rate 1000
```

```python
from src.strain_gauge import emi_mode

# Analyze spectrum for power grid harmonics and VRM switching
results = emi_mode(duration=30, sample_rate=500)
print(f"60 Hz harmonics: {results['n_harmonics']}/4")
print(f"VRM peaks: {len(results['vrm_peaks'])}")
```

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

## Configuration

```python
from src.strain_gauge import StrainGaugeConfig

config = StrainGaugeConfig(
    sample_rate=4000,           # O2 validated
    detection_window_ms=10,     # O4: 2.5ms latency
    mean_shift_threshold=20.0,  # >20% = workload detected
    dt=0.025,                   # Lorenz critical point
    warm_up_enabled=True,
)
```

## File Structure

```
src/
├── strain_gauge.py        # PRODUCTION - Use this

experiments/
├── o1_replicate_b1e.py    # O1: Mean shift validation
├── o2_sample_rate_sweep.py # O2: Sample rate optimization
├── o2b_investigate_2khz.py # O2b: 2050 Hz dip investigation
├── o3_o5_4khz_validation.py # O3+O5: 4000 Hz validation
├── o4_o7_latency_cv.py    # O4+O7: Latency and CV tests
```

## Research History

| Phase | Finding | Status |
|-------|---------|--------|
| Original | PDN coupling via correlations | INVALIDATED |
| Exp 28 | Correlations 100% algorithmic | VALIDATED |
| O1 | Mean shift +153% at 1790 Hz | VALIDATED |
| O2 | 4000 Hz optimal (lowest variance) | VALIDATED |
| O2b | Avoid 1900-2100 Hz (dip) | VALIDATED |
| O4 | 2.5 ms detection latency | VALIDATED |
| O5 | 1% detection floor | VALIDATED |
| O7 | 3.4% CV at 70% load | VALIDATED |

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
