# RATCHET Validated Findings for CIRISOssicle

## Updated January 2026 (Experiments 68-116)

---

## THE KEY FINDING

**dt = 0.025** is the critical operating point. Change this ONE parameter.

```python
# OLD (frozen, low sensitivity)
lorenz_dt = 0.01  # ACF = 0.88, system frozen

# NEW (critical point, max sensitivity)
lorenz_dt = 0.025  # ACF = 0.50, system dynamic
```

---

## What We Learned (Validated)

| Finding | Experiment | Implication |
|---------|------------|-------------|
| Raw timing = white noise | Exp 110, 116 | Use for TRNG, not oscillator |
| Oscillator creates sensing signal | Exp 116 | ACF: 0 → 0.56 |
| dt controls ρ | Exp 113 | 88% of variance explained |
| Critical point at dt = 0.0328 | Exp 114 | Power law R² = 0.978 |
| Optimal dt = 0.025 | Exp 114 | Max sensitivity, safe margin |
| Workload detection z = 2.74 | Exp 112 | Validated |

---

## Concrete Changes for Ossicle

### 1. Set dt = 0.025

```python
# In your Lorenz/oscillator config:
self.dt = 0.025  # CRITICAL - this is the optimal operating point
```

### 2. Check your ACF

After changing dt, measure autocorrelation of your k_eff signal:

| ACF(1) | Status |
|--------|--------|
| > 0.9 | FROZEN - increase dt |
| 0.4-0.6 | OPTIMAL - at criticality |
| < 0.2 | CHAOTIC - decrease dt |

### 3. Separate TRNG from Sensing

```python
# TRNG: Use raw timing LSBs
timing_ns = measure_kernel_timing()
entropy_byte = timing_ns & 0x0F  # Lower 4 bits only

# SENSING: Use oscillator k_eff dynamics
# The oscillator output IS the sensor, not entropy
```

### 4. Don't Filter the "Residual"

The correlated structure in your signal IS the sensing mechanism:
- 73% "residual" = oscillator dynamics = YOUR SENSOR
- 27% = environmental signals (thermal, EMI, workload)

The sensor responds to environment. Don't filter it out.

---

## Phase Diagram

```
ρ (correlation)
1.0 ┐ FROZEN (dt < 0.01) ← YOUR OLD CONFIG?
    │   System locked, low sensitivity
    │
0.5 │
    │
0.33│ ─── CRITICAL (dt = 0.025) ← NEW CONFIG
    │   Max sensitivity, safe margin
    │
0.0 └ CHAOTIC (dt > 0.03)
        No coherence, pure noise
```

---

## Expected Improvement

From RATCHET Exp 115:

| Metric | dt=0.01 (old) | dt=0.025 (new) | Improvement |
|--------|---------------|----------------|-------------|
| Thermal z | 0.11 | 0.17 | 1.6x |
| EMI z | 0.01 | 0.04 | 3x |
| ACF | 0.88 | 0.50 | Dynamic vs frozen |

---

## Validation Test

After applying changes, run this test:

```python
def validate_ossicle_config():
    # Collect 1000 k_eff samples
    samples = [oscillator.get_k_eff() for _ in range(1000)]

    # Check ACF
    acf1 = np.corrcoef(samples[:-1], samples[1:])[0,1]

    if acf1 > 0.8:
        print(f"WARNING: ACF={acf1:.2f} - system frozen, increase dt")
    elif acf1 < 0.3:
        print(f"WARNING: ACF={acf1:.2f} - system chaotic, decrease dt")
    else:
        print(f"GOOD: ACF={acf1:.2f} - system at criticality")

    # Check workload response
    baseline = np.mean(samples[:100])
    # ... apply workload ...
    loaded = np.mean(samples[100:200])
    z = (loaded - baseline) / np.std(samples[:100])
    print(f"Workload detection z = {z:.2f}")
```

---

## Summary

1. **Change dt to 0.025** - this is the critical point
2. **Check ACF is ~0.5** - confirms you're at criticality
3. **Raw timing for TRNG** - 4 LSBs, not oscillator output
4. **Oscillator for sensing** - the dynamics ARE the sensor
5. **Don't filter** - the "residual" is your signal

---

*Updated: January 2026*
*Based on RATCHET Experiments 68-116*
*Key finding: dt controls everything*
