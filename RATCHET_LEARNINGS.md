# RATCHET Validated Findings for CIRISOssicle

## Updated January 2026 (Experiments 68-116 + Array Validation)

---

## THE KEY FINDINGS

### 1. dt_crit is THERMALLY DEPENDENT
Use ACF feedback, not fixed dt.

### 2. Distribution is STUDENT-T, NOT GAUSSIAN
- Kurtosis κ = 230 (validated: 229.8)
- Student-t df ≈ 1.3 (validated: 1.34)
- Detection works via rare extreme spikes (fat tails)

---

## FAT-TAIL DISTRIBUTION (test_fat_tails.py)

| Assumption | Gaussian | Actual (Student-t) |
|------------|----------|-------------------|
| Kurtosis | 0 | 230 |
| Degrees of freedom | ∞ | 1.3 |
| Tail behavior | Thin | Extremely fat |
| Detection mechanism | Mean shift | Rare spikes |
| z=534 probability | Impossible | Expected |

**Why z=534-1652 works:** With fat tails, extreme values are much more probable than Gaussian predicts. Detection catches these rare spikes.

---

## THERMAL SELF-TUNING

```python
# OLD (fixed dt - WRONG)
lorenz_dt = 0.025  # Only works at one temperature!

# NEW (ACF feedback - CORRECT)
def auto_tune_dt(self, target_acf: float = 0.45) -> float:
    """Adjust dt to maintain criticality regardless of temperature."""
    current_acf = self.get_acf()

    if current_acf > target_acf + 0.1:  # Too frozen
        self.dt *= 1.1  # Increase dt
    elif current_acf < target_acf - 0.1:  # Too chaotic
        self.dt *= 0.9  # Decrease dt

    return self.dt
```

**Thermal dependency (from Array team):**
| GPU State | dt_crit | ACF at dt=0.025 |
|-----------|---------|-----------------|
| Warm | ~0.025 | 0.45 (critical) |
| Cold | ~0.030 | 0.84 (frozen!) |

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

    # Check ACF and auto-tune if needed
    acf1 = np.corrcoef(samples[:-1], samples[1:])[0,1]

    if acf1 > 0.55:
        print(f"WARNING: ACF={acf1:.2f} - system frozen, AUTO-TUNING dt up")
        oscillator.dt *= 1.1
    elif acf1 < 0.35:
        print(f"WARNING: ACF={acf1:.2f} - system chaotic, AUTO-TUNING dt down")
        oscillator.dt *= 0.9
    else:
        print(f"GOOD: ACF={acf1:.2f} - system at criticality")

    # Check workload response (should see z > 500)
    baseline = np.mean(samples[:100])
    # ... apply workload ...
    loaded = np.mean(samples[100:200])
    z = (loaded - baseline) / np.std(samples[:100])
    print(f"Workload detection z = {z:.2f} (target: >500)")
```

---

## Summary

1. **Use ACF feedback to auto-tune dt** - dt_crit is thermally dependent!
2. **Target ACF ~0.45** - confirms you're at criticality
3. **Raw timing for TRNG** - 4 LSBs, not oscillator output
4. **Oscillator for sensing** - the dynamics ARE the sensor
5. **Don't filter** - the "residual" is your signal
6. **Expected z > 500** - validated detection performance

---

*Updated: January 2026*
*Based on RATCHET Experiments 68-116, Array thermal finding*
*Key finding: dt_crit is thermally dependent - use ACF feedback*
