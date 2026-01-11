# CIRISOssicle Validation Results

**Date:** January 10, 2026
**Platform:** RTX 4090 (16GB)
**Methodology:** Null hypothesis testing with α = 0.05

## Summary Table

| Claim | H0 (Null) | Result | p-value | Effect Size |
|-------|-----------|--------|---------|-------------|
| Local tamper detection | No signal difference | **VALIDATED** | 0.007 | Mean shift -0.006 |
| Baseline noise floor | σ > 0.05 | **VALIDATED** | - | σ = 0.003 |
| Reset improves sensitivity | No improvement | **VALIDATED** | 0.032 | 7x z-score |
| Workload discrimination | Crypto = Memory | **NOT VALIDATED** | 0.49 | Indistinguishable |
| Startup transient | No transient | **NOT DETECTED** | 0.14 | Variance ratio 1.02x |

---

## Detailed Results

### CLAIM 1: Local Tamper Detection ✓

**Hypothesis:**
- H0: Workload does not change correlation
- H1: Workload changes correlation

**Method:** 10 trials baseline vs 10 trials with 90% crypto workload

**Results:**
- Baseline mean: -0.000192 ± 0.002920
- Workload mean: -0.006037 ± 0.004970
- t-statistic: 3.042
- **p-value: 0.007** (significant)

**Conclusion:** Tamper detection via correlation mean shift is **VALIDATED**.

---

### CLAIM 2: Baseline Noise Floor ✓

**Hypothesis:**
- H0: Noise floor too large (σ > 0.05)
- H1: Noise floor bounded (σ < 0.05)

**Method:** 8 consecutive baseline trials

**Results:**
- Mean correlation: ~0 (as expected for chaotic oscillators)
- Standard deviation: **0.003**
- 2σ detection threshold: 0.006
- 3σ detection threshold: 0.010

**Conclusion:** Noise floor is **VALIDATED** as bounded at σ ≈ 0.003.

---

### CLAIM 3: Reset Improves Sensitivity ✓

**Hypothesis:**
- H0: Reset has no effect on z-score
- H1: Reset improves z-score

**Method:** Compare detection z-scores with/without reset between trials

**Results:**
- Without reset: z = 0.08 ± 0.04
- With reset: z = 0.54 ± 0.35
- Improvement: **~7x**
- **p-value: 0.032** (significant)

**Conclusion:** Reset strategy is **VALIDATED**. Resetting oscillator states between measurements significantly improves detection sensitivity.

---

### CLAIM 4: Workload Discrimination ✗

**Hypothesis:**
- H0: Crypto and memory have same effect
- H1: Different workloads produce distinguishable signatures

**Method:** 6 trials crypto vs 6 trials memory, compare delta directions

**Results:**
- Crypto delta: +0.000281 ± 0.008115
- Memory delta: +0.003274 ± 0.004801
- **p-value: 0.49** (not significant)
- Both shift in same direction

**Conclusion:** **NOT VALIDATED**. Cannot distinguish crypto from memory workloads with current methodology.

---

### CLAIM 5: Startup Transient ✗

**Hypothesis:**
- H0: Early samples = Late samples (no transient)
- H1: Early samples differ (transient exists)

**Method:** Compare variance of first 2 seconds vs samples after 5 seconds

**Results:**
- Early variance: 0.00036929
- Late variance: 0.00036077
- Variance ratio: 1.02x
- **p-value: 0.14** (not significant)

**Conclusion:** **NOT DETECTED**. No evidence of startup transient in variance.

---

## Implications for Documentation

### Claims to Keep

1. **"Detects unauthorized GPU workloads"** - Validated (p=0.007)
2. **"Reset maintains sensitivity"** - Validated (p=0.032)
3. **"Noise floor ~0.003"** - Validated

### Claims to Remove or Revise

1. ~~"4:1 asymmetry between negentropic/entropic"~~ - Not validated
2. ~~"Startup transient requires warmup"~~ - Not detected
3. ~~"Can distinguish crypto from memory"~~ - Not validated

### Revised Capability Table

| Capability | Status | Evidence |
|------------|--------|----------|
| Local tamper detection | VALIDATED | p=0.007 |
| Reset strategy improves detection | VALIDATED | p=0.032, 7x improvement |
| Bounded noise floor | VALIDATED | σ=0.003 |
| Workload type classification | NOT VALIDATED | p=0.49 |
| Startup transient | NOT DETECTED | p=0.14 |

---

## Test Commands

```bash
# Run all validation tests
python -m pytest tests/test_claims.py -v -s

# Run individual tests
python -m pytest tests/test_claims.py::TestNullHypotheses::test_claim1_tamper_detection_variance -v -s
python -m pytest tests/test_claims.py::TestNullHypotheses::test_claim2_baseline_stability -v -s
python -m pytest tests/test_claims.py::TestNullHypotheses::test_claim4_reset_sensitivity -v -s
```

---

## Notes

- All tests run on RTX 4090 laptop GPU
- Crypto workload simulates SHA-256-like operations at 90% intensity
- Memory workload uses large buffer copies
- Statistical significance threshold: α = 0.05
