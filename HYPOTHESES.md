# CIRISOssicle: Hypotheses

## Confirmed Hypotheses

### H1: Correlation fingerprinting detects unauthorized workloads ✓

**Statement:** Different GPU workloads create distinguishable correlation signatures in coupled chaotic oscillators.

**Result:** CONFIRMED (exp8-10)
- Crypto mining: detectable at z=3.59
- Memory bandwidth: detectable at z=3.09
- Detection time: < 0.1 seconds

---

### H2: Smaller sensors are more sensitive (Ossicle Effect) ✓

**Statement:** Reducing sensor size (cells, oscillators) increases sensitivity, not decreases it.

**Result:** CONFIRMED (exp24)

| Config | Memory | z-score |
|--------|--------|---------|
| 7 osc, 512 cells | 14 KB | 0.07 |
| 4 osc, 256 cells | 4 KB | 2.98 |
| 3 osc, 64 cells | **0.75 KB** | **4.25** |

**Interpretation:** Like inner ear ossicles, minimal mass enables maximum sensitivity.

---

### H3: Magic angle exists at 1.1 degrees ✓

**Statement:** There exists an optimal twist angle that maximizes detection sensitivity.

**Result:** CONFIRMED (exp21-22)
- At DOF=21 (7 oscillators), optimal twist = 1.1 deg
- Matches graphene's magic angle exactly!
- At DOF=6 (4 oscillators), 90 deg (quadrature) is optimal

---

### H4: Correlation patterns form moire interference ✓

**Statement:** The three correlation pairs (ρ_AB, ρ_BC, ρ_AC) create a moire-like interference pattern affected by PDN strain.

**Result:** CONFIRMED (exp25)
- Entropic strain blurs pattern (correlations → 0)
- Negentropic strain sharpens pattern (correlations → ±1)
- Sensor acts as coherence interferometer

---

## Open Hypotheses

### H5: Workload types have distinguishable signatures

**Statement:** Crypto mining and memory bandwidth create different correlation shift directions.

**Test:** Compare sign of Δρ across workload types.

**Preliminary:**
- Crypto tends toward positive Δρ
- Memory tends toward negative Δρ
- Needs more data for statistical confirmation

**Status:** Partially tested

---

### H6: Cross-GPU generalization

**Statement:** CIRISOssicle works on GPUs other than RTX 4090.

**Test:**
1. Deploy on Jetson Orin (ARM GPU)
2. Test on AMD GPU
3. Test on older NVIDIA architectures

**Status:** Untested

---

### H7: Magic angle depends on DOF

**Statement:** Optimal twist angle is a function of degrees of freedom: θ_opt = f(DOF)

**Observed:**
| DOF | Optimal θ |
|-----|-----------|
| 6 | 90 deg |
| 21 | 1.1 deg |
| 28 | 0.55 deg |

**Hypothesis:** θ_opt ∝ 1/DOF for DOF > 10

**Status:** Partially tested, needs theoretical explanation

---

### H8: Minimum viable size

**Statement:** There exists a theoretical lower bound on sensor size.

**Current minimum:** 3 osc × 64 cells = 0.75 KB

**Test:**
1. Try 3 osc × 32 cells (0.375 KB)
2. Try 3 osc × 16 cells (0.1875 KB)
3. Find where detection breaks down

**Status:** Untested

---

## Disproven Hypotheses

### H_old: Physical acceleration detection

**Original claim:** GPU chaotic oscillators can detect physical shaking.

**Result:** DISPROVEN
- The "6× shake effect" was actually GPU load variation
- No evidence of sub-g acceleration sensitivity
- PDN voltage noise explains all observations

---

## Research Questions

1. **Why 1.1 degrees?** Connection to twisted bilayer graphene?
2. **Is there a theoretical basis for the ossicle effect?** Information theory?
3. **Can we predict optimal parameters from first principles?** Lyapunov analysis?
4. **What is the detection limit?** Minimum GPU utilization detectable?
