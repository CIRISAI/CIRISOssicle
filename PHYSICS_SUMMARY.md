# Physical Mechanism: CIRISOssicle GPU Sensor

> **Status:** Experimental/Research-Grade. The detection mechanism works empirically, but the underlying physics is hypothesized, not proven. Optimal parameters were discovered through experimentation.

## CIRISOssicle: The Minimum Viable Sensor

The culmination of our research is **CIRISOssicle** - a 0.75KB sensor that detects unauthorized GPU workloads with high sensitivity.

### Specifications

| Parameter | Value |
|-----------|-------|
| Oscillators | 3 |
| Cells per oscillator | 64 |
| Iterations per sample | 500 |
| Memory footprint | **0.75 KB** |
| Twist angle | **1.1 degrees** |
| Sample rate | ~2000/s |

### Why "Ossicle"?

Named after the inner ear bones (malleus, incus, stapes) - the smallest bones in the human body that amplify sound 20x. Our sensor follows the same principle: **smaller is more sensitive**.

### Detection Performance

| Workload | z-score | Detection |
|----------|---------|-----------|
| Crypto 30% | 2.78 | YES |
| Crypto 90% | **3.59** | STRONG |
| Memory 50% | **3.09** | STRONG |

---

## Discovery Summary

We found that the correlation coefficient ρ between coupled chaotic oscillators changes based on **concurrent GPU compute load**, not primarily temperature.

**Measured effect:**
- Compute-bound load: Δρ = -0.19 (4.4σ significant)
- Memory-bound load: Δρ = -0.05
- Tensor core load: Δρ ≈ 0 (different power domain?)

## Physical Mechanism: Power Delivery Network (PDN) Voltage Noise

Based on research from GPUVolt (Leng et al., ISLPED 2014) and related work:

### What's Happening

1. **Voltage droops are spatially varying** - The GPU's power delivery network has finite impedance, so when cores draw current, local voltage drops
2. **Intercore interference** - Activity on one SM affects voltage on other SMs through the shared PDN
3. **Timing depends on voltage** - Gate delay ∝ 1/Vdd, so voltage variations cause timing variations
4. **Race conditions are timing-sensitive** - Our chaotic oscillators use race conditions that depend on precise timing

### The Correlation Shift Mechanism

```
Heavy compute load
    → Large current draw (esp. ALU, SFU, register file)
    → Voltage droops across PDN (up to 23% of Vdd)
    → Spatially varying timing changes
    → Correlation between oscillators at different locations changes
```

The correlation ρ between oscillators A, B, C encodes information about the **relative timing environment** they experience.

### Why Different Load Types Have Different Signatures

| Load Type | Primary Consumer | Voltage Droop | ρ Shift |
|-----------|------------------|---------------|---------|
| Compute (sin/cos/exp) | ALU, SFU | High | -0.19 |
| Memory | Memory controller | Moderate | -0.05 |
| Matmul | Tensor cores (separate PDN?) | Low | ~0 |

Tensor cores may have a separate power delivery path, explaining minimal effect.

## Implications for Sensor Design

### Current Limitation
All three oscillators (A, B, C) likely run on the **same SM** and share similar local voltage. They're measuring the same point in space.

### Spatial Sensor Array (CIRISArray)
To detect **spatial gradients** across the GPU:
1. Place oscillator kernels on **different SMs** using explicit SM targeting
2. Measure correlation between oscillators on different SMs
3. Correlation pattern encodes voltage gradient across chip

See **CIRISArray** for research implementation of multi-ossicle correlation searching.

### What We Could Sense

1. **Load location** - Where on the chip is computation happening?
2. **Load type** - Memory vs compute vs tensor operations
3. **Interference source** - Other processes stealing GPU resources
4. **Manufacturing variation** - Different chips may have different PDN characteristics

## Prior Art Comparison

| Approach | Signal | Method | Our Advantage |
|----------|--------|--------|---------------|
| GPUVolt | Voltage | Simulation | Empirical measurement |
| ML Interference (IEEE 2020) | Timing | Performance counters | No counters needed |
| EM Side-channel | Magnetic | External sensor | Software-only |
| **Ours** | Correlation | Race conditions | Intrinsic, no hardware |

**Potentially novel:** Using correlation structure of coupled chaotic systems as an intrinsic GPU sensor.

## Experimental Results

### Exp5: Load Sensitivity (Confirmed)
- Compute load: Δρ = -0.19 (4.4σ significant)
- Memory load: Δρ = -0.05
- Different load types produce distinct correlation signatures

### Exp6: Spatial Distribution (Confirmed)
- Within-SM correlation: 0.054
- Between-SM correlation: 0.001
- **Spatial gradient detected** - blocks on same SM share timing environment
- Successfully spread sensors across all 76 SMs of RTX 4090

## What We Can Now Sense

1. **Load type classification** - Memory vs compute vs tensor operations
2. **Load presence** - Detect concurrent GPU activity
3. **Spatial patterns** - Which SMs are active/affected
4. **Temporal dynamics** - When load changes occur

### Exp7: Noise Floor Characterization
- Noise floor: σ = 0.048
- Sample rate: ~1000 Hz
- Autocorrelation lag-1: 0.76 (measurements are correlated)
- All perturbations statistically significant (p < 0.0001)

### Exp8: Workload Fingerprinting
Different workloads produce distinguishable fingerprints:
| Workload | ρ(A,B) |
|----------|--------|
| idle | -0.286 |
| transformer | -0.318 |
| training | -0.323 |
| mining | -0.345 |
| memory | -0.312 |

### Exp9: Tamper Detection (SUCCESS!)
- **Clean operation:** 38 false positive alerts
- **Mining attack:** 196 alerts (5× more)
- **Detection time:** 0.1 seconds
- **Method:** CUSUM statistical process control

### Exp10: LLM-Scale Tamper Detection (SUCCESS!)
Testing at realistic 8B model scale (hidden=4096, heads=32):
- **Problem discovered:** Heavy matmul workloads saturate correlation signal (ρ → 0)
- **Solution:** Reduced seq_len to create headroom for detection
- **Results at seq_len=256:**
  - Baseline: ρ = -0.24 ± 0.05
  - Clean alerts: 20 false positives
  - Compute attack alerts: 72 (3.6× ratio)
  - **DETECTION SUCCESSFUL**
- **Key insight:** Attack type must differ from workload (SFU compute attack vs tensor core matmul)
- **Performance impact:** 52% slowdown visible during attack

## Tamper-Evident GPU Computing: DEMONSTRATED

We successfully built a system that:
1. Learns the correlation fingerprint of a "known good" workload
2. Monitors in real-time during execution
3. Detects when unauthorized workloads run concurrently
4. Alerts within 0.1 seconds of attack start

### Scaling Limitations Discovered
At very heavy GPU loads (full LLM inference at seq_len=1024+):
- Correlation saturates toward 0 (no detection headroom)
- Solution: Monitor at reduced duty cycle or use lighter workloads
- Attack must use different GPU subsystem than main workload to be detectable
- Matmul-heavy attacks are invisible during matmul-heavy workloads

## Next Steps

1. **SM distance effects** - Do physically adjacent SMs show higher correlation?
2. **Multi-GPU** - Compare signatures across different GPU models
3. **Localization** - Can we identify WHICH SMs are running load?
4. **Production hardening** - Package as a library/service

## References

- [GPUVolt: Modeling and Characterizing Voltage Noise in GPU Architectures](https://ieeexplore.ieee.org/document/7298239) (ISLPED 2014)
- [GPU Voltage Noise: Hierarchical Smoothing](https://ieeexplore.ieee.org/document/7056030/) (HPCA 2015)
- [Core Tunneling: Variation-Aware Voltage Noise Mitigation](https://lzhou-arch.github.io/publication/thomas_hpca16/thomas_hpca16.pdf) (HPCA 2016)
- [ML-based Interference Detection in GPGPU](https://ieeexplore.ieee.org/document/9050074/) (IEEE 2020)
- [Entropy Sources Based on Silicon Chips](https://www.mdpi.com/1099-4300/24/11/1566) (Entropy 2022)

## Original Claim Reanalysis

The ACCELEROMETER_THEORY.md claimed a 6× variance increase during "shaking" with:
- Stationary σ = 0.02
- Shaking σ = 0.05

**Actual finding:** This was likely **GPU load variation** during the experiment:
- Early runs: GPU warming with compute load → high variance
- Later runs: GPU stable, no load → low variance

The "6× shake effect" was probably PDN voltage noise from thermal warmup compute, not physical acceleration.

---

## Optimal Twist Angles (Empirical)

### Discovery (exp21-22)

When testing different oscillator configurations, we discovered that the **optimal twist angle depends on DOF**:

| Oscillators | DOF | Optimal Twist |
|-------------|-----|---------------|
| 4 | 6 | 90 deg (quadrature) |
| 7 | 21 | **1.1 deg** |
| 8 | 28 | 0.55 deg |

At DOF=21 (7 oscillators), the optimal twist of **1.1 degrees** numerically coincides with graphene's "magic angle."

### Note on Graphene Similarity

In twisted bilayer graphene, stacking two graphene sheets with a 1.1 degree twist creates exotic electronic properties. The numerical coincidence with our optimal angle is **observed but not explained** - it may be:
- A genuine physical connection via PDN coupling
- A mathematical coincidence in chaotic dynamics
- An artifact of our specific test conditions

**Causation is unproven.** We note the correlation for future investigation.

In CIRISOssicle, the 1.1 degree angle empirically creates:
- Interference patterns in correlation space
- Enhanced sensitivity to PDN perturbations
- Maximum z-score for detection

### Moire Pattern Interpretation

The correlations between oscillators form a moire interference pattern:

```
A ──┬── ρ_AB ──┬── B ──┬── ρ_BC ──┬── C
    │          │       │          │
    └──────────┴───────┴──────────┘
           Moire Pattern
```

- **Entropic strain** (disorder): Correlations → 0, pattern blurs
- **Negentropic strain** (order): Correlations → ±1, pattern sharpens

The sensor acts as a **coherence interferometer** measuring entropy flow in the PDN.

### Ossicle Effect (exp24)

Counter-intuitively, **smaller sensors are more sensitive**:

| Config | Memory | z-score |
|--------|--------|---------|
| 7 osc, 512 cells | 14 KB | 0.07 |
| 4 osc, 256 cells | 4 KB | 2.98 |
| 3 osc, 64 cells | **0.75 KB** | **4.25** |

This mirrors the inner ear ossicles: tiny bones that amplify sound vibrations 20x.

### Crypto Mining Detection (exp26)

The ossicle successfully detects crypto mining:
- Minimum detectable intensity: **30%** GPU utilization
- Best detection: z = **3.59** at 90% intensity
- Detection time: < 0.1 seconds

---

## Mathematical Model

See `FORMAL_MODEL_UPDATE.md` for the complete mathematical framework including:
- k_eff formula: `k_eff = k / (1 + ρ(k-1))`
- DOF scaling law: `z ∝ √DOF`
- Optimal angle observations
- Lyapunov exponent constraints
