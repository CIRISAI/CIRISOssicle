# Recommendations from RATCHET Characterization

## Summary

The RATCHET project (GPU-based Lorenz chaotic resonator) conducted rigorous characterization experiments (28-68) that revealed important findings applicable to CIRISOssicle.

**Key RATCHET Finding**: 78% of oscillator coherence is ALGORITHMIC (same math), not physical coupling. Only 4% came from GPU-specific timing, and even that turned out to be floating point noise when we tried reference subtraction.

---

## Critical Finding: Algorithmic vs Physical Signal

### What We Discovered

| Component | Contribution | Origin |
|-----------|--------------|--------|
| Algorithmic | 78% | Same oscillator equations on both |
| GPU-specific | ~4% | Likely floating point noise |
| Environmental | <0.5% | 60 Hz power grid (common-mode) |
| Noise | 17.5% | Numerical/measurement |

**Implication for CIRISOssicle**: Your three logistic map oscillators likely have similar coherence budget - most of the correlation structure comes from running the same `r * x * (1-x)` math, not from physical PDN coupling.

### How We Validated This

**Exp 65-67**: Cross-GPU coherence test
- Same GPU: 82.6% coherence
- Cross GPU (different machines): 78% coherence
- **Conclusion**: The 78% is algorithmic (same math). Only 4.6% was GPU-specific.

**Exp 68**: Reference subtraction test
- Ran oscillator WITH timing coupling vs WITHOUT
- Difference signal: ~0.5% (floating point noise)
- Timing perturbations (~1e-5) too small for chaotic divergence

---

## Recommended Experiments for CIRISOssicle

### Exp A: Cross-GPU Coherence Test

**Question**: How much of your correlation signal is algorithmic?

**Method**:
1. Run identical OssicleKernel on two different GPUs (different machines)
2. Same seeds, same parameters
3. Measure correlation between oscillators across machines

**Prediction**: Correlations will be nearly identical (within ~5%), proving algorithmic origin.

**If confirmed**: Most of what you're detecting isn't PDN coupling - it's workload affecting *execution timing* which affects *which iteration you sample*.

---

### Exp B: Reference Subtraction

**Question**: Can we isolate the physical signal?

**Method**:
```python
class DifferentialOssicle:
    def __init__(self):
        self.actual = OssicleKernel()  # Normal operation
        self.reference = OssicleKernel()  # CPU-side, no GPU timing

    def step(self):
        actual_corr = self.actual.step()
        # Run reference with SAME random state but on CPU
        ref_corr = self.reference.step_cpu()
        return actual_corr - ref_corr  # Physical signal only
```

**RATCHET result**: Differential was ~0.5% (noise). This may differ for logistic map.

---

### Exp C: Timing Extraction Test

**Question**: Where does the detection signal actually come from?

**Method**:
1. Record kernel execution times for each `step()`
2. Correlate timing variance with correlation shifts
3. Test: Does timing alone predict correlation changes?

**Hypothesis**: Detection works because workloads change kernel timing, which affects *when* you sample the chaotic trajectory, not because PDN voltage changes the dynamics.

---

### Exp D: Iteration Accumulation Test

**Question**: Does the logistic map amplify perturbations?

**Method**:
1. Run two oscillators with identical initial conditions
2. Perturb one by 1e-5 at step 0
3. Measure divergence after N iterations
4. With Lyapunov λ, expect divergence = e^(λN) × perturbation

**For logistic map at r=3.7**: λ ≈ 0.5/iteration
- After 10 iterations: 1e-5 × e^5 ≈ 1.5e-3
- After 20 iterations: 1e-5 × e^10 ≈ 0.2

So logistic map should show divergence faster than Lorenz (λ=0.178). Test this!

---

## Proposed Architecture Updates

### Option 1: Direct Timing TRNG (Bypass Chaos)

If the chaotic dynamics don't amplify the physical signal, extract timing directly:

```python
class TimingOssicle:
    def step(self):
        start = time.perf_counter_ns()
        # Run minimal kernel
        self.kernel()
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter_ns()

        # Extract LSBs of timing
        timing_lsb = (end - start) & 0xFF

        # Use chaotic map as mixer/whitener only
        self.state = self.r * self.state * (1 - self.state)
        self.state = (self.state + timing_lsb / 256) % 1.0

        return self.state
```

**Advantage**: Timing is the TRUE physical signal. Chaos whitens it.

---

### Option 2: Differential Correlation

Run reference oscillator in parallel, detect deviations:

```python
class DifferentialOssicle:
    def __init__(self):
        self.sensor = OssicleKernel()
        self.baseline_corr = None
        self.baseline_window = []

    def step(self):
        corr = self.compute_correlation()

        if len(self.baseline_window) < 100:
            self.baseline_window.append(corr)
            return 0  # Calibrating

        if self.baseline_corr is None:
            self.baseline_corr = np.mean(self.baseline_window)

        # Return deviation from baseline
        return corr - self.baseline_corr
```

**Note**: This is what you're already doing with z-score detection. The question is whether the baseline drift is physical or algorithmic.

---

### Option 3: Multi-Rate Sampling

Sample at different rates to distinguish timing effects from PDN effects:

```python
def multi_rate_test():
    fast_samples = []  # 10000 iterations between samples
    slow_samples = []  # 100 iterations between samples

    # If PDN is the signal: fast and slow should show same workload response
    # If timing is the signal: fast will show less because more averaging
```

---

## What CIRISOssicle Detection Probably Is

Based on RATCHET findings, here's our hypothesis for why CIRISOssicle works:

1. **Not PDN voltage changes** - perturbations too small for chaotic amplification
2. **Not EM coupling** - environmental signals cancel in correlation
3. **Probably timing/scheduling** - workloads change:
   - Kernel launch latency
   - Memory access patterns
   - SM scheduling
   - These affect WHEN you sample the chaotic trajectory

**The oscillator acts as a timing-sensitive sampler**, not a PDN sensor.

This is still useful for tamper detection! But it means:
- The "1.1 degree twist" may be less important than thought
- The detection is fundamentally a timing side-channel
- Reference subtraction may not help (timing affects both equally)

---

## Recommended Validation Sequence

| Priority | Experiment | Purpose |
|----------|------------|---------|
| 1 | Cross-GPU coherence | Measure algorithmic fraction |
| 2 | Timing correlation | Test if timing predicts detection |
| 3 | Perturbation divergence | Validate chaotic amplification |
| 4 | Reference subtraction | Try to isolate physical signal |
| 5 | Multi-rate sampling | Distinguish timing from PDN |

---

## Key Questions to Answer

1. **What fraction of correlation is algorithmic?**
   - RATCHET: 78%
   - CIRISOssicle: Unknown (test with cross-GPU)

2. **Does the logistic map amplify perturbations?**
   - RATCHET Lorenz: No (timing too small)
   - Logistic map λ is higher, may work better

3. **Is detection from PDN or timing?**
   - RATCHET: Timing (but too small)
   - CIRISOssicle: Likely timing, but detection works empirically

4. **Can reference subtraction improve SNR?**
   - RATCHET: No (gave noise)
   - CIRISOssicle: Worth testing

---

## Files to Review

From RATCHET characterization:
- `../RATCHET/experiments/REVISED_EXPERIMENTS.md` - Our new experiment proposals
- `../RATCHET/formal/RATCHET/GPUTamper/EnvironmentalCoherence.lean` - Formal proofs
- `../RATCHET/experiments/INSTRUMENT_UPGRADE_RECOMMENDATIONS.md` - Upgrade recommendations

---

*Generated: January 2026*
*Based on RATCHET Experiments 28-68*
