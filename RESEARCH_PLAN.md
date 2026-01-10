# CIRISOssicle: Research Plan

## Project Summary

**CIRISOssicle** is a 0.75KB GPU sensor that detects unauthorized workloads (crypto mining, resource hijacking) using chaotic oscillator correlation fingerprinting.

**Related:** CIRISArray - Research implementation for searching correlations across multiple ossicles.

### Key Achievements

| Milestone | Result |
|-----------|--------|
| Crypto detection | z=3.59 at 90% intensity |
| Minimum detectable | 30% GPU utilization |
| Memory footprint | **0.75 KB** |
| Detection time | < 0.1 seconds |
| Magic angle | 1.1 deg (graphene!) |

---

## Current Configuration

```python
OssicleKernel(
    n_oscillators=3,      # DOF = 3
    n_cells=64,
    n_iterations=500,
    twist_deg=1.1,        # Magic angle
    r_base=3.70,
    spacing=0.03,
    coupling=0.05
)
```

---

## Research Phases

### Phase 1: Foundation (COMPLETE)

1. ✓ PDN voltage noise discovery (exp5-6)
2. ✓ Workload fingerprinting (exp8)
3. ✓ Tamper detection demonstrated (exp9-10)
4. ✓ LLM-scale validation (exp10)

### Phase 2: Optimization (COMPLETE)

1. ✓ Magic angle discovery (exp19-22)
   - 90 deg quadrature at DOF=6
   - 1.1 deg at DOF=21 (graphene magic angle!)
2. ✓ Ossicle effect (exp24)
   - Smaller sensors are MORE sensitive
   - 0.75KB achieves z=4.25
3. ✓ Crypto mining detection (exp26)
   - z=3.59 at 90% intensity
   - Minimum 30% detectable

### Phase 3: Production (IN PROGRESS)

1. □ Package as library/service
2. □ Cross-GPU validation
3. □ Long-term stability testing
4. □ False positive characterization

### Phase 4: Commercialization (FUTURE)

1. □ API documentation
2. □ Integration examples
3. □ Performance benchmarks
4. □ License compliance tools

---

## Open Research Questions

### Theoretical

1. **Why 1.1 degrees?**
   - Connection to twisted bilayer graphene?
   - Moire interference in correlation space?
   - Flat band analogy?

2. **Ossicle effect mechanism**
   - Why does smaller = more sensitive?
   - Information-theoretic explanation?
   - Noise averaging vs signal coherence?

3. **DOF scaling law**
   - Observed: z ∝ √DOF (from formal model)
   - But ossicle (DOF=3) beats larger sensors
   - Non-monotonic relationship?

### Practical

1. **Cross-GPU generalization**
   - Does ossicle work on AMD GPUs?
   - Jetson Orin (ARM) validation?
   - Older NVIDIA architectures?

2. **Attack classification**
   - Can we distinguish crypto from memory?
   - Workload fingerprinting with ossicle?
   - Multi-class detection?

3. **Evasion resistance**
   - Can attackers hide their workload?
   - Low-and-slow attacks?
   - Mimicry attacks?

---

## Next Experiments

### exp27: Cross-GPU Validation

Deploy ossicle on:
- Jetson Orin (ARM GPU)
- RTX 3080 (Ampere)
- A100 (datacenter)

### exp28: Minimum Size Limit

Test smaller configurations:
- 3 osc × 32 cells (0.375 KB)
- 3 osc × 16 cells (0.1875 KB)
- Find detection breakdown point

### exp29: Attack Classification

Train classifier to distinguish:
- Crypto mining
- Memory bandwidth
- Compute (matmul)
- Mixed workloads

### exp30: Long-term Stability

Run continuous monitoring for 24+ hours:
- Track baseline drift
- False positive rate
- Detection consistency

---

## Hardware

### Primary Development
- RTX 4090 Laptop (16GB)
- CUDA 12.x
- CuPy, NumPy, SciPy

### Remote Testing
- Jetson Orin (jetson.local)
- SSH: `ssh emoore@jetson.local`

---

## Key Files

| File | Purpose |
|------|---------|
| experiments/exp26_ossicle_crypto.py | Crypto detection |
| experiments/exp25_entropy_strain.py | Entropy measurement |
| experiments/exp24_minimum_antenna.py | Ossicle discovery |
| experiments/exp22_7osc_prime.py | Magic angle |
| PHYSICS_SUMMARY.md | Physical mechanism |
| FORMAL_MODEL_UPDATE.md | Mathematical model |

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Memory footprint | < 1 KB | 0.75 KB ✓ |
| Detection z-score | > 3.0 | 3.59 ✓ |
| Min detectable intensity | < 50% | 30% ✓ |
| Detection time | < 1 sec | < 0.1 sec ✓ |
| False positive rate | < 5% | TBD |
| Cross-GPU support | 3+ GPUs | 1 GPU |

---

## References

- GPUVolt (Leng et al., ISLPED 2014) - PDN voltage noise
- Twisted Bilayer Graphene - Magic angle physics
- CUSUM Process Control - Detection algorithm
- Chaos Theory - Lyapunov exponents, logistic map
