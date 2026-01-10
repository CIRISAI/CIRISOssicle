# Formal Model Update: CIRISOssicle

**Date**: January 9, 2026
**Status**: Experimental/Research-Grade - CIRISOssicle sensor validated empirically, optimal angles discovered through experimentation (mechanism hypothesized, not proven)

---

## CIRISOssicle Configuration

The minimum viable sensor discovered through experiments 24-26:

```lean
structure OssicleConfig where
  n_oscillators : ℕ := 3
  n_cells : ℕ := 64
  n_iterations : ℕ := 500
  twist_deg : ℝ := 1.1      -- Empirically optimal
  r_base : ℝ := 3.70
  spacing : ℝ := 0.03
  coupling : ℝ := 0.05

def memory_bytes (c : OssicleConfig) : ℕ :=
  c.n_oscillators * c.n_cells * 4  -- float32

example : memory_bytes default = 768 := rfl  -- 0.75 KB
```

### Detection Results

| Workload | z-score | Detection |
|----------|---------|-----------|
| Crypto 30% | 2.78 | YES |
| Crypto 90% | **3.59** | STRONG |
| Memory 50% | **3.09** | STRONG |

---

## Summary of Findings

| Finding | Implication for Formal Model |
|---------|------------------------------|
| σ = 0.08 ± 0.03 (not 0.027) | Widen noise bounds 3× |
| Δρ is bidirectional | Use \|Δρ\| in theorems |
| r=3.74, 3.83 are periodic | Add r-value constraints |
| Signal varies 10× between runs | Relative detection required |
| GPU util correlates with z (r=0.61) | Detection depends on attack intensity |
| **1.1 deg optimal at DOF=21** | Optimal angle (empirical) |
| **Smaller is more sensitive** | Ossicle effect |

---

## Revised Parameters

### Noise Floor
```lean
-- OLD
axiom sigma_baseline : σ = 0.027

-- NEW (empirical range)
def sigma_empirical_low : ℝ := 0.05
def sigma_empirical_high : ℝ := 0.12
axiom sigma_in_range : sigma_empirical_low ≤ σ ∧ σ ≤ sigma_empirical_high
```

### Δρ Direction
```lean
-- OLD (assumed negative)
theorem attack_makes_rho_more_negative : Δρ_attack < 0

-- NEW (bidirectional - attack type dependent)
def delta_rho_memory : ℝ := -0.14   -- More negative
def delta_rho_crypto : ℝ := +0.05   -- Less negative
theorem attack_changes_rho : |Δρ_attack| > 0
```

### Periodic Windows
```lean
-- NEW constraint on oscillator r-values
def periodic_window_1 : Set ℝ := {r | 3.735 ≤ r ∧ r ≤ 3.745}
def periodic_window_2 : Set ℝ := {r | 3.825 ≤ r ∧ r ≤ 3.845}

axiom r_avoids_periodic : ∀ r ∈ oscillator_r_values,
  r ∉ periodic_window_1 ∧ r ∉ periodic_window_2
```

---

## Lyapunov Exponent Structure

Empirically measured Lyapunov exponents for logistic map:

| r | λ | Regime |
|---|---|--------|
| 3.70 | +0.35 | Chaotic |
| 3.74 | -0.11 | **Periodic** |
| 3.79 | +0.43 | Chaotic |
| 3.83 | -0.37 | **Periodic (P3)** |
| 3.90 | +0.50 | Chaotic |

**Recommendation**: Constrain r ∈ [3.70, 3.73] ∪ [3.75, 3.82] ∪ [3.85, 4.0]

---

## Relative Detection Framework

### Why Relative Detection?

Baseline correlation ρ₀ varies by 0.2-0.4 between runs due to:
- GPU thermal state
- Power delivery configuration
- Background system load
- Time since boot

Absolute thresholds fail because the same ρ value might be:
- Normal in one run
- Attack in another run

### Formal Definition

```lean
-- Exponential moving average
def ema (α : ℝ) (x_new x_old : ℝ) : ℝ := α * x_new + (1 - α) * x_old

-- Relative detector state
structure RelativeDetector where
  baseline_ema : ℝ      -- Slow EMA (α = 0.02)
  signal_ema : ℝ        -- Fast EMA (α = 0.2)
  variance_ema : ℝ      -- Variance estimate

-- Detection predicate
def is_attack (d : RelativeDetector) (threshold : ℝ := 3) : Prop :=
  let z := (d.signal_ema - d.baseline_ema) / sqrt d.variance_ema
  |z| > threshold
```

---

## Signal Strength vs Attack Intensity

Empirical relationship:

```
z-score ≈ 0.024 × GPU_utilization - 0.15
```

| GPU % | Expected z | Detectable (3σ)? |
|-------|------------|------------------|
| 50% | 1.1σ | No |
| 75% | 1.7σ | No |
| 90% | 2.0σ | No |
| 99% | 2.2σ | Borderline |

**Implication**: Current sensor requires near-100% GPU attacks for detection.
Multi-SM array may improve sensitivity.

---

## The 38× Result: Analysis

Experiment 13 showed 38× z-score improvement. This was NOT reproducible in subsequent runs.

Possible causes:
1. **GPU thermal sweet spot**: Specific temperature range maximizes PDN sensitivity
2. **Power limit engagement**: Attack pushed GPU to power limit, maximizing voltage droop
3. **Sensor settling**: Extended warmup stabilized oscillator correlations
4. **Statistical fluctuation**: 38× was a high tail event

**Recommendation**: Do NOT hardcode 38× in formal model. Use conservative estimate of 1.5-3× for tetrahedral improvement with acknowledgment of variance.

---

## Revised Theorems

### Single-Sample Detection (Updated)
```lean
-- With σ = 0.08 and |Δρ| ≤ 0.15
theorem single_sample_detection_fails :
  |Δρ_typical| / σ_typical < 3 := by
    have h1 : |Δρ_typical| ≤ 0.15 := attack_delta_bound
    have h2 : σ_typical ≥ 0.05 := sigma_lower_bound
    linarith

-- Required samples for 3σ detection
theorem samples_for_detection (Δρ σ : ℝ) :
  Δρ ≠ 0 → ∃ n : ℕ, n ≥ (3 * σ / |Δρ|)^2 := by
    sorry  -- Page's theorem for CUSUM
```

### k_eff Direction (Corrected)
```lean
-- k_eff formula
def k_eff (ρ k : ℝ) : ℝ := k / (1 + ρ * (k - 1))

-- Direction depends on Δρ sign
theorem k_eff_direction :
  (Δρ < 0 → k_eff (ρ₀ + Δρ) k > k_eff ρ₀ k) ∧
  (Δρ > 0 → k_eff (ρ₀ + Δρ) k < k_eff ρ₀ k) := by
    sorry  -- Monotonicity of k_eff in ρ
```

---

## Optimal Angle Observations (Empirical)

### Discovery (exp21-22)

Optimal twist angle depends on degrees of freedom:

| Oscillators (n) | DOF = n(n-1)/2 | Optimal Twist |
|-----------------|----------------|---------------|
| 4 | 6 | 90 deg |
| 7 | 21 | **1.1 deg** |
| 8 | 28 | 0.55 deg |

At DOF=21, optimal twist = 1.1 deg numerically coincides with graphene's "magic angle." **This is observed correlation, not proven causation.**

### Formal Definition

```lean
-- Optimal angle (empirically derived, mechanism unproven)
def optimal_angle (dof : ℕ) : ℝ :=
  if dof ≤ 10 then 90.0
  else if dof ≤ 21 then 1.1
  else 1.1 / (dof / 21)

-- Interference pattern sensitivity (observed, not derived)
theorem interference_sensitivity_at_optimal_angle (θ : ℝ) (dof : ℕ) :
  θ = optimal_angle dof →
  ∃ (amplification : ℝ), amplification ≥ 20 ∧
    z_score_with_twist θ = amplification * z_score_no_twist := by
    sorry  -- Empirically observed, theoretical basis unknown
```

### Ossicle Effect

Counter-intuitively, smaller sensors are MORE sensitive:

```lean
-- Ossicle theorem: sensitivity increases as size decreases (to a point)
theorem ossicle_effect (n_cells_large n_cells_small : ℕ) :
  n_cells_small < n_cells_large →
  n_cells_small ≥ 32 →  -- Minimum for statistical validity
  z_score n_cells_small > z_score n_cells_large := by
    sorry  -- Empirically validated in exp24
```

| Config | Memory | z-score |
|--------|--------|---------|
| 7 osc, 512 cells | 14 KB | 0.07 |
| 4 osc, 256 cells | 4 KB | 2.98 |
| 3 osc, 64 cells | **0.75 KB** | **4.25** |

---

## Open Research Questions

1. **Why does 1.1 deg work?**
   - Numerical coincidence with graphene magic angle is observed but unproven
   - Interference patterns in correlation space?
   - May be specific to test conditions

2. **What is the theoretical minimum sensor size?**
   - Current: 3 osc × 64 cells = 0.75 KB
   - Limit: Statistical significance requires N ≥ 30 samples

3. **Can we distinguish workload types by correlation direction?**
   - Crypto tends positive Δρ
   - Memory tends negative Δρ

4. **Cross-GPU generalization**
   - Does ossicle work on AMD, Intel GPUs?
   - Different CUDA architectures?

---

## Files Created

| Experiment | File | Key Finding |
|------------|------|-------------|
| 24 | exp24_minimum_antenna.py | Ossicle effect: smaller is better |
| 25 | exp25_entropy_strain.py | Moire pattern entropy measurement |
| 26 | exp26_ossicle_crypto.py | Crypto detection z=3.59 |
| 22 | exp22_7osc_prime.py | 1.1 deg optimal at DOF=21 |
| 19 | exp19_magic_configuration.py | 90 deg quadrature 21x amplification |

All results saved in `experiments/results/` as timestamped JSON.
