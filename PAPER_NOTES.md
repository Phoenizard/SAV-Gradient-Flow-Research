# Paper Notes: Key Settings for Reproduction

**Reference:** Ma, Mao & Shen (2024). *Efficient and stable SAV-based methods for gradient flows arising from deep learning.* Journal of Computational Physics 505, 112911. DOI: 10.1016/j.jcp.2024.112911

This file contains **verbatim or closely paraphrased** settings from the paper. Claude Code must follow these exactly during Phase 1. Do not deviate without a PAUSE_REPORT.

---

## Model Architecture (Section 2, Eq. 2)

The neural network is a **one-hidden-layer network**:

$$f(\theta; x) = \frac{1}{m} \sum_{k=1}^{m} a_k \sigma(w_k \cdot x + b_k)$$

where:
- $\sigma$ = **ReLU** activation (paper uses ReLU for DL examples)
- $m$ = number of neurons in the hidden layer (varies per experiment, see below)
- Parameters $\theta = \{w_k, b_k, a_k\}_{k=1}^m$
- $w_k \in \mathbb{R}^D$, $b_k \in \mathbb{R}$, $a_k \in \mathbb{R}$
- **CRITICAL: The 1/m factor is part of the architecture, not just notation**

**Implementation note:** Bias $b_k$ can be absorbed into $w_k$ by augmenting $x$ with a constant 1, giving $W \in \mathbb{R}^{(D+1) \times m}$ and $a \in \mathbb{R}^{m \times 1}$.

**Data preprocessing:** Z-score normalization: $x \Rightarrow (x - \mu) / \sigma$, where $\mu$ and $\sigma$ are computed from the training set.

**Training loss (I(θ) for optimization):** Standard MSE:

$$I(\theta) = \frac{1}{N}\sum_{i=1}^{N} (f(\theta; x_i) - y_i)^2$$

**Evaluation metric (for figures):** Relative error:

$$\text{RelError}(\theta) = \frac{\sum_{i=1}^{N} (f(\theta; x_i) - y_i)^2}{\sum_{i=1}^{N} y_i^2}$$

**Note:** The paper's figures plot relative error on the y-axis, but the gradient flow
and SAV formulations use MSE (or sum-of-squares) as I(θ). Using relative error as
the training loss suppresses gradients by a factor of N/Σy² — negligible for Example 1
(y~O(1)) but catastrophic for Example 2 (y~O(100), factor ~28000x slower).

**Data split:** 80% training, 20% testing from a single pool of M data points.

---

## Example 1: Sin + Cos Regression (Section 3.1.1)

**Target function:**
$$f^*(x_1, \ldots, x_D) = \sin\!\left(\sum_{i=1}^{D} p_i x_i\right) + \cos\!\left(\sum_{i=1}^{D} q_i x_i\right)$$

where $p_i, q_i$ are fixed random coefficients drawn from $\text{Uniform}(0,1)$ and **fixed** (use seed).

**Data:**
- Input $x \sim \text{Uniform}(0, 1)^D$
- No noise added (clean targets)

**Random seed:** Fix seed = 42 for data generation and model initialization.

---

## Example 2: Polynomial Regression (Section 3.1.2)

**Target function:**
$$f^*(x_1, \ldots, x_D) = \sum_{i=1}^{D} c_i x_i^2$$

where $c_i \sim \text{Uniform}(0, 1)$, fixed with seed.

**Data:**
- Input $x \sim \text{Uniform}(0, 5)^D$
- No noise added

---

## Per-Figure Experimental Settings

All figures use: ReLU activation, Z-score normalized inputs, MSE training loss, relative error for evaluation, 80/20 train/test split from M total.

| Fig | Example | D  | m    | M      | l (batch) | lr     | C   | λ  | Epochs | PM methods shown        |
|-----|---------|----|------|--------|-----------|--------|-----|----|--------|-------------------------|
| 1   | Ex1     | 20 | 1000 | 10000  | 8000(full)| 0.6    | 1   | 10 | 8000   | Euler,SAV,RelSAV        |
| 2   | Ex1     | 40 | 1000 | 10000  | 64        | 0.2    | 1   | 0  | 10000  | Euler,SAV               |
| 3   | Ex1     | 40 | 1000 | 100000 | 256       | 0.5/1.0| 100 | 4  | 10000  | (SPM only)              |
| 5   | Ex1     | 40 | 1000 | 100000 | 256       | 0.5    | 100 | varies | 10000 | SAV (λ study)       |
| 7   | Ex2     | 40 | 100  | 10000  | 64        | 0.01   | 1   | 0  | 10000  | Euler,SAV               |
| 8   | Ex2     | 40 | 100  | 10000  | 64        | 0.4/1.0| 1   | 4  | 10000  | (SPM only)              |

**Note:** We implement PM (particle method) only. SPM requires Gaussian smoothing kernel (out of scope).

---

## Reproduction Targets (PM only)

### Primary: Fig 2 (Example 1, λ=0)
- D=40, m=1000, M=10000, l=64, lr=0.2, C=1, λ=0, 10000 epochs
- Methods: PM-Euler (SGD), PM-SAV, PM-ResSAV, PM-RelSAV
- Expected: SAV train relative error ~1e-10

### Primary: Fig 7 (Example 2, λ=0)
- D=40, m=100, M=10000, l=64, lr=0.01, C=1, λ=0, 10000 epochs
- Methods: PM-Euler (SGD), PM-SAV, PM-ResSAV, PM-RelSAV
- Expected: SAV train relative error ~1e-5

### Energy verification: Fig 1 (Example 1, λ=10)
- D=20, m=1000, M=10000, l=8000 (full batch), lr=0.6, C=1, λ=10, 8000 epochs
- Methods: GD, SAV, RelSAV
- Expected: energy strictly non-increasing

---

## SAV Algorithm Formulas (Paper's Version)

### Vanilla SAV (Paper's Algorithm 2, PM version)
- α = 1/(1 + λΔt)
- μ^n = ∇I(θ^n) / √(I(θ^n) + C)
- b = ‖μ^n‖²
- r^{n+1} = r^n / (1 + αΔt/2 · b)
- θ^{n+1} = θ^n − αΔt · r^{n+1} · μ^n

When λ=0: α=1, and this simplifies to the same as our original code.

### Restart SAV (Paper's Algorithm 3, PM version)
- Same as Vanilla but r̂^n = √(I(θ^n) + C) replaces r^n in the r update
- r^{n+1} = r̂^n / (1 + αΔt/2 · b)
- θ^{n+1} = θ^n − αΔt · r^{n+1} · μ^n

### Relax SAV (Paper's Algorithm 4, PM version)
1. Do vanilla SAV step → r̃, θ_new
2. r̂ = √(I(θ_new) + C)
3. Solve for ξ₀ using η=0.99:
   - a = (r̃ − r̂)²
   - b = 2r̂(r̃ − r̂)
   - c = r̂² − r̃² − η·‖θ_new − θ_old‖²/Δt
   - ξ₀ = max{0, (−b − √(b²−4ac)) / (2a)} if discriminant ≥ 0
4. r_new = ξ₀·r̃ + (1−ξ₀)·r̂

---

## Important Notes from Paper

- The paper uses **full-batch gradient** for the theoretical analysis, but **mini-batch** in experiments. Mini-batch means the energy stability is approximate (per-batch loss is used in the $r$ update, not the full training loss).
- The auxiliary variable $r^n$ is initialized as $r^0 = \sqrt{I(\theta^0) + C}$ using the **full training set** loss at initialization.
- For Restart SAV, $\hat{r}^n$ is recomputed from the **current mini-batch** loss at each step.
- The paper does **not** report exact final loss values in a table; results are read from log-scale figures. Expect order-of-magnitude comparisons only.
