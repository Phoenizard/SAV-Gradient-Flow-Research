# Paper Notes: Key Settings for Reproduction

**Reference:** Ma, Mao & Shen (2024). *Efficient and stable SAV-based methods for gradient flows arising from deep learning.* Journal of Computational Physics 505, 112911. DOI: 10.1016/j.jcp.2024.112911

This file contains **verbatim or closely paraphrased** settings from the paper. Claude Code must follow these exactly during Phase 1. Do not deviate without a PAUSE_REPORT.

---

## Model Architecture (Section 2, Eq. 2)

The neural network is a **one-hidden-layer network**:

$$f(\theta; x) = \sum_{k=1}^{m} a_k \sigma(w_k \cdot x + b_k)$$

where:
- $\sigma$ = **ReLU** activation (paper uses ReLU for DL examples)
- $m$ = number of neurons in the hidden layer (varies per experiment, see below)
- Parameters $\theta = \{w_k, b_k, a_k\}_{k=1}^m$
- $w_k \in \mathbb{R}^D$, $b_k \in \mathbb{R}$, $a_k \in \mathbb{R}$

**Implementation note:** Bias $b_k$ can be absorbed into $w_k$ by augmenting $x$ with a constant 1, giving $W \in \mathbb{R}^{(D+1) \times m}$ and $a \in \mathbb{R}^{m \times 1}$.

**Loss function:** Mean Squared Error (MSE)

$$I(\theta) = \frac{1}{N} \sum_{i=1}^{N} (f(\theta; x_i) - y_i)^2$$

Note: Paper uses $\frac{1}{2N}$ convention in some places — use standard PyTorch `nn.MSELoss()` (which divides by $N$) for consistency.

---

## Example 1: Sin + Cos Regression (Section 3.1.1)

**Target function:**
$$f^*(x_1, \ldots, x_D) = \sin\!\left(\sum_{i=1}^{D} p_i x_i\right) + \cos\!\left(\sum_{i=1}^{D} q_i x_i\right)$$

where $p_i, q_i$ are fixed random coefficients drawn from $\text{Uniform}(0,1)$ and **fixed** (use seed).

**Data:**
- Input $x \sim \text{Uniform}(0, 1)^D$
- Dimension: $D = 20$ (primary), $D = 40$ (secondary)
- Training samples: $N_{\text{train}} = 1000$
- Test samples: $N_{\text{test}} = 200$
- No noise added (clean targets)

**Model size:** $m = 100$ neurons (primary experiments)

**Training epochs:** 50,000 (for SAV-type methods); 50,000 for SGD/Adam baselines

**Random seed:** Fix seed = 42 for data generation and model initialization.

---

## Example 2: Polynomial Regression (Section 3.1.2)

**Target function:**
$$f^*(x_1, \ldots, x_D) = \sum_{i=1}^{D} c_i x_i^2$$

where $c_i \sim \text{Uniform}(0, 1)$, fixed with seed.

**Data:**
- Input $x \sim \text{Uniform}(0, 5)^D$
- Dimension: $D = 20$
- Training samples: $N_{\text{train}} = 1000$
- Test samples: $N_{\text{test}} = 200$
- No noise added

**Model size:** $m = 100$ neurons

**Training epochs:** 50,000

---

## Hyperparameters (Table 1 in Paper)

### SAV Parameters

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| SAV constant | $C$ | 1 | Ensures $I + C > 0$ |
| Linear operator coeff | $\lambda$ | 0 | No regularization in primary experiments |
| Time step (lr) | $\Delta t$ | 0.1 | Primary value; paper also tests 0.5, 1.0 |
| Batch size | $l$ | 256 | Mini-batch SGD over training data |
| Neurons | $m$ | 100 | Hidden layer width |

### SGD Baseline Parameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.1 |
| Batch size | 256 |
| Momentum | 0 (vanilla SGD) |

### Adam Baseline Parameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| Batch size | 256 |

---

## Paper's Expected Qualitative Results (Phase 1 Verification)

From Figures 1–4 in the paper, the following qualitative trends should hold:

1. **SAV converges faster than SGD** in terms of loss vs. epoch, especially in early training (< 10,000 epochs).
2. **Restart SAV achieves lower final loss than Vanilla SAV** — Vanilla SAV's $r^n$ drifts, causing it to plateau earlier.
3. **Relax SAV achieves the best or equal final loss** among the three SAV variants.
4. All three SAV variants are **energy-stable**: the modified energy $\mathcal{E}^n = (r^n)^2 + \frac{\lambda}{2}\|\theta^n\|^2$ should be non-increasing at every step.
5. SAV allows **larger learning rates** than SGD without divergence (paper tests $\Delta t = 0.5, 1.0$ which SGD cannot use stably).

**Pause threshold (Phase 1):** If after 5,000 epochs, ALL THREE SAV variants have test loss ≥ SGD baseline test loss, write PAUSE_REPORT and stop.

---

## Important Notes from Paper

- The paper uses **full-batch gradient** for the theoretical analysis, but **mini-batch** in experiments. Mini-batch means the energy stability is approximate (per-batch loss is used in the $r$ update, not the full training loss).
- The auxiliary variable $r^n$ is initialized as $r^0 = \sqrt{I(\theta^0) + C}$ using the **full training set** loss at initialization.
- For Restart SAV, $\hat{r}^n$ is recomputed from the **full training set** loss at each epoch (not each mini-batch step). Clarification: the paper is ambiguous here — Claude Code should implement both and note the difference.
- The paper does **not** report exact final loss values in a table; results are read from log-scale figures. Expect order-of-magnitude comparisons only.
