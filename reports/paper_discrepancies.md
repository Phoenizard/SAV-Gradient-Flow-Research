# Paper Discrepancies Report

**Date:** 2026-02-27
**Paper:** Ma, Mao & Shen (2024). *Efficient and stable SAV-based methods for gradient flows arising from deep learning.* JCP 505, 112911.

## Summary

After running Phase 1 Example 1 for 50k epochs and comparing results against the paper PDF, **6 critical discrepancies** were found between our implementation/PAPER_NOTES.md and the paper's actual experimental setup.

---

## Critical Discrepancies

### 1. Missing 1/m factor in network output

**Paper (Eq. 2):**
$$f(\theta; x) = \frac{1}{m} \sum_{k=1}^{m} a_k \sigma(\omega_k^T x)$$

**Our code (`network.py:40`):**
```python
out = h @ self.a  # Missing 1/m
```

**Impact:** Without 1/m, gradients scale with m. When m=1000, gradients are 1000x too large, causing Vanilla SAV's r to collapse rapidly.

### 2. No Z-score normalization

**Paper (Section 3.1):** "we process data with Z-score normalization: x ⇒ (x−μ)/σ"

**Our code:** Uses raw inputs directly.

**Impact:** Different input scale means different gradient magnitudes and loss landscape geometry.

### 3. Wrong loss function

**Paper (Section 3.1):** Uses relative error as the loss function:
$$I(\theta) = \frac{\sum_i (f(\theta; x_i) - y_i)^2}{\sum_i y_i^2}$$

**Our code:** Uses `nn.MSELoss()` = (1/N) Σ(f−y)².

**Impact:** Relative error is scale-invariant; MSE is not. Different loss landscapes and gradient magnitudes.

### 4. Wrong data split

**Paper:** "80% training, 20% testing" from a single pool of M data points.

**Our code:** Generates train and test sets independently with separate random draws.

**Impact:** Paper's test set comes from the same distribution instance; ours doesn't.

### 5. Wrong hyperparameters

**PAPER_NOTES.md claimed:** D=20, m=100, M=1000 (these were initial guesses before reading the paper carefully).

**Paper figures actually use:** D=40, m=1000, M=10000+ depending on figure.

**Impact:** Completely different problem scale. m=1000 vs m=100 is a 10x difference in model capacity.

### 6. SAV formula mismatch for λ>0

**Paper's Algorithm 2 (θ update):**
$$\theta^{n+1} = \theta^n - \alpha \Delta t \cdot r^{n+1} \cdot \mu^n$$

**Our MATH_REFERENCE / code:**
$$\theta^{n+1} = \alpha (\theta^n - \Delta t \cdot r^{n+1} \cdot \mu^n)$$

These are equivalent only when λ=0 (α=1). For λ>0, the paper's version applies α only to the gradient term, while ours scales the entire θ.

**Paper's r update:** Has no "a" term (no ⟨μ,θ⟩ contribution):
$$r^{n+1} = \frac{r^n}{1 + \alpha \Delta t / 2 \cdot b}$$

---

## Minor Discrepancy

### 7. Relax SAV η parameter

**Paper (Eq. 23):** Uses η=0.99 in the relaxation constraint for energy bound.

**Our code:** Does not include η.

---

## Paper's Per-Figure Experimental Settings

All figures use: ReLU activation, Z-score normalized inputs, relative error loss, 80/20 train/test split from M total.

| Fig | Example | D  | m    | M      | l (batch) | lr     | C   | λ  | Epochs | PM methods shown        |
|-----|---------|----|------|--------|-----------|--------|-----|----|--------|-------------------------|
| 1   | Ex1     | 20 | 1000 | 10000  | 8000(full)| 0.6    | 1   | 10 | 8000   | Euler,SAV,RelSAV        |
| 2   | Ex1     | 40 | 1000 | 10000  | 64        | 0.2    | 1   | 0  | 10000  | Euler,SAV               |
| 3   | Ex1     | 40 | 1000 | 100000 | 256       | 0.5/1.0| 100 | 4  | 10000  | (SPM only)              |
| 5   | Ex1     | 40 | 1000 | 100000 | 256       | 0.5    | 100 | varies | 10000 | SAV (λ study)       |
| 7   | Ex2     | 40 | 100  | 10000  | 64        | 0.01   | 1   | 0  | 10000  | Euler,SAV               |
| 8   | Ex2     | 40 | 100  | 10000  | 64        | 0.4/1.0| 1   | 4  | 10000  | (SPM only)              |

**Note:** We implement PM (particle method) only. SPM requires Gaussian smoothing kernel (out of scope).

## Reproduction Targets

1. **Fig 2 (Example 1):** D=40, m=1000, M=10000, l=64, lr=0.2, C=1, λ=0, 10000 epochs
2. **Fig 7 (Example 2):** D=40, m=100, M=10000, l=64, lr=0.01, C=1, λ=0, 10000 epochs
3. **Fig 1 (Energy verification):** D=20, m=1000, M=10000, l=8000(full), lr=0.6, C=1, λ=10, 8000 epochs
