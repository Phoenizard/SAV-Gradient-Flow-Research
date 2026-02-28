# Phase 3 Report: IEQ Algorithms

**Date:** 2026-02-28
**Phase:** 3 (IEQ Algorithms — Vanilla, Restart, Relax)
**Status:** COMPLETE

---

## Experiments Run

### Example 1: sin+cos (lambda=0)
- **Settings:** D=40, m=1000, batch_size=64, dt=0.2, lambda=0, 5000 epochs
- **Methods:** SGD, Vanilla IEQ, Restart IEQ, Relax IEQ

### Example 2: polynomial (lambda=0)
- **Settings:** D=40, m=100, batch_size=64, dt=0.01, lambda=0, 5000 epochs
- **Methods:** SGD, Vanilla IEQ, Restart IEQ, Relax IEQ

---

## Internal Consistency Verification

### Criterion 1: Energy non-increasing?
**PASS for Restart and Relax. FAIL for Vanilla (expected).**
- Vanilla IEQ: q-drift causes energy to fluctuate (stale q in mini-batch mode)
- Restart IEQ: energy monotonically decreasing (q reset each step)
- Relax IEQ: energy monotonically decreasing (relax correction enforces stability)

### Criterion 2: Restart ≤ Vanilla final loss?
**PASS.** Restart IEQ converges properly while Vanilla IEQ diverges due to q-drift.

### Criterion 3: IEQ converges (loss decreasing)?
**PASS for Restart and Relax. FAIL for Vanilla (expected).**
- Vanilla IEQ diverges in mini-batch mode due to stale q
- Restart IEQ and Relax IEQ converge to accuracy comparable to SGD

---

## Key Findings

### 1. IEQ Restart = Levenberg-Marquardt (μ = 1/dt)

The most significant finding of Phase 3. Through Woodbury identity:

θ^{n+1} = θ^n - (J^TJ + (1/dt)·I_d)^{-1} · J^T · (f(θ^n) - y)

This is exactly LM with damping μ = 1/dt. See `reports/IEQ-Review.md` for full proof.

Implications:
- dt → 0: IEQ → SGD (large damping)
- dt → ∞: IEQ → Gauss-Newton (no damping)
- IEQ provides a unified gradient flow perspective connecting first-order and second-order methods

### 2. Vanilla IEQ q-drift in mini-batch mode

Vanilla IEQ maintains q across steps without resetting. In mini-batch mode, q[idx] becomes stale because θ has been updated by other batches since q[idx] was computed. This causes divergence.

Fix applied: Relax IEQ now uses fresh q_hat_old = f(θ)-y in its Vanilla step (commit 8977012).

### 3. Computational cost

IEQ requires ~2x wall time per epoch compared to SAV/ESAV due to:
- Jacobian computation: O(m·d) via vmap
- Linear system solve: O(m³) via torch.linalg.solve
- batch_size limited to 64 for 4GB GPU

### 4. Traditional IEQ comparison

Yang (2016) PDE construction uses scalar q = √(F(u)+C), which is mathematically equivalent to SAV. Our vector construction q = f(θ)-y leads to the LM equivalence instead. See `reports/IEQ-Review.md` Section 3.

---

## Cross-Family Summary (All 3 Phases)

| Family | Vanilla | Restart | Relax | Cost/epoch |
|--------|---------|---------|-------|------------|
| SAV | r-collapse → poor | = SGD | ≈ SGD | O(d) |
| ESAV | 13-229x better than SAV | = SGD | ≈ SGD | O(d) |
| IEQ | q-drift → diverges | = LM (μ=1/dt) | ≈ LM | O(md + m³) |

**Practical recommendation:** Restart SAV or Restart ESAV for lightweight energy-stable optimization. IEQ only for theoretical analysis.

---

## Detailed Analysis

See `reports/IEQ-Review.md` for:
- Complete IEQ = LM equivalence proof
- Traditional IEQ community review
- Research value assessment
- Alternative construction analysis
- Paper positioning strategy
