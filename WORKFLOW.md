# Research Workflow: SAV-Gradient-Flow-Research

This document defines the complete execution protocol for this research project. Claude Code must follow this workflow precisely and autonomously.

---

## Phase 0: Environment Setup (Run Once, Collaboratively With Human)

Phase 0 is completed **collaboratively with the human**. Claude Code executes each step, reports output, and waits for confirmation before proceeding. Do not start Phase 1 until all checks pass.

---

### Step 0.1: Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .   # installs src/ as importable Python package
```

Verify:
```python
import torch, wandb, numpy, matplotlib, scipy
print(torch.__version__)
print(wandb.__version__)
```

Report all versions to human. If any ImportError → stop and report.

Commit: `chore(setup): install and lock dependencies`

---

### Step 0.2: GPU Verification

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM total (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
    print("VRAM free (GB):", torch.cuda.mem_get_info()[0] / 1e9)
```

Expected: CUDA=True, Device contains "3050 Ti", VRAM ≥ 3.5 GB free.

- If CUDA not available → stop, report to human immediately.
- If VRAM free < 3.5 GB → note in report; IEQ batch size will need to be 32 instead of 64.

---

### Step 0.3: Git Setup

```bash
git init
git remote add origin <URL_PROVIDED_BY_HUMAN>
git add .
git commit -m "chore(init): initial project structure and documentation"
git push -u origin main
```

Verify:
```bash
git remote -v
git log --oneline
```

Report remote URL and commit hash to human.

---

### Step 0.4: wandb Configuration

**Human provides API key interactively.**

```bash
wandb login   # human enters key
```

Smoke test:
```python
import wandb
run = wandb.init(project="sav-gradient-flow-research", name="setup-test", mode="online")
wandb.log({"test_metric": 1.0})
run.finish()
print("wandb OK")
```

Expected: run visible at https://wandb.ai under project `sav-gradient-flow-research`.

**Run naming convention for all experiments:**
```
{phase}_{algorithm}_{example}  # e.g. phase1_vanilla-sav_example1
```

**Metrics to log for every experiment:** `train_rel_error`, `test_rel_error`, `train_mse`, `energy`, `r` (or `q_norm` for IEQ), step (= epoch).

---

### Step 0.5: Slack Integration

**Human configures webhook/clawbot token interactively.**

After setup, verify with a test message:
```
✅ SAV-Gradient-Flow-Research: environment setup complete. Ready for Phase 1.
```

**Claude Code must send Slack notifications automatically at these exact events:**

| Trigger | Message |
|---------|---------|  
| `phase1-paper-verified` tag created | `📊 Phase 1 paper check: [PASS/FAIL]. [one-line summary]` |
| Any PAUSE_REPORT written | `⚠️ PAUSE (Phase N): [trigger reason]. Human review needed.` |
| `phaseN-complete` tag created | `✅ Phase N complete. Final test loss: [X]. See results/phaseN_*/` |

---

### Step 0.6: Phase 0 Complete

Send Slack: `🚀 Phase 0 complete. Starting Phase 1 (SAV reproduction).`

Commit: `chore(setup): phase 0 complete - all systems verified`
Tag: `git tag phase0-complete && git push origin phase0-complete`

---

## Overview

**Goal:** Implement and validate three families of energy-stable optimization algorithms (SAV, ESAV, IEQ) for neural network training viewed as gradient flows.

**Three Phases:**
- Phase 1: SAV reproduction (strictly following paper)
- Phase 2: ESAV algorithm (innovation)
- Phase 3: IEQ algorithm (innovation)

**Tasks in all phases:** Regression only (Example 1: sin+cos, Example 2: polynomial). No image classification. No PDE tasks.

**Mathematical authority:** `MATH_REFERENCE.md` — all algorithm formulas come from here.
**Paper settings authority:** `PAPER_NOTES.md` — all hyperparameters and data settings come from here.

---

## General Principles

1. **Self-contained:** Claude Code works autonomously. Debug issues independently before escalating.
2. **Math-first:** If code behavior contradicts `MATH_REFERENCE.md`, trust the math and fix the code.
3. **Allowed to question math:** If after careful analysis Claude Code believes a formula in `MATH_REFERENCE.md` is inconsistent or incorrect, it must document the specific concern in `PAUSE_REPORT.md` and stop for human review.
4. **Reproducibility:** All experiments use fixed random seeds. Results must be reproducible.
5. **Incremental commits:** Commit after every meaningful unit of work (see Git Protocol).

---

## Directory Structure

```
SAV-Gradient-Flow-Research/
├── CLAUDE.md                  # Role definition and constraints
├── WORKFLOW.md                # This file
├── MATH_REFERENCE.md          # Mathematical derivations (authoritative)
├── PAPER_NOTES.md             # Paper settings (authoritative for Phase 1)
├── src/
│   ├── algorithms/
│   │   ├── baselines.py       # SGD, Adam
│   │   ├── sav.py             # Vanilla, Restart, Relax SAV
│   │   ├── esav.py            # Vanilla, Restart, Relax ESAV
│   │   └── ieq.py             # Vanilla, Restart, Relax IEQ
│   ├── models/
│   │   └── network.py         # One-hidden-layer neural network
│   └── utils/
│       ├── data.py            # Data generation for Example 1 & 2
│       ├── trainer.py         # Training loop and history recording
│       └── plotting.py        # Loss curve plotting utilities
├── experiments/
│   ├── phase1_example1.py     # Phase 1, Example 1
│   ├── phase1_example2.py     # Phase 1, Example 2
│   ├── phase2_example1.py     # Phase 2, Example 1
│   ├── phase2_example2.py     # Phase 2, Example 2
│   ├── phase3_example1.py     # Phase 3, Example 1
│   └── phase3_example2.py     # Phase 3, Example 2
├── results/
│   ├── phase1_sav/
│   ├── phase2_esav/
│   └── phase3_ieq/
└── reports/
    ├── phase1_report.md
    ├── phase2_report.md
    ├── phase3_report.md
    └── PAUSE_REPORT.md        # Written ONLY when pausing for human review
```

---

## Git Protocol

### Commit Message Format

```
type(scope): short description

[optional body: what changed and why]
```

**Types:**
- `feat` — new algorithm or feature implementation
- `exp` — experiment run and results recorded
- `fix` — bug fix
- `report` — phase report or pause report written
- `chore` — setup, refactor, dependency changes

**Examples:**
```
feat(sav): implement Vanilla SAV (Algorithm 1 from MATH_REFERENCE)
feat(sav): implement Restart SAV (Algorithm 2) and Relax SAV (Algorithm 3)
exp(phase1): run Example1 all SAV variants, 50k epochs
fix(sav): correct r^{n+1} numerator sign in Restart SAV
report(phase1): phase 1 complete, energy stability verified
```

### Mandatory Tags

After each phase completion:
```bash
git tag phase1-paper-verified   # After paper comparison passes
git tag phase1-complete         # After free exploration complete
git tag phase2-complete
git tag phase3-complete
```

### Commit Frequency

- After completing each algorithm implementation: commit
- After each experiment run with results saved: commit
- After each bug fix: commit
- After writing any report: commit
- **Never leave uncommitted changes when stopping work**

---

## Phase 1: SAV Reproduction

### Step 1: Environment Setup

```bash
pip install torch numpy matplotlib scipy
```

Verify GPU availability:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # Should show 3050Ti
```

Device priority: CUDA > MPS > CPU.

### Step 2: Data Generation

Implement `src/utils/data.py`:

- `generate_example1(D, N_train, N_test, seed)`: Example 1 (sin+cos), per PAPER_NOTES.md
- `generate_example2(D, N_train, N_test, seed)`: Example 2 (polynomial), per PAPER_NOTES.md
- Fixed seed = 42 for all experiments
- Save generated data to `results/phase1_sav/data_example1.pt` and `data_example2.pt`
- Verify: print train/test shapes, print min/max of targets

Commit: `feat(data): implement Example 1 and 2 data generation`

### Step 3: Model Implementation

Implement `src/models/network.py`:

- Class `OneHiddenLayerNet(D, m)`: one hidden layer, ReLU activation
- Parameters: `W ∈ R^{(D+1) × m}` (bias absorbed), `a ∈ R^{m × 1}`
- Forward: `x_aug = [x, 1]`, `h = ReLU(x_aug @ W)`, `out = h @ a`
- Initialization: Kaiming normal for both W and a
- Expose `flatten_params()` and `unflatten_params()` as methods or utilities

Commit: `feat(model): implement one-hidden-layer network per paper spec`

### Step 4: Baseline Implementations

Implement `src/algorithms/baselines.py`:

- `train_sgd(model, data, lr, batch_size, epochs, device)` → returns history dict
- `train_adam(model, data, lr, batch_size, epochs, device)` → returns history dict
- History dict format: `{'train_loss': [...], 'test_loss': [...], 'energy': [...]}`
- Loss: `nn.MSELoss()`
- Parameters per PAPER_NOTES.md

Commit: `feat(baseline): implement SGD and Adam trainers`

### Step 5: SAV Implementations

Implement `src/algorithms/sav.py` with three functions:
- `train_vanilla_sav(model, data, C, lambda_, dt, batch_size, epochs, device)`
- `train_restart_sav(model, data, C, lambda_, dt, batch_size, epochs, device)`
- `train_relax_sav(model, data, C, lambda_, dt, batch_size, epochs, device)`

**Algorithm source:** `MATH_REFERENCE.md` Algorithms 1, 2, 3 — implement exactly as written.

Key implementation details:
- $\lambda = 0$ as default (simplifies formulas significantly)
- Initialize $r^0 = \sqrt{I(\theta^0) + C}$ using **full training set** loss
- For Restart SAV: recompute $\hat{r}^n$ from full training set at the **start of each epoch**
- For Relax SAV: compute $\hat{r}^{n+1}$ after $\theta^{n+1}$ is updated, before committing $r^{n+1}$
- Track and record `energy = r^2` at every epoch
- Record `train_loss` (full training set MSE) and `test_loss` at every epoch

**Self-verification after implementation:**
1. Run 100 epochs, verify `energy` array is non-increasing (Vanilla SAV theorem)
2. Print $r^n$ vs $\sqrt{I(\theta^n)+C}$ every 10 epochs for Vanilla SAV — they should diverge over time (this is expected)
3. For Restart SAV, the two should always be equal (by definition)

Commit: `feat(sav): implement Vanilla, Restart, Relax SAV`

### Step 6: Run Phase 1 Experiments

Run `experiments/phase1_example1.py`:
- All methods: SGD, Adam, Vanilla SAV, Restart SAV, Relax SAV
- Example 1, D=20, m=100, 50,000 epochs
- Parameters strictly from PAPER_NOTES.md
- Save results to `results/phase1_sav/example1_results.pt`
- Save loss curves plot to `results/phase1_sav/example1_loss_curves.png`

Then run `experiments/phase1_example2.py` for Example 2.

Commit: `exp(phase1): run all SAV variants on Example 1 and 2`

### Step 7: Paper Comparison and Pause Decision

After experiments complete, evaluate:

**Pause threshold:** If after 5,000 epochs, ALL THREE SAV variants have test loss ≥ SGD baseline test loss on Example 1 → write PAUSE_REPORT and stop.

**If not triggered:** Verify qualitative trends from PAPER_NOTES.md:
1. SAV variants faster than SGD? ✓/✗
2. Restart ≤ Vanilla final loss? ✓/✗
3. Relax ≤ Restart final loss? ✓/✗
4. Energy non-increasing for all SAV variants? ✓/✗
5. Larger $\Delta t$ (0.5, 1.0) stable for SAV but unstable for SGD? ✓/✗

Record all checks in `reports/phase1_report.md`.

If 3 or more checks fail → write PAUSE_REPORT and stop.
If ≥ 3 checks pass → proceed.

Commit tag: `git tag phase1-paper-verified`

### Step 8: Free Exploration (Phase 1)

After paper verification passes, Claude Code may freely explore:
- Different $m$ values (200, 500, 1000)
- Different $\Delta t$ values
- Different $C$ values
- Higher dimensions D=40
- Any architectural modifications deemed appropriate

Record findings in `reports/phase1_report.md`.

Commit: `report(phase1): phase 1 complete`
Tag: `git tag phase1-complete`

---

## Phase 2: ESAV Algorithm

### Overview

Implement Vanilla, Restart, Relax ESAV from `MATH_REFERENCE.md` Algorithms 4, 5, 6.

Key difference from SAV: auxiliary variable $r = \exp(I(\theta)/2)$ instead of $\sqrt{I(\theta)+C}$.
Normalization factor: $\nu^n = \nabla I(\theta^n) \cdot \exp(-I(\theta^n)/2)$.

### Implementation Notes

- `src/algorithms/esav.py`
- Initialize $r^0 = \exp(I(\theta^0)/2)$ using full training set loss
- **Numerical safeguard:** if $I(\theta)$ is large (> 50), use log-space arithmetic to avoid overflow. Store $s = \ln r = I/2$ and compute $\exp(-s)$ carefully.
- All three variants follow identical structure to SAV with $\mu^n$ replaced by $\nu^n$

### Verification Standard (Phase 2)

No paper comparison — verify internal consistency:
1. Energy $\mathcal{E}^n = (r^n)^2 + \frac{\lambda}{2}\|\theta^n\|^2$ non-increasing ✓/✗
2. Restart ESAV final loss ≤ Vanilla ESAV final loss ✓/✗
3. Relax ESAV final loss ≤ Restart ESAV final loss ✓/✗
4. ESAV converges (loss decreasing trend) on both examples ✓/✗

If any of 1 or 4 fail → write PAUSE_REPORT.

Commit tag: `git tag phase2-complete`

---

## Phase 3: IEQ Algorithm

### Overview

Implement Vanilla, Restart, Relax IEQ from `MATH_REFERENCE.md` Algorithms 7, 8, 9.

Key features:
- Auxiliary variable $q = f(\theta) - y \in \mathbb{R}^m$ (vector, not scalar)
- Requires computing Jacobian $J^n = \partial f/\partial\theta \in \mathbb{R}^{m \times d}$
- Requires solving $m \times m$ linear system per step: cost $O(m^3)$
- **Only valid for MSE loss** — do not use CrossEntropy

### GPU Memory Constraint (3050Ti, 4GB)

IEQ requires storing the Jacobian $J^n \in \mathbb{R}^{m \times d}$:
- $m$ = batch_size, $d$ = num_parameters
- For $m=64$, $d \approx 10,100$ (Example 1, D=20, m_neurons=100): $J$ is 64×10100 ≈ 5MB — feasible
- For $m=256$: $J$ is 256×10100 ≈ 20MB — feasible but linear solve is 256×256
- **Recommended batch size for IEQ: 64**
- If OOM occurs: reduce batch size to 32 or 16

### Implementation Notes

- `src/algorithms/ieq.py`
- Compute Jacobian via `torch.autograd.functional.jacobian`
- Solve $(I_m + \alpha G)q^{n+1} = q^n - \alpha\lambda J^n\theta^n$ via `torch.linalg.solve`
- **Restart IEQ is the recommended default**: $\hat{q}^n = f(\theta^n) - y$ is free since Jacobian needs a forward pass anyway
- For Relax IEQ: requires extra forward pass for $\hat{q}^{n+1} = f(\theta^{n+1}) - y$

### Verification Standard (Phase 3)

Internal consistency:
1. Energy $\mathcal{E}^n = \frac{1}{2}\|q^n\|^2 + \frac{\lambda}{2}\|\theta^n\|^2$ non-increasing ✓/✗
2. Restart IEQ final loss ≤ Vanilla IEQ final loss ✓/✗  
3. IEQ converges faster than SGD baseline ✓/✗ (expected from theory)

If 1 fails → write PAUSE_REPORT.

Note: IEQ may be slow per epoch due to Jacobian cost. Run fewer epochs (5,000) and compare per-epoch loss.

Commit tag: `git tag phase3-complete`

---

## PAUSE_REPORT Protocol

When pausing is required, create/update `reports/PAUSE_REPORT.md` with:

```markdown
# PAUSE REPORT

**Date:** [date]
**Phase:** [1/2/3]
**Trigger:** [exact condition that triggered pause]

## What was running
[experiment name, parameters]

## Results observed
[actual numbers/trends]

## Expected results
[what should have happened per paper or math]

## Self-diagnosis
[what Claude Code already checked and ruled out]

## Specific question for human
[precise question — e.g., "Is the formula for r^{n+1} correct when lambda=0?"]

## Code reference
[file name and line numbers of relevant code]
```

After writing PAUSE_REPORT:
1. Save all current results
2. Commit: `report(pause): pause for human review - [brief reason]`
3. Stop all experiments
4. Wait for human response before resuming

---

## Result Recording Format

Every experiment must save:

```python
results = {
    'method': 'Vanilla SAV',
    'params': {'C': 1, 'lambda': 0, 'dt': 0.1, 'batch_size': 64},
    'train_loss': [...],   # list of float, relative error per epoch (for paper comparison)
    'test_loss': [...],    # list of float, relative error per epoch
    'energy': [...],       # list of float, one per epoch (r^2 for SAV/ESAV, MSE for baselines)
    'r_values': [...],     # list of float (r^n trajectory, for SAV/ESAV only)
    'final_train_loss': float,  # final relative error
    'final_test_loss': float,   # final relative error
    'wall_time': float,
}
torch.save(results, 'results/phaseX_xxx/method_example_results.pt')
```

**Note:** `train_loss`/`test_loss` store relative error (matching paper figures). MSE is logged separately to wandb as `train_mse`.

Every experiment must also save a plot `*_loss_curves.png` showing:
- Train loss vs epoch (log scale y-axis)
- Test loss vs epoch (log scale y-axis)
- All methods on same plot for comparison
- Clear legend, title, axis labels in English

---

## Debug Protocol

When a bug or unexpected result is found:

1. **First: check the formula.** Re-read the relevant algorithm in `MATH_REFERENCE.md`. Verify every line of code maps to a formula step.
2. **Second: check special cases.** Run with $\lambda=0$, small $m=10$, 10 epochs, single batch. Verify r update by hand for 2-3 steps.
3. **Third: check numerical issues.** Print $r^n$, $\|\mu^n\|^2$, $\langle\mu^n,\theta^n\rangle$ for first 5 steps. Look for NaN, inf, or extremely large values.
4. **Fourth: bisect.** Comment out mini-batch loop and run full-batch. If that works, the issue is in batching.
5. **If still unresolved after all above:** Write PAUSE_REPORT.

---

## Phase Completion Checklist

### Phase 1 Complete When:
- [x] All 3 SAV variants implemented and pass energy stability check
- [x] Example 1 and Example 2 experiments run to 10k epochs (Fig 2 + Fig 7)
- [x] Paper comparison evaluation done (results in phase1_report.md)
- [x] All results saved as .pt files and plots saved as .png
- [x] `phase1-paper-verified` tag created
- [x] Free exploration done and documented
- [x] `phase1-complete` tag created

**Phase 1 Completion Notes:**
- 4/5 paper comparison criteria pass (Criterion 3 fails: Relax > Restart)
- Vanilla SAV r-collapse confirmed; Restart SAV is most practical variant
- Energy stability (Theorem 1) verified: 0 violations for Vanilla SAV across all experiments
- See `reports/phase1_report.md` for full analysis

### Phase 2 Complete When:
- [x] All 3 ESAV variants implemented
- [x] Example 1 and Example 2 experiments run
- [x] Internal consistency verified
- [x] `phase2-complete` tag created

**Phase 2 Completion Notes:**
- C1 (energy): Vanilla ESAV violations due to drift safeguard (documented in esav_analysis.md)
- C2 (Restart ≤ Vanilla): Pass on Ex1, marginal fail on Ex2 (within noise)
- C3 (Relax ≤ Restart): Mixed (Ex1 fail, Ex2 pass)
- C4 (convergence): All pass
- ESAV Vanilla 13-229x better than SAV Vanilla (log-space prevents r-collapse)

### Phase 3 Complete When:
- [ ] All 3 IEQ variants implemented
- [ ] Example 1 and Example 2 experiments run (batch_size ≤ 64)
- [ ] Internal consistency verified
- [ ] `phase3-complete` tag created
- [ ] Final summary written in reports/
