# ESAV 算法实现分析报告

**日期：** 2026-02-28
**范围：** `src/algorithms/esav.py` — Vanilla/Restart/Relax ESAV (Algorithms 4-6)
**目的：** 对照 MATH_REFERENCE.md 检查公式正确性，分析数值策略的合理性，总结实验洞察

---

## 一、公式正确性验证

### 1.1 核心公式对照

MATH_REFERENCE Algorithm 4 (Vanilla ESAV):

```
1. ν^n = ∇I(θ^n) / exp(I(θ^n)/2)
2. a = ⟨ν^n, θ^n⟩,  b = ‖ν^n‖²,  α = 1/(1+λΔt)
3. r^{n+1} = (r^n + (α-1)/2 · a) / (1 + αΔt/2 · b)
4. θ^{n+1} = α(θ^n - Δt · r^{n+1} · ν^n)
```

代码中的 ratio 形式（`_esav_vanilla_step`, 第81-149行）:

```
log_R = I/2                          # ln(exp(I/2)) = I/2
scale2 = exp(-I)                     # = exp(-2·log_R)
a_scaled = scale2 · ⟨gradI, θ⟩       # = ⟨ν,θ⟩ / exp(I/2) = a/R
b_scaled = scale2 · ‖gradI‖²         # = ‖ν‖² = b
exp_delta = exp(log_r - log_R)       # = r/R

ratio = (exp_delta + (α-1)/2 · a_scaled) / (1 + αΔt/2 · b_scaled)
      = r^{n+1} / R

θ_new = α(θ - Δt · ratio · gradI)
      = α(θ - Δt · (r^{n+1}/R) · gradI)
      = α(θ - Δt · r^{n+1} · gradI/R)
      = α(θ - Δt · r^{n+1} · ν)      ✓ (因为 ν = gradI/R)
```

**验证结论：ratio 形式与原始公式数学等价。** 关键恒等式是 `r^{n+1}·ν = ratio·gradI`，指数因子完全约去。

### 1.2 b_scaled 的推导验证

需要确认 `b_scaled = exp(-I) · ‖gradI‖²` 确实等于 `b = ‖ν‖²`：

```
ν = gradI / exp(I/2)
‖ν‖² = ‖gradI‖² / exp(I) = exp(-I) · ‖gradI‖² = b_scaled  ✓
```

### 1.3 a_scaled 的推导验证

需要确认 `a_scaled = exp(-I) · ⟨gradI, θ⟩` 是 `a/R`（而非 `a`）：

```
a = ⟨ν, θ⟩ = ⟨gradI/exp(I/2), θ⟩ = exp(-I/2) · ⟨gradI, θ⟩
a/R = a / exp(I/2) = exp(-I) · ⟨gradI, θ⟩ = a_scaled  ✓
```

ratio 公式的分子是 `exp_delta + (α-1)/2 · a_scaled`，即 `r/R + (α-1)/2 · a/R`。
原始公式的分子是 `r + (α-1)/2 · a`，除以 R 后得到 `r/R + (α-1)/2 · a/R`。✓

### 1.4 Restart ESAV (第265-389行)

Restart 将 `r^n` 替换为 `r_hat = exp(I/2)`，即 `log_r = I/2`，`delta_log = 0`，`exp_delta = 1`。
代码中直接设 `numer = 1.0 + (α-1)/2 · a_scaled`。✓

### 1.5 Relax ESAV (第392-570行)

对照 Algorithm 6：
1. Vanilla step → `θ_new`, `log_r_tilde` ✓
2. `r_hat = exp(I(θ_new)/2)` ✓
3. `S^n = (r^n)² + λ/2·(‖θ^n‖² - ‖θ^{n+1}‖²)` ✓
4. 若 `r_hat² ≤ S^n`：ξ* = 0 ✓
5. 否则：`ξ* = (√S^n - r_hat) / (r_tilde - r_hat)`，clamp [0,1] ✓
6. `r_new = ξ·r_tilde + (1-ξ)·r_hat` ✓

**注意：Relax ESAV 与 Relax SAV 使用了不同的松弛公式。**
- SAV 版本：使用论文 Algorithm 4 中的 eta 参数二次方程（`eta=0.99`，基于 `‖Δθ‖²/Δt` 约束）
- ESAV 版本：使用 MATH_REFERENCE Algorithm 6 中的 S^n 能量预算方法

两者都是合法的松弛策略，但来源不同。SAV 版本更保守（eta<1 缩小了稳定性预算），ESAV 版本直接使用完整的理论预算 S^n。

---

## 二、数值策略分析

### 2.1 对数空间算术

**动机：** ESAV 的辅助变量 `r = exp(I/2)`，当 I 较大时（如 Example 2 初始 I≈34000），`exp(17000)` 直接溢出。

**方案：** 存储 `log_r = ln(r) = I/2`，所有计算通过 ratio 形式避免显式求 exp(I/2)。

**评价：** 这是合理且必要的数值策略。对数空间的核心优势在于乘除法变为加减法，避免了极大数的直接运算。

### 2.2 漂移安全机制 (`_DRIFT_MAX=5`, `_I_RESTART_THRESH=30`)

**机制：** 当 `I > 30` 或 `|log_r - log_R| > 5` 时，强制将 `log_r` 重置为 `log_R = I/2`。

**为什么需要这个机制：**

当 `I > 30` 时，`scale2 = exp(-I) ≈ 0`（float64下 exp(-30) ≈ 9.4e-14 勉强可用，exp(-700) = 0）。此时：
- `a_scaled ≈ 0`, `b_scaled ≈ 0`
- 分子退化为 `exp_delta`，分母退化为 `1`
- `ratio = exp(delta_log)` — 完全由漂移量决定

r-更新退化为 `r_new = R · exp(delta_log) = r`（不变），而 θ 在梯度下降中持续变化。这导致 `log_r` 固定但 `log_R` 持续变化，`delta_log` 无界增长。若不加限制，`ratio = exp(delta_log)` 会指数爆炸，导致 θ 更新发散。

**但这个机制有代价：**

实验数据显示，Vanilla ESAV 有 **10 次能量违反**，每次幅度巨大（2,000,000%+）。例如：
- epoch 681→682: energy 1.82e-04 → 4.37e+00（r 从 0.0135 跳到 2.09）
- epoch 1211→1212: energy 1.02e-04 → 2.60e+00（r 从 0.0101 跳到 1.61）

**机制的因果链：** r 在 Vanilla 跟踪下逐渐衰减（因为 `ratio < 1`），经过几百个 epoch 衰减到 r≈0.01。此时 `|delta_log| = |ln(0.01) - I/2|` 超过阈值 5，触发强制重启，r 跳回 exp(I/2)≈1.2。能量 = r² 从 0.0001 跳到 1.44。然后 r 又开始衰减，如此循环。

**影响：** 从理论角度，这打破了 Theorem 2 的无条件能量稳定性保证。但从实用角度，如果不加这个机制，Vanilla ESAV 的 r 要么固定不变（I 大时），要么 ratio 爆炸导致 NaN。这是一个**无法避免的数值妥协**。

### 2.3 SAV vs ESAV 的 r-衰减对比

| | Vanilla SAV | Vanilla ESAV |
|---|---|---|
| 衰减机制 | `r_new = r / (1 + dt/2·b)` 每步乘以 <1 因子 | `ratio = exp(delta_log) / (1+dt/2·b_scaled)` 在对数空间衰减 |
| 10k epochs 后的 r | 6.47e-40 | 0.411 |
| 能否恢复 | 不能（r 单调递减，趋于0） | 部分可以（漂移安全机制强制重启） |
| 最终 test error | 0.443 | 0.033 |

**关键洞察：SAV 的 r-崩溃是乘法性的，ESAV 的 r-漂移是加法性的（在对数空间）。**

SAV 每步将 r 乘以一个小于 1 的因子，N 步后 `r ∝ ∏(1/(1+dt·b_i/2))` → 指数衰减到 0。
ESAV 在对数空间中，`delta_log` 是每步的增量，漂移速度是线性的而非指数的。加上漂移安全机制定期重置，r 不会彻底崩溃。

这解释了为什么 Vanilla ESAV (test=0.033) 比 Vanilla SAV (test=0.443) 好 **13 倍**。

---

## 三、三个变体的行为模式

### 3.1 Restart ESAV — 最实用

| 指标 | Restart SAV | Restart ESAV | SGD |
|------|-----------|-------------|-----|
| Test RelError | 1.273e-02 | 1.273e-02 | 1.273e-02 |
| Final r | 1.005 | 1.005 | — |
| 能量违反 | 4424/9999 | 4435/9999 | — |

两者完全一致。原因：当 λ=0 时，Restart 每步都重置 `r = sqrt(I+C)`（SAV）或 `r = exp(I/2)`（ESAV）。α=1 消除了 a-项。两者退化为相同的梯度缩放：

```
SAV:  θ_new = θ - dt · r_new · μ = θ - dt · [r_hat/(1+dt/2·b)] · gradI/r_hat
                                  = θ - dt · gradI / (1 + dt/2·b)

ESAV: θ_new = θ - dt · ratio · gradI = θ - dt · gradI / (1 + dt/2·b_scaled)
```

当 λ=0 时，`b = ‖μ‖² = ‖gradI‖²/(I+C)` vs `b_scaled = ‖gradI‖²·exp(-I)`。
这两个值不同（`1/(I+C)` vs `exp(-I)`），但实验中差异微小（I≈0.01 时，1/(0.01+1)≈0.99 vs exp(-0.01)≈0.99）。当 I 趋近 0 时两者趋同，因此最终结果几乎一致。

### 3.2 Vanilla ESAV — 改善但仍不理想

Vanilla ESAV 有 48.6% 的 epochs r < 0.1，4.35% 的 epochs r < 0.01。r 在 0.008~2.25 之间振荡（漂移安全机制导致的锯齿形态）。最终 test=0.033，是 SGD 的 2.6 倍，但远好于 Vanilla SAV (0.443)。

### 3.3 Relax ESAV — 能量稳定且接近最优

| 指标 | Relax SAV | Relax ESAV |
|------|----------|-----------|
| Test RelError | 1.270e-02 | 1.282e-02 |
| Final r | 0.700 | 1.002 |
| 能量违反 | 0/9999 | 0/9999 |

**Relax ESAV 的 r 几乎不衰减**（1.88 → 1.00），而 Relax SAV 的 r 明显下降（1.50 → 0.70）。
原因：ESAV 的 Relax 使用 S^n 能量预算，由于 Vanilla ESAV 步产生的 r_tilde 接近 r_hat（漂移小），ξ* 经常为 0，退化为 restart。而 SAV 的 Relax 使用 eta 约束，ξ > 0 的频率更高，保留了更多 vanilla 成分（即 r 衰减的成分）。

---

## 四、已知限制与风险

### 4.1 Vanilla ESAV 的理论保证失效

由于漂移安全机制强制重启，Theorem 2 的能量稳定性不再成立。实验中观察到 10 次大幅能量跳升（每次约 2,000,000%）。这不是 bug，而是数值妥协的必然代价。

**如果要恢复理论保证**，需要让 r 自然跟踪而不强制重启，但这在 I 较大时不可能（scale2 下溢）。

### 4.2 Example 2 (多项式) 的特殊困难

Example 2 初始 I ≈ 34000，`exp(-I) = 0`（下溢到零）。在这种情况下，Vanilla ESAV 从第一步起就被漂移安全机制拦截，退化为 Restart ESAV 的行为。理论上的 Vanilla 跟踪完全不可行。

### 4.3 Relax SAV vs Relax ESAV 的公式不一致

两者使用了不同来源的松弛公式：
- SAV: 论文 Algorithm 4 的 eta 二次方程
- ESAV: MATH_REFERENCE Algorithm 6 的 S^n 方法

这不是错误，但意味着 SAV 和 ESAV 之间的 Relax 变体**不是严格的苹果对苹果比较**。如果要做公平对比，应统一使用同一种松弛策略。

---

## 五、总结

| 维度 | 评价 |
|------|------|
| **公式正确性** | ratio 形式推导正确，与 MATH_REFERENCE 数学等价 |
| **数值策略** | 对数空间 + 漂移安全机制是合理的工程妥协 |
| **Vanilla ESAV** | 比 Vanilla SAV 好 13 倍，但仍不如 SGD，且能量保证失效 |
| **Restart ESAV** | 与 Restart SAV 和 SGD 等价，最实用 |
| **Relax ESAV** | 零能量违反，r 衰减极小，接近 Restart 性能 |
| **主要风险** | Example 2 的 Vanilla ESAV 退化为 Restart；Relax 公式不统一 |
