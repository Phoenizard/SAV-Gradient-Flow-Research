# 损失函数调查报告：训练损失差异的根因分析

**日期：** 2026-02-27
**项目：** SAV-Gradient-Flow-Research
**阶段：** Phase 1 — SAV 论文复现

---

## 1. 问题描述

在 Phase 1 的论文复现实验中，我们发现了以下异常：

### Fig 7（Example 2：多项式回归，D=40, m=100, lr=0.01, λ=0）
- **论文目标：** SGD 训练损失（train relative error）应在 10000 epochs 内降至 ~1e-5
- **实验结果：** SGD 训练损失停滞在 **0.992**，几乎没有下降
- **SAV 方法同样停滞**，无明显收敛

### Fig 2（Example 1：sin+cos 回归，D=40, m=1000, lr=0.2, λ=0）
- **论文目标：** SAV 训练损失 ~1e-10
- **实验结果：** 训练损失可以下降，但速度较论文慢

### Fig 1（Example 1：D=20, m=1000, lr=0.6, λ=10, 能量稳定性验证）
- 能量曲线行为与论文一致：SAV/RelSAV 能量严格非递增，GD 在此学习率下发散（NaN）

---

## 2. 调查过程

### Step 1：排除超参数问题
- 核对 D、m、M、batch_size、lr、C、λ 等所有设置：与 PAPER_NOTES.md 和论文 Table 完全一致
- 排除数据生成问题：Example 2 的 x ~ Uniform(0,5)^D、目标函数 f*(x) = Σ c_i x_i² 实现正确

### Step 2：排除网络架构问题
- 网络结构 f(θ;x) = (1/m) Σ a_k σ(w_k·x + b_k) 已包含 1/m 因子
- ReLU 激活、Z-score 归一化均正确实现

### Step 3：排除 SAV 公式问题
- SAV 的 r-update 公式经过两轮修正，确认与论文公式一致
- 在 Fig 1 (λ=10) 场景下成功验证能量稳定性

### Step 4：聚焦损失函数
- 注意到关键线索：**Fig 7 的 SGD（不涉及 SAV 公式）也完全不收敛**
- 这排除了 SAV 公式错误的可能性——问题出在更基础的层面

### Step 5：分析 Example 1 vs Example 2 的差异
- Example 1 (sin+cos): y 值范围 ∈ [-2, 2]，|Σ y²| ≈ N（量级 ~1000）
- Example 2 (polynomial): y = Σ c_i x_i², x ∈ [0,5]^40，y 值范围 ∈ [0, ~500]，|Σ y²| ≈ **数百万**

这个差异指向了 **RelativeErrorLoss 的梯度抑制效应**。

---

## 3. 根因分析：RelativeError vs MSE 的梯度差异

### 当前实现的损失函数

我们使用 Relative Error Loss 作为训练损失（用于梯度计算）：

$$I_{rel}(\theta) = \frac{\sum_{i=1}^{N}(f(\theta; x_i) - y_i)^2}{\sum_{i=1}^{N} y_i^2}$$

### 标准 MSE 损失函数

$$I_{MSE}(\theta) = \frac{1}{N}\sum_{i=1}^{N}(f(\theta; x_i) - y_i)^2$$

### 梯度对比

两者的梯度关系为：

$$\nabla I_{rel} = \frac{1}{\sum y_i^2} \cdot \nabla\left[\sum (f - y)^2\right]$$

$$\nabla I_{MSE} = \frac{1}{N} \cdot \nabla\left[\sum (f - y)^2\right]$$

因此：

$$\nabla I_{rel} = \frac{N}{\sum y_i^2} \cdot \nabla I_{MSE}$$

**梯度缩放因子** = N / Σy²

### Example 1 (sin+cos)
- y ∈ [-2, 2]，平均 y² ≈ 1
- Σy² ≈ N × 1 = N
- 缩放因子 ≈ N/N = **1**（RelError ≈ MSE，几乎无差异）

### Example 2 (polynomial)
- y = Σ c_i x_i²，x ∈ [0,5]^40，c_i ∈ [0,1]
- 平均 y ≈ 0.5 × 40 × E[x²] = 0.5 × 40 × 25/3 ≈ 167
- 平均 y² ≈ 167² ≈ 28000
- Σy² ≈ 8000 × 28000 ≈ 2.24 × 10⁸
- 缩放因子 ≈ 8000 / (2.24 × 10⁸) = **3.57 × 10⁻⁵**

**结论：在 Example 2 中，RelativeErrorLoss 的有效学习率比 MSE 小 ~28000 倍！**

以 Fig 7 的 lr=0.01 为例：
- MSE 等效学习率 = 0.01
- RelError 等效学习率 ≈ 0.01 × 3.57e-5 ≈ **3.57 × 10⁻⁷**

如此微小的有效学习率完全解释了为什么训练损失停滞在 0.992。

---

## 4. 论文中损失函数的正确理解

重新审读论文后，我们认为：

1. **论文的 I(θ) 定义**（Eq. 3）为标准 MSE 或 sum-of-squares 形式，**不包含** 1/Σy² 归一化因子
2. **Relative Error 仅用于结果报告**（论文图中 y 轴标注为 "relative error"），不用于梯度计算
3. 这是一个 **报告指标 vs 优化目标** 的混淆

佐证：
- 论文 Figure 2/7 的 y 轴标注是 "relative error"，但这是评估指标
- 对于 Example 1，MSE ≈ RelError（因为 Σy² ≈ N），所以即使混用也不会有显著差异
- 对于 Example 2，差异巨大，而这正是我们观测到问题的地方

---

## 5. 修复方案

### 方案：分离训练损失和评估指标

1. **训练优化**：使用标准 MSE = (1/N) Σ(f-y)²
   - SAV 的 I(θ) 用 MSE 计算
   - SGD/Adam 的 loss 用 MSE 计算
   - SAV 的 r = sqrt(I_MSE + C)

2. **评估报告**：使用 Relative Error = Σ(f-y)²/Σy²
   - train_rel_error 和 test_rel_error 用于日志和图表
   - 与论文图表直接对比

3. **wandb 日志**：同时记录 train_mse、test_mse（调试用）和 train_rel_error、test_rel_error（与论文对比）

### 预期影响

| 实验 | 修复前 | 修复后（预期） |
|------|--------|---------------|
| Fig 7 SGD train_rel_error | 0.992（停滞） | < 0.1（正常收敛） |
| Fig 2 结果 | 可以收敛 | 保持不变（MSE ≈ RelError for Example 1） |
| Fig 1 能量稳定性 | 通过 | 保持通过 |

---

## 6. 结论

**根本原因：** 将论文的评估指标（Relative Error）误用为优化目标，导致 Example 2 的梯度被 Σy² ≈ 10⁸ 量级的分母严重抑制，使有效学习率降低约 28000 倍。

**影响范围：**
- Example 2（Fig 7, Fig 8）：严重影响，训练完全停滞
- Example 1（Fig 1, Fig 2）：几乎无影响（MSE ≈ RelError for sin+cos）

**修复难度：** 低。只需在训练循环中将损失函数从 RelativeErrorLoss 改为 MSELoss，保留 RelativeErrorLoss 用于评估报告。
