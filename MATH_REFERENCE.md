# Energy-Stable Optimization Algorithms via Auxiliary Variable Methods

## Notation and Setup

We minimize a loss $I(\theta)$ over parameters $\theta \in \mathbb{R}^d$ via the stabilized gradient flow:

$$\frac{d\theta}{dt} = -\mathcal{L}\theta - \nabla_\theta I(\theta), \quad \mathcal{L}\theta = \lambda\theta, \quad \lambda \geq 0$$

The stabilization term $\lambda\theta$ is added and subtracted from the energy. Define the **total energy**:

$$E(\theta) = \frac{\lambda}{2}\|\theta\|^2 + I(\theta)$$

Our goal: after time discretization with step size $\Delta t > 0$, we require **unconditional energy stability**: $\mathcal{E}^{n+1} \leq \mathcal{E}^n$ for all $\Delta t > 0$.

---

# Part 1 — SAV (Scalar Auxiliary Variable)

## 1.1 Continuous System

**Auxiliary variable:** $r(t) := \sqrt{I(\theta(t)) + C}$, where $C > 0$ ensures $I + C > 0$.

Differentiate:

$$\dot{r} = \frac{\langle \nabla I, \dot{\theta} \rangle}{2\sqrt{I + C}} = \frac{\langle \nabla I, \dot{\theta} \rangle}{2r}$$

Rewrite the original flow using the identity $\nabla I = \frac{r}{\sqrt{I+C}} \nabla I = r \cdot \frac{\nabla I}{\sqrt{I+C}}$. Define $\mu(\theta) := \frac{\nabla_\theta I(\theta)}{\sqrt{I(\theta)+C}}$.

The equivalent **(θ, r) system** is:

$$\boxed{\frac{d\theta}{dt} = -\lambda\theta - r\,\mu(\theta), \quad \frac{dr}{dt} = \frac{1}{2}\langle \mu(\theta), \dot{\theta} \rangle}$$

**Energy dissipation:** The modified energy is $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + r^2$.

$$\frac{d\mathcal{E}}{dt} = \lambda\langle\theta,\dot{\theta}\rangle + 2r\dot{r} = \lambda\langle\theta,\dot{\theta}\rangle + \langle\mu,\dot{\theta}\rangle \cdot r$$

Substituting $\dot{\theta} = -\lambda\theta - r\mu$:

$$= \lambda\langle\theta, -\lambda\theta - r\mu\rangle + r\langle\mu, -\lambda\theta - r\mu\rangle$$
$$= -\lambda^2\|\theta\|^2 - \lambda r\langle\theta,\mu\rangle - \lambda r\langle\mu,\theta\rangle - r^2\|\mu\|^2$$
$$= -\lambda^2\|\theta\|^2 - 2\lambda r\langle\theta,\mu\rangle - r^2\|\mu\|^2$$
$$= -\|\lambda\theta + r\mu\|^2 = -\|\dot{\theta}\|^2 \leq 0 \quad \checkmark$$

In particular, $\frac{d}{dt}(r^2) = \langle\mu,\dot{\theta}\rangle \cdot r$, and combined with the $\lambda$ term, the full energy dissipates.

---

## 1.2 Vanilla SAV

**Semi-implicit discretization:** Treat $\mathcal{L}\theta$ implicitly, and $\mu$ explicitly at $\theta^n$:

$$\frac{\theta^{n+1} - \theta^n}{\Delta t} = -\lambda\theta^{n+1} - r^{n+1}\mu^n$$

$$\frac{r^{n+1} - r^n}{\Delta t} = \frac{1}{2}\left\langle \mu^n, \frac{\theta^{n+1}-\theta^n}{\Delta t} \right\rangle$$

where $\mu^n := \frac{\nabla I(\theta^n)}{\sqrt{I(\theta^n)+C}}$.

**Decoupling trick.** From the first equation:

$$\theta^{n+1} = \frac{1}{1+\lambda\Delta t}\left(\theta^n - \Delta t\, r^{n+1}\mu^n\right)$$

Define $\alpha := \frac{1}{1+\lambda\Delta t}$. Then:

$$\theta^{n+1} = \alpha\theta^n - \alpha\Delta t\, r^{n+1}\mu^n$$

Substitute into the second equation:

$$r^{n+1} - r^n = \frac{1}{2}\langle \mu^n, \theta^{n+1} - \theta^n \rangle$$

$$= \frac{1}{2}\langle\mu^n, (\alpha-1)\theta^n - \alpha\Delta t\,r^{n+1}\mu^n\rangle$$

$$= \frac{1}{2}(\alpha-1)\langle\mu^n,\theta^n\rangle - \frac{\alpha\Delta t}{2}\|\mu^n\|^2 r^{n+1}$$

Let $a := \langle\mu^n,\theta^n\rangle$ and $b := \|\mu^n\|^2$. Then:

$$r^{n+1}\left(1 + \frac{\alpha\Delta t}{2}b\right) = r^n + \frac{\alpha-1}{2}a$$

Since $\alpha = \frac{1}{1+\lambda\Delta t}$, we have $\alpha - 1 = \frac{-\lambda\Delta t}{1+\lambda\Delta t}$.

$$\boxed{r^{n+1} = \frac{r^n - \frac{\lambda\Delta t}{2(1+\lambda\Delta t)}\langle\mu^n,\theta^n\rangle}{1 + \frac{\Delta t}{2(1+\lambda\Delta t)}\|\mu^n\|^2}}$$

$$\boxed{\theta^{n+1} = \frac{\theta^n - \Delta t\, r^{n+1}\mu^n}{1+\lambda\Delta t}}$$

### Algorithm 1: Vanilla SAV

> 1. Compute $\mu^n = \nabla I(\theta^n) / \sqrt{I(\theta^n)+C}$
> 2. Compute $a = \langle\mu^n,\theta^n\rangle$, $b = \|\mu^n\|^2$, $\alpha = 1/(1+\lambda\Delta t)$
> 3. $r^{n+1} = \displaystyle\frac{r^n + \frac{\alpha-1}{2}\,a}{1 + \frac{\alpha\Delta t}{2}\,b}$
> 4. $\theta^{n+1} = \alpha(\theta^n - \Delta t\, r^{n+1}\mu^n)$

**Note:** When $\lambda=0$, $\alpha=1$ and the $a$ term vanishes, giving the simplified form $r^{n+1} = r^n/(1 + \Delta t\,b/2)$, $\theta^{n+1} = \theta^n - \Delta t\,r^{n+1}\mu^n$.

**Theorem 1 (Unconditional Stability).** $(r^{n+1})^2 + \frac{\lambda}{2}\|\theta^{n+1}\|^2 \leq (r^n)^2 + \frac{\lambda}{2}\|\theta^n\|^2$ for all $\Delta t > 0$.

**Proof sketch.** Take the inner product of the $\theta$-equation with $\frac{\theta^{n+1}-\theta^n}{\Delta t}$, and multiply the $r$-equation by $2r^{n+1}$:

From the $\theta$-equation:

$$\frac{\|\theta^{n+1}-\theta^n\|^2}{\Delta t} + \lambda\langle\theta^{n+1},\theta^{n+1}-\theta^n\rangle = -r^{n+1}\langle\mu^n,\theta^{n+1}-\theta^n\rangle$$

Using the identity $\langle a, a-b\rangle = \frac{1}{2}(\|a\|^2 - \|b\|^2 + \|a-b\|^2)$:

$$\frac{\|\theta^{n+1}-\theta^n\|^2}{\Delta t} + \frac{\lambda}{2}(\|\theta^{n+1}\|^2 - \|\theta^n\|^2 + \|\theta^{n+1}-\theta^n\|^2) = -r^{n+1}\langle\mu^n,\theta^{n+1}-\theta^n\rangle$$

From the $r$-equation: $2r^{n+1}(r^{n+1}-r^n) = r^{n+1}\langle\mu^n,\theta^{n+1}-\theta^n\rangle$.

Using $(r^{n+1})^2 - (r^n)^2 \leq 2r^{n+1}(r^{n+1}-r^n)$ (since $|r^{n+1}-r^n|^2 \geq 0$), we add:

$$\frac{\|\theta^{n+1}-\theta^n\|^2}{\Delta t} + \frac{\lambda}{2}\|\theta^{n+1}-\theta^n\|^2 + (r^{n+1})^2 - (r^n)^2 + \frac{\lambda}{2}(\|\theta^{n+1}\|^2 - \|\theta^n\|^2) \leq 0$$

Hence $\mathcal{E}^{n+1} \leq \mathcal{E}^n$ where $\mathcal{E}^n = (r^n)^2 + \frac{\lambda}{2}\|\theta^n\|^2$. $\square$

**Practical notes:** Requires one gradient evaluation per step. Cost is $O(d)$ where $d = \dim(\theta)$. The scalar $r^n$ drifts from $\sqrt{I(\theta^n)+C}$ over time — this is Vanilla's key weakness.

---

## 1.3 Restart SAV

**Problem with Vanilla SAV.** As training proceeds, the auxiliary variable $r^n$ satisfies the discrete dynamics but may drift significantly from the "true" value $\sqrt{I(\theta^n)+C}$. Since $r^n$ multiplies $\mu^n$ in the update, this mismatch means the effective gradient $r^n\mu^n \neq \nabla I(\theta^n)$, causing the algorithm to follow an incorrect descent direction. Specifically, if $r^n \ll \sqrt{I(\theta^n)+C}$, the step is too small (stalling); if $r^n \gg \sqrt{I(\theta^n)+C}$, the step is too large (instability in $I$ even though $\mathcal{E}$ decreases).

**Restart strategy.** At each step, reset $r^n \leftarrow \hat{r}^n := \sqrt{I(\theta^n)+C}$.

### Algorithm 2: Restart SAV

> 1. Compute $\hat{r}^n = \sqrt{I(\theta^n)+C}$, $\mu^n = \nabla I(\theta^n)/\hat{r}^n$
> 2. Compute $a = \langle\mu^n,\theta^n\rangle$, $b = \|\mu^n\|^2$, $\alpha = 1/(1+\lambda\Delta t)$
> 3. $r^{n+1} = \displaystyle\frac{\hat{r}^n - \frac{\lambda\Delta t\,\alpha}{2}\,a}{1 + \frac{\Delta t\,\alpha}{2}\,b}$
> 4. $\theta^{n+1} = \alpha(\theta^n - \Delta t\, r^{n+1}\mu^n)$

**Energy dissipated:** At each step, we have $\mathcal{E}^n = (\hat{r}^n)^2 + \frac{\lambda}{2}\|\theta^n\|^2 = I(\theta^n) + C + \frac{\lambda}{2}\|\theta^n\|^2$. The scheme guarantees:

$$(r^{n+1})^2 + \frac{\lambda}{2}\|\theta^{n+1}\|^2 \leq I(\theta^n) + C + \frac{\lambda}{2}\|\theta^n\|^2$$

However, since $r^{n+1}$ is *not* reset until the next step, we only know $(r^{n+1})^2 \leq (\hat{r}^n)^2$, not $I(\theta^{n+1})+C \leq I(\theta^n)+C$. The restart ensures the energy at each step is always "fresh," but we lose guaranteed monotone decrease of $I(\theta^n)$ itself.

**Practical notes:** Requires one extra function evaluation $I(\theta^n)$ per step (beyond the gradient). Ensures $r^n$ is always synchronized with $\theta^n$.

---

## 1.4 Relax SAV

Define two quantities after computing $\theta^{n+1}$:

- $\tilde{r}^{n+1}$: the Vanilla update from §1.2 (starting from $r^n$)
- $\hat{r}^{n+1} := \sqrt{I(\theta^{n+1})+C}$: the "true" value at the new iterate

Set $r^{n+1} = \xi\,\tilde{r}^{n+1} + (1-\xi)\,\hat{r}^{n+1}$ for $\xi \in [0,1]$.

**Optimal ξ derivation.** We want to minimize:

$$|r^{n+1} - \hat{r}^{n+1}|^2 = |\xi(\tilde{r}^{n+1} - \hat{r}^{n+1})|^2 = \xi^2|\tilde{r}^{n+1} - \hat{r}^{n+1}|^2$$

This is minimized at $\xi = 0$ (pure restart). But we impose the stability constraint $(r^{n+1})^2 \leq (r^n)^2 + \frac{\lambda}{2}(\|\theta^n\|^2 - \|\theta^{n+1}\|^2)$, which from the Vanilla proof is guaranteed when $r^{n+1} = \tilde{r}^{n+1}$ (i.e., $\xi = 1$).

Let $\delta = \tilde{r}^{n+1} - \hat{r}^{n+1}$ and write $r^{n+1} = \hat{r}^{n+1} + \xi\delta$. The constraint is:

$$(\hat{r}^{n+1} + \xi\delta)^2 \leq (r^n)^2 + \frac{\lambda}{2}(\|\theta^n\|^2 - \|\theta^{n+1}\|^2) =: S^n$$

where $S^n$ is the right-hand side from the stability proof (computable). This gives:

$$(\hat{r}^{n+1})^2 + 2\xi\delta\hat{r}^{n+1} + \xi^2\delta^2 \leq S^n$$

Define $R := S^n - (\hat{r}^{n+1})^2$ (the "stability budget"). The constraint becomes:

$$\xi^2\delta^2 + 2\xi\delta\hat{r}^{n+1} - R \leq 0$$

This is a quadratic in $\xi$. Solving with the quadratic formula and taking the smaller root gives the maximum feasible $\xi$:

If $\delta \neq 0$:

$$\xi^* = \frac{-2\delta\hat{r}^{n+1} + \sqrt{4\delta^2(\hat{r}^{n+1})^2 + 4\delta^2 R}}{2\delta^2} = \frac{-\hat{r}^{n+1} + \sqrt{(\hat{r}^{n+1})^2 + R}}{|\delta|}$$

Wait — we want to *minimize* $\xi^2\delta^2$, so we want the *smallest* $\xi \geq 0$ satisfying the constraint. Since the constraint is $(r^{n+1})^2 \leq S^n$, and $\xi = 0$ gives $(r^{n+1})^2 = (\hat{r}^{n+1})^2$:

- If $(\hat{r}^{n+1})^2 \leq S^n$ (restart is already stable), then $\xi^* = 0$.
- Otherwise, we need $\xi > 0$ to satisfy the constraint.

More precisely, the optimal $\xi$ is:

$$\boxed{\xi^* = \begin{cases} 0 & \text{if } (\hat{r}^{n+1})^2 \leq S^n \\ \xi_{\min} & \text{otherwise}\end{cases}}$$

where in the "otherwise" case, we need the constraint to hold with equality (to use the minimum amount of Vanilla mixing):

$$(\hat{r}^{n+1} + \xi\delta)^2 = S^n$$

$$\xi = \frac{-\hat{r}^{n+1}\delta + \operatorname{sgn}(\delta)\sqrt{S^n}\,|\delta|}{\delta^2} = \frac{\operatorname{sgn}(\delta)\sqrt{S^n} - \hat{r}^{n+1}}{\delta}$$

Since $\tilde{r}^{n+1}$ satisfies $(\tilde{r}^{n+1})^2 \leq S^n$ (Vanilla stability), and we set $r^{n+1} = \hat{r}^{n+1} + \xi\delta$ where $\delta = \tilde{r}^{n+1} - \hat{r}^{n+1}$, we want $r^{n+1} > 0$. The simplest practical formula: clamp $\xi \in [0, 1]$ after solving the quadratic.

### Algorithm 3: Relax SAV

> 1. Run Vanilla SAV (Algorithm 1) to get $\tilde{r}^{n+1}$, $\theta^{n+1}$
> 2. Compute $\hat{r}^{n+1} = \sqrt{I(\theta^{n+1})+C}$
> 3. Compute $S^n = (r^n)^2 + \frac{\lambda}{2}(\|\theta^n\|^2 - \|\theta^{n+1}\|^2)$
> 4. If $(\hat{r}^{n+1})^2 \leq S^n$: set $\xi^* = 0$
> 5. Else: $\delta = \tilde{r}^{n+1} - \hat{r}^{n+1}$, $\xi^* = \frac{\sqrt{S^n} - \hat{r}^{n+1}}{\delta}$ (taking the root with correct sign ensuring $r^{n+1} > 0$), clamp to $[0,1]$
> 6. $r^{n+1} = \xi^*\tilde{r}^{n+1} + (1-\xi^*)\hat{r}^{n+1}$

**Energy statement:** $(r^{n+1})^2 + \frac{\lambda}{2}\|\theta^{n+1}\|^2 \leq (r^n)^2 + \frac{\lambda}{2}\|\theta^n\|^2$ by construction of $\xi^*$.

**Practical notes:** Requires one extra forward pass to evaluate $I(\theta^{n+1})$ for $\hat{r}^{n+1}$. Best of both worlds: stays close to the true energy (like Restart) while guaranteeing unconditional stability (like Vanilla).

---

# Part 2 — ESAV (Exponential SAV)

## Motivation and Choice of ϕ

**Problem with SAV when $I(\theta) \to 0$:** As training succeeds, $I(\theta) \to 0$. Then $r = \sqrt{I+C} \to \sqrt{C}$ and $\mu = \nabla I / \sqrt{I+C} \to \nabla I / \sqrt{C}$, which is well-defined. However, if $C$ is small, the ratio $r / \sqrt{I+C}$ amplifies small perturbations. More critically, if we choose $C$ poorly and $I$ takes negative values (possible with regularization), $I + C < 0$ makes $r$ undefined.

**Design criteria for ϕ:**
1. $\phi: \mathbb{R} \to (0, \infty)$ — ensures $r > 0$ for **any** value of $I$, no additive constant $C$ needed.
2. $\phi'(s) > 0$ — monotonicity, so $r$ tracks $I$.
3. The modified energy $\psi(r) := \phi^{-1}(r)$ should be expressible simply in terms of $r$.

**Natural choice:** $\phi(s) = \exp(s/2)$, i.e.,

$$\boxed{r(t) = \exp\!\left(\frac{I(\theta(t))}{2}\right)}$$

Then $I = 2\ln r$, and the modified energy is $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + 2\ln r$, but this is not bounded below. Instead, we work with the energy $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + r^2$ and prove dissipation of this quantity (matching the structure of SAV).

Actually, the more natural ESAV energy is $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + r$ (since $r = e^{I/2}$ is itself the exponential of the energy). But for structural consistency with SAV and to enable the same decoupling trick, we use:

**Refined choice:** $r(t) = \exp(I(\theta(t)))$, so $I = \ln r$ and $r > 0$ always.

$$\dot{r} = r\,\langle\nabla I, \dot{\theta}\rangle$$

Define $\mu(\theta) := \nabla_\theta I(\theta)$ (the raw gradient — no normalization needed!).

The modified energy: $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + r$.

$$\frac{d\mathcal{E}}{dt} = \lambda\langle\theta,\dot{\theta}\rangle + \dot{r} = \lambda\langle\theta,\dot{\theta}\rangle + r\langle\mu,\dot{\theta}\rangle$$

With $\dot{\theta} = -\lambda\theta - r\mu$:

$$= \lambda\langle\theta,-\lambda\theta - r\mu\rangle + r\langle\mu,-\lambda\theta - r\mu\rangle = -\|\lambda\theta + r\mu\|^2... $$

Wait — the original flow is $\dot{\theta} = -\lambda\theta - \nabla I$, but in the ESAV system we need to express $\nabla I$ via the auxiliary variable. Since $r = e^I$, we have $\nabla I = \frac{\nabla r}{r}$, so $\nabla I = \mu$. But the $(θ, r)$ system should use $r$ to replace $\nabla I$.

Let me redo this more carefully.

**Correct ESAV formulation.** The original PDE: $\dot{\theta} = -\lambda\theta - \nabla I$. With $r = e^I$:
- $\nabla I = \nabla(\ln r) = \frac{1}{r}\nabla_\theta r$... but $r$ is a scalar, so $\nabla_\theta r = r\nabla I$.

The key identity: $\nabla I(\theta) = \frac{r}{r}\nabla I = r \cdot \frac{\nabla I}{r} = r \cdot \frac{\nabla I}{e^I}$. Define:

$$\mu^n := \frac{\nabla I(\theta^n)}{r^n} = \frac{\nabla I(\theta^n)}{e^{I(\theta^n)}} = \nabla I(\theta^n) e^{-I(\theta^n)}$$

Then $\nabla I = r\mu$, exactly as in SAV! The system becomes:

$$\dot{\theta} = -\lambda\theta - r\mu, \quad \dot{r} = r\langle\mu, \dot{\theta}\rangle$$

But the $r$-equation has $r$ on the right side, unlike SAV. For the continuous energy:

$$\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + r$$

$$\frac{d\mathcal{E}}{dt} = \lambda\langle\theta,\dot\theta\rangle + r\langle\mu,\dot\theta\rangle = \langle\lambda\theta + r\mu, \dot\theta\rangle = \langle -\dot\theta,\dot\theta\rangle = -\|\dot\theta\|^2 \leq 0$$

This works. But for the **discrete** scheme, the multiplicative $r$ in $\dot{r}$ creates difficulty. Following the ESAV literature (e.g., Liu & Li 2020), we use $\ln r$ instead.

**Final ESAV choice.** Let $s(t) := \ln r(t) = I(\theta(t))$. Then:

$$\dot{s} = \langle\nabla I, \dot\theta\rangle$$

And $\nabla I = e^s \mu$ where $\mu = \nabla I \cdot e^{-s} = \nabla I \cdot e^{-I}$.

This is getting circuitous. Let me adopt the clean standard form.

### ESAV: Final Clean Formulation

**Auxiliary variable:** $r(t) = \exp\!\big(\frac{I(\theta(t))}{2}\big)$

**Derivative:** $\dot{r} = \frac{r}{2}\langle\nabla I, \dot\theta\rangle$

**Define:** $\mu(\theta) := \frac{\nabla I(\theta)}{2r} = \frac{\nabla I(\theta)}{2\exp(I/2)}$

Then $\nabla I = 2r\mu$, and:

$$\dot{\theta} = -\lambda\theta - 2r\mu, \quad \dot{r} = r\langle\mu,\dot\theta\rangle$$

**Modified energy:** $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + r^2$

$$\frac{d\mathcal{E}}{dt} = \lambda\langle\theta,\dot\theta\rangle + 2r\dot{r} = \lambda\langle\theta,\dot\theta\rangle + 2r^2\langle\mu,\dot\theta\rangle$$

But $\dot\theta = -\lambda\theta - 2r\mu$, so $\lambda\theta + 2r\mu = -\dot\theta$. However:

$$\frac{d\mathcal{E}}{dt} = \langle \lambda\theta + 2r^2\mu, \dot\theta\rangle$$

This doesn't factor cleanly because of $r^2$ vs $r$. The issue is the extra factor of $r$ in $\dot{r}$.

**Resolution:** Use $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + 2r$ (linear in $r$):

$$\frac{d\mathcal{E}}{dt} = \lambda\langle\theta,\dot\theta\rangle + 2\dot{r} = \lambda\langle\theta,\dot\theta\rangle + r\langle\nabla I,\dot\theta\rangle$$

With $\nabla I = 2r\mu$: $= \lambda\langle\theta,\dot\theta\rangle + 2r^2\langle\mu,\dot\theta\rangle$. Same issue.

**Actually**, let me use the approach from the ESAV literature directly. The key insight: with $r = e^{I/2}$, we have $r^2 = e^I$, so:

$$\frac{d(r^2)}{dt} = 2r\dot{r} = r^2\langle\nabla I,\dot\theta\rangle = e^I\langle\nabla I,\dot\theta\rangle$$

The continuous system is:

$$\dot\theta = -\lambda\theta - \nabla I$$

$$\frac{d(r^2)}{dt} = r^2\langle\nabla I,\dot\theta\rangle = -r^2\langle\nabla I, \lambda\theta + \nabla I\rangle$$

For energy stability, we consider $\mathcal{E}_{\rm ESAV} = \frac{\lambda}{2}\|\theta\|^2 + r^2$ (noting $r^2 = e^I$ is an exponentially-weighted version of the loss):

$$\frac{d\mathcal{E}}{dt} = \lambda\langle\theta,\dot\theta\rangle + r^2\langle\nabla I,\dot\theta\rangle = \langle\lambda\theta + r^2\nabla I/1, \dot\theta\rangle$$

This doesn't simplify to $-\|\dot\theta\|^2$ because $r^2\nabla I \neq \nabla I$ (unless $r^2 = 1$, i.e., $I = 0$).

**The correct ESAV approach** (following Liu & Li 2020): The system is modified to *replace* $\nabla I$ with a term involving $r$. Define:

$$\dot\theta = -\lambda\theta - \frac{r}{R(\theta)}\nabla I(\theta)$$

where $R(\theta) = e^{I(\theta)/2}$ is the "true" value. The ratio $r/R(\theta)$ equals 1 in continuous time (since $r = R(\theta)$), but differs after discretization. Then:

$$\dot{r} = \frac{r}{2R(\theta)}\langle\nabla I,\dot\theta\rangle$$

This is identical to SAV with $\sqrt{I+C}$ replaced by $e^{I/2}$! The structure is:

$$\dot\theta = -\lambda\theta - r\nu, \quad \dot{r} = \frac{1}{2}\langle\nu,\dot\theta\rangle$$

where $\nu(\theta) := \frac{\nabla I(\theta)}{e^{I(\theta)/2}}$.

**Energy:** $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + r^2$, and $\frac{d\mathcal{E}}{dt} = -\|\dot\theta\|^2 \leq 0$ (identical algebra to SAV §1.1).

This is the correct ESAV formulation. The only difference from SAV is:

| | SAV | ESAV |
|---|---|---|
| Auxiliary variable | $r = \sqrt{I+C}$ | $r = \exp(I/2)$ |
| Normalization factor | $\mu = \nabla I / \sqrt{I+C}$ | $\nu = \nabla I / \exp(I/2)$ |
| Well-definedness | Requires $I + C > 0$ | Always $r > 0$ |

---

## 2.1 Continuous ESAV System

$$\boxed{\dot\theta = -\lambda\theta - r\nu(\theta), \quad \dot{r} = \frac{1}{2}\langle\nu(\theta),\dot\theta\rangle}$$

where $\nu(\theta) := \nabla I(\theta) \cdot e^{-I(\theta)/2}$.

**Energy:** $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + r^2$. Dissipation: $\frac{d\mathcal{E}}{dt} = -\|\lambda\theta + r\nu\|^2 = -\|\dot\theta\|^2 \leq 0$. (Proof identical to SAV §1.1.)

---

## 2.2 Vanilla ESAV

The discretization is structurally identical to Vanilla SAV with $\mu^n$ replaced by $\nu^n$:

### Algorithm 4: Vanilla ESAV

> 1. Compute $\nu^n = \nabla I(\theta^n) / \exp(I(\theta^n)/2)$
> 2. Compute $a = \langle\nu^n,\theta^n\rangle$, $b = \|\nu^n\|^2$, $\alpha = 1/(1+\lambda\Delta t)$
> 3. $r^{n+1} = \displaystyle\frac{r^n - \frac{\lambda\Delta t\,\alpha}{2}\,a}{1 + \frac{\Delta t\,\alpha}{2}\,b}$
> 4. $\theta^{n+1} = \alpha(\theta^n - \Delta t\, r^{n+1}\nu^n)$

**Theorem 2.** $(r^{n+1})^2 + \frac{\lambda}{2}\|\theta^{n+1}\|^2 \leq (r^n)^2 + \frac{\lambda}{2}\|\theta^n\|^2$ unconditionally.

**Proof:** Identical to Theorem 1, replacing $\mu^n$ with $\nu^n$ throughout. $\square$

**Practical notes:** No $C$ constant needed. $r > 0$ is automatic since $r = e^{I/2} > 0$. However, if $I$ is large, $e^{I/2}$ can overflow — in practice, use $\ln r$ for storage and compute $\nu^n = \nabla I \cdot e^{-I/2}$ via $\nabla I \cdot \exp(-I/2)$ with appropriate numerical safeguards.

---

## 2.3 Restart ESAV

Reset $r^n \leftarrow \hat{r}^n := \exp(I(\theta^n)/2)$ at each step.

### Algorithm 5: Restart ESAV

> 1. Compute $\hat{r}^n = \exp(I(\theta^n)/2)$, $\nu^n = \nabla I(\theta^n)/\hat{r}^n$
> 2–4. Same as Vanilla ESAV with $r^n$ replaced by $\hat{r}^n$

**Energy dissipated:** $(r^{n+1})^2 + \frac{\lambda}{2}\|\theta^{n+1}\|^2 \leq e^{I(\theta^n)} + \frac{\lambda}{2}\|\theta^n\|^2$.

**Practical notes:** Same as Restart SAV — one extra evaluation of $I(\theta^n)$ per step.

---

## 2.4 Relax ESAV

Identical structure to Relax SAV (Algorithm 3), with:
- $\tilde{r}^{n+1}$: Vanilla ESAV output
- $\hat{r}^{n+1} = \exp(I(\theta^{n+1})/2)$: true value at new iterate
- $S^n = (r^n)^2 + \frac{\lambda}{2}(\|\theta^n\|^2 - \|\theta^{n+1}\|^2)$: stability budget

### Algorithm 6: Relax ESAV

> 1. Run Vanilla ESAV to get $\tilde{r}^{n+1}$, $\theta^{n+1}$
> 2. Compute $\hat{r}^{n+1} = \exp(I(\theta^{n+1})/2)$
> 3. Compute $S^n = (r^n)^2 + \frac{\lambda}{2}(\|\theta^n\|^2 - \|\theta^{n+1}\|^2)$
> 4. If $(\hat{r}^{n+1})^2 \leq S^n$: $\xi^* = 0$
> 5. Else: $\delta = \tilde{r}^{n+1} - \hat{r}^{n+1}$, $\xi^* = \frac{\sqrt{S^n} - \hat{r}^{n+1}}{\delta}$, clamp to $[0,1]$
> 6. $r^{n+1} = \xi^*\tilde{r}^{n+1} + (1-\xi^*)\hat{r}^{n+1}$

**Optimal ξ derivation:** Identical to §1.4 — the algebra depends only on the $r^2 + \frac{\lambda}{2}\|\theta\|^2$ energy structure, not on the specific form of $\phi$. $\square$

---

# Part 3 — IEQ (Invariant Energy Quadratization)

## Setup and Auxiliary Variable

For $I(\theta) = \frac{1}{2}\|f(\theta) - y\|^2$ where $f: \mathbb{R}^d \to \mathbb{R}^m$ (e.g., $m$ = batch size × output dim):

**Natural IEQ variable:**

$$\boxed{q(t) := f(\theta(t)) - y \in \mathbb{R}^m}$$

This is a **vector** (not scalar like SAV/ESAV) because it preserves the full residual structure. The loss becomes:

$$I(\theta) = \frac{1}{2}\|q\|^2$$

which is **quadratic** in $q$ — this is the defining property of IEQ. No square root, no exponential, no additive constant.

**Jacobian:** $J(\theta) := \frac{\partial f}{\partial \theta} \in \mathbb{R}^{m \times d}$.

**Derivative of q:**

$$\dot{q} = J(\theta)\dot{\theta}$$

**Gradient:** $\nabla_\theta I = J^\top q$.

---

## 3.1 Continuous IEQ System

The original flow $\dot\theta = -\lambda\theta - \nabla I = -\lambda\theta - J^\top q$ becomes the $(\theta, q)$ system:

$$\boxed{\dot\theta = -\lambda\theta - J^\top q, \quad \dot{q} = J\dot\theta}$$

**Modified energy:** $\mathcal{E} = \frac{\lambda}{2}\|\theta\|^2 + \frac{1}{2}\|q\|^2$

$$\frac{d\mathcal{E}}{dt} = \lambda\langle\theta,\dot\theta\rangle + \langle q,\dot{q}\rangle = \lambda\langle\theta,\dot\theta\rangle + \langle q, J\dot\theta\rangle = \langle\lambda\theta + J^\top q, \dot\theta\rangle = \langle -\dot\theta, \dot\theta\rangle = -\|\dot\theta\|^2 \leq 0 \quad \checkmark$$

---

## 3.2 Vanilla IEQ

**Semi-implicit discretization:**

$$\frac{\theta^{n+1} - \theta^n}{\Delta t} = -\lambda\theta^{n+1} - (J^n)^\top q^{n+1}$$

$$\frac{q^{n+1} - q^n}{\Delta t} = J^n \frac{\theta^{n+1} - \theta^n}{\Delta t}$$

where $J^n := J(\theta^n)$. From the second equation:

$$q^{n+1} = q^n + J^n(\theta^{n+1} - \theta^n)$$

Substitute into the first:

$$\frac{\theta^{n+1} - \theta^n}{\Delta t} = -\lambda\theta^{n+1} - (J^n)^\top\big[q^n + J^n(\theta^{n+1} - \theta^n)\big]$$

Let $\delta\theta = \theta^{n+1} - \theta^n$:

$$\frac{\delta\theta}{\Delta t} = -\lambda(\theta^n + \delta\theta) - (J^n)^\top q^n - (J^n)^\top J^n \delta\theta$$

$$\frac{\delta\theta}{\Delta t} + \lambda\delta\theta + (J^n)^\top J^n \delta\theta = -\lambda\theta^n - (J^n)^\top q^n$$

$$\left(\frac{1}{\Delta t}I_d + \lambda I_d + (J^n)^\top J^n\right)\delta\theta = -\lambda\theta^n - (J^n)^\top q^n$$

This requires solving an **$d \times d$ linear system** with matrix $A = \frac{1}{\Delta t}I_d + \lambda I_d + (J^n)^\top J^n$. Since $d$ (number of parameters) is huge, this is impractical.

**Alternative: work in output space.** Apply $J^n$ to both sides. Let $\delta q = J^n\delta\theta = q^{n+1} - q^n$. From the $\theta$-equation:

$$\theta^{n+1} = \frac{1}{1+\lambda\Delta t}(\theta^n - \Delta t(J^n)^\top q^{n+1})$$

$$\delta\theta = \frac{-\lambda\Delta t\,\theta^n - \Delta t(J^n)^\top q^{n+1}}{1+\lambda\Delta t}$$

Then $\delta q = J^n\delta\theta$:

$$q^{n+1} - q^n = \frac{\Delta t}{1+\lambda\Delta t}\left(-\lambda J^n\theta^n - J^n(J^n)^\top q^{n+1}\right)$$

Let $\alpha = \frac{\Delta t}{1+\lambda\Delta t}$ and $G = J^n(J^n)^\top \in \mathbb{R}^{m \times m}$:

$$(I_m + \alpha G)\,q^{n+1} = q^n - \alpha\lambda J^n\theta^n$$

This is an **$m \times m$ system** where $m$ = output dimension (typically batch_size × num_classes or just batch_size for regression). Solving costs $O(m^3)$.

### Algorithm 7: Vanilla IEQ

> 1. Compute $J^n = \frac{\partial f}{\partial\theta}(\theta^n)$, $q^n$ (the current residual variable), $G = J^n(J^n)^\top$
> 2. Solve $(I_m + \alpha G)\,q^{n+1} = q^n - \alpha\lambda J^n\theta^n$ where $\alpha = \Delta t/(1+\lambda\Delta t)$
> 3. $\theta^{n+1} = \frac{1}{1+\lambda\Delta t}(\theta^n - \Delta t(J^n)^\top q^{n+1})$
> 4. (Do NOT reset $q^{n+1}$ — keep it as computed)

**Theorem 3 (Unconditional Stability).** $\frac{1}{2}\|q^{n+1}\|^2 + \frac{\lambda}{2}\|\theta^{n+1}\|^2 \leq \frac{1}{2}\|q^n\|^2 + \frac{\lambda}{2}\|\theta^n\|^2$.

**Proof sketch.** Take inner product of the $\theta$-equation with $(\theta^{n+1}-\theta^n)/\Delta t$ and of the $q$-equation with $q^{n+1}$:

$$\frac{\|\delta\theta\|^2}{\Delta t} + \frac{\lambda}{2}(\|\theta^{n+1}\|^2 - \|\theta^n\|^2) + \frac{\lambda}{2}\|\delta\theta\|^2 = -\langle (J^n)^\top q^{n+1}, \delta\theta\rangle = -\langle q^{n+1}, J^n\delta\theta\rangle = -\langle q^{n+1}, \delta q\rangle$$

Using $\langle q^{n+1}, q^{n+1}-q^n\rangle = \frac{1}{2}(\|q^{n+1}\|^2 - \|q^n\|^2 + \|\delta q\|^2)$:

$$\frac{\|\delta\theta\|^2}{\Delta t} + \frac{\lambda}{2}\|\delta\theta\|^2 + \frac{1}{2}\|\delta q\|^2 + \frac{1}{2}(\|q^{n+1}\|^2 - \|q^n\|^2) + \frac{\lambda}{2}(\|\theta^{n+1}\|^2 - \|\theta^n\|^2) = 0$$

All terms on the left except the differences are non-negative, giving $\mathcal{E}^{n+1} \leq \mathcal{E}^n$. $\square$

**Practical notes:**
- Requires Jacobian $J^n \in \mathbb{R}^{m\times d}$: computed via one forward + backward pass
- Requires solving $m \times m$ system: $O(m^3)$ where $m$ is output dimension
- For classification with batch $B$ and classes $K$: $m = BK$, cost $O(B^3K^3)$ — prohibitive for large batches
- Memory: storing $J^n$ costs $O(md)$ — also potentially prohibitive
- The auxiliary variable $q^n$ can drift from $f(\theta^n)-y$ over iterations (same Vanilla drift issue)

---

## 3.3 Restart IEQ

Reset $q^n \leftarrow \hat{q}^n := f(\theta^n) - y$ at each step.

### Algorithm 8: Restart IEQ

> 1. Compute $\hat{q}^n = f(\theta^n) - y$, $J^n = \frac{\partial f}{\partial\theta}(\theta^n)$, $G = J^n(J^n)^\top$
> 2. Solve $(I_m + \alpha G)\,q^{n+1} = \hat{q}^n - \alpha\lambda J^n\theta^n$
> 3. $\theta^{n+1} = \frac{1}{1+\lambda\Delta t}(\theta^n - \Delta t(J^n)^\top q^{n+1})$

**Is restart trivially exact?** In IEQ, the auxiliary variable $q = f(\theta) - y$ is *exactly* the physical quantity (the residual). In SAV/ESAV, the auxiliary variable was a *transformation* of $I$. Here, setting $\hat{q}^n = f(\theta^n) - y$ is simply re-evaluating the residual, which we already do in the forward pass to compute $J^n$. Therefore:

> **Restart is essentially free in IEQ** — $\hat{q}^n$ is a byproduct of the forward pass needed to compute $J^n$.

This means there is almost no reason to use Vanilla IEQ over Restart IEQ. The Vanilla variant's only advantage is avoiding one function evaluation $f(\theta^n)$, but since computing $J^n$ requires a forward pass anyway, this evaluation comes for free.

**Energy dissipated:** $\frac{1}{2}\|q^{n+1}\|^2 + \frac{\lambda}{2}\|\theta^{n+1}\|^2 \leq \frac{1}{2}\|f(\theta^n)-y\|^2 + \frac{\lambda}{2}\|\theta^n\|^2 = I(\theta^n) + \frac{\lambda}{2}\|\theta^n\|^2$.

---

## 3.4 Relax IEQ

Define:
- $\tilde{q}^{n+1}$: Vanilla IEQ output
- $\hat{q}^{n+1} := f(\theta^{n+1}) - y$: true residual at new iterate
- $q^{n+1} = \xi\tilde{q}^{n+1} + (1-\xi)\hat{q}^{n+1}$

**Optimal ξ.** Minimize $\|q^{n+1} - \hat{q}^{n+1}\|^2 = \xi^2\|\tilde{q}^{n+1} - \hat{q}^{n+1}\|^2$ subject to:

$$\frac{1}{2}\|q^{n+1}\|^2 + \frac{\lambda}{2}\|\theta^{n+1}\|^2 \leq \frac{1}{2}\|q^n\|^2 + \frac{\lambda}{2}\|\theta^n\|^2$$

Let $\delta = \tilde{q}^{n+1} - \hat{q}^{n+1} \in \mathbb{R}^m$, so $q^{n+1} = \hat{q}^{n+1} + \xi\delta$. The constraint becomes:

$$\frac{1}{2}\|\hat{q}^{n+1} + \xi\delta\|^2 \leq S^n := \frac{1}{2}\|q^n\|^2 + \frac{\lambda}{2}(\|\theta^n\|^2 - \|\theta^{n+1}\|^2)$$

Expand:

$$\frac{1}{2}\|\hat{q}^{n+1}\|^2 + \xi\langle\hat{q}^{n+1},\delta\rangle + \frac{\xi^2}{2}\|\delta\|^2 \leq S^n$$

Define $R = S^n - \frac{1}{2}\|\hat{q}^{n+1}\|^2$. The constraint is:

$$\frac{\xi^2}{2}\|\delta\|^2 + \xi\langle\hat{q}^{n+1},\delta\rangle \leq R$$

If $R \geq 0$ (restart is already feasible): $\xi^* = 0$.

Otherwise, solve the quadratic $\frac{\|\delta\|^2}{2}\xi^2 + \langle\hat{q}^{n+1},\delta\rangle\xi - R = 0$ for the smallest positive root:

$$\xi^* = \frac{-\langle\hat{q}^{n+1},\delta\rangle + \sqrt{\langle\hat{q}^{n+1},\delta\rangle^2 + \|\delta\|^2 R}}{\|\delta\|^2}$$

(taking the positive root), clamped to $[0,1]$.

### Algorithm 9: Relax IEQ

> 1. Run Vanilla IEQ to get $\tilde{q}^{n+1}$, $\theta^{n+1}$
> 2. Compute $\hat{q}^{n+1} = f(\theta^{n+1}) - y$ (extra forward pass!)
> 3. Compute $S^n = \frac{1}{2}\|q^n\|^2 + \frac{\lambda}{2}(\|\theta^n\|^2 - \|\theta^{n+1}\|^2)$
> 4. Compute $R = S^n - \frac{1}{2}\|\hat{q}^{n+1}\|^2$
> 5. If $R \geq 0$: $\xi^* = 0$
> 6. Else: $\delta = \tilde{q}^{n+1} - \hat{q}^{n+1}$, solve quadratic for $\xi^*$, clamp to $[0,1]$
> 7. $q^{n+1} = \xi^*\tilde{q}^{n+1} + (1-\xi^*)\hat{q}^{n+1}$

**Computational feasibility:** The extra forward pass for $\hat{q}^{n+1} = f(\theta^{n+1}) - y$ is the main additional cost. For IEQ, where we already need the Jacobian (which requires a forward pass), the total becomes 2 forward passes + 1 backward pass per step, plus the $O(m^3)$ linear solve. The inner products $\langle\hat{q}^{n+1},\delta\rangle$ and $\|\delta\|^2$ are $O(m)$.

---

# Comparison Table

| Scheme | Aux. Variable | Modified Energy $\mathcal{E}^n$ | Update Complexity | Key Trade-off |
|--------|--------------|-------------------------------|------------------|---------------|
| **Vanilla SAV** | $r = \sqrt{I+C}$ (scalar) | $r^2 + \frac{\lambda}{2}\|\theta\|^2$ | $O(d)$ | Stable but $r^n$ drifts from truth |
| **Restart SAV** | $\hat{r}^n = \sqrt{I(\theta^n)+C}$ | $I(\theta^n)+C+\frac{\lambda}{2}\|\theta^n\|^2$ | $O(d)$ + 1 eval of $I$ | No drift; no guarantee on $I(\theta^n)$ decrease |
| **Relax SAV** | $\xi\tilde{r}+(1-\xi)\hat{r}$ | $(r^{n+1})^2+\frac{\lambda}{2}\|\theta^{n+1}\|^2$ | $O(d)$ + 1 eval of $I$ | Best of both; requires $I(\theta^{n+1})$ |
| **Vanilla ESAV** | $r = e^{I/2}$ (scalar) | $r^2 + \frac{\lambda}{2}\|\theta\|^2$ | $O(d)$ | No $C$ needed; overflow risk for large $I$ |
| **Restart ESAV** | $\hat{r}^n = e^{I(\theta^n)/2}$ | $e^{I(\theta^n)}+\frac{\lambda}{2}\|\theta^n\|^2$ | $O(d)$ + 1 eval of $I$ | Same as Restart SAV + exponential energy |
| **Relax ESAV** | $\xi\tilde{r}+(1-\xi)\hat{r}$ | $(r^{n+1})^2+\frac{\lambda}{2}\|\theta^{n+1}\|^2$ | $O(d)$ + 1 eval of $I$ | Same as Relax SAV + no $C$ |
| **Vanilla IEQ** | $q = f(\theta)-y$ (vector) | $\frac{1}{2}\|q\|^2+\frac{\lambda}{2}\|\theta\|^2$ | $O(m^3) + O(md)$ | Exact energy quadratization; Jacobian cost |
| **Restart IEQ** | $\hat{q}^n = f(\theta^n)-y$ | $I(\theta^n)+\frac{\lambda}{2}\|\theta^n\|^2$ | $O(m^3) + O(md)$ | Free restart (Jacobian needs forward pass anyway) |
| **Relax IEQ** | $\xi\tilde{q}+(1-\xi)\hat{q}$ | $\frac{1}{2}\|q^{n+1}\|^2+\frac{\lambda}{2}\|\theta^{n+1}\|^2$ | $O(m^3)+O(md)$ + extra fwd | Optimal tracking; extra forward pass |

### Key Takeaways

1. **SAV vs ESAV:** Structurally identical after the change of auxiliary variable. ESAV avoids the artificial constant $C$ and is well-defined for any $I \in \mathbb{R}$, but risks numerical overflow when $I$ is large. In practice, store $\ln r$ and compute exponentials carefully.

2. **SAV/ESAV vs IEQ:** SAV/ESAV use scalar auxiliary variables with $O(d)$ cost — ideal for large-scale deep learning. IEQ uses a vector auxiliary variable with $O(m^3)$ cost from the Gram matrix solve — more expensive but preserves richer structure of the loss landscape.

3. **Vanilla vs Restart vs Relax:** Vanilla guarantees energy decrease of the *modified* energy but allows drift. Restart eliminates drift but may not decrease the original loss monotonically. Relax is the theoretically optimal compromise — it stays as close to the true energy as possible while maintaining unconditional stability.

4. **For neural network training:** SAV/ESAV Relax variants are most practical: $O(d)$ per step, unconditionally stable, and approximately track the true loss. IEQ variants are mainly of theoretical interest unless $m \ll d$ (e.g., regression with few outputs).

### Assumptions Made

- **SAV/ESAV:** $I(\theta) \geq 0$ (for SAV, $I + C > 0$; for ESAV, no constraint needed). $I$ and $\nabla I$ are bounded and Lipschitz on bounded sets.
- **IEQ:** $I(\theta) = \frac{1}{2}\|f(\theta)-y\|^2$ (squared loss). $f$ is differentiable with bounded Jacobian. The Gram matrix $J(J)^\top + \frac{1}{\alpha}I_m$ is positive definite (always true since $\alpha > 0$).
- **All schemes:** The stabilization parameter $\lambda \geq 0$ is chosen such that the splitting $E = \frac{\lambda}{2}\|\theta\|^2 + I(\theta)$ makes $I$ "easier" (smaller Lipschitz constant of $\nabla I$). In practice, $\lambda$ plays the role of weight decay.
