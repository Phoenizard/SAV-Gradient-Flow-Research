"""ESAV (Exponential SAV) family: Vanilla, Restart, Relax.

Implements Algorithms 4-6 from MATH_REFERENCE.md.
Training uses MSE loss; relative error is reported for comparison.

Key difference from SAV: r = exp(I/2) instead of sqrt(I + C).
Normalization: nu = gradI / exp(I/2) = gradI * exp(-I/2).
No additive constant C needed; r > 0 always.

Numerical strategy: store log_r = ln(r) instead of r directly.
All updates computed via ratios to avoid exp overflow/underflow.
When I is large, the algorithm degenerates gracefully to gradient descent.
"""

import math
import time
import torch

from src.models.network import OneHiddenLayerNet
from src.utils.trainer import (
    get_device, make_batches, evaluate_loss, evaluate_rel_error,
    MSELoss, RelativeErrorLoss
)
from src.utils.slack import send_slack

# Threshold for safe direct exp computation (float64 safe up to ~709)
_LOG_SAFE = 300.0

# Maximum allowed |log_r - log_R| before forcing a restart.
# dt * exp(delta_log) is the effective step multiplier; we need this ≈ O(dt).
# delta_log = 5 → ratio ≈ 148, which is already aggressive.
_DRIFT_MAX = 5.0

# When I_batch exceeds this, exp(-I) underflows to 0 and the vanilla
# r-tracking mechanism is non-functional. Force restart in this regime.
# With I < 30, scale2 = exp(-30) ≈ 9.4e-14 which is still nonzero in float64.
_I_RESTART_THRESH = 30.0


def _esav_config_str(method, model, X_train, epochs, batch_size, dt, lambda_,
                     device):
    """Build config string for Slack."""
    return (
        f"Starting {method} on current example\n"
        f"Config: D={X_train.shape[1]}, m={model.m}, epochs={epochs}, "
        f"batch_size={batch_size}, dt={dt}, lambda={lambda_}, "
        f"seed=42\nDevice: {device}"
    )


def _compute_grad_and_loss(model, theta, X_batch, y_batch, train_loss_fn):
    """Forward + backward to get I_batch and grad_I.

    Returns (I_batch, grad_I) where I_batch is MSE loss scalar,
    grad_I is the flat gradient tensor.
    """
    model.unflatten_params(theta)
    model.zero_grad()
    pred = model(X_batch)
    loss = train_loss_fn(pred, y_batch)
    loss.backward()

    I_batch = loss.item()
    grad_I = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    return I_batch, grad_I


def _evaluate_batch_loss(model, theta, X_batch, y_batch, train_loss_fn):
    """Forward pass only (no grad) to get batch MSE loss."""
    model.unflatten_params(theta)
    with torch.no_grad():
        pred = model(X_batch)
        return train_loss_fn(pred, y_batch).item()


def _safe_exp(x):
    """Compute exp(x) with clamping to avoid overflow."""
    return math.exp(min(max(x, -_LOG_SAFE), _LOG_SAFE))


def _esav_vanilla_step(theta, log_r, I_batch, grad_I, alpha, dt):
    """One Vanilla ESAV step using log-space arithmetic.

    The ESAV update (Algorithm 4) is structurally identical to SAV:
      nu^n = gradI / exp(I/2)
      a = <nu, theta>, b = ||nu||^2
      r_new = (r + (alpha-1)/2 * a) / (1 + alpha*dt/2 * b)
      theta_new = alpha * (theta - dt * r_new * nu)

    In ratio form (to avoid overflow):
      log_R = I_batch / 2  (log of true normalizer)
      scale2 = exp(-I_batch)  (= exp(-2*log_R))
      a_scaled = scale2 * <gradI, theta>
      b_scaled = scale2 * ||gradI||^2
      exp_delta = exp(log_r - log_R)  (ratio of tracked r to true R)
      ratio = (exp_delta + (alpha-1)/2 * a_scaled) / (1 + alpha*dt/2 * b_scaled)
      log_r_new = log_R + ln(ratio)
      theta_new = alpha * (theta - dt * ratio * gradI)

    Drift safeguard: when I_batch is large (>_I_RESTART_THRESH), the exponential
    normalization factor exp(-I/2) underflows, making the r-tracking mechanism
    non-functional (scale2 ≈ 0 → a_scaled ≈ 0, b_scaled ≈ 0 → ratio = exp(delta_log)
    which can be catastrophically large). In this regime, we force a restart.

    Additionally, if |log_r - log_R| exceeds _DRIFT_MAX, we force a restart
    even when I is moderate, to prevent ratio overflow.

    Returns (theta_new, log_r_new, drifted) where drifted is True if a restart
    was forced.
    """
    log_R = I_batch / 2.0
    delta_log = log_r - log_R
    drifted = False

    # Drift safeguard 1: when I is too large, vanilla tracking doesn't work
    # because scale2 = exp(-I) ≈ 0 and the r-update is trivial.
    # Force restart to prevent ratio = exp(delta_log) from exploding.
    if I_batch > _I_RESTART_THRESH or abs(delta_log) > _DRIFT_MAX:
        delta_log = 0.0
        log_r = log_R
        drifted = True

    # Compute scale2 = exp(-I_batch) = exp(-2*log_R)
    # For large I_batch, this underflows to 0 — that's fine
    scale2 = _safe_exp(-I_batch)

    a_raw = torch.dot(grad_I, theta).item()
    b_raw = torch.dot(grad_I, grad_I).item()

    a_scaled = scale2 * a_raw
    b_scaled = scale2 * b_raw

    # exp_delta = r / R = exp(log_r - log_R)
    exp_delta = _safe_exp(delta_log)

    # r-update in ratio form
    numer = exp_delta + (alpha - 1.0) / 2.0 * a_scaled
    denom = 1.0 + alpha * dt / 2.0 * b_scaled
    ratio = numer / denom

    if ratio <= 0 or math.isnan(ratio):
        ratio = max(ratio, 1e-15) if not math.isnan(ratio) else 1.0

    log_r_new = log_R + math.log(ratio)

    # theta update: theta_new = alpha * (theta - dt * ratio * gradI)
    theta_new = alpha * (theta - dt * ratio * grad_I)

    return theta_new, log_r_new, drifted


def train_vanilla_esav(model, X_train, y_train, X_test, y_test,
                       lambda_=0.0, dt=0.1, batch_size=256,
                       epochs=10000, device=None, slack_interval=1000,
                       wandb_run=None):
    """Vanilla ESAV (Algorithm 4 from MATH_REFERENCE.md).

    Structurally identical to Vanilla SAV with:
      - r = exp(I/2) instead of sqrt(I+C)
      - nu = gradI / exp(I/2) instead of mu = gradI / sqrt(I+C)
      - No C constant needed
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_loss_fn = MSELoss()
    alpha = 1.0 / (1.0 + lambda_ * dt)

    send_slack(_esav_config_str("Vanilla ESAV", model, X_train, epochs,
                                batch_size, dt, lambda_, device))

    # Initialize theta and log_r
    theta = model.flatten_params().clone()
    full_loss = evaluate_loss(model, X_train, y_train, train_loss_fn)
    log_r = full_loss / 2.0  # ln(r) = I/2

    train_losses = []
    test_losses = []
    energy_values = []
    r_values = []

    gen = torch.Generator().manual_seed(42)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        batches = make_batches(len(X_train), batch_size, generator=gen)

        for idx in batches:
            X_b, y_b = X_train[idx], y_train[idx]

            I_batch, grad_I = _compute_grad_and_loss(
                model, theta, X_b, y_b, train_loss_fn
            )

            theta, log_r, _ = _esav_vanilla_step(
                theta, log_r, I_batch, grad_I, alpha, dt
            )

            if math.isnan(log_r) or torch.isnan(theta).any():
                print(f"[Vanilla ESAV] NaN at epoch {epoch}!")
                break

        # Epoch-end: evaluate full-dataset metrics
        model.unflatten_params(theta)
        tr_rel = evaluate_rel_error(model, X_train, y_train)
        te_rel = evaluate_rel_error(model, X_test, y_test)

        # Energy = r^2 + lambda/2 * ||theta||^2 = exp(2*log_r) + lambda/2*||theta||^2
        theta_norm_sq = torch.dot(theta, theta).item()
        if log_r < _LOG_SAFE:
            energy = math.exp(2 * log_r) + lambda_ / 2.0 * theta_norm_sq
        else:
            energy = float('inf')

        # Store log_r as r_value for plotting (actual r = exp(log_r))
        if log_r < _LOG_SAFE:
            r_actual = math.exp(log_r)
        else:
            r_actual = float('inf')

        train_losses.append(tr_rel)
        test_losses.append(te_rel)
        energy_values.append(energy)
        r_values.append(r_actual)

        if wandb_run is not None:
            tr_mse = evaluate_loss(model, X_train, y_train, train_loss_fn)
            wandb_run.log({"epoch": epoch,
                           "train_rel_error": tr_rel, "test_rel_error": te_rel,
                           "train_mse": tr_mse, "energy": energy,
                           "r": r_actual, "log_r": log_r})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Vanilla ESAV | epoch {epoch}/{epochs} | "
                f"train_rel={tr_rel:.4e} | test_rel={te_rel:.4e} | "
                f"energy={energy:.4e} | log_r={log_r:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Vanilla ESAV done | final_test_rel={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Vanilla ESAV",
        "params": {"lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }


def train_restart_esav(model, X_train, y_train, X_test, y_test,
                       lambda_=0.0, dt=0.1, batch_size=256,
                       epochs=10000, device=None, slack_interval=1000,
                       wandb_run=None):
    """Restart ESAV (Algorithm 5 from MATH_REFERENCE.md).

    Same as Vanilla ESAV but reset r_hat = exp(I/2) at each step.
    In log-space: log_r is set to I_batch/2 at each step (delta_log=0, exp_delta=1).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_loss_fn = MSELoss()
    alpha = 1.0 / (1.0 + lambda_ * dt)

    send_slack(_esav_config_str("Restart ESAV", model, X_train, epochs,
                                batch_size, dt, lambda_, device))

    theta = model.flatten_params().clone()
    full_loss = evaluate_loss(model, X_train, y_train, train_loss_fn)
    log_r = full_loss / 2.0

    train_losses = []
    test_losses = []
    energy_values = []
    r_values = []

    gen = torch.Generator().manual_seed(42)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        batches = make_batches(len(X_train), batch_size, generator=gen)

        for idx in batches:
            X_b, y_b = X_train[idx], y_train[idx]

            I_batch, grad_I = _compute_grad_and_loss(
                model, theta, X_b, y_b, train_loss_fn
            )

            # Restart: set log_r = I_batch / 2 (i.e., r = exp(I/2))
            log_r_hat = I_batch / 2.0

            # With restart, delta_log = 0, exp_delta = 1
            # ratio = (1 + (alpha-1)/2 * a_scaled) / (1 + alpha*dt/2 * b_scaled)
            scale2 = _safe_exp(-I_batch)
            a_raw = torch.dot(grad_I, theta).item()
            b_raw = torch.dot(grad_I, grad_I).item()
            a_scaled = scale2 * a_raw
            b_scaled = scale2 * b_raw

            numer = 1.0 + (alpha - 1.0) / 2.0 * a_scaled
            denom = 1.0 + alpha * dt / 2.0 * b_scaled
            ratio = numer / denom

            if ratio <= 0 or math.isnan(ratio):
                ratio = max(ratio, 1e-15) if not math.isnan(ratio) else 1.0

            log_r = log_r_hat + math.log(ratio)

            # theta update
            theta = alpha * (theta - dt * ratio * grad_I)

            if math.isnan(log_r) or torch.isnan(theta).any():
                print(f"[Restart ESAV] NaN at epoch {epoch}!")
                break

        # Epoch-end evaluation
        model.unflatten_params(theta)
        tr_rel = evaluate_rel_error(model, X_train, y_train)
        te_rel = evaluate_rel_error(model, X_test, y_test)

        theta_norm_sq = torch.dot(theta, theta).item()
        if log_r < _LOG_SAFE:
            energy = math.exp(2 * log_r) + lambda_ / 2.0 * theta_norm_sq
        else:
            energy = float('inf')

        if log_r < _LOG_SAFE:
            r_actual = math.exp(log_r)
        else:
            r_actual = float('inf')

        train_losses.append(tr_rel)
        test_losses.append(te_rel)
        energy_values.append(energy)
        r_values.append(r_actual)

        if wandb_run is not None:
            tr_mse = evaluate_loss(model, X_train, y_train, train_loss_fn)
            wandb_run.log({"epoch": epoch,
                           "train_rel_error": tr_rel, "test_rel_error": te_rel,
                           "train_mse": tr_mse, "energy": energy,
                           "r": r_actual, "log_r": log_r})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Restart ESAV | epoch {epoch}/{epochs} | "
                f"train_rel={tr_rel:.4e} | test_rel={te_rel:.4e} | "
                f"energy={energy:.4e} | log_r={log_r:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Restart ESAV done | final_test_rel={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Restart ESAV",
        "params": {"lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }


def train_relax_esav(model, X_train, y_train, X_test, y_test,
                     lambda_=0.0, dt=0.1, batch_size=256,
                     epochs=10000, device=None, slack_interval=1000,
                     wandb_run=None):
    """Relax ESAV (Algorithm 6 from MATH_REFERENCE.md).

    1. Do vanilla ESAV step -> log_r_tilde, theta_new
    2. Compute log_r_hat = I(theta_new) / 2
    3. S^n = r^2 + lambda/2*(||theta^n||^2 - ||theta^{n+1}||^2)
    4. If r_hat^2 <= S^n: xi* = 0 (restart is stable)
    5. Else: solve quadratic for xi*, clamp to [0,1]
    6. r_new = xi*r_tilde + (1-xi)*r_hat

    All done in log-space where possible.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_loss_fn = MSELoss()
    alpha = 1.0 / (1.0 + lambda_ * dt)

    send_slack(_esav_config_str("Relax ESAV", model, X_train, epochs,
                                batch_size, dt, lambda_, device))

    theta = model.flatten_params().clone()
    full_loss = evaluate_loss(model, X_train, y_train, train_loss_fn)
    log_r = full_loss / 2.0

    train_losses = []
    test_losses = []
    energy_values = []
    r_values = []

    gen = torch.Generator().manual_seed(42)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        batches = make_batches(len(X_train), batch_size, generator=gen)

        for idx in batches:
            X_b, y_b = X_train[idx], y_train[idx]

            theta_old = theta.clone()
            log_r_old = log_r

            # === Step 1: Vanilla ESAV step ===
            I_batch, grad_I = _compute_grad_and_loss(
                model, theta, X_b, y_b, train_loss_fn
            )

            theta_new, log_r_tilde, _ = _esav_vanilla_step(
                theta, log_r, I_batch, grad_I, alpha, dt
            )

            # === Step 2: Compute r_hat = exp(I(theta_new)/2) ===
            I_new = _evaluate_batch_loss(model, theta_new, X_b, y_b,
                                         train_loss_fn)
            log_r_hat = I_new / 2.0

            # === Step 3-6: Relax correction ===
            # S^n = r^2 + lambda/2*(||theta^n||^2 - ||theta^{n+1}||^2)
            # In log space: S^n = exp(2*log_r) + lambda/2*(...)
            theta_old_norm_sq = torch.dot(theta_old, theta_old).item()
            theta_new_norm_sq = torch.dot(theta_new, theta_new).item()
            lambda_term = lambda_ / 2.0 * (theta_old_norm_sq - theta_new_norm_sq)

            # Check feasibility: r_hat^2 <= S^n
            # i.e., exp(2*log_r_hat) <= exp(2*log_r_old) + lambda_term
            # When values are huge, work in log space for the comparison
            if log_r_old < _LOG_SAFE and log_r_hat < _LOG_SAFE:
                r_hat_sq = math.exp(2 * log_r_hat)
                S_n = math.exp(2 * log_r_old) + lambda_term

                if r_hat_sq <= S_n:
                    xi_star = 0.0
                else:
                    # Need xi > 0 for stability
                    r_tilde = math.exp(log_r_tilde)
                    r_hat = math.exp(log_r_hat)
                    delta = r_tilde - r_hat

                    if abs(delta) < 1e-15:
                        xi_star = 0.0
                    else:
                        # Solve (r_hat + xi*delta)^2 = S_n
                        # xi = (sqrt(S_n) - r_hat) / delta  (taking correct sign)
                        if S_n > 0:
                            xi_star = (math.sqrt(S_n) - r_hat) / delta
                        else:
                            xi_star = 1.0
                        xi_star = max(0.0, min(1.0, xi_star))

                # Blend r
                r_tilde = math.exp(log_r_tilde)
                r_hat = math.exp(log_r_hat)
                r_new = xi_star * r_tilde + (1.0 - xi_star) * r_hat
                if r_new > 0:
                    log_r = math.log(r_new)
                else:
                    log_r = log_r_hat  # fallback to restart
            else:
                # Values too large for direct computation — use log comparison
                # For lambda=0: S^n = exp(2*log_r_old), check 2*log_r_hat <= 2*log_r_old
                if lambda_ == 0.0:
                    if log_r_hat <= log_r_old:
                        xi_star = 0.0
                        log_r = log_r_hat  # pure restart
                    else:
                        xi_star = 1.0
                        log_r = log_r_tilde  # keep vanilla
                else:
                    # Conservative fallback: use restart
                    xi_star = 0.0
                    log_r = log_r_hat

            theta = theta_new

            if math.isnan(log_r) or torch.isnan(theta).any():
                print(f"[Relax ESAV] NaN at epoch {epoch}!")
                break

        # Epoch-end evaluation
        model.unflatten_params(theta)
        tr_rel = evaluate_rel_error(model, X_train, y_train)
        te_rel = evaluate_rel_error(model, X_test, y_test)

        theta_norm_sq = torch.dot(theta, theta).item()
        if log_r < _LOG_SAFE:
            energy = math.exp(2 * log_r) + lambda_ / 2.0 * theta_norm_sq
        else:
            energy = float('inf')

        if log_r < _LOG_SAFE:
            r_actual = math.exp(log_r)
        else:
            r_actual = float('inf')

        train_losses.append(tr_rel)
        test_losses.append(te_rel)
        energy_values.append(energy)
        r_values.append(r_actual)

        if wandb_run is not None:
            tr_mse = evaluate_loss(model, X_train, y_train, train_loss_fn)
            wandb_run.log({"epoch": epoch,
                           "train_rel_error": tr_rel, "test_rel_error": te_rel,
                           "train_mse": tr_mse, "energy": energy,
                           "r": r_actual, "log_r": log_r})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Relax ESAV | epoch {epoch}/{epochs} | "
                f"train_rel={tr_rel:.4e} | test_rel={te_rel:.4e} | "
                f"energy={energy:.4e} | log_r={log_r:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Relax ESAV done | final_test_rel={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Relax ESAV",
        "params": {"lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }
