"""IEQ (Invariant Energy Quadratization) family: Vanilla, Restart, Relax.

Implements Algorithms 7-9 from MATH_REFERENCE.md.
Training uses MSE loss; relative error is reported for comparison.

Key difference from SAV/ESAV: q is a VECTOR (batch_size,) not a scalar.
  q(t) = f(theta(t)) - y  (the residual)
  I(theta) = 1/2 ||q||^2   (quadratic in q — defining property of IEQ)

The IEQ update requires:
  - Jacobian J = df/dtheta in R^{m x d}  (m = batch_size, d = num_params)
  - Gram matrix G = J @ J^T in R^{m x m}
  - Solving the m x m linear system (I_m + alpha * G) q^{n+1} = rhs
  - Cost: O(m^3) for the solve + O(m*d) for the Jacobian

Memory constraint: batch_size <= 64 to avoid OOM from Jacobian storage on 4GB GPU.
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


def _ieq_config_str(method, model, X_train, epochs, batch_size, dt, lambda_,
                    device):
    """Build config string for Slack."""
    return (
        f"Starting {method} on current example\n"
        f"Config: D={X_train.shape[1]}, m={model.m}, epochs={epochs}, "
        f"batch_size={batch_size}, dt={dt}, lambda={lambda_}, "
        f"seed=42\nDevice: {device}"
    )


def _compute_jacobian(model, theta, X_batch):
    """Compute the Jacobian J = df/dtheta for the batch using per-sample gradients.

    Args:
        model: OneHiddenLayerNet
        theta: flat parameter tensor (d,)
        X_batch: input batch (m, D)

    Returns:
        J: Jacobian matrix (m, d) where J[i] = grad_theta f_i(theta)
        f_batch: model predictions (m,)
    """
    model.unflatten_params(theta)

    # Forward pass with gradient tracking
    # We need per-sample gradients, so we compute them via backward on each output
    # More efficient: use torch.func.jacrev with vmap if available,
    # but for clarity and compatibility, use the loop approach with
    # retain_graph for all but the last sample.
    m = X_batch.shape[0]
    d = theta.numel()

    # Forward pass
    f_batch = model(X_batch)  # (m,)

    # Compute Jacobian row by row
    J = torch.zeros(m, d, device=theta.device, dtype=theta.dtype)

    for i in range(m):
        model.zero_grad()
        f_batch[i].backward(retain_graph=(i < m - 1))
        J[i] = torch.cat([p.grad.reshape(-1) for p in model.parameters()])

    return J, f_batch.detach()


def _compute_jacobian_vmap(model, theta, X_batch):
    """Compute Jacobian using torch.func.vmap + grad for efficiency.

    This is faster than the loop approach when torch.func is available.

    Args:
        model: OneHiddenLayerNet
        theta: flat parameter tensor (d,)
        X_batch: input batch (m, D)

    Returns:
        J: Jacobian matrix (m, d)
        f_batch: model predictions (m,)
    """
    model.unflatten_params(theta)

    # Define a function that takes flat params and a single input, returns scalar
    def func_single(flat_params, x_single):
        """Forward pass for a single sample given flat parameters."""
        # Unflatten params manually
        offset = 0
        params = []
        for p in model.parameters():
            numel = p.numel()
            params.append(flat_params[offset:offset + numel].reshape(p.shape))
            offset += numel

        # Manual forward: x_aug @ W -> relu -> @ a / m
        W, a = params[0], params[1]
        ones = torch.ones(1, device=x_single.device, dtype=x_single.dtype)
        x_aug = torch.cat([x_single, ones], dim=0)  # (D+1,)
        h = torch.relu(x_aug @ W)  # (m_neurons,)
        out = (h @ a) / model.m  # scalar (1,) -> squeeze
        return out.squeeze()

    try:
        from torch.func import grad, vmap

        # grad of func_single w.r.t. flat_params, for each sample
        grad_fn = grad(func_single, argnums=0)  # gradient w.r.t. first arg
        # vmap over the batch dimension (second argument)
        J = vmap(grad_fn, in_dims=(None, 0))(theta, X_batch)  # (m, d)

        # Also compute f_batch
        with torch.no_grad():
            f_batch = model(X_batch)  # (m,)

        return J, f_batch

    except (ImportError, Exception):
        # Fallback to loop-based approach
        return _compute_jacobian(model, theta, X_batch)


def _solve_ieq_system(G, q_rhs, alpha):
    """Solve (I_m + alpha * G) @ q_new = q_rhs.

    Args:
        G: Gram matrix J @ J^T, shape (m, m)
        q_rhs: right-hand side vector, shape (m,)
        alpha: dt / (1 + lambda * dt)

    Returns:
        q_new: solution vector, shape (m,)
    """
    m = G.shape[0]
    A = torch.eye(m, device=G.device, dtype=G.dtype) + alpha * G
    q_new = torch.linalg.solve(A, q_rhs)
    return q_new


# ─────────────────────────────────────────────────────────────────────
# Algorithm 7: Vanilla IEQ
# ─────────────────────────────────────────────────────────────────────

def train_vanilla_ieq(model, X_train, y_train, X_test, y_test,
                      lambda_=0.0, dt=0.1, batch_size=64,
                      epochs=5000, device=None, slack_interval=1000,
                      wandb_run=None):
    """Vanilla IEQ (Algorithm 7 from MATH_REFERENCE.md).

    Semi-implicit discretization with vector auxiliary variable q = f(theta) - y.

    1. Compute J^n, q^n (current auxiliary), G = J^n @ (J^n)^T
    2. Solve (I_m + alpha*G) q^{n+1} = q^n - alpha*lambda*J^n@theta^n
    3. theta^{n+1} = 1/(1+lambda*dt) * (theta^n - dt * (J^n)^T @ q^{n+1})
    4. Do NOT reset q — keep as computed

    Energy: E^n = 1/2 ||q^n||^2 + lambda/2 ||theta^n||^2
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_loss_fn = MSELoss()
    alpha_coeff = 1.0 / (1.0 + lambda_ * dt)  # for theta update
    alpha = dt / (1.0 + lambda_ * dt)          # for linear system

    send_slack(_ieq_config_str("Vanilla IEQ", model, X_train, epochs,
                                batch_size, dt, lambda_, device))

    # Initialize theta and q
    theta = model.flatten_params().clone()
    model.unflatten_params(theta)
    with torch.no_grad():
        f_init = model(X_train)
    q = (f_init - y_train).clone()  # full-dataset q (N,)

    train_losses = []
    test_losses = []
    energy_values = []
    r_values = []  # store ||q|| as the "r-value" analog

    gen = torch.Generator().manual_seed(42)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        batches = make_batches(len(X_train), batch_size, generator=gen)

        for idx in batches:
            X_b, y_b = X_train[idx], y_train[idx]
            q_b = q[idx]  # current batch slice of q

            # Step 1: Compute Jacobian J^n and predictions
            J, f_b = _compute_jacobian_vmap(model, theta, X_b)
            # J: (m, d), f_b: (m,)

            # Gram matrix G = J @ J^T: (m, m)
            G = J @ J.t()

            # Step 2: Solve (I_m + alpha * G) q^{n+1} = q^n - alpha*lambda*J@theta
            rhs = q_b.clone()
            if lambda_ > 0:
                rhs = rhs - alpha * lambda_ * (J @ theta)
            q_new = _solve_ieq_system(G, rhs, alpha)

            # Step 3: theta update
            # theta^{n+1} = 1/(1+lambda*dt) * (theta^n - dt * J^T @ q^{n+1})
            theta = alpha_coeff * (theta - dt * (J.t() @ q_new))

            # Step 4: Update q for this batch (do NOT reset)
            q[idx] = q_new

            # NaN check
            if torch.isnan(theta).any() or torch.isnan(q_new).any():
                print(f"[Vanilla IEQ] NaN at epoch {epoch}!")
                break

        # Epoch-end: evaluate full-dataset metrics
        model.unflatten_params(theta)
        tr_rel = evaluate_rel_error(model, X_train, y_train)
        te_rel = evaluate_rel_error(model, X_test, y_test)

        q_norm = torch.norm(q).item()
        theta_norm_sq = torch.dot(theta, theta).item()
        energy = 0.5 * q_norm ** 2 + lambda_ / 2.0 * theta_norm_sq

        train_losses.append(tr_rel)
        test_losses.append(te_rel)
        energy_values.append(energy)
        r_values.append(q_norm)

        if wandb_run is not None:
            tr_mse = evaluate_loss(model, X_train, y_train, train_loss_fn)
            wandb_run.log({"epoch": epoch,
                           "train_rel_error": tr_rel, "test_rel_error": te_rel,
                           "train_mse": tr_mse, "energy": energy,
                           "q_norm": q_norm})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Vanilla IEQ | epoch {epoch}/{epochs} | "
                f"train_rel={tr_rel:.4e} | test_rel={te_rel:.4e} | "
                f"energy={energy:.4e} | ||q||={q_norm:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Vanilla IEQ done | final_test_rel={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Vanilla IEQ",
        "params": {"lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }


# ─────────────────────────────────────────────────────────────────────
# Algorithm 8: Restart IEQ
# ─────────────────────────────────────────────────────────────────────

def train_restart_ieq(model, X_train, y_train, X_test, y_test,
                      lambda_=0.0, dt=0.1, batch_size=64,
                      epochs=5000, device=None, slack_interval=1000,
                      wandb_run=None):
    """Restart IEQ (Algorithm 8 from MATH_REFERENCE.md).

    Same as Vanilla IEQ but reset q^n = f(theta^n) - y at each step.
    This is essentially free since the Jacobian computation requires
    a forward pass anyway.

    1. Compute q_hat^n = f(theta^n) - y, J^n, G = J^n @ (J^n)^T
    2. Solve (I_m + alpha*G) q^{n+1} = q_hat^n - alpha*lambda*J^n@theta^n
    3. theta^{n+1} = 1/(1+lambda*dt) * (theta^n - dt * (J^n)^T @ q^{n+1})
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_loss_fn = MSELoss()
    alpha_coeff = 1.0 / (1.0 + lambda_ * dt)
    alpha = dt / (1.0 + lambda_ * dt)

    send_slack(_ieq_config_str("Restart IEQ", model, X_train, epochs,
                                batch_size, dt, lambda_, device))

    theta = model.flatten_params().clone()

    # q is maintained for energy tracking between batches,
    # but is always reset from the forward pass within each batch step
    model.unflatten_params(theta)
    with torch.no_grad():
        f_init = model(X_train)
    q = (f_init - y_train).clone()

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

            # Step 1: Compute Jacobian and RESTART q from forward pass
            J, f_b = _compute_jacobian_vmap(model, theta, X_b)
            q_hat = f_b - y_b  # reset q to true residual

            # Gram matrix
            G = J @ J.t()

            # Step 2: Solve (I_m + alpha * G) q^{n+1} = q_hat - alpha*lambda*J@theta
            rhs = q_hat.clone()
            if lambda_ > 0:
                rhs = rhs - alpha * lambda_ * (J @ theta)
            q_new = _solve_ieq_system(G, rhs, alpha)

            # Step 3: theta update
            theta = alpha_coeff * (theta - dt * (J.t() @ q_new))

            # Update q for tracking
            q[idx] = q_new

            if torch.isnan(theta).any() or torch.isnan(q_new).any():
                print(f"[Restart IEQ] NaN at epoch {epoch}!")
                break

        # Epoch-end evaluation
        model.unflatten_params(theta)
        tr_rel = evaluate_rel_error(model, X_train, y_train)
        te_rel = evaluate_rel_error(model, X_test, y_test)

        q_norm = torch.norm(q).item()
        theta_norm_sq = torch.dot(theta, theta).item()
        energy = 0.5 * q_norm ** 2 + lambda_ / 2.0 * theta_norm_sq

        train_losses.append(tr_rel)
        test_losses.append(te_rel)
        energy_values.append(energy)
        r_values.append(q_norm)

        if wandb_run is not None:
            tr_mse = evaluate_loss(model, X_train, y_train, train_loss_fn)
            wandb_run.log({"epoch": epoch,
                           "train_rel_error": tr_rel, "test_rel_error": te_rel,
                           "train_mse": tr_mse, "energy": energy,
                           "q_norm": q_norm})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Restart IEQ | epoch {epoch}/{epochs} | "
                f"train_rel={tr_rel:.4e} | test_rel={te_rel:.4e} | "
                f"energy={energy:.4e} | ||q||={q_norm:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Restart IEQ done | final_test_rel={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Restart IEQ",
        "params": {"lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }


# ─────────────────────────────────────────────────────────────────────
# Algorithm 9: Relax IEQ
# ─────────────────────────────────────────────────────────────────────

def train_relax_ieq(model, X_train, y_train, X_test, y_test,
                    lambda_=0.0, dt=0.1, batch_size=64,
                    epochs=5000, device=None, slack_interval=1000,
                    wandb_run=None):
    """Relax IEQ (Algorithm 9 from MATH_REFERENCE.md).

    1. Run Vanilla IEQ step -> q_tilde^{n+1}, theta^{n+1}
    2. Compute q_hat^{n+1} = f(theta^{n+1}) - y (extra forward pass)
    3. S^n = 1/2 ||q^n||^2 + lambda/2 (||theta^n||^2 - ||theta^{n+1}||^2)
    4. R = S^n - 1/2 ||q_hat^{n+1}||^2
    5. If R >= 0: xi* = 0 (pure restart is stable)
    6. Else: solve quadratic for xi*, clamp to [0,1]
       xi* = (-<q_hat, delta> + sqrt(<q_hat,delta>^2 + ||delta||^2 * R)) / ||delta||^2
       where delta = q_tilde - q_hat
    7. q^{n+1} = xi * q_tilde + (1 - xi) * q_hat
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_loss_fn = MSELoss()
    alpha_coeff = 1.0 / (1.0 + lambda_ * dt)
    alpha = dt / (1.0 + lambda_ * dt)

    send_slack(_ieq_config_str("Relax IEQ", model, X_train, epochs,
                                batch_size, dt, lambda_, device))

    theta = model.flatten_params().clone()

    model.unflatten_params(theta)
    with torch.no_grad():
        f_init = model(X_train)
    q = (f_init - y_train).clone()

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
            theta_old_norm_sq = torch.dot(theta_old, theta_old).item()

            # === Step 1: Vanilla IEQ step (using FRESH q, not stale q[idx]) ===
            # In mini-batch mode, q[idx] is stale (from last epoch's update of
            # this batch). Use fresh q_hat_old = f(theta) - y to avoid q-drift.
            J, f_b = _compute_jacobian_vmap(model, theta, X_b)
            q_hat_old = f_b - y_b  # fresh residual at current theta
            q_old_norm_sq = torch.dot(q_hat_old, q_hat_old).item()
            G = J @ J.t()

            rhs = q_hat_old.clone()
            if lambda_ > 0:
                rhs = rhs - alpha * lambda_ * (J @ theta)
            q_tilde = _solve_ieq_system(G, rhs, alpha)

            # theta update
            theta_new = alpha_coeff * (theta - dt * (J.t() @ q_tilde))

            # === Step 2: Compute q_hat = f(theta_new) - y (extra forward pass) ===
            model.unflatten_params(theta_new)
            with torch.no_grad():
                f_new = model(X_b)
            q_hat = f_new - y_b

            # === Steps 3-7: Relax correction ===
            theta_new_norm_sq = torch.dot(theta_new, theta_new).item()

            # S^n = 1/2 ||q^n||^2 + lambda/2 (||theta^n||^2 - ||theta^{n+1}||^2)
            S_n = 0.5 * q_old_norm_sq + \
                  lambda_ / 2.0 * (theta_old_norm_sq - theta_new_norm_sq)

            # R = S^n - 1/2 ||q_hat||^2
            q_hat_norm_sq = torch.dot(q_hat, q_hat).item()
            R = S_n - 0.5 * q_hat_norm_sq

            if R >= 0:
                # Restart is already stable: xi* = 0
                xi_star = 0.0
            else:
                # Need xi > 0 for stability
                delta = q_tilde - q_hat  # (m,)
                delta_norm_sq = torch.dot(delta, delta).item()
                q_hat_dot_delta = torch.dot(q_hat, delta).item()

                if delta_norm_sq < 1e-30:
                    # q_tilde ~= q_hat, no correction needed
                    xi_star = 0.0
                else:
                    # Solve: ||delta||^2/2 * xi^2 + <q_hat,delta> * xi - R = 0
                    # Using quadratic formula for the positive root:
                    # xi* = (-<q_hat,delta> + sqrt(<q_hat,delta>^2 + ||delta||^2 * R)) / ||delta||^2
                    discriminant = q_hat_dot_delta ** 2 + delta_norm_sq * R
                    if discriminant < 0:
                        # No real root; fallback to vanilla (xi=1)
                        xi_star = 1.0
                    else:
                        xi_star = (-q_hat_dot_delta + math.sqrt(discriminant)) / delta_norm_sq
                        xi_star = max(0.0, min(1.0, xi_star))

            # Step 7: Blend q
            if xi_star == 0.0:
                q_new = q_hat
            elif xi_star == 1.0:
                q_new = q_tilde
            else:
                q_new = xi_star * q_tilde + (1.0 - xi_star) * q_hat

            # Update state
            theta = theta_new
            q[idx] = q_new

            if torch.isnan(theta).any() or torch.isnan(q_new).any():
                print(f"[Relax IEQ] NaN at epoch {epoch}!")
                break

        # Epoch-end evaluation
        model.unflatten_params(theta)
        tr_rel = evaluate_rel_error(model, X_train, y_train)
        te_rel = evaluate_rel_error(model, X_test, y_test)

        q_norm = torch.norm(q).item()
        theta_norm_sq = torch.dot(theta, theta).item()
        energy = 0.5 * q_norm ** 2 + lambda_ / 2.0 * theta_norm_sq

        train_losses.append(tr_rel)
        test_losses.append(te_rel)
        energy_values.append(energy)
        r_values.append(q_norm)

        if wandb_run is not None:
            tr_mse = evaluate_loss(model, X_train, y_train, train_loss_fn)
            wandb_run.log({"epoch": epoch,
                           "train_rel_error": tr_rel, "test_rel_error": te_rel,
                           "train_mse": tr_mse, "energy": energy,
                           "q_norm": q_norm})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Relax IEQ | epoch {epoch}/{epochs} | "
                f"train_rel={tr_rel:.4e} | test_rel={te_rel:.4e} | "
                f"energy={energy:.4e} | ||q||={q_norm:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Relax IEQ done | final_test_rel={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Relax IEQ",
        "params": {"lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }
