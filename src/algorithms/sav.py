"""SAV (Scalar Auxiliary Variable) family: Vanilla, Restart, Relax.

Implements Algorithms 1-3 from MATH_REFERENCE.md.
"""

import math
import time
import torch
import torch.nn as nn

from src.models.network import OneHiddenLayerNet
from src.utils.trainer import get_device, make_batches, evaluate_loss
from src.utils.slack import send_slack


def _sav_config_str(method, model, X_train, epochs, batch_size, dt, C, lambda_,
                    device):
    """Build config string for Slack."""
    return (
        f"Starting {method} on current example\n"
        f"Config: D={X_train.shape[1]}, m={model.m}, epochs={epochs}, "
        f"batch_size={batch_size}, dt={dt}, C={C}, lambda={lambda_}, "
        f"seed=42\nDevice: {device}"
    )


def _compute_mu_and_loss(model, theta, X_batch, y_batch, C, loss_fn):
    """Compute mu^n = grad_I / sqrt(I + C) for a mini-batch.

    Sets model params from theta, does forward+backward.
    Returns (I_batch, mu, grad_I) where I_batch is the loss value (float),
    mu is the normalized gradient (flat tensor), grad_I is the raw gradient.
    """
    model.unflatten_params(theta)
    model.zero_grad()
    pred = model(X_batch)
    loss = loss_fn(pred, y_batch)
    loss.backward()

    I_batch = loss.item()
    grad_I = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    r_true = math.sqrt(I_batch + C)
    mu = grad_I / r_true

    return I_batch, mu, grad_I, r_true


def _evaluate_batch_loss(model, theta, X_batch, y_batch, loss_fn):
    """Forward pass only (no grad) to get loss on a mini-batch."""
    model.unflatten_params(theta)
    with torch.no_grad():
        pred = model(X_batch)
        return loss_fn(pred, y_batch).item()


def train_vanilla_sav(model, X_train, y_train, X_test, y_test,
                      C=1.0, lambda_=0.0, dt=0.1, batch_size=256,
                      epochs=50000, device=None, slack_interval=5000,
                      wandb_run=None):
    """Algorithm 1: Vanilla SAV.

    Per MATH_REFERENCE lines 84-90:
      1. mu^n = grad_I(theta^n) / sqrt(I(theta^n) + C)
      2. a = <mu^n, theta^n>, b = ||mu^n||^2, alpha = 1/(1 + lambda*dt)
      3. r^{n+1} = (r^n - lambda*dt*alpha/2 * a) / (1 + dt*alpha/2 * b)
      4. theta^{n+1} = alpha * (theta^n - dt * r^{n+1} * mu^n)
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    loss_fn = nn.MSELoss()
    alpha = 1.0 / (1.0 + lambda_ * dt)

    send_slack(_sav_config_str("Vanilla SAV", model, X_train, epochs,
                               batch_size, dt, C, lambda_, device))

    # Initialize theta and r from full dataset
    theta = model.flatten_params().clone()
    full_loss = evaluate_loss(model, X_train, y_train, loss_fn)
    r = math.sqrt(full_loss + C)

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

            # Step 1: compute mu
            I_batch, mu, grad_I, _ = _compute_mu_and_loss(
                model, theta, X_b, y_b, C, loss_fn
            )

            # Step 2: compute a, b
            a = torch.dot(mu, theta).item()
            b = torch.dot(mu, mu).item()

            # Step 3: r update
            r_new = (r - lambda_ * dt * alpha / 2.0 * a) / \
                    (1.0 + dt * alpha / 2.0 * b)

            # Step 4: theta update
            theta = alpha * (theta - dt * r_new * mu)

            r = r_new

            # NaN check
            if math.isnan(r) or torch.isnan(theta).any():
                print(f"[Vanilla SAV] NaN at epoch {epoch}!")
                break

        # Epoch-end: evaluate full-dataset metrics
        model.unflatten_params(theta)
        tr_loss = evaluate_loss(model, X_train, y_train, loss_fn)
        te_loss = evaluate_loss(model, X_test, y_test, loss_fn)
        energy = r ** 2 + lambda_ / 2.0 * torch.dot(theta, theta).item()

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        energy_values.append(energy)
        r_values.append(r)

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": tr_loss,
                           "test_loss": te_loss, "energy": energy, "r": r})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Vanilla SAV | epoch {epoch}/{epochs} | "
                f"train_loss={tr_loss:.4e} | test_loss={te_loss:.4e} | "
                f"energy={energy:.4e} | r={r:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Vanilla SAV done | final_test_loss={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Vanilla SAV",
        "params": {"C": C, "lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }


def train_restart_sav(model, X_train, y_train, X_test, y_test,
                      C=1.0, lambda_=0.0, dt=0.1, batch_size=256,
                      epochs=50000, device=None, slack_interval=5000,
                      wandb_run=None):
    """Algorithm 2: Restart SAV.

    Per MATH_REFERENCE lines 121-126:
    Same as Vanilla but reset r_hat = sqrt(I_batch + C) at each step.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    loss_fn = nn.MSELoss()
    alpha = 1.0 / (1.0 + lambda_ * dt)

    send_slack(_sav_config_str("Restart SAV", model, X_train, epochs,
                               batch_size, dt, C, lambda_, device))

    # Initialize theta
    theta = model.flatten_params().clone()
    # r is reset per-step, but we track it for energy logging
    full_loss = evaluate_loss(model, X_train, y_train, loss_fn)
    r = math.sqrt(full_loss + C)

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

            # Step 1: compute mu with RESTART: r_hat = sqrt(I_batch + C)
            I_batch, mu, grad_I, r_hat = _compute_mu_and_loss(
                model, theta, X_b, y_b, C, loss_fn
            )

            # Step 2: compute a, b
            a = torch.dot(mu, theta).item()
            b = torch.dot(mu, mu).item()

            # Step 3: r update (using r_hat instead of r)
            r_new = (r_hat - lambda_ * dt * alpha / 2.0 * a) / \
                    (1.0 + dt * alpha / 2.0 * b)

            # Step 4: theta update
            theta = alpha * (theta - dt * r_new * mu)

            r = r_new

            if math.isnan(r) or torch.isnan(theta).any():
                print(f"[Restart SAV] NaN at epoch {epoch}!")
                break

        # Epoch-end evaluation
        model.unflatten_params(theta)
        tr_loss = evaluate_loss(model, X_train, y_train, loss_fn)
        te_loss = evaluate_loss(model, X_test, y_test, loss_fn)
        energy = r ** 2 + lambda_ / 2.0 * torch.dot(theta, theta).item()

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        energy_values.append(energy)
        r_values.append(r)

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": tr_loss,
                           "test_loss": te_loss, "energy": energy, "r": r})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Restart SAV | epoch {epoch}/{epochs} | "
                f"train_loss={tr_loss:.4e} | test_loss={te_loss:.4e} | "
                f"energy={energy:.4e} | r={r:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Restart SAV done | final_test_loss={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Restart SAV",
        "params": {"C": C, "lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }


def train_relax_sav(model, X_train, y_train, X_test, y_test,
                    C=1.0, lambda_=0.0, dt=0.1, batch_size=256,
                    epochs=50000, device=None, slack_interval=5000,
                    wandb_run=None):
    """Algorithm 3: Relax SAV.

    Per MATH_REFERENCE lines 188-195:
      1. Run Vanilla step -> r_tilde, theta_new
      2. r_hat_new = sqrt(I(theta_new) + C)
      3. S_n = r^2 + lambda/2 * (||theta||^2 - ||theta_new||^2)
      4. If r_hat_new^2 <= S_n: xi=0
      5. Else: delta = r_tilde - r_hat_new, xi = (sqrt(S_n) - r_hat_new) / delta, clamp [0,1]
      6. r_new = xi * r_tilde + (1-xi) * r_hat_new
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    loss_fn = nn.MSELoss()
    alpha = 1.0 / (1.0 + lambda_ * dt)

    send_slack(_sav_config_str("Relax SAV", model, X_train, epochs,
                               batch_size, dt, C, lambda_, device))

    # Initialize
    theta = model.flatten_params().clone()
    full_loss = evaluate_loss(model, X_train, y_train, loss_fn)
    r = math.sqrt(full_loss + C)

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
            r_old = r

            # === Vanilla step (Algorithm 1) ===
            I_batch, mu, grad_I, _ = _compute_mu_and_loss(
                model, theta, X_b, y_b, C, loss_fn
            )
            a = torch.dot(mu, theta).item()
            b = torch.dot(mu, mu).item()

            r_tilde = (r - lambda_ * dt * alpha / 2.0 * a) / \
                      (1.0 + dt * alpha / 2.0 * b)
            theta_new = alpha * (theta - dt * r_tilde * mu)

            # === Relax correction ===
            # Step 2: evaluate I(theta_new) via forward pass
            I_new = _evaluate_batch_loss(model, theta_new, X_b, y_b, loss_fn)
            r_hat_new = math.sqrt(I_new + C)

            # Step 3: S_n = r_old^2 + lambda/2 * (||theta_old||^2 - ||theta_new||^2)
            S_n = r_old ** 2 + lambda_ / 2.0 * (
                torch.dot(theta_old, theta_old).item() -
                torch.dot(theta_new, theta_new).item()
            )
            # Guard against numerical issues
            S_n = max(S_n, 0.0)

            # Step 4-5: compute xi
            if r_hat_new ** 2 <= S_n:
                xi = 0.0
            else:
                delta = r_tilde - r_hat_new
                if abs(delta) < 1e-12:
                    xi = 0.0
                else:
                    xi = (math.sqrt(S_n) - r_hat_new) / delta
                    xi = max(0.0, min(1.0, xi))

            # Step 6: blend
            r = xi * r_tilde + (1.0 - xi) * r_hat_new
            theta = theta_new

            if math.isnan(r) or torch.isnan(theta).any():
                print(f"[Relax SAV] NaN at epoch {epoch}!")
                break

        # Epoch-end evaluation
        model.unflatten_params(theta)
        tr_loss = evaluate_loss(model, X_train, y_train, loss_fn)
        te_loss = evaluate_loss(model, X_test, y_test, loss_fn)
        energy = r ** 2 + lambda_ / 2.0 * torch.dot(theta, theta).item()

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        energy_values.append(energy)
        r_values.append(r)

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": tr_loss,
                           "test_loss": te_loss, "energy": energy, "r": r})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"Relax SAV | epoch {epoch}/{epochs} | "
                f"train_loss={tr_loss:.4e} | test_loss={te_loss:.4e} | "
                f"energy={energy:.4e} | r={r:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"Relax SAV done | final_test_loss={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": "Relax SAV",
        "params": {"C": C, "lambda": lambda_, "dt": dt, "batch_size": batch_size},
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": r_values,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }
