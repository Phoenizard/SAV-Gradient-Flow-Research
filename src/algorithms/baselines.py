"""SGD and Adam baseline trainers."""

import time
import torch

from src.utils.trainer import get_device, make_batches, evaluate_loss, RelativeErrorLoss
from src.utils.slack import send_slack


def _train_optimizer(method_name, optimizer_cls, optimizer_kwargs,
                     model, X_train, y_train, X_test, y_test,
                     batch_size=256, epochs=50000, device=None,
                     slack_interval=5000, wandb_run=None):
    """Generic training loop for torch optimizers (SGD, Adam).

    Returns standardized result dict.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    loss_fn = RelativeErrorLoss()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    # Config string for Slack
    config_str = (
        f"Config: D={X_train.shape[1]}, m={model.m}, epochs={epochs}, "
        f"batch_size={batch_size}, "
        + ", ".join(f"{k}={v}" for k, v in optimizer_kwargs.items())
        + f", seed=42\nDevice: {device}"
    )
    send_slack(f"Starting {method_name} on current example\n{config_str}")

    train_losses = []
    test_losses = []
    energy_values = []  # energy = train_loss for baselines

    gen = torch.Generator().manual_seed(42)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        batches = make_batches(len(X_train), batch_size, generator=gen)

        for idx in batches:
            optimizer.zero_grad()
            pred = model(X_train[idx])
            loss = loss_fn(pred, y_train[idx])
            loss.backward()
            optimizer.step()

        # Epoch-end evaluation
        tr_loss = evaluate_loss(model, X_train, y_train, loss_fn)
        te_loss = evaluate_loss(model, X_test, y_test, loss_fn)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        energy_values.append(tr_loss)

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": tr_loss,
                           "test_loss": te_loss, "energy": tr_loss})

        if epoch % slack_interval == 0 or epoch == 1:
            send_slack(
                f"{method_name} | epoch {epoch}/{epochs} | "
                f"train_loss={tr_loss:.4e} | test_loss={te_loss:.4e}"
            )

    wall_time = time.time() - t0
    send_slack(
        f"{method_name} done | final_test_loss={test_losses[-1]:.4e} | "
        f"wall_time={wall_time:.1f}s"
    )

    return {
        "method": method_name,
        "params": optimizer_kwargs,
        "train_loss": train_losses,
        "test_loss": test_losses,
        "energy": energy_values,
        "r_values": [],
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "wall_time": wall_time,
    }


def train_sgd(model, X_train, y_train, X_test, y_test,
              lr=0.1, batch_size=256, epochs=50000, device=None,
              slack_interval=5000, wandb_run=None, weight_decay=0.0):
    """Train with vanilla SGD (no momentum).

    weight_decay: L2 regularization coefficient (corresponds to lambda in the
    gradient flow dtheta/dt = -lambda*theta - grad_I).
    """
    return _train_optimizer(
        "SGD",
        torch.optim.SGD,
        {"lr": lr, "momentum": 0, "weight_decay": weight_decay},
        model, X_train, y_train, X_test, y_test,
        batch_size=batch_size, epochs=epochs, device=device,
        slack_interval=slack_interval, wandb_run=wandb_run,
    )


def train_adam(model, X_train, y_train, X_test, y_test,
               lr=0.001, batch_size=256, epochs=50000, device=None,
               slack_interval=5000, wandb_run=None):
    """Train with Adam optimizer."""
    return _train_optimizer(
        "Adam",
        torch.optim.Adam,
        {"lr": lr, "betas": (0.9, 0.999)},
        model, X_train, y_train, X_test, y_test,
        batch_size=batch_size, epochs=epochs, device=device,
        slack_interval=slack_interval, wandb_run=wandb_run,
    )
