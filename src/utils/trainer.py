"""Shared training utilities for all algorithms."""

import torch
import torch.nn as nn


def get_device():
    """Return best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_batches(N, batch_size, generator=None):
    """Return list of index tensors for mini-batches.

    Shuffles indices using generator for reproducibility.
    """
    perm = torch.randperm(N, generator=generator)
    return [perm[i:i + batch_size] for i in range(0, N, batch_size)]


def evaluate_loss(model, X, y, loss_fn=None):
    """Compute full-dataset loss with no gradient tracking.

    Returns scalar float.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    with torch.no_grad():
        pred = model(X)
        return loss_fn(pred, y).item()


def compute_batch_gradient(model, X_batch, y_batch, loss_fn=None):
    """Forward + backward on a mini-batch.

    Returns (loss_value: float, grad_flat: Tensor of shape (d,)).
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    model.zero_grad()
    pred = model(X_batch)
    loss = loss_fn(pred, y_batch)
    loss.backward()

    grad_flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    return loss.item(), grad_flat
