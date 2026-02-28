"""Shared training utilities for all algorithms."""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Standard mean squared error: mean((pred - target)^2).

    Used as the training loss (optimization target) for gradient computation.
    """

    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)


class RelativeErrorLoss(nn.Module):
    """Relative error: sum((pred - target)^2) / sum(target^2).

    Used as the evaluation metric for reporting (matches paper figures).
    NOT used for gradient computation.
    """

    def forward(self, pred, target):
        return torch.sum((pred - target) ** 2) / torch.sum(target ** 2)


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

    Returns scalar float. Default loss is MSE (training loss).
    """
    if loss_fn is None:
        loss_fn = MSELoss()
    with torch.no_grad():
        pred = model(X)
        return loss_fn(pred, y).item()


def evaluate_rel_error(model, X, y):
    """Compute full-dataset relative error (evaluation metric).

    Returns scalar float.
    """
    rel_fn = RelativeErrorLoss()
    with torch.no_grad():
        pred = model(X)
        return rel_fn(pred, y).item()


def compute_batch_gradient(model, X_batch, y_batch, loss_fn=None):
    """Forward + backward on a mini-batch.

    Returns (loss_value: float, grad_flat: Tensor of shape (d,)).
    Default loss is MSE.
    """
    if loss_fn is None:
        loss_fn = MSELoss()

    model.zero_grad()
    pred = model(X_batch)
    loss = loss_fn(pred, y_batch)
    loss.backward()

    grad_flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    return loss.item(), grad_flat
