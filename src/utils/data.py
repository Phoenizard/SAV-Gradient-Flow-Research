"""Data generation for regression examples from Ma, Mao & Shen (2024).

Corrected to match paper: Z-score normalization, 80/20 split from single pool.
"""

import torch


def _zscore_normalize(X_train, X_test):
    """Apply Z-score normalization: x => (x - mu) / sigma.

    Statistics computed from training set, applied to both.
    """
    mu = X_train.mean(dim=0)
    sigma = X_train.std(dim=0)
    sigma = torch.clamp(sigma, min=1e-8)  # avoid division by zero
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def generate_example1(D=40, M=10000, seed=42):
    """Example 1: sin + cos regression.

    Target: f*(x) = sin(sum(p_i * x_i)) + cos(sum(q_i * x_i))
    Input: x ~ Uniform(0, 1)^D
    Split: 80% train, 20% test from M total
    Preprocessing: Z-score normalization on X

    Returns (X_train, y_train, X_test, y_test) as float32 tensors.
    """
    gen = torch.Generator().manual_seed(seed)

    # Fixed coefficients (from seed)
    p = torch.rand(D, generator=gen)
    q = torch.rand(D, generator=gen)

    # Generate M total data points
    X_all = torch.rand(M, D, generator=gen)

    def target(X):
        return torch.sin(X @ p) + torch.cos(X @ q)

    y_all = target(X_all)

    # 80/20 split
    N_train = int(0.8 * M)
    perm = torch.randperm(M, generator=gen)
    train_idx = perm[:N_train]
    test_idx = perm[N_train:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # Z-score normalization
    X_train, X_test = _zscore_normalize(X_train, X_test)

    return X_train, y_train, X_test, y_test


def generate_example2(D=40, M=10000, seed=42):
    """Example 2: polynomial regression.

    Target: f*(x) = sum(c_i * x_i^2)
    Input: x ~ Uniform(0, 5)^D
    Split: 80% train, 20% test from M total
    Preprocessing: Z-score normalization on X

    Returns (X_train, y_train, X_test, y_test) as float32 tensors.
    """
    gen = torch.Generator().manual_seed(seed)

    # Fixed coefficients (from seed)
    c = torch.rand(D, generator=gen)

    # Generate M total data points: x ~ Uniform(0, 5)^D
    X_all = 5.0 * torch.rand(M, D, generator=gen)

    def target(X):
        return (X ** 2) @ c

    y_all = target(X_all)

    # 80/20 split
    N_train = int(0.8 * M)
    perm = torch.randperm(M, generator=gen)
    train_idx = perm[:N_train]
    test_idx = perm[N_train:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # Z-score normalization
    X_train, X_test = _zscore_normalize(X_train, X_test)

    return X_train, y_train, X_test, y_test
