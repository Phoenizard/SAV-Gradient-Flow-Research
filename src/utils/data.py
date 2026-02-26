"""Data generation for regression examples from Ma, Mao & Shen (2024)."""

import torch


def generate_example1(D=20, N_train=1000, N_test=200, seed=42):
    """Example 1: sin + cos regression.

    Target: f*(x) = sin(sum(p_i * x_i)) + cos(sum(q_i * x_i))
    Input: x ~ Uniform(0, 1)^D

    Returns (X_train, y_train, X_test, y_test) as float32 tensors.
    """
    gen = torch.Generator().manual_seed(seed)

    # Fixed coefficients (from seed)
    p = torch.rand(D, generator=gen)
    q = torch.rand(D, generator=gen)

    # Generate data
    X_train = torch.rand(N_train, D, generator=gen)
    X_test = torch.rand(N_test, D, generator=gen)

    def target(X):
        return torch.sin(X @ p) + torch.cos(X @ q)

    y_train = target(X_train)
    y_test = target(X_test)

    return X_train, y_train, X_test, y_test


def generate_example2(D=20, N_train=1000, N_test=200, seed=42):
    """Example 2: polynomial regression.

    Target: f*(x) = sum(c_i * x_i^2)
    Input: x ~ Uniform(0, 5)^D

    Returns (X_train, y_train, X_test, y_test) as float32 tensors.
    """
    gen = torch.Generator().manual_seed(seed)

    # Fixed coefficients (from seed)
    c = torch.rand(D, generator=gen)

    # Generate data: x ~ Uniform(0, 5)^D
    X_train = 5.0 * torch.rand(N_train, D, generator=gen)
    X_test = 5.0 * torch.rand(N_test, D, generator=gen)

    def target(X):
        return (X ** 2) @ c

    y_train = target(X_train)
    y_test = target(X_test)

    return X_train, y_train, X_test, y_test
