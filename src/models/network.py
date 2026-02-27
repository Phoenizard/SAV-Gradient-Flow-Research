"""One-hidden-layer ReLU network per Ma, Mao & Shen (2024)."""

import math
import torch
import torch.nn as nn


class OneHiddenLayerNet(nn.Module):
    """f(x) = ReLU(x_aug @ W) @ a, where x_aug = [x, 1].

    Parameters:
        W: (D+1, m) — input weights with bias absorbed
        a: (m, 1)  — output weights

    Total params d = (D+1)*m + m.
    """

    def __init__(self, D, m, seed=42):
        super().__init__()
        self.D = D
        self.m = m

        gen = torch.Generator().manual_seed(seed)

        # Kaiming normal init for W (fan_in = D+1)
        self.W = nn.Parameter(
            torch.randn(D + 1, m, generator=gen) * math.sqrt(2.0 / (D + 1))
        )

        # Kaiming normal init for a (fan_in = m)
        self.a = nn.Parameter(
            torch.randn(m, 1, generator=gen) * math.sqrt(2.0 / m)
        )

    def forward(self, x):
        """x: (batch, D) -> (batch,)"""
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, ones], dim=1)  # (batch, D+1)
        h = torch.relu(x_aug @ self.W)       # (batch, m)
        out = (h @ self.a) / self.m           # (batch, 1) — paper's 1/m factor
        return out.squeeze(-1)                # (batch,)

    def flatten_params(self):
        """Return all parameters as a single flat (d,) tensor."""
        return torch.cat([p.data.reshape(-1) for p in self.parameters()])

    def unflatten_params(self, flat):
        """Set parameters from a flat (d,) tensor in-place."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
            offset += numel

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
