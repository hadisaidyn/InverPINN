"""A data-only concentration regressor; no physics equations or PDE loss."""

import torch
from torch import nn


class ConcentrationMLP(nn.Module):
    """Map physical coordinates [...,3] = [x,y,t] to concentration [...,1].

    Inputs are scaled to [-1,1] using supplied domain bounds. Smooth tanh
    hidden layers learn a regression function; the linear output is unconstrained
    and may be negative. This model imposes no boundary or initial conditions.
    Bounds are stored as buffers so checkpoints preserve input normalization.
    """

    def __init__(self, lower: list[float], upper: list[float], width: int = 64, depth: int = 3):
        super().__init__()
        if type(width) is not int or type(depth) is not int or min(width, depth) < 1:
            raise ValueError("width and depth must be positive integers.")
        lo, hi = torch.tensor(lower, dtype=torch.float32), torch.tensor(upper, dtype=torch.float32)
        if lo.shape != (3,) or hi.shape != (3,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not torch.all(hi > lo):
            raise ValueError("Bounds must be three finite, strictly increasing intervals.")
        self.register_buffer("lower", lo)
        self.register_buffer("upper", hi)
        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(3 if i == 0 else width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Predict C at each supplied [x,y,t] point without a physics constraint."""
        return self.network(2 * (coordinates - self.lower) / (self.upper - self.lower) - 1)
