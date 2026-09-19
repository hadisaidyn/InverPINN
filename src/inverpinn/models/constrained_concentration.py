"""Positive/zero-IC-and-BC concentration representations; no PDE definition."""

import torch
from torch.nn import functional as F

from inverpinn.models.mlp import ConcentrationMLP


class ConstrainedConcentrationMLP(ConcentrationMLP):
    """Same tanh MLP, optionally C=softplus(f) or C=E*softplus(f).

    E=16*x*(1-x)*y*(1-y)*t/T on [0,1]² x [0,T]. E is nonnegative,
    zero on all four edges and at t=0, and at most one. The complete C is
    differentiated by the external physics module, including product rules.
    Output scale is one synthetic concentration unit for both transforms.
    The exact values, not normal derivatives, are constrained at the boundary.
    """

    def __init__(self, final_time, width=64, depth=3, constraint="positive"):
        if constraint not in ("unconstrained", "positive", "hard"):
            raise ValueError("Unknown concentration constraint.")
        super().__init__([0., 0., 0.], [1., 1., final_time], width, depth)
        self.constraint = constraint

    def forward(self, coordinates):
        raw = super().forward(coordinates)
        if self.constraint == "unconstrained":
            return raw
        concentration = F.softplus(raw)
        if self.constraint == "hard":
            x, y, t = coordinates.split(1, dim=-1)
            concentration = concentration*(16*x*(1-x)*y*(1-y)*t/self.upper[2])
        return concentration
