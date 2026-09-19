"""Controlled positive-strength parameterization comparison for revision 4."""

import math

import torch
from torch import nn

from inverpinn.physics.trainable_source import TrainableGaussianSource


class DiagnosticGaussianSource(TrainableGaussianSource):
    """Keep sigmoid locations and fixed width; compare softplus Q with exp Q.

    exp has derivative Q (apart from the underflow floor); softplus has
    derivative 1-exp(-Q). Neither removes the NN/source tradeoff. No upper or
    lower prior-bound constraint is introduced. Overflow must fail loudly.
    """

    def __init__(self, x_s, y_s, Q, sigma, parameterization="softplus"):
        if parameterization not in ("softplus", "exp"):
            raise ValueError("Unknown Q parameterization.")
        super().__init__(x_s, y_s, Q, sigma)
        self.parameterization = parameterization
        if parameterization == "exp":
            self.raw_Q = nn.Parameter(torch.tensor(math.log(Q-torch.finfo(torch.float32).tiny)))

    @property
    def Q(self):
        if self.parameterization == "exp":
            return torch.exp(self.raw_Q)+torch.finfo(self.raw_Q.dtype).tiny
        return super().Q
