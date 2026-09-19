"""Unit-consistent scaling of the unchanged advection--diffusion equation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CharacteristicScales:
    """L,T,C*: positive physical reference scales, never fitted to source truth.

    x=L xi, t=T tau, C=C* c imply r_hat=(T/C*) r. The diffusion,
    velocity and source scales are L²/T, L/T and C*/T respectively.
    """

    length: float
    time: float
    concentration: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.length, self.time, self.concentration)):
            raise ValueError("Characteristic scales must be finite and positive.")

    def transport(self, u, v, D):
        return dict(u=u*self.time/self.length, v=v*self.time/self.length,
                    D=D*self.time/self.length**2)

    def residual(self, physical_residual):
        return physical_residual*self.time/self.concentration

    def loss_factors(self):
        """Multipliers on physical MSEs, separate from preference weights."""
        return dict(data=1/self.concentration**2, initial=1/self.concentration**2,
                    boundary=1/self.concentration**2, pde=(self.time/self.concentration)**2)
