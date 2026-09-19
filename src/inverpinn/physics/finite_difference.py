"""Explicit forward simulation on the unit square using PyTorch CPU float64."""

import math
from collections.abc import Callable, Mapping, Sequence

import torch

from inverpinn.physics.sources import gaussian_source


def stability_number(*, dx: float, dy: float, dt: float, u: float, v: float, D: float) -> float:
    """Return dt*(|u|/dx + |v|/dy + 2D/dx² + 2D/dy²).

    This dimensionless sum must be <= 1 for the explicit upwind/centered
    scheme. Then all stencil weights are nonnegative: an old nonnegative
    field plus nonnegative emissions stays nonnegative without clipping.
    Length, time, wind, and diffusion must use consistent units.
    """
    if not all(math.isfinite(a) for a in (dx, dy, dt, u, v, D)):
        raise ValueError("Grid, time, and transport parameters must be finite.")
    if min(dx, dy, dt) <= 0 or D < 0:
        raise ValueError("dx, dy, dt must be positive and D must be nonnegative.")
    return dt * (abs(u) / dx + abs(v) / dy + 2 * D * (dx**-2 + dy**-2))


def solve_advection_diffusion(
    *, nx: int, ny: int, dt: float, steps: int, u: float, v: float, D: float,
    sources: Sequence[Mapping[str, float]] = (),
    forcing: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] | None = None,
    return_history: bool = True,
) -> torch.Tensor:
    r"""Return C[t,y,x] with shape (steps+1, ny, nx), including initial zeros.

    Solve C_t = -u C_x - v C_y + D(C_xx+C_yy) + S on [0,1]².
    x_i=i/(nx-1), y_j=j/(ny-1), and t_n=n*dt. Each source mapping contains
    x_s, y_s, Q, sigma, as in gaussian_source; sources add linearly and remain
    constant in time. Q is a peak concentration/time rate, not total emissions.

    Optional forcing(x_grid, y_grid, t) adds a signed, time-dependent source
    rate for manufactured-solution validation. It must return a finite tensor
    of shape (ny, nx), evaluated at t_n (not t_{n+1}) for forward Euler.
    It does not change the zero initial or boundary conditions. With
    return_history=False, return only C[y,x] at the final time, using O(nx*ny)
    storage and exactly the same update as the default full-history mode.

    Forward Euler advances time. Wind uses first-order upwind differences:
    positive u looks left, negative u looks right (and similarly for v).
    Diffusion uses centered second differences. The method is first-order in
    time and advection, second-order in diffusion; upwinding adds numerical
    spreading. A timestep violating the combined CFL/diffusion bound raises
    ValueError before allocating the history; it is never silently adjusted.

    Boundary condition: C=0 on all four edges at all times, representing an
    ideal clean-air/absorbing boundary. Only interior cells receive emissions.
    This is not a no-flux or mass-conserving closed box and is not a general
    open-outflow model. Keep sources/plumes away from edges when studying
    unbounded transport. No initial pollution is present anywhere.

    All inputs use compatible length/time units. Full history is retained,
    requiring approximately 8*(steps+1)*ny*nx bytes; use modest grids first.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in (nx, ny)):
        raise ValueError("nx and ny must be integers >= 3.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    dx, dy = 1 / (nx - 1), 1 / (ny - 1)
    cfl = stability_number(dx=dx, dy=dy, dt=dt, u=u, v=v, D=D)
    if cfl > 1:
        raise ValueError(f"Unstable timestep: stability number {cfl:g} > 1; reduce dt.")
    x = torch.linspace(0, 1, nx, dtype=torch.float64)
    y = torch.linspace(0, 1, ny, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    emission = torch.zeros_like(xx)
    for source in sources:
        emission += gaussian_source(xx, yy, **source)
    history = torch.zeros((steps + 1, ny, nx), dtype=torch.float64) if return_history else None
    old = history[0] if return_history else torch.zeros_like(xx)
    for n in range(steps):
        rate = emission
        if forcing is not None:
            extra = forcing(xx, yy, n * dt)
            if extra.shape != xx.shape or not torch.isfinite(extra).all():
                raise ValueError("forcing must return a finite (ny, nx) tensor.")
            rate = emission + extra
        new = history[n + 1] if return_history else torch.zeros_like(old)
        center = old[1:-1, 1:-1]
        # Wind carries the value from the upstream neighbor into each cell.
        upstream_x = old[1:-1, :-2] if u >= 0 else old[1:-1, 2:]
        upstream_y = old[:-2, 1:-1] if v >= 0 else old[2:, 1:-1]
        # Nonnegative weights make the stability argument visible in the code.
        new[1:-1, 1:-1] = (
            (1 - cfl) * center
            + dt * abs(u) / dx * upstream_x
            + dt * abs(v) / dy * upstream_y
            + dt * D / dx**2 * (old[1:-1, :-2] + old[1:-1, 2:])
            + dt * D / dy**2 * (old[:-2, 1:-1] + old[2:, 1:-1])
            + dt * rate[1:-1, 1:-1]
        )
        old = new
    return history if return_history else old
