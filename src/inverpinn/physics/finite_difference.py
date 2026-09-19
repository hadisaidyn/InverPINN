"""Explicit forward simulation on the unit square using PyTorch CPU float64."""

import math
from collections.abc import Callable, Mapping, Sequence

import torch

from inverpinn.data.sensors import sample_sensors
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


def _output_schedule(values, dt, steps):
    """Plan physical times using at most two adjacent Euler states.

    Native times (within floating-point representation error) use the native
    state exactly. Other times use (1-alpha)*C_left+alpha*C_right. This does
    not alter the integration step or silently round physical observation times.
    Schedule storage is O(number of requested times), not O(steps).
    """
    times = torch.as_tensor([] if values is None else values, dtype=torch.float64).clone()
    if (times.ndim != 1 or not torch.isfinite(times).all()
            or torch.any(times < 0) or torch.any(times > steps*dt)
            or torch.any(times[1:] <= times[:-1])):
        raise ValueError("Output times must be finite, strictly increasing, unique, and in [0, steps*dt].")
    schedule = {}
    for index, t in enumerate(times.tolist()):
        ratio = t/dt
        nearest = round(ratio)
        if abs(ratio-nearest) <= 32*torch.finfo(torch.float64).eps*max(1, abs(ratio)):
            right, alpha = nearest, 1.0
        else:
            right = math.ceil(ratio)
            alpha = ratio-(right-1)
        schedule.setdefault(right, []).append((index, alpha))
    return times, schedule


def solve_advection_diffusion(
    *, nx: int, ny: int, dt: float, steps: int, u: float, v: float, D: float,
    sources: Sequence[Mapping[str, float]] = (),
    forcing: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] | None = None,
    return_history: bool = True,
    output_mode: str | None = None,
    output_times: Sequence[float] | None = None,
    sensor_positions: torch.Tensor | None = None,
    measurement_times: Sequence[float] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
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

    Output modes (all use the identical update below):
    - full_history (default): C[steps+1,ny,nx], including initial zeros.
    - final_only: C[ny,nx]. Legacy return_history=False selects this mode.
    - selected_times: dictionary with fields[K,ny,nx], field_times[K], x, y,
      sensor_positions[N,2], measurement_times[M], measurements[M,N].
      output_times is required; sensors may be collected in the same run.
    - sensor_only: same dictionary, but no field snapshots (K=0).
      Fixed sensor_positions and measurement_times are required.

    Non-native physical times use documented linear interpolation between
    adjacent Euler states; no dt changes or silent rounding. Point sensors
    use the same bilinear spatial interpolation as sample_sensors, without
    noise. Apply measurement noise after generation with an explicit seed.
    Sensor/field times are independent, strictly increasing and in [0,T].

    Selective modes use two working fields plus only requested outputs:
    O(nx*ny + K*nx*ny + M*N) storage. Full history, intended for small debugging
    runs, instead needs about 8*(steps+1)*ny*nx bytes. All units are compatible
    synthetic length/time/concentration units; no atmospheric unit conversion.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in (nx, ny)):
        raise ValueError("nx and ny must be integers >= 3.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    dx, dy = 1 / (nx - 1), 1 / (ny - 1)
    cfl = stability_number(dx=dx, dy=dy, dt=dt, u=u, v=v, D=D)
    if cfl > 1:
        raise ValueError(f"Unstable timestep: stability number {cfl:g} > 1; reduce dt.")
    mode = output_mode or ("full_history" if return_history else "final_only")
    if mode not in ("full_history", "final_only", "selected_times", "sensor_only"):
        raise ValueError(f"Unknown output_mode: {mode}")
    if output_mode is not None and not return_history and mode != "final_only":
        raise ValueError("return_history=False conflicts with explicit output_mode.")
    selective = mode in ("selected_times", "sensor_only")
    if not selective and any(a is not None for a in (output_times, sensor_positions, measurement_times)):
        raise ValueError("Output/sensor times and positions require a selective mode.")
    if mode == "sensor_only" and output_times is not None:
        raise ValueError("sensor_only must not request field snapshots.")
    field_times, field_schedule = _output_schedule(output_times, dt, steps)
    sensor_times, sensor_schedule = _output_schedule(measurement_times, dt, steps)
    if mode == "selected_times" and not len(field_times):
        raise ValueError("selected_times requires nonempty output_times.")
    if (sensor_positions is None) != (len(sensor_times) == 0) or (mode == "sensor_only" and not len(sensor_times)):
        raise ValueError("Sensors require both positions and nonempty measurement_times.")
    x = torch.linspace(0, 1, nx, dtype=torch.float64)
    y = torch.linspace(0, 1, ny, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    emission = torch.zeros_like(xx)
    for source in sources:
        emission += gaussian_source(xx, yy, **source)
    history = torch.zeros((steps + 1, ny, nx), dtype=torch.float64) if mode == "full_history" else None
    old = history[0] if history is not None else torch.zeros_like(xx)
    spare = torch.zeros_like(old) if history is None else None
    snapshots = torch.empty((len(field_times), ny, nx), dtype=torch.float64)
    positions = torch.empty((0, 2), dtype=torch.float64)
    if sensor_positions is not None:
        positions, _ = sample_sensors(old.unsqueeze(0), x, y, positions=sensor_positions)
    measurements = torch.empty((len(sensor_times), len(positions)), dtype=torch.float64)

    def capture(step, left, right):
        for index, alpha in field_schedule.get(step, ()):
            snapshots[index].copy_(right if alpha == 1 else (1-alpha)*left+alpha*right)
        requests = sensor_schedule.get(step, ())
        if requests:
            _, right_values = sample_sensors(right.unsqueeze(0), x, y, positions=positions)
            if any(alpha != 1 for _, alpha in requests):
                _, left_values = sample_sensors(left.unsqueeze(0), x, y, positions=positions)
            for index, alpha in requests:
                measurements[index].copy_(right_values[0] if alpha == 1 else
                    (1-alpha)*left_values[0]+alpha*right_values[0])

    capture(0, old, old)
    for n in range(steps):
        rate = emission
        if forcing is not None:
            extra = forcing(xx, yy, n * dt)
            if extra.shape != xx.shape or not torch.isfinite(extra).all():
                raise ValueError("forcing must return a finite (ny, nx) tensor.")
            rate = emission + extra
        new = history[n + 1] if history is not None else spare
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
        capture(n+1, old, new)
        spare, old = old, new
    if selective:
        return dict(fields=snapshots, field_times=field_times, x=x, y=y,
                    sensor_positions=positions, measurement_times=sensor_times, measurements=measurements)
    return history if history is not None else old
