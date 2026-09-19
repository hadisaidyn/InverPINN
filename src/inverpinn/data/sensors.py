"""Sample fixed point sensors from a gridded concentration history."""

import math

import torch


def relative_noise_std(clean_measurements: torch.Tensor, percent: float) -> float:
    """Absolute Gaussian std = percent/100 * RMS of noise-free sensor readings.

    The RMS uses all times and sensors for a run, including the initial zeros.
    This is a constant additive noise std, not pointwise multiplicative noise.
    Scaling uses synthetic clean readings only during data generation; training
    receives noisy readings. A positive percentage with zero RMS is undefined.
    """
    if not math.isfinite(percent) or percent < 0:
        raise ValueError("Noise percentage must be finite and nonnegative.")
    if clean_measurements.numel() == 0 or not torch.isfinite(clean_measurements).all():
        raise ValueError("Clean measurements must be nonempty and finite.")
    rms = clean_measurements.double().square().mean().sqrt().item()
    if rms == 0 and percent > 0:
        raise ValueError("Positive relative noise requires nonzero clean signal RMS.")
    return percent / 100 * rms


def sample_sensors(
    concentration: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *,
    n_sensors: int | None = None, positions: torch.Tensor | None = None,
    noise_std: float = 0.0, seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return (positions[N,2], measurements[T,N]) from C[T,ny,nx].

    Specify exactly one of n_sensors (uniform random placement over the grid
    rectangle) or positions (manual rows of [x,y]). Positions stay fixed for
    every time slice. x and y are strictly increasing coordinate vectors;
    nonuniform grids and positions on the boundary are supported.

    A sensor between four grid nodes reads their bilinear weighted average:
    (1-a)(1-b)C00 + a(1-b)C10 + (1-a)bC01 + abC11, where a,b are fractional
    distances across the cell. No time interpolation is performed: every
    supplied time slice is sampled. Interpolation is exact for bilinear fields,
    but the input simulation itself may have discretization error.

    Observations are Y = interpolated(C) + epsilon, with independent
    epsilon ~ Normal(0, noise_std**2) for each time and sensor. noise_std is
    an absolute standard deviation in concentration units. Negative noisy
    readings are retained; clipping would bias the Gaussian noise model.

    Uses local seeded generators (placement seed, noise seed+1) without
    changing global random state. Inputs must be finite CPU float32/float64
    tensors. The output uses concentration's dtype. Only sparse measurements
    and their positions are returned, never the full field.
    """
    if concentration.device.type != "cpu" or concentration.dtype not in (torch.float32, torch.float64):
        raise ValueError("concentration must be a CPU float32/float64 tensor.")
    if concentration.ndim != 3 or concentration.shape[0] < 1 or not torch.isfinite(concentration).all():
        raise ValueError("concentration must be finite with shape [T,ny,nx], T >= 1.")
    x = torch.as_tensor(x, dtype=concentration.dtype, device="cpu")
    y = torch.as_tensor(y, dtype=concentration.dtype, device="cpu")
    for axis, size in ((x, concentration.shape[2]), (y, concentration.shape[1])):
        if (axis.ndim != 1 or axis.numel() != size or size < 2
                or not torch.isfinite(axis).all() or not torch.all(axis[1:] > axis[:-1])):
            raise ValueError("Coordinates must match the field and be finite, strictly increasing vectors.")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63-2].")
    if (n_sensors is None) == (positions is None):
        raise ValueError("Specify exactly one of n_sensors or positions.")
    if positions is None:
        if isinstance(n_sensors, bool) or not isinstance(n_sensors, int) or n_sensors < 1:
            raise ValueError("n_sensors must be a positive integer.")
        positions = torch.rand((n_sensors, 2), dtype=concentration.dtype,
                               generator=torch.Generator().manual_seed(seed))
        positions[:, 0] = x[0] + positions[:, 0] * (x[-1] - x[0])
        positions[:, 1] = y[0] + positions[:, 1] * (y[-1] - y[0])
    else:
        positions = torch.as_tensor(positions, dtype=concentration.dtype, device="cpu").clone()
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 1 or not torch.isfinite(positions).all():
        raise ValueError("positions must be finite with shape [N,2], N >= 1.")
    px, py = positions[:, 0].contiguous(), positions[:, 1].contiguous()
    if torch.any((px < x[0]) | (px > x[-1]) | (py < y[0]) | (py > y[-1])):
        raise ValueError("Sensor positions must lie inside the coordinate domain.")
    # Find the cell around each sensor, including points on the outer edges.
    ix = (torch.searchsorted(x, px, right=True) - 1).clamp(0, len(x)-2)
    iy = (torch.searchsorted(y, py, right=True) - 1).clamp(0, len(y)-2)
    a = (px - x[ix]) / (x[ix+1] - x[ix])
    b = (py - y[iy]) / (y[iy+1] - y[iy])
    readings = ((1-a)*(1-b)*concentration[:, iy, ix]
                + a*(1-b)*concentration[:, iy, ix+1]
                + (1-a)*b*concentration[:, iy+1, ix]
                + a*b*concentration[:, iy+1, ix+1])
    noise = torch.randn(readings.shape, dtype=readings.dtype,
                        generator=torch.Generator().manual_seed(seed + 1))
    return positions, readings + noise_std * noise
