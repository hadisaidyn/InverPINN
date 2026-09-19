"""Spatial interpolation independently at each observed time, without physics."""

import math

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist


def _inputs(positions, measurements, queries):
    positions, measurements, queries = (np.asarray(a, dtype=np.float64) for a in (positions, measurements, queries))
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions must have shape [N,2].")
    if measurements.ndim != 2 or measurements.shape[1] != len(positions) or measurements.shape[0] == 0:
        raise ValueError("measurements must have shape [T,N].")
    if queries.ndim != 2 or queries.shape[1] != 2 or len(queries) == 0:
        raise ValueError("queries must have shape [M,2].")
    if any(not np.isfinite(a).all() for a in (positions, measurements, queries)):
        raise ValueError("Interpolation inputs must be finite.")
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError("Duplicate sensor positions require explicit aggregation before interpolation.")
    return positions, measurements, queries


def idw_interpolate(positions, measurements, queries, *, power=2.):
    r"""Return [T,M] values: C_hat(q)=sum_i w_i(q) C_i / sum_i w_i(q).

    w_i=distance(q,p_i)^(-power). At a sensor, return its exact observation.
    Relative distances avoid overflowing inverse powers near sensor locations.
    One fixed spatial weight matrix is applied independently to each time.
    No extrapolation clipping or additional zero boundary values are supplied.
    """
    p, values, q = _inputs(positions, measurements, queries)
    if not math.isfinite(power) or power <= 0:
        raise ValueError("IDW power must be positive and finite.")
    distance = cdist(q, p)
    weights = np.zeros_like(distance)
    exact = (distance == 0).any(axis=1)
    weights[exact] = (distance[exact] == 0).astype(float)
    d = distance[~exact]
    weights[~exact] = (d.min(axis=1, keepdims=True)/d)**power
    weights /= weights.sum(axis=1, keepdims=True)
    return values @ weights.T


def rbf_interpolate(positions, measurements, queries, *, smoothing=0.):
    r"""Thin-plate RBF phi(r)=r² log(r), with degree-1 polynomial augmentation.

    Returns [T,M]. Thin-plate kernel avoids choosing a shape parameter. Zero
    smoothing interpolates sensor readings exactly, including their noise;
    smoothing is explicit and fixed, not fitted to dense truth. Requires at
    least three distinct noncollinear positions. Each time is an independent
    right-hand side of the same spatial interpolation system.
    """
    p, values, q = _inputs(positions, measurements, queries)
    if not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError("RBF smoothing must be finite and nonnegative.")
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(p)), p])) < 3:
        raise ValueError("Thin-plate RBF needs at least three noncollinear sensors.")
    return RBFInterpolator(p, values.T, kernel="thin_plate_spline", degree=1, smoothing=smoothing)(q).T


def ordinary_kriging(positions, measurements, queries, *, range_length=0.2, nugget=0.):
    r"""Ordinary kriging with fixed unit-sill exponential covariance exp(-d/range).

    Solve [K,1;1ᵀ,0][w;lambda]=[k(q);1] so weights sum to one (unknown constant
    mean). K has nugget on its diagonal; range has coordinate length units and
    nugget is a ratio to unit sill. With nugget=0, this exactly interpolates.
    Returns [T,M], applying the same weights independently to each time.

    Covariance settings are prescribed, not estimated from ground truth or
    optimized per time. This is a simple fixed-covariance baseline, not a
    calibrated geostatistical uncertainty model. No uncertainty is reported.
    """
    p, values, q = _inputs(positions, measurements, queries)
    if not math.isfinite(range_length) or range_length <= 0 or not math.isfinite(nugget) or nugget < 0:
        raise ValueError("Kriging range must be positive and nugget nonnegative, both finite.")
    n = len(p)
    system = np.ones((n+1, n+1))
    system[:n, :n] = np.exp(-cdist(p,p)/range_length)+nugget*np.eye(n)
    system[-1,-1] = 0
    rhs = np.vstack([np.exp(-cdist(p,q)/range_length), np.ones(len(q))])
    weights = np.linalg.solve(system, rhs)[:n]
    return values @ weights
