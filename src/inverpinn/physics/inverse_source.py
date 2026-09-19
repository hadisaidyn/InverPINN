"""Classical single-source inversion using the existing discrete PDE model."""

import numpy as np
import torch
from scipy.optimize import least_squares

from inverpinn.data.sensors import sample_sensors
from inverpinn.physics.finite_difference import solve_advection_diffusion


def fit_source(positions, observations, *, solver, sigma, initial,
               max_nfev=100, tolerance=1e-9):
    r"""Fit (x_s,y_s,Q) by minimizing 1/2 sum (H C(theta)-observations)^2.

    Each trial solves the same finite-difference PDE with zero initial and
    boundary concentrations; H is the existing bilinear sensor sampler.
    Observations must have shape [steps+1,N], including t=0. The solver dict
    contains nx,ny,dt,steps,u,v,D, but NO true sources. Width sigma is known.
    Q is Gaussian peak intensity (concentration/time), not integrated emission.

    Think of moving and brightening a pollution emitter until its simulated
    sensor readings resemble the measured readings. Equal residual weights
    correspond to independent Gaussian noise with a common variance. Negative
    noisy readings are retained. No dense concentration truth enters the fit.

    SciPy trust-region reflective least squares uses finite-difference
    Jacobians and bounds 0<=x_s,y_s<=1, Q>=1e-12. This tiny positive floor
    approximates strict positivity without an arbitrary upper strength bound.
    It is a LOCAL fit: convergence is not a guarantee of source identifiability
    or a global optimum. History records ALL objective calls, including
    Jacobian probes/rejected trials, not optimizer iterations. max_nfev is
    SciPy's budget (which excludes numerical Jacobian probes).
    """
    required = {"nx", "ny", "dt", "steps", "u", "v", "D"}
    if set(solver) != required:
        raise ValueError(f"solver must contain exactly {sorted(required)}; no sources.")
    positions = torch.as_tensor(np.asarray(positions), dtype=torch.float64)
    observations = np.asarray(observations, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    if (initial.shape != (3,) or not np.isfinite(initial).all()
            or np.any(initial[:2] < 0) or np.any(initial[:2] > 1) or initial[2] < 1e-12):
        raise ValueError("initial must be finite [x_s,y_s,Q] within bounds.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")
    if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0
            or observations.shape != (solver["steps"]+1, len(positions))
            or not np.isfinite(observations).all()):
        raise ValueError("Require positions[N,2] and finite observations[steps+1,N].")
    if solver["steps"] < 1:
        raise ValueError("Need observations after the zero initial condition.")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= np.finfo(float).eps:
        raise ValueError("tolerance must exceed machine epsilon.")
    x = torch.linspace(0, 1, solver["nx"], dtype=torch.float64)
    y = torch.linspace(0, 1, solver["ny"], dtype=torch.float64)
    history = []

    def predict(parameters):
        source = dict(zip(("x_s", "y_s", "Q"), parameters)) | {"sigma": sigma}
        field = solve_advection_diffusion(**solver, sources=[source])
        _, readings = sample_sensors(field, x, y, positions=positions)
        if not torch.isfinite(readings).all():
            raise FloatingPointError("Nonfinite inverse-source predictions.")
        return field.numpy(), readings.numpy()

    def residual(parameters):
        _, readings = predict(parameters)
        error = (readings-observations).ravel()
        mse = float(np.mean(error**2))
        if not np.isfinite(mse):
            raise FloatingPointError("Nonfinite inverse-source objective.")
        history.append(dict(evaluation=len(history)+1, x_s=float(parameters[0]),
                            y_s=float(parameters[1]), Q=float(parameters[2]), mse=mse))
        return error

    result = least_squares(residual, initial, bounds=([0, 0, 1e-12], [1, 1, np.inf]),
                           method="trf", jac="2-point", max_nfev=max_nfev,
                           ftol=tolerance, xtol=tolerance, gtol=tolerance)
    field, readings = predict(result.x)
    # Returning the OptimizeResult preserves nonconvergence rather than hiding it.
    return result, field, readings, history
