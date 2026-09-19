"""Read-only diagnostics of frozen development models; never fit or select.

Physical PDE terms share concentration/time units. Comparing their signed
values and RMS exposes scale disparities without changing the PDE or weights.
Source gradients additionally require parameter scales: dJ/d(x/L)=L*dJ/dx
and dJ/d(Q/Q*)=Q* dJ/dQ. Raw-logit and physical gradients are not interchangeable.
"""

import numpy as np
import torch

from inverpinn.physics.autodiff import concentration_derivatives


def pde_terms(C, coordinates, *, u, v, D, S):
    """Six signed physical terms and their left-minus-right sum, [N,1].

    Advection terms include the sign of the wind; curvature terms may have
    either sign. The source and diffusion terms are SUBTRACTED in the residual.
    Parameters here are finite constant transport scalars and a source column.
    Derivatives include any input scaling and the entire constraint envelope.
    """
    if not all(np.isfinite(p) for p in (u, v, D)) or D < 0:
        raise ValueError("Finite transport and nonnegative constant D required.")
    if S.shape != (len(coordinates), 1):
        raise ValueError("Source must be an [N,1] column; refuse broadcasting.")
    d = concentration_derivatives(C, coordinates)
    terms = dict(C_t=d["C_t"], u_C_x=u*d["C_x"], v_C_y=v*d["C_y"],
                 D_C_xx=D*d["C_xx"], D_C_yy=D*d["C_yy"], S=S)
    terms["residual"] = terms["C_t"]+terms["u_C_x"]+terms["v_C_y"]-D*(d["C_xx"]+d["C_yy"])-S
    if not all(torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError("Nonfinite PDE diagnostic; do not mask NaNs.")
    return terms


def statistics(values):
    """Finite nonempty sampled values; no clipping and no zero imputation."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Statistics require nonempty finite samples.")
    return dict(count=values.size, mean=float(values.mean()),
                rmse=float(np.sqrt(np.mean(values**2))),
                mae=float(np.mean(np.abs(values))), maxabs=float(np.max(np.abs(values))))


def region_masks(coordinates, estimate, *, source_radius, boundary_width):
    """Overlapping spatial regions in the unit-square OPEN spacetime interior.

    A sigma-wide edge band is a geometric diagnostic, not a physical inversion
    threshold or an adaptive sampler. Global, source and boundary masks overlap;
    their RMSs must not be added. Exclude t=0 and exact edges from PDE statistics.
    Regions depend only on estimated source coordinates, never on source truth.
    """
    points = np.asarray(coordinates, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or estimate.shape != (2,)
            or not np.isfinite(points).all() or not np.isfinite(estimate).all()
            or not np.isfinite(source_radius) or source_radius <= 0
            or not np.isfinite(boundary_width) or not 0 < boundary_width < .5):
        raise ValueError("Finite [N,3] coordinates, [2] estimate and valid radii required.")
    xy = points[:, :2]
    edge_distance = np.minimum(xy, 1-xy).min(axis=1)
    interior = (edge_distance > 0) & (points[:, 2] > 0)
    near = np.sum((xy-estimate)**2, axis=1) <= source_radius**2
    return dict(global_interior=interior, near_estimate=interior & near,
                away_from_estimate=interior & ~near,
                boundary_band=interior & (edge_distance <= boundary_width))


def source_motion(history, start, stop, Q_scale):
    """Endpoint movement from POST-update histories, not a convergence test.

    Displacement can cancel oscillations, so also retain the last-step change
    and maximum excursion from the start. Q change/Q_scale uses a fixed prior
    scale, not true Q. Windows are fixed before computing this supplement.
    """
    if not (0 < start < stop and np.isfinite(Q_scale) and Q_scale > 0):
        raise ValueError("Ordered positive update numbers and Q scale required.")
    epochs = history["epoch"].to_numpy()
    if not np.array_equal(epochs, np.arange(1, len(history)+1)) or stop > len(history):
        raise ValueError("Require complete consecutive post-update history.")
    values = history[["source/x_s", "source/y_s", "source/Q"]].to_numpy(dtype=float)[start-1:stop]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source history.")
    delta = values[-1]-values[0]
    return dict(start=start, stop=stop, x_change=float(delta[0]), y_change=float(delta[1]),
                location_displacement=float(np.linalg.norm(delta[:2])), Q_change=float(delta[2]),
                Q_change_over_prior_scale=float(delta[2]/Q_scale),
                max_location_excursion=float(np.linalg.norm(values[:, :2]-values[0, :2], axis=1).max()),
                max_Q_excursion=float(np.abs(values[:, 2]-values[0, 2]).max()),
                last_step_location_change=float(np.linalg.norm(values[-1, :2]-values[-2, :2])),
                last_step_Q_change=float(values[-1, 2]-values[-2, 2]))


def inferred_raw_parameters(frame):
    """Inverse-transform saved B0/B5 physical estimates in float64.

    Intermediate raw values were NOT checkpointed. These are reconstructions
    from rounded CSV physical values, not exact historical logits. Final raw
    values are separately loaded exactly from checkpoints. All these runs use
    softplus Q and sigmoid locations; no arbitrary saturation cutoff is used.
    """
    x, y, q = (frame[key].to_numpy(dtype=float) for key in ("x_s", "y_s", "Q"))
    q = q-torch.finfo(torch.float32).tiny
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(q).all()
            and ((x > 0) & (x < 1) & (y > 0) & (y < 1) & (q > 0)).all()):
        raise ValueError("Cannot reconstruct finite logits from saturated physical values.")
    return dict(raw_x_inferred=np.log(x)-np.log1p(-x),
                raw_y_inferred=np.log(y)-np.log1p(-y),
                raw_Q_inferred=q+np.log(-np.expm1(-q)))


def scaled_gradient_rows(frame, scales):
    """Compare signed gradients in raw and dimensionless physical coordinates.

    Only source PDE rows are used by the caller. A zero Q gradient makes a
    location/Q ratio undefined (NaN), not infinity or an invented zero. The
    ratio is a conditioning diagnostic, not an optimizer update or causal score.
    """
    result = frame.copy()
    result["location_raw_norm"] = np.hypot(result.x_s_raw_gradient, result.y_s_raw_gradient)
    result["location_scaled_norm"] = scales.length*np.hypot(result.x_s_physical_gradient, result.y_s_physical_gradient)
    result["Q_scaled_gradient"] = scales.concentration/scales.time*result.Q_physical_gradient
    denominator = result.Q_scaled_gradient.abs().replace(0, np.nan)
    result["location_to_Q_scaled_ratio"] = result.location_scaled_norm/denominator
    for name, values in inferred_raw_parameters(result).items():
        result[name] = values
    return result
