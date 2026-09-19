"""Single-source localization and concentration reconstruction metrics."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def matched_source_metrics(predicted_positions, predicted_Q, true_positions, true_Q):
    """Match equal-sized source sets by minimum TOTAL Euclidean distance.

    Hungarian assignment handles label permutation globally, not greedily.
    Strength errors use the SAME spatial assignment; strength does not influence
    matching. Returns per-pair metrics and their arithmetic means. Equal-cost
    ties have no unique physical assignment (especially coincident sources).
    Unequal source counts require an explicit missed/extra-source penalty and
    are rejected here. Matching is evaluation-only, never used for training.
    """
    p, t = _finite_array(predicted_positions, "predicted positions"), _finite_array(true_positions, "true positions")
    pq, tq = _finite_array(predicted_Q, "predicted Q"), _finite_array(true_Q, "true Q")
    if p.ndim != 2 or p.shape[1] != 2 or t.shape != p.shape or pq.shape != (len(p),) or tq.shape != (len(p),):
        raise ValueError("Require equal K sources: positions[K,2] and strengths[K].")
    if (pq < 0).any() or (tq < 0).any():
        raise ValueError("Strengths must be nonnegative.")
    distances = np.linalg.norm(p[:, None, :]-t[None, :, :], axis=-1)
    pred_ids, true_ids = linear_sum_assignment(distances)
    pairs = [dict(predicted_index=int(i), true_index=int(j), localization_error=float(distances[i,j]),
                  relative_strength_error=relative_source_strength_error(pq[i], tq[j])) for i,j in zip(pred_ids,true_ids)]
    strengths = [pair["relative_strength_error"] for pair in pairs]
    return {"matching": "minimum total Euclidean distance", "pairs": pairs,
            "mean_localization_error": float(np.mean([pair["localization_error"] for pair in pairs])),
            "mean_relative_strength_error": float(np.mean(strengths)) if all(s is not None for s in strengths) else None}


def _finite_array(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite.")
    return array


def source_localization_error(predicted, true) -> float:
    r"""Euclidean distance sqrt((x_hat-x_true)²+(y_hat-y_true)²).

    Both coordinates must have shape [2] and use the same length units.
    This is a single-source metric; multiple sources require explicit matching.
    """
    predicted, true = _finite_array(predicted, "predicted position"), _finite_array(true, "true position")
    if predicted.shape != (2,) or true.shape != (2,):
        raise ValueError("Each source position must have shape [2], ordered [x,y].")
    return float(np.linalg.norm(predicted-true))


def relative_source_strength_error(predicted: float, true: float) -> float | None:
    """Absolute relative error |Q_hat-Q_true|/|Q_true|, as a fraction.

    Emission strengths must be nonnegative scalars. For zero reference strength,
    relative error is undefined and returned as None (JSON null), even if both
    strengths are zero. Q denotes peak Gaussian intensity, not total emissions.
    """
    predicted, true = _finite_array(predicted, "predicted strength"), _finite_array(true, "true strength")
    if predicted.ndim != 0 or true.ndim != 0 or predicted < 0 or true < 0:
        raise ValueError("Source strengths must be nonnegative scalars.")
    return float(abs(predicted-true)/abs(true)) if true != 0 else None


def concentration_metrics(predicted, true) -> dict[str, float | None]:
    r"""Return relative L2, MAE and RMSE over all supplied grid entries.

    L2_relative=||C_hat-C||₂/||C||₂, MAE=mean(|C_hat-C|), and
    RMSE=sqrt(mean((C_hat-C)²)). All entries receive equal weight; intended for
    a uniform space-time grid. Shapes must match exactly, with no broadcasting.
    Relative L2 is dimensionless (fraction); MAE/RMSE have concentration units.
    A zero-norm reference yields None for relative L2, not a misleading zero.
    Negative finite predictions are retained without clipping.
    """
    predicted, true = _finite_array(predicted, "predicted concentration"), _finite_array(true, "true concentration")
    if predicted.shape != true.shape:
        raise ValueError("Concentration shapes must match exactly.")
    error = predicted-true
    denominator = np.linalg.norm(true.ravel())
    result = {"relative_l2": float(np.linalg.norm(error.ravel())/denominator) if denominator > 0 else None,
              "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2)))}
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite concentration metrics (possible numerical overflow).")
    return result
