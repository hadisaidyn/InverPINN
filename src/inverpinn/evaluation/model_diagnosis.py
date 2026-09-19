"""Development-only paired summaries and predeclared candidate eligibility.

No benchmark data is loaded here. Metric tables are supplied after fitting.
Ten selected development cases do not supply independent confirmatory evidence.
"""

import numpy as np
import pandas as pd

PRIMARY = ("localization_error", "relative_strength_error", "relative_l2")
PHYSICAL = ("negative_fraction", "pde_rmse", "bc_rmse", "ic_rmse")


def candidate_summary(rows):
    """Retain means, medians, maxima, failures and full runtime for each method."""
    records = []
    for candidate, group in rows.groupby("candidate", sort=False):
        row = dict(candidate=candidate, count=len(group),
                   successful=int(group.computational_success.sum()), recovered=int(group.recovery_success.sum()))
        for key in (*PRIMARY, *PHYSICAL, "sensor_mse", "objective", "rmse", "mae", "training_seconds"):
            values = group[key].to_numpy(dtype=float)
            row.update({f"{key}_mean":float(np.mean(values)), f"{key}_median":float(np.median(values)),
                        f"{key}_max":float(np.max(values)), f"{key}_std":float(np.std(values, ddof=1))})
        records.append(row)
    return pd.DataFrame(records)


def select_revised(summary):
    """Apply docs/model_diagnosis_selection.md without an outcome-fitted score.

    Reject any missing/nonfinite metric; all ten cases must complete. Pareto
    filtering is on the three medians, followed by the declared lexicographic
    priority Q, concentration, location, runtime. Returning None is allowed.
    """
    baseline = summary.set_index("candidate").loc["B0"]
    eligible, reasons = [], {}
    for _, row in summary.iterrows():
        name = row["candidate"]
        why = []
        if row["count"] != 10 or row["successful"] != 10:
            why.append("not all ten cases successful")
        numeric = row.drop("candidate").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            why.append("missing/nonfinite diagnostic")
        if row["negative_fraction_max"] != 0:
            why.append("negative concentration")
        for key in ("bc_rmse", "ic_rmse"):
            for statistic in ("mean", "max"):
                field = f"{key}_{statistic}"
                if row[field] > baseline[field]:
                    why.append(f"worse {field}")
        if row["pde_rmse_mean"] > baseline["pde_rmse_mean"]:
            why.append("worse mean physical PDE RMS")
        for key in PRIMARY:
            for statistic in ("mean", "median"):
                field = f"{key}_{statistic}"
                if row[field] > baseline[field]:
                    why.append(f"worse {field}")
        reasons[name] = why
        if not why and name != "B0":
            eligible.append(row)
    pareto = []
    for row in eligible:
        values = np.array([row[f"{k}_median"] for k in PRIMARY])
        dominated = False
        for other in eligible:
            comparison = np.array([other[f"{k}_median"] for k in PRIMARY])
            if np.all(comparison <= values) and np.any(comparison < values):
                dominated = True
        if not dominated:
            pareto.append(row)
    selected = min(pareto, key=lambda r:(r["relative_strength_error_median"], r["relative_l2_median"],
                   r["localization_error_median"], r["training_seconds_mean"]))["candidate"] if pareto else None
    return dict(selected=selected, eligible=[r["candidate"] for r in eligible],
                pareto=[r["candidate"] for r in pareto], exclusion_reasons=reasons)


def violation_correlations(rows):
    """Within-candidate Pearson and Spearman correlations (n=10, descriptive).

    A zero-variance quantity, such as hard BC=0, has undefined correlation.
    Return missing, not a fabricated zero. No pooled cross-candidate inference.
    """
    records = []
    keys = [*PRIMARY, "bc_rmse", "ic_rmse", "negative_fraction", "sensor_mse"]
    for candidate, group in rows.groupby("candidate", sort=False):
        for method in ("pearson", "spearman"):
            matrix = group[keys].corr(method=method)
            for i, left in enumerate(keys):
                for right in keys[i+1:]:
                    records.append(dict(candidate=candidate, method=method, left=left, right=right,
                                        correlation=matrix.loc[left,right], count=len(group)))
    return pd.DataFrame(records)


def amplitude_ray_diagnostic(predictions, observations, weighted_physics_penalty, data_weight):
    r"""Truth-independent analytic diagnostic, NOT a fitted candidate.

    At fixed location and shape, jointly scale (C,Q) -> (a C,a Q). By
    linearity of the PDE and homogeneous IC/BC, all physics MSEs scale as a².
    J(a)=w_d mean((a C_sensor-y)²)+a² P, where P is the weighted PDE+IC+BC
    penalty at a=1. The unconstrained scalar minimizers are
    a_data=<C,y>/<C,C> and a_joint=w_d<C,y>/(w_d<C,C>+P).
    Thus a_joint/a_data=A/(A+P)<=1 for A=w_d<C,C>, P>=0. Residual/constraint
    errors create an amplitude-shrinkage mechanism relative to data alone.
    This mechanism disappears for an exact PDE/IC/BC solution (P=0). It is
    NOT proof that every observed Q error has this sole cause. No source
    truth or dense fields enter these minimizers. The model is not updated.
    """
    predictions, observations = np.asarray(predictions), np.asarray(observations)
    if predictions.shape != observations.shape or predictions.size == 0:
        raise ValueError("Matched nonempty sensor arrays required.")
    if not (np.isfinite(predictions).all() and np.isfinite(observations).all()
            and np.isfinite(weighted_physics_penalty) and weighted_physics_penalty >= 0
            and np.isfinite(data_weight) and data_weight > 0):
        raise ValueError("Finite sensor arrays and nonnegative physics penalty required.")
    square=float(np.mean(predictions**2))
    cross=float(np.mean(predictions*observations))
    if square == 0:
        raise ValueError("Zero field has no identifiable amplitude along this ray.")
    A=data_weight*square
    return dict(data_only_scale=cross/square, joint_objective_scale=data_weight*cross/(A+weighted_physics_penalty),
                objective_shrinkage_ratio=A/(A+weighted_physics_penalty),
                weighted_sensor_prediction_energy=A, weighted_physics_penalty=weighted_physics_penalty)
