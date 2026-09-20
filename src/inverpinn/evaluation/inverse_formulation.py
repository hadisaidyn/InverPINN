"""Predeclared distribution-wide Revision-7 selection, not best-median tuning."""

import numpy as np


METRICS=("localization_error","relative_strength_error","signed_Q_error","relative_l2","rmse","mae","pde_rmse","runtime_seconds")


def distribution(values):
    """All finite outcomes; missing/failed counts must be reported separately."""
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):
        return dict.fromkeys(("mean","median","sd","q25","q75","p90","worst"))
    return dict(mean=float(a.mean()),median=float(np.median(a)),sd=float(a.std(ddof=1)) if len(a)>1 else None,
        q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),worst=float(a.max()))


def candidate_statistics(frame,tolerance):
    """Signed Q bias in physical units; recovery denominator retains failures."""
    result={k:distribution(frame[k]) for k in METRICS}
    bias=frame.signed_Q_error
    def extremum(column,kind):
        values=np.asarray(frame[column],dtype=float);values=values[np.isfinite(values)]
        return float(values.max() if kind=="max" else values.min()) if len(values) else None
    result.update(count=len(frame),failed_fits=int((~frame.computational_success).sum()),
        failed_evaluations=int((~frame.evaluation_success).sum()),recovered=int(frame.recovery_success.sum()),
        recovery_rate=float(frame.recovery_success.sum()/len(frame)),
        Q_under=int((bias < -tolerance).sum()),Q_over=int((bias > tolerance).sum()),
        Q_equal=int((abs(bias)<=tolerance).sum()),Q_missing=int(bias.isna().sum()),
        max_negative_fraction=extremum("negative_fraction","max"),minimum_concentration=extremum("minimum_concentration","min"),
        max_BC=extremum("bc_maxabs","max"),max_IC=extremum("ic_maxabs","max"))
    return result


def advancement_checks(stats,baseline,rule):
    """Individual promised advancement criteria, with explicit PDE margin."""
    if any(stats[k]["median"] is None or baseline[k]["median"] is None for k in METRICS):
        return {k:False for k in ("median_localization","median_Q_error","nonuniversal_underestimation","median_L2",
                                  "positivity","BC_exact","IC_exact","PDE_median","PDE_p90")}
    return dict(median_localization=stats["localization_error"]["median"]<=rule["promising_median_localization_max"],
        median_Q_error=stats["relative_strength_error"]["median"]<=rule["promising_median_Q_relative_max"],
        nonuniversal_underestimation=stats["Q_under"]<stats["count"] and stats["Q_missing"]==0,
        median_L2=stats["relative_l2"]["median"]<=rule["promising_median_L2_max"],
        positivity=stats["max_negative_fraction"]==0,BC_exact=stats["max_BC"]==0,IC_exact=stats["max_IC"]==0,
        PDE_median=stats["pde_rmse"]["median"]<=baseline["pde_rmse"]["median"]+max(rule["material_PDE_absolute_tolerance"],rule["material_PDE_relative_tolerance"]*baseline["pde_rmse"]["median"]),
        PDE_p90=stats["pde_rmse"]["p90"]<=baseline["pde_rmse"]["p90"]+max(rule["material_PDE_absolute_tolerance"],rule["material_PDE_relative_tolerance"]*baseline["pde_rmse"]["p90"]))


def selection_checks(stats,baseline,sensitivity,baseline_sensitivity,rule):
    """All gates must pass; a single improved median never selects a model."""
    checks=advancement_checks(stats,baseline,rule)
    if any(stats[k]["median"] is None or baseline[k]["median"] is None for k in METRICS):
        return checks | dict(fits_complete=False)
    checks.update(fits_complete=stats["failed_fits"]==0 and stats["failed_evaluations"]==0,
        Q_median_improvement=stats["relative_strength_error"]["median"]<=(1-rule["minimum_Q_median_improvement"])*baseline["relative_strength_error"]["median"],
        signed_bias_improvement=abs(stats["signed_Q_error"]["mean"])<=(1-rule["minimum_absolute_signed_bias_improvement"])*abs(baseline["signed_Q_error"]["mean"]),
        recovery_noninferior=stats["recovered"]>=baseline["recovered"],
        sensitivity_recovery=sensitivity["recovered"]>=baseline_sensitivity["recovered"],
        sensitivity_complete=sensitivity["failed_fits"]==0 and sensitivity["failed_evaluations"]==0,
        sensitivity_constraints=sensitivity["max_negative_fraction"]==0 and sensitivity["max_BC"]==0 and sensitivity["max_IC"]==0)
    for metric,quantiles in (("localization_error",("median","p90","worst")),("relative_strength_error",("p90","worst")),
        ("relative_l2",("median","p90","worst")),("rmse",("mean",)),("mae",("mean",))):
        for q in quantiles:
            checks[f"{metric}_{q}_noninferior"]=stats[metric][q]<=rule["maximum_relative_noninferiority"]*baseline[metric][q]
    return checks


def trajectory_summary(gradients,conditional_Q):
    """Descriptive checkpoint-to-checkpoint drift; no post-fit optimization."""
    data=gradients.sort_values("epoch")
    dq=np.diff(abs(data.pre_Q.to_numpy()-conditional_Q))
    dl=np.diff(data.full_sensor_mse.to_numpy())
    improving=dl<0
    away=dq>0
    n=int(improving.sum())
    cosine=data.data_pde_gradient_cosine.dropna()
    return dict(intervals=len(dl),data_improving_intervals=n,data_improves_Q_moves_away=int((improving&away).sum()),
        away_fraction_of_improving=float((improving&away).sum()/n) if n else None,
        conditional_Q=conditional_Q,final_logged_Q=float(data.pre_Q.iloc[-1]),
        final_logged_conditional_gap=float(abs(data.pre_Q.iloc[-1]-conditional_Q)),
        median_data_PDE_cosine=float(cosine.median()) if len(cosine) else None,
        median_x_gradient=float(data.raw_x_gradient_norm.median()),median_y_gradient=float(data.raw_y_gradient_norm.median()),
        source_data_gradient_max=float(data.data_source_gradient_norm.max()))
