"""Prospective confirmation criteria and paired descriptive uncertainty."""
import numpy as np
import pandas as pd
from inverpinn.data.final_confirmatory import IDS,METHODS
from inverpinn.evaluation.single_source_benchmark import distribution,wilson_interval
from inverpinn.evaluation.fresh_blind import bootstrap_difference

METRICS=("localization_error","relative_strength_error","relative_l2","rmse","mae","pde_rmse","pde_mae",
         "runtime_seconds","signed_Q_error","signed_relative_Q_error","negative_fraction","bc_maxabs","ic_maxabs")


def validate_rows(frame):
    """Exactly one paired row per fixed method/case, failures included."""
    for method in METHODS:
        rows=frame[frame.method==method]
        if len(rows)!=30 or set(rows.scenario_id)!=set(IDS) or rows.scenario_id.duplicated().any():
            raise ValueError("Incomplete, duplicated, or substituted primary cases.")
        if not (rows.optimization_seed==42).all():raise ValueError("Primary seed changed.")
    if len(frame)!=90:raise ValueError("Unexpected methods.")
    for key in ("observations_hash","reference_data_hash"):
        if frame[key].isna().any() or (frame.groupby("scenario_id")[key].nunique()!=1).any():raise ValueError("Methods do not share identical observations/reference.")


def statistics(rows,config):
    tol=config["analysis"]["Q_equality_atol"]
    signed=rows.signed_Q_error
    n=len(rows);success=int(rows.recovery_success.sum())
    return dict(count=n,recovered=success,recovery_rate=success/n,recovery_Wilson95=wilson_interval(success,n,.95),
        computational_successes=int(rows.computational_success.sum()),evaluations=int(rows.evaluation_success.sum()),
        Q_under=int((signed < -tol).sum()),Q_over=int((signed > tol).sum()),Q_equal=int((signed.abs()<=tol).sum()),
        Q_missing=int(signed.isna().sum()),metrics={k:distribution(rows[k].tolist(),n) for k in METRICS})


def criteria(summaries,rule):
    """ALL gates, unrounded; missing required outcomes cannot pass."""
    j=summaries["J1"];b=summaries["B_revised"];m=j["metrics"]
    def below(metric,limit):
        value=m[metric]["median"]
        return value is not None and np.isfinite(value) and value<=limit
    complete=all(v["count"]==30 and v["computational_successes"]==v["evaluations"]==30 for v in summaries.values())
    jp,bp=m["pde_rmse"]["median"],b["metrics"]["pde_rmse"]["median"]
    signed=m["signed_relative_Q_error"]["median"]
    return dict(all_runs_evaluable=complete,joint_recovery=j["recovered"]>=rule["min_recovered"],
        median_localization=below("localization_error",rule["median_localization_max"]),
        median_Q_error=below("relative_strength_error",rule["median_relative_Q_max"]),
        median_L2=below("relative_l2",rule["median_L2_max"]),
        signed_relative_Q_bias=signed is not None and abs(signed)<=rule["absolute_median_signed_relative_Q_max"],
        nonuniversal_underestimation=j["Q_under"]<30 and j["Q_missing"]==0,
        positivity=m["negative_fraction"]["count"]==30 and m["negative_fraction"]["max"]==0,
        BC_exact=m["bc_maxabs"]["count"]==30 and m["bc_maxabs"]["max"]==0,
        IC_exact=m["ic_maxabs"]["count"]==30 and m["ic_maxabs"]["max"]==0,
        paired_median_PDE=jp is not None and bp is not None and jp<=rule["PDE_median_max_ratio"]*bp)


def paired(joint,comparator,settings):
    """J1 minus comparator; same scenario resamples across metric intervals."""
    a=joint[joint.method=="J1"].set_index("scenario_id");b=joint[joint.method==comparator].set_index("scenario_id")
    keys=["localization_error","relative_strength_error","relative_l2","rmse","mae","runtime_seconds"]
    if comparator=="B_revised":keys.append("pde_rmse")
    records=[]
    for identifier in IDS:
        row=dict(scenario_id=identifier,observations_hash=a.loc[identifier,"observations_hash"])
        for key in keys:
            x,y=a.loc[identifier,key],b.loc[identifier,key]
            finite=np.isfinite([x,y]).all();delta=float(x-y) if finite else None
            tol=settings["metric_tie_atol"]+settings["metric_tie_rtol"]*max(abs(x),abs(y)) if finite else 0
            row[f"delta_{key}"]=delta
            row[f"comparison_{key}"]="missing" if not finite else ("tied" if abs(delta)<=tol else ("J1_better" if delta<0 else f"{comparator}_better"))
        row["runtime_ratio"]=float(a.loc[identifier,"runtime_seconds"]/b.loc[identifier,"runtime_seconds"])
        records.append(row)
    frame=pd.DataFrame(records);summary={}
    for key in keys:
        counts=frame[f"comparison_{key}"].value_counts()
        summary[key]=bootstrap_difference(frame[f"delta_{key}"],settings["bootstrap_seed"],settings["bootstrap_samples"])|{
            k:int(counts.get(k,0)) for k in ("J1_better",f"{comparator}_better","tied","missing")}
    summary["runtime_ratio"]=distribution(frame.runtime_ratio.tolist(),30)
    return frame,summary
