"""Summarize all Revision-7 outcomes under the predeclared selection rule."""

import argparse
import json
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from inverpinn.data.development_v3 import IDS, manifest, verify, verify_consumed
from inverpinn.data.single_source_benchmark import write_csv
from inverpinn.data.synthetic_reference import ROOT, sha256, write_json, load_model_inputs
from inverpinn.evaluation.inverse_formulation import candidate_statistics, selection_checks, advancement_checks, trajectory_summary, METRICS
from inverpinn.visualization.inverse_formulation import figures
from audit_inverse_formulation_revision import verify_truth_against_plan
from run_model_diagnosis import snapshot


def require_audit(root,plan):
    """Require complete checkpoint audits and recheck truth/table immutability."""
    audit=json.loads((root/"integrity_audit.json").read_text())
    if audit["audited_runs"]!=128 or not audit["model_and_selection_code_unchanged"]:
        raise ValueError("Incomplete model-integrity audit.")
    for name,digest in audit["table_hashes"].items():
        if sha256(root/name)!=digest:
            raise ValueError("Audited result table changed.")
    for item in plan:
        case=root/"datasets"/item["scenario_id"]
        verify_truth_against_plan(json.loads((case/"truth/source.json").read_text()),
            json.loads((case/"generation.json").read_text()),item["source"])
    return audit


def oracle_location_diagnostic(root,frame,config):
    """Post-hoc explanatory oracle, NOT J3 or a model-selection candidate.

    The pre-fit information calculation saved J_Q=h*s_Q/C*, exactly because
    the zero-IC/BC forward solution is linear in Q. Recover h at TRUE location
    from that column and profile against readings. True Q is used only for
    scoring. This diagnoses mislocalization versus amplitude ambiguity without
    changing any fitted estimate or using consumed historical outcomes.
    """
    rows=[]
    for row in frame[frame.candidate=="J0"].itertuples():
        inputs=load_model_inputs(root/"datasets"/row.scenario_id)
        with np.load(root/"information"/row.scenario_id/"jacobian.npz") as arrays:
            h=arrays["scaled_jacobian"][:,2]*config["scaling"]["concentration_scale"]/config["scaling"]["parameter_scales"][2]
        observed=inputs["measurements"].numpy().reshape(-1)
        q=max(np.finfo(float).tiny,float(h@observed/(h@h)))
        rows.append(dict(scenario_id=row.scenario_id,oracle_true_location=True,used_for_selection=False,
            Q_true=row.Q_true,Q_oracle=q,signed_Q_error=q-row.Q_true,relative_strength_error=abs(q-row.Q_true)/abs(row.Q_true)))
    write_csv(root/"oracle_location_Q_diagnostic.csv",rows)
    oracle=pd.DataFrame(rows)
    return dict(label="posthoc_true_location_oracle_not_an_inference_candidate",count=len(rows),
        median_relative_Q_error=float(oracle.relative_strength_error.median()),mean_relative_Q_error=float(oracle.relative_strength_error.mean()),
        worst_relative_Q_error=float(oracle.relative_strength_error.max()),mean_signed_Q_error=float(oracle.signed_Q_error.mean()),
        under=int((oracle.signed_Q_error < -1e-6).sum()),over=int((oracle.signed_Q_error > 1e-6).sum()))


def rank_association(x,y):
    """Report undefined rank correlations explicitly, never as zero or NaN.

    Spearman correlation divides by both rank standard deviations. A constant
    input has zero rank variance, so there is no defined coefficient. Failed
    evaluations are retained elsewhere; this exploratory statistic reports its
    finite-pair count and exclusion count instead of silently imputing values.
    """
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    valid=np.isfinite(x)&np.isfinite(y)
    a,b=x[valid],y[valid]
    result=dict(n=int(valid.sum()),missing_pairs=int((~valid).sum()))
    if len(a)<2 or np.unique(a).size<2 or np.unique(b).size<2:
        return result|dict(rho=None,status="undefined_insufficient_pairs_or_constant_ranks")
    return result|dict(rho=float(spearmanr(a,b).statistic),status="defined_exploratory")


def report(root):
    config,plan,lock=verify(root);manifest(root)
    if (root/"summary.json").exists():
        raise FileExistsError("Completed report preserved, no silent overwrite.")
    audit=require_audit(root,plan)
    execution=root/"reporting_execution"
    execution.mkdir(exist_ok=False);snapshot(execution)
    frames=[]
    for name in ("J0","J1","J2","J3"):
        frame=pd.read_csv(root/f"{name}_results.csv")
        if frame.scenario_id.tolist()!=list(IDS) or not frame.primary.all() or not (frame.optimization_seed==42).all():
            raise ValueError("Primary results incomplete, reordered, or replaced by repeated seeds.")
        frames.append(frame)
    frame=pd.concat(frames,ignore_index=True)
    if frame.groupby("scenario_id").observations_hash.nunique().max()!=1:
        raise ValueError("Candidate observations are not paired.")
    seeds=pd.read_csv(root/"seed_sensitivity.csv")
    for name in ("J0","J1","J2","J3"):
        subset=seeds[seeds.candidate==name]
        pairs={(r.scenario_id,int(r.optimization_seed)) for r in subset.itertuples()}
        expected={(k,s) for k in config["optimization"]["sensitivity_ids"] for s in [42,142,242]}
        if pairs!=expected or len(subset)!=len(expected):
            raise ValueError("Incomplete or duplicated sensitivity runs.")
    info=pd.read_csv(root/"observability.csv")
    tolerance=config["selection"]["Q_equality_tolerance"]
    stats={name:candidate_statistics(group,tolerance) for name,group in frame.groupby("candidate")}
    sensitivity={name:candidate_statistics(group,tolerance) for name,group in seeds.groupby("candidate")}
    checks={name:selection_checks(stats[name],stats["J0"],sensitivity[name],sensitivity["J0"],config["selection"]) for name in ("J1","J2")}
    eligible=[name for name,values in checks.items() if all(values.values())]
    selected=min(eligible,key=lambda n:(-stats[n]["recovered"],stats[n]["relative_strength_error"]["median"],
        stats[n]["localization_error"]["median"],stats[n]["runtime_seconds"]["median"])) if eligible else None
    summary_rows=[];bias=[]
    for name,values in stats.items():
        summary_rows.append(dict(candidate=name,**{f"{m}_{q}":v for m in METRICS for q,v in values[m].items()},
            recovered=values["recovered"],count=values["count"],recovery_rate=values["recovery_rate"],failed_fits=values["failed_fits"],
            negative_fraction_max=values["max_negative_fraction"],BC_max=values["max_BC"],IC_max=values["max_IC"]))
        bias.append(dict(candidate=name,under=values["Q_under"],over=values["Q_over"],equal=values["Q_equal"],missing=values["Q_missing"],
            mean_signed_Q_error=values["signed_Q_error"]["mean"],median_signed_Q_error=values["signed_Q_error"]["median"],
            median_relative_Q_error=values["relative_strength_error"]["median"],worst_relative_Q_error=values["relative_strength_error"]["worst"]))
    write_csv(root/"candidate_summary.csv",summary_rows);write_csv(root/"Q_bias_summary.csv",bias)
    trajectories=[]
    for row in frame[frame.candidate!="J3"].itertuples():
        if row.computational_success:
            run=root/"runs"/row.candidate/row.scenario_id/"seed_42"
            trajectory=trajectory_summary(pd.read_csv(run/"gradient_pathway.csv"),row.conditional_Q_at_estimated_location)
            trajectory.update(candidate=row.candidate,scenario_id=row.scenario_id,
                final_Q=row.Q_pred,final_conditional_gap=abs(row.Q_pred-row.conditional_Q_at_estimated_location),
                PDE_Q_at_fixed_field=row.PDE_Q_at_fixed_field,field_observation_Q_gap=row.field_observation_Q_gap)
            history=pd.DataFrame([json.loads(s) for s in (run/"training_history.jsonl").read_text().splitlines()])
            trajectory["sum_absolute_Q_updates"]=float(history.effective_Q_update.abs().sum())
            trajectory["terminal_Q_update"]=float(row.Q_pred-history["source/Q"].iloc[-1])
            trajectories.append(trajectory)
    write_csv(root/"trajectory_summary.csv",trajectories)
    weak=[]
    joined=frame.merge(info[["scenario_id","information_group"]],on="scenario_id",validate="many_to_one")
    for (name,group),values in joined.groupby(["candidate","information_group"]):
        record=candidate_statistics(values,tolerance)
        weak.append(dict(candidate=name,information_group=group,count=len(values),recovered=record["recovered"],
            localization_median=record["localization_error"]["median"],Q_error_median=record["relative_strength_error"]["median"],
            signed_Q_mean=record["signed_Q_error"]["mean"],L2_median=record["relative_l2"]["median"]))
    write_csv(root/"information_strata_summary.csv",weak)
    ranges=[]
    for (name,identifier),values in seeds.groupby(["candidate","scenario_id"]):
        ranges.append(dict(candidate=name,scenario_id=identifier,count=len(values),recovered=int(values.recovery_success.sum()),
            **{f"{col}_{suffix}":float(value) for col in ("localization_error","Q_pred","relative_l2")
               for suffix,value in (("min",values[col].min()),("max",values[col].max()),("range",values[col].max()-values[col].min()),("sd",values[col].std(ddof=1)))}))
    write_csv(root/"seed_sensitivity_ranges.csv",ranges)
    paired=[]
    baseline=frame[frame.candidate=="J0"].set_index("scenario_id")
    for row in frame[frame.candidate!="J0"].itertuples():
        base=baseline.loc[row.scenario_id]
        paired.append(dict(candidate=row.candidate,scenario_id=row.scenario_id,observations_hash=row.observations_hash,
            **{f"delta_{key}":getattr(row,key)-base[key] for key in ("localization_error","relative_strength_error","relative_l2","rmse","mae","pde_rmse")}))
    write_csv(root/"paired_results.csv",paired)
    consistency=[r for r in audit.get("runs",[]) if r["seed"]==42 and "neural_sensor_rmse" in r]
    coupling={}
    if consistency:
        write_csv(root/"source_field_consistency.csv",consistency)
        for name,group in pd.DataFrame(consistency).groupby("candidate"):
            coupling[name]=dict(neural_sensor_rmse_median=float(group.neural_sensor_rmse.median()),
                forward_from_inferred_source_sensor_rmse_median=float(group.forward_from_neural_source_sensor_rmse.median()),
                neural_forward_sensor_gap_relative_l2_median=float(group.neural_forward_sensor_gap_relative_l2.median()),
                forward_fit_worse_count=int((group.forward_from_neural_source_sensor_rmse>group.neural_sensor_rmse).sum()))
    oracle=oracle_location_diagnostic(root,frame,config)
    hybrid=frame[frame.candidate=="J3"]
    associations={metric:rank_association(hybrid.localization_error,hybrid[metric])
        for metric in ("relative_strength_error","relative_l2")}
    figures(root,frame,info,config)
    preserved=verify_consumed(root)
    advances={n:advancement_checks(s,stats["J0"],config["selection"]) for n,s in stats.items()}
    advances["J3"].update(PDE_median=None,PDE_p90=None)  # noncommensurate diagnostics
    result=dict(development_only=True,counts=dict(primary_cases=24,neural_fits=96,hybrid_fits=32),candidates=stats,sensitivity=sensitivity,
        selection_checks=checks,selected_candidate=selected,B_revised_2_created=False,
        advancement_checks=advances,posthoc_source_field_consistency=coupling,
        posthoc_true_location_oracle=oracle,exploratory_J3_location_error_spearman=associations,
        J3_physics_caveat="Native centered FD, not identical to off-grid neural autograd; hybrid ineligible for B_revised_2 here.",
        J4_run=False,J4_reason="J2 already implements field block, location block, and analytic Q; no distinct fourth intervention preregistered.",
        reference_target_misses=int(frames[0].reference_target_met.eq(False).sum()),
        maximum_reference_error_estimate=float(frames[0].reference_error_estimate.max()),
        historical_files_unchanged=preserved,protocol_hash=lock["protocol_hash"],
        prior_revision_commit=config["preserved_revision_6_commit"],frozen_B_revised_hash=sha256(root/"B_revised_frozen.yaml"),
        fitting_archive_hash=sha256(root/"fitting_execution/source_snapshot.zip"),
        result_hashes={f"{n}_results.csv":sha256(root/f"{n}_results.csv") for n in stats})
    write_json(root/"summary.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    report(ROOT/"results/inverse_formulation_revision")
