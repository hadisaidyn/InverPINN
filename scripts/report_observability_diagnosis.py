"""Join completed diagnostic evidence without fitting, selection or model changes."""

import argparse
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import yaml

from inverpinn.data.diagnostic_v2 import (IDS, manifest, require_information_complete,
    verify, verify_consumed)
from inverpinn.data.synthetic_reference import ROOT, sha256, write_json
from inverpinn.data.single_source_benchmark import write_csv
from inverpinn.evaluation.observability import failure_evidence
from inverpinn.evaluation.single_source_benchmark import wilson_interval
from run_observability_diagnosis import correlations, summarize


def paired_inputs(classical,pinn):
    """Every planned case remains, including failures; pairing is by input hash."""
    for frame in (classical,pinn):
        if frame.scenario_id.duplicated().any() or set(frame.scenario_id)!=set(IDS):
            raise ValueError("Expected each of the 30 planned cases, including failures, exactly once.")
    a=classical.set_index("scenario_id").observations_hash.sort_index()
    b=pinn.set_index("scenario_id").observations_hash.sort_index()
    if not a.equals(b):
        raise ValueError("Classical/PINN observations differ.")


def signed_strength(predicted,truth):
    """Predeclared 1e-8 + 1e-6|Q| numerical equality convention, not recovery."""
    delta=np.asarray(predicted,dtype=float)-np.asarray(truth,dtype=float)
    tolerance=1e-8+1e-6*np.abs(truth)
    return dict(underestimates=int(np.sum(delta < -tolerance)),overestimates=int(np.sum(delta > tolerance)),
        approximately_equal=int(np.sum(np.abs(delta)<=tolerance)),missing=int(np.sum(~np.isfinite(delta))),
        **summarize(pd.DataFrame(dict(signed_error=delta,signed_relative_error=delta/np.asarray(truth))),
                    ["signed_error","signed_relative_error"]))


def report(root):
    config,_,_=verify(root)
    frozen_rows=manifest(root)
    require_information_complete(root)
    completion=json.loads((root/"pinn_complete.json").read_text())
    if completion["raw_hash"]!=sha256(root/"pinn_results.csv"):
        raise ValueError("Completed PINN table changed.")
    if (root/"summary.json").exists() or (root/"failure_decomposition.csv").exists():
        raise FileExistsError("Existing diagnostic report preserved; no silent overwrite.")
    preserved=verify_consumed(root)
    tables={name:pd.read_csv(root/f"{name}.csv") for name in (
        "classical_inverse_results","pinn_results","conditional_Q_results","resolution_bias_results",
        "jacobian_metrics","geometry_metrics","map_metrics","envelope_reference","envelope_network")}
    classical,pinn=tables["classical_inverse_results"],tables["pinn_results"]
    paired_inputs(classical,pinn)
    info=json.loads((root/"information_summary.json").read_text())
    combined=tables["geometry_metrics"].merge(tables["jacobian_metrics"],on="scenario_id",validate="one_to_one")
    keys=["recovery_success","computational_success","localization_error","relative_strength_error","relative_l2",
          "rmse","mae","sensor_mse","x_pred","y_pred","Q_pred","signed_Q_error"]
    for name,frame in (("classical",classical),("pinn",pinn)):
        combined=combined.merge(frame[["scenario_id"]+keys].rename(columns={k:f"{name}_{k}" for k in keys}),
                                on="scenario_id",validate="one_to_one")
    conditional=tables["conditional_Q_results"]
    combined=combined.merge(conditional[["scenario_id","relative_Q_error","signed_relative_Q_error"]].rename(
        columns={"relative_Q_error":"conditional_relative_Q_error","signed_relative_Q_error":"conditional_signed_relative_Q_error"}),
        on="scenario_id",validate="one_to_one")
    finest=tables["resolution_bias_results"].query("grid == 321")
    combined=combined.merge(finest[["scenario_id","relative_Q_error"]].rename(
        columns={"relative_Q_error":"conditional_321_relative_Q_error"}),on="scenario_id",validate="one_to_one")
    write_csv(root/"paired_diagnostic_evidence.csv",combined.to_dict("records"))
    categories=pd.DataFrame([dict(scenario_id=row["scenario_id"],
        classical_recovered=row["classical_recovery_success"],pinn_recovered=row["pinn_recovery_success"],
        **failure_evidence(row,info["relative_coverage_thresholds"],config["recovery"])) for row in combined.to_dict("records")])
    write_csv(root/"failure_decomposition.csv",categories.to_dict("records"))
    chain=combined[["scenario_id","Q_true","classical_Q_pred","pinn_Q_pred"]].rename(
        columns={"classical_Q_pred":"Q_classical","pinn_Q_pred":"Q_PINN"})
    chain=chain.merge(conditional[["scenario_id","Q_analytic","amplitude_ratio","relative_observation_l2","signed_observation_bias"]].rename(
        columns={"Q_analytic":"Q_conditional"}),on="scenario_id",validate="one_to_one")
    write_csv(root/"Q_bias_chain.csv",chain.to_dict("records"))
    features=["sensor_rms","maximum_sensor_signal","x_true","y_true","Q_true","nearest_sensor_distance","boundary_distance",
        "downwind_sensor_count","nearest_downwind_distance","min_downwind_crosswind_distance","reachable_downwind_count",
        "plume_corridor_count","x_sensitivity_norm","y_sensitivity_norm","Q_sensitivity_norm","sigma_min",
        "jacobian_condition","Q_effective_information","conditional_relative_Q_error","classical_recovery_success"]
    outcomes=[f"{method}_{key}" for method in ("classical","pinn") for key in ("localization_error","relative_strength_error","relative_l2")]
    corr=correlations(combined,outcomes,features)
    write_csv(root/"diagnostic_associations.csv",corr)
    base_metrics=["localization_error","relative_strength_error","relative_l2","rmse","mae","sensor_mse","signed_Q_error"]
    methods={}
    for name,frame in (("classical",classical),("pinn",pinn)):
        recovered=int(frame.recovery_success.sum())
        extra=[] if name=="classical" else ["pde_rmse","pde_mae","bc_rmse","bc_maxabs","ic_rmse","ic_maxabs","negative_fraction","minimum_concentration"]
        methods[name]=dict(total=len(frame),computational_successes=int(frame.computational_success.sum()),
            recovered=recovered,recovery_rate=recovered/len(frame),
            Wilson95_descriptive_only=wilson_interval(recovered,len(frame),config["recovery"]["confidence_level"]),
            metrics=summarize(frame,base_metrics+extra),
            Q_direction=signed_strength(frame.Q_pred,frame.Q_true))
    overlap={f"classical_{a}_pinn_{b}":int(((combined.classical_recovery_success==a)&(combined.pinn_recovery_success==b)).sum())
             for a in (True,False) for b in (True,False)}
    failed=categories.loc[~(categories.classical_recovered & categories.pinn_recovered)]
    pinn_failed=categories.loc[~categories.pinn_recovered]
    def attribution(frame):
        names=["observation/identifiability limitation","numerical forward-model mismatch","PINN optimization/model limitation","mixed/uncertain"]
        return dict(denominator=len(frame),counts={k:int((frame.primary_evidence_attribution==k).sum()) for k in names},
            fractions={k:float((frame.primary_evidence_attribution==k).mean()) if len(frame) else None for k in names})
    resolution_counts={str(g):signed_strength(group.Q_analytic,group.Q_true)
                       for g,group in tables["resolution_bias_results"].groupby("grid")}
    reference=tables["envelope_reference"].query("scope == 'active' and time == 0.8")
    network=tables["envelope_network"].query("scope == 'active' and time == 0.8")
    envelope=reference.merge(network,on="scenario_id",suffixes=("_reference","_network"),validate="one_to_one").merge(
        combined[["scenario_id","pinn_relative_strength_error","pinn_localization_error"]],on="scenario_id",validate="one_to_one")
    envelope_features=["required_positive_latent_q95","required_raw_latent_q95","transform_gain_q05_reference",
        "positive_latent_q95","raw_latent_q95","transform_gain_q05_network","raw_spatial_gradient_q95","C_spatial_gradient_q95"]
    envelope_correlations=correlations(envelope,["pinn_relative_strength_error","pinn_localization_error","x_true","y_true"],envelope_features)
    write_csv(root/"envelope_associations.csv",envelope_correlations)
    from inverpinn.visualization.observability import final_figures
    final_figures(root,config,combined,chain,categories,tables["envelope_reference"],tables["envelope_network"])
    summary=dict(experiment="diagnostic_v2",completed_at_utc=datetime.now(timezone.utc).isoformat(),
        design="30-point fixed Latin hypercube; diagnostic development, not fresh blind validation",
        preserved_files=preserved,model_config_sha256=config["models"]["B_revised"]["sha256"],
        methods=methods,overlap=overlap,information=info,resolution_Q_direction=resolution_counts,
        reference_target_misses=[r["scenario_id"] for r in frozen_rows if not r["reference_target_met"]],
        maximum_reference_error_estimate=max(r["max_reference_error_estimate"] for r in frozen_rows),
        failure_evidence=dict(union_failures=attribution(failed),PINN_failures=attribution(pinn_failed),
            nonexclusive_flag_counts={f"flag_{k}":int(categories[f"flag_{k}"].sum()) for k in "ABCDE"},
            causal_claim=False,weak_information_is_relative_map_quartile=True),
        Q_bias_chain={key:signed_strength(chain[key],chain.Q_true) for key in ("Q_conditional","Q_classical","Q_PINN")},
        jacobian_all=summarize(tables["jacobian_metrics"],["sigma_min","sigma_max","jacobian_condition",
            "x_sensitivity_norm","y_sensitivity_norm","Q_sensitivity_norm","Q_effective_information",
            "physical_x_std_proxy","physical_y_std_proxy","physical_Q_std_proxy"]),
        envelope_active_final=summarize(envelope,envelope_features),associations=corr,envelope_associations=envelope_correlations,
        limitations=["Local Jacobian is not global identifiability or posterior uncertainty.",
            "Unit-normalized noise is an information convention; no measurement noise was added.",
            "LHS diagnostic cases are not IID: Wilson intervals are descriptive, not design-valid confidence guarantees.",
            "A multistart optimizer is not a global inverse oracle certificate.",
            "Finite-difference inference bias cannot by itself establish PINN bias causality.",
            "Coordinate gradients are not optimizer parameter gradients or an intervention on the envelope.",
            "Retained reference-target misses are Richardson estimates, not certified error bounds."])
    write_json(root/"summary.json",summary)
    write_csv(root/"summary.csv",[dict(method=name,metric=metric,**stats)
        for name,value in methods.items() for metric,stats in value["metrics"].items()])
    verify(root);manifest(root);require_information_complete(root)
    write_json(root/"report_complete.json",dict(summary_hash=sha256(root/"summary.json"),
        files={str(p.relative_to(root)):sha256(p) for p in sorted(root.iterdir()) if p.is_file()}))
    print("Diagnostic report complete; no model or sensor changes performed.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/observability_diagnosis.yaml")
    args=parser.parse_args()
    settings=yaml.safe_load((ROOT/args.config).read_text())
    report(ROOT/settings["output_dir"])
