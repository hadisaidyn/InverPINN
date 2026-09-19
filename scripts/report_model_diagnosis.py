"""Aggregate the complete frozen revision-4 plan; nominate at most one model.

Reads only revision-4 run artifacts and the ten allowlisted development hashes.
Never reads the old held-out plan, truth, metrics or figures. No new fits run.
"""

import argparse
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import torch
import yaml

from inverpinn.data.development_only import DEVELOPMENT_IDS, development_hashes, install_development_read_guard, sparse_inputs
from inverpinn.data.synthetic_reference import ROOT, sha256, write_json
from inverpinn.evaluation.model_diagnosis import candidate_summary, select_revised, violation_correlations, amplitude_ray_diagnostic
from inverpinn.evaluation.single_source_benchmark import diagnostic_points
from inverpinn.models.mlp import ConcentrationMLP
from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
from inverpinn.physics.diagnostic_source import DiagnosticGaussianSource
from inverpinn.physics.trainable_source import TrainableGaussianSource
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.training.development_candidates import select_multistart, loss_factors
from inverpinn.visualization.model_diagnosis import comparison_figures


def amplitude_diagnostics(config, rows):
    """Reload only development-fitted checkpoints; compute analytic controls."""
    torch.set_num_threads(1)
    output=ROOT/config["output_dir"]
    records=[]
    for _, row in rows[rows.candidate != "M0"].iterrows():
        inputs=sparse_inputs(ROOT/config["benchmark_root"],row.scenario_id)
        stored=torch.load(output/"runs"/row.candidate/row.scenario_id/"model.pt",weights_only=True)
        recipe=stored["recipe"]
        training,formulation=recipe["training"],recipe["formulation"]
        if formulation == "baseline":
            model=ConcentrationMLP([0.,0.,0.],[1.,1.,.8],training["width"],training["depth"])
            source=TrainableGaussianSource(training["initial_x"],training["initial_y"],training["initial_Q"],recipe["sigma"])
            factors=dict(data=1.,pde=1.,initial=1.,boundary=1.)
        else:
            model=ConstrainedConcentrationMLP(.8,training["width"],training["depth"],formulation["constraint"])
            source=DiagnosticGaussianSource(training["initial_x"],training["initial_y"],training["initial_Q"],recipe["sigma"],formulation["Q_parameterization"])
            factors=loss_factors(formulation)
        model.load_state_dict(stored["model_state_dict"])
        source.load_state_dict(stored["source_state_dict"])
        weights={k:training["loss_weights"][k]*factors[k] for k in factors}
        with torch.no_grad():
            predictions=model(inputs["features"].float()).numpy().astype(float)
        penalty=sum(weights[k]*row[f"{prefix}_rmse"]**2 for k,prefix in (("pde","pde"),("initial","ic"),("boundary","bc")))
        record=amplitude_ray_diagnostic(predictions,inputs["targets"].numpy(),penalty,weights["data"])
        interior,_,_=diagnostic_points(config["diagnostics"],.8)
        interior.requires_grad_()
        source_values=source(interior[:,:1],interior[:,1:2])
        r=autograd_pde_residual(model(interior),interior,**inputs["physics"]["transport"],S=source_values)
        # L(C)=Ct+uCx+vCy-D Lap(C), g=unit-peak Gaussian at estimated x,y.
        lhs=(r+source_values).detach().double()
        g=(source_values/source.Q).detach().double()
        conditional_Q=float((lhs*g).mean()/g.square().mean())
        record.update(candidate=row.candidate,scenario_id=row.scenario_id,Q_fitted=row.Q_pred,
                      PDE_only_optimal_Q_given_fixed_field=conditional_Q,
                      Q_after_joint_ray_scale=row.Q_pred*record["joint_objective_scale"])
        records.append(record)
    pd.DataFrame(records).to_csv(output/"amplitude_diagnostics.csv",index=False)


def baseline_reproduction(config):
    """Bitwise checkpoint audit against ONLY the prior development fits."""
    from inverpinn.data.development_only import validate_benchmark_open
    output=ROOT/config["output_dir"]
    records=[]
    keys=("localization_error","relative_strength_error","relative_l2","rmse","mae",
          "negative_fraction","pde_rmse","bc_rmse","ic_rmse")
    for identifier in DEVELOPMENT_IDS:
        current=output/"runs/B0"/identifier
        previous=ROOT/config["benchmark_root"]/"runs"/identifier/f"seed_{config['optimization_seed']}"
        for name in ("model.pt","metrics.json"):
            validate_benchmark_open(ROOT/config["benchmark_root"],previous/name)
        a=torch.load(current/"model.pt",weights_only=True)
        b=torch.load(previous/"model.pt",weights_only=True)
        equal=all(a[k].keys()==b[k].keys() and all(torch.equal(a[k][name],b[k][name]) for name in a[k])
                  for k in ("model_state_dict","source_state_dict"))
        ma,mb=json.loads((current/"metrics.json").read_text()),json.loads((previous/"metrics.json").read_text())
        difference=max(abs(ma[k]-mb[k]) for k in keys)
        if not equal or difference != 0:
            raise AssertionError(f"B0 reproduction differs for {identifier}; investigate before interpretation")
        records.append(dict(scenario_id=identifier,checkpoint_tensors_equal=equal,maximum_metric_difference=difference))
    write_json(output/"baseline_reproduction.json",records)


def report(config):
    install_development_read_guard(ROOT/config["benchmark_root"])
    output=ROOT/config["output_dir"]
    final_lock=output/"report_complete.json"
    if final_lock.exists():
        raise FileExistsError("Completed report is frozen; refusing to silently overwrite it.")
    plan=yaml.safe_load((output/"candidate_protocol/plan.yaml").read_text())
    candidates=yaml.safe_load((output/"candidate_protocol/resolved.yaml").read_text())
    names=["B0"]+[c["name"] for c in candidates]
    rows=[]
    for name in names:
        for identifier in DEVELOPMENT_IDS:
            path=output/"runs"/name/identifier/"metrics.json"
            row=json.loads(path.read_text())
            if row["scenario_id"] != identifier or row["candidate"] != name:
                raise ValueError("Unexpected case label")
            rows.append(row)
    # Separate truth-independent scalar selection from access to metric rows.
    multistart=[]
    for identifier in DEVELOPMENT_IDS:
        subset=[r for r in rows if r["scenario_id"]==identifier and r["candidate"] in plan["multistart"]["candidates"]]
        chosen=select_multistart({r["candidate"]:r["objective"] for r in subset})
        record=next(r.copy() for r in subset if r["candidate"]==chosen)
        record.update(candidate="M0",selected_start=chosen,
                      training_seconds=sum(r["training_seconds"] for r in subset))
        selection_dir=output/"multistart"/identifier
        selection_dir.mkdir(parents=True,exist_ok=False)
        selected_run=output/"runs"/chosen/identifier
        record["selected_checkpoint"]=str((selected_run/"model.pt").relative_to(output))
        write_json(selection_dir/"selection.json",dict(selected_start=chosen,
            objectives={r["candidate"]:r["objective"] for r in subset},
            checkpoint=record["selected_checkpoint"],checkpoint_sha256=sha256(selected_run/"model.pt"),
            plots=str(selected_run.relative_to(output)),total_training_seconds=record["training_seconds"],
            truth_used_for_selection=False))
        (selection_dir/"config.yaml").write_text(yaml.safe_dump(plan["multistart"]))
        write_json(selection_dir/"metrics.json",record)
        multistart.append(record)
    rows=pd.DataFrame(rows+multistart)
    baseline_reproduction(config)
    signal_rows=[]
    for identifier in DEVELOPMENT_IDS:
        inputs=sparse_inputs(ROOT/config["benchmark_root"],identifier)
        observed=inputs["measurements"].numpy()
        energy=float(np.mean(observed**2))
        if energy <= 0:
            raise ValueError("Relative sensor error is undefined for zero observations")
        mask=rows.scenario_id==identifier
        rows.loc[mask,"sensor_relative_l2"]=np.sqrt(rows.loc[mask,"sensor_mse"]/energy)
        baseline_row=rows[mask & (rows.candidate=="B0")].iloc[0]
        positions=inputs["positions"].numpy()
        distance=np.linalg.norm(positions-np.array([baseline_row.x_true,baseline_row.y_true]),axis=1)
        signal_rows.append(dict(scenario_id=identifier,observed_sensor_rms=float(np.sqrt(energy)),
            observed_peak=float(observed.max()),nearest_sensor_to_true_source=float(distance.min()),
            B0_sensor_relative_l2=float(rows.loc[mask & (rows.candidate=="B0"),"sensor_relative_l2"].iloc[0]),
            B0_source_strength_error=baseline_row.relative_strength_error,
            B0_concentration_relative_l2=baseline_row.relative_l2))
    pd.DataFrame(signal_rows).to_csv(output/"sensor_signal_diagnostics.csv",index=False)
    rows.to_csv(output/"raw_results.csv",index=False)
    summary=candidate_summary(rows)
    summary.to_csv(output/"candidate_summary.csv",index=False)
    violation_correlations(rows).to_csv(output/"violation_correlations.csv",index=False)
    decision=select_revised(summary)
    decision.update(selected_at_utc=datetime.now(timezone.utc).isoformat(),
                    selection_rule_sha256=sha256(output/"candidate_protocol/model_diagnosis_selection.md"))
    write_json(output/"selection_decision.json",decision)
    comparisons=[]
    baseline=rows[rows.candidate=="B0"].set_index("scenario_id")
    for name in names[1:]+["M0"]:
        subset=rows[rows.candidate==name].set_index("scenario_id")
        for identifier in DEVELOPMENT_IDS:
            record=dict(candidate=name,scenario_id=identifier)
            for key in ("localization_error","relative_strength_error","relative_l2","negative_fraction",
                        "sensor_mse","objective","pde_rmse","bc_rmse","ic_rmse","training_seconds"):
                record[f"delta_{key}"]=float(subset.loc[identifier,key]-baseline.loc[identifier,key])
            comparisons.append(record)
    pd.DataFrame(comparisons).to_csv(output/"paired_changes.csv",index=False)
    location=[]
    for identifier in DEVELOPMENT_IDS:
        subset=rows[(rows.scenario_id==identifier)&rows.candidate.isin(plan["multistart"]["candidates"])]
        xy=subset[["x_pred","y_pred"]].to_numpy()
        distance=np.sqrt(np.sum((xy[:,None]-xy[None,:])**2,axis=-1))
        location.append(dict(scenario_id=identifier,max_pairwise_source_distance=float(distance.max()),
            Q_range=float(subset.Q_pred.max()-subset.Q_pred.min()),
            localization_range=float(subset.localization_error.max()-subset.localization_error.min()),
            minimum_objective=float(subset.objective.min()),maximum_objective=float(subset.objective.max()),
            selected_start=next(r["selected_start"] for r in multistart if r["scenario_id"]==identifier)))
    pd.DataFrame(location).to_csv(output/"source_initialization_diagnostics.csv",index=False)
    histories=[]
    for name in names:
        for identifier in DEVELOPMENT_IDS:
            path=output/"runs"/name/identifier/"loss_components.csv"
            frame=pd.read_csv(path)
            first,last=frame.iloc[:500],frame.iloc[-500:]
            record=dict(candidate=name,scenario_id=identifier,Q_min=float(frame["source/Q"].min()),
                        Q_final=float(frame["source/Q"].iloc[-1]))
            for key in ("data","pde","initial","boundary","total"):
                record[f"first500_{key}"]=float(first[f"loss/{key}"].mean())
                record[f"last500_{key}"]=float(last[f"loss/{key}"].mean())
            # Last-1000 to previous-1000 ratio measures fixed-budget trends,
            # not a proof of stationarity or a stopping rule.
            record["late_objective_ratio"]=float(frame["loss/total"].iloc[-1000:].mean()/frame["loss/total"].iloc[-2000:-1000].mean())
            histories.append(record)
    pd.DataFrame(histories).to_csv(output/"convergence_diagnostics.csv",index=False)
    prefix_audit=[]
    for short,long in (("B0","O1"),("B4","B5")):
        for identifier in DEVELOPMENT_IDS:
            a=pd.read_csv(output/"runs"/short/identifier/"loss_components.csv")
            b=pd.read_csv(output/"runs"/long/identifier/"loss_components.csv").iloc[:len(a)]
            columns=[c for c in a if c.startswith(("loss/","source/","weighted/"))]
            difference=float(np.abs(a[columns].to_numpy()-b[columns].to_numpy()).max())
            if difference != 0:
                raise AssertionError(f"Longer-training prefix differs: {short}/{long}/{identifier}")
            prefix_audit.append(dict(short=short,long=long,scenario_id=identifier,identical_updates=len(a),
                                     maximum_record_difference=difference))
    write_json(output/"optimizer_prefix_audit.json",prefix_audit)
    amplitude_diagnostics(config,rows)
    comparison_figures(rows,summary,output,config["plot_dpi"])
    after=development_hashes(ROOT/config["benchmark_root"])
    before=json.loads((output/"development_data_hashes.json").read_text())
    if before != after:
        raise RuntimeError("Reference data were modified")
    write_json(output/"reference_integrity.json",dict(unchanged=True,file_count=len(after),
                                                    development_cases=10,
                                                    heldout_access="No held-out inputs used; development path allowlist enforced",
                                                    guard_scope="Python/path guards, not an OS sandbox"))
    if decision["selected"] is not None:
        if decision["selected"] == "M0":
            selected=dict(name="M0",training=config["training"],
                formulation=dict(constraint="unconstrained",Q_parameterization="softplus",scales=None),
                multistart=plan["multistart"],
                starts=[dict(name="B0",training=config["training"])]
                       +[c for c in candidates if c["name"] in plan["multistart"]["candidates"]])
        else:
            selected=next(c for c in candidates if c["name"]==decision["selected"])
        frozen=dict(name="B_revised",selected_development_candidate=selected["name"],
            inference=selected,source_family=dict(kind="gaussian_peak",sigma=config["sigma"],K=1),
            optimization_seed=config["optimization_seed"],optimizer=dict(name="Adam",learning_rate=selected["training"]["learning_rate"],
                betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,maximize=False,foreach=None,
                capturable=False,differentiable=False,fused=None,decoupled_weight_decay=False),
            architecture=dict(input_columns=["x","y","t"],hidden_width=selected["training"]["width"],
                hidden_layers=selected["training"]["depth"],hidden_activation="tanh",output_features=1,
                input_normalization="2*(coordinates-lower)/(upper-lower)-1",concentration_output_scale=1.0,
                linear_initialization="PyTorch nn.Linear defaults under archived package version and seed",
                concentration_transform=selected["formulation"]["constraint"]),
            collocation=dict(strategy="uniform_resampled_each_update",batch=selected["training"]["collocation_batch"],
                             generator_seed=config["optimization_seed"]+1,adaptive=False),
            observation_sampling=dict(strategy="uniform_with_replacement_each_update",batch=selected["training"]["data_batch"]),
            domain=dict(x=[0,1],y=[0,1],t=[0,.8]),physics=dict(u=.35,v=-.15,D=.005),
            dtype="torch.float32",device="cpu",torch_threads=1,deterministic_algorithms=True,
            source_position_transform="sigmoid",
            initialization_strategy=("five fixed starts; truth-independent common objective selection" if selected["name"]=="M0"
                                     else "single fixed truth-independent location and configured Q"),
            selection_rule_sha256=decision["selection_rule_sha256"],
            candidate_source_archive_sha256=sha256(output/"candidate_protocol/source_snapshot.zip"),
            validation_status="development-selected; NEVER evaluated on old held-out set; fresh test deferred")
        path=output/"B_revised.yaml"
        if path.exists() and yaml.safe_load(path.read_text()) != frozen:
            raise FileExistsError("Refusing to overwrite a different B_revised configuration")
        path.write_text(yaml.safe_dump(frozen,sort_keys=False))
    # Archive postprocessing separately: it was developed while the already
    # frozen fitting code ran, and must not replace either execution snapshot.
    import importlib.util
    spec=importlib.util.spec_from_file_location("diagnosis_runner",ROOT/"scripts/run_model_diagnosis.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provenance_dir=output/"report_provenance"
    provenance_dir.mkdir(exist_ok=False)
    module.snapshot(provenance_dir)
    artifact_hashes={str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob("*"))
                     if path.is_file() and path.name not in ("artifact_hashes.json","report_complete.json")}
    write_json(output/"artifact_hashes.json",artifact_hashes)
    write_json(final_lock,dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        raw_results_sha256=sha256(output/"raw_results.csv"),summary_sha256=sha256(output/"candidate_summary.csv"),
        artifact_manifest_sha256=sha256(output/"artifact_hashes.json"),
        selected=decision["selected"]))
    print(summary[["candidate","recovered","localization_error_median","relative_strength_error_median",
                   "relative_l2_median","negative_fraction_mean","pde_rmse_mean","bc_rmse_mean","ic_rmse_mean",
                   "training_seconds_mean"]].to_string(index=False))
    print(json.dumps(decision,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/model_diagnosis.yaml")
    args=parser.parse_args()
    report(yaml.safe_load((ROOT/args.config).read_text()))
