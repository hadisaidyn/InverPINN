"""Run predeclared development-only information diagnostics and J0--J3 fits.

Stages: information, J0, J1, J2, J3, sensitivity. Results never select seeds
or trigger retries. Partial failures are preserved; unexpected bugs stop work.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from inverpinn.data.development_v3 import IDS, case_path, install_guard, manifest, verify
from inverpinn.data.single_source_benchmark import write_csv
from inverpinn.data.synthetic_reference import ROOT, load_model_inputs, sha256, write_json
from inverpinn.evaluation.metrics import concentration_metrics
from inverpinn.evaluation.inverse_formulation import trajectory_summary
from inverpinn.evaluation.observability import unit_operator, jacobian_stability, information_metrics, geometry
from inverpinn.evaluation.single_source_benchmark import parameter_metrics, physical_diagnostics, predict_snapshots, recovery_decision, diagnostic_points
from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
from inverpinn.physics.profiled_source import ProfiledGaussianSource, profile_Q
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.training.development_candidates import fit_candidate
from inverpinn.training.hybrid_source import observation_profile, numerical_field
from inverpinn.training.inverse_formulation import GradientPathway, fit_profiled
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings
from inverpinn.visualization.single_source_benchmark import training_figures
from run_fresh_blind_benchmark import load_fit, recipe_for
from run_model_diagnosis import snapshot


def load_sparse(root, identifier):
    inputs=load_model_inputs(case_path(root,identifier))
    physics=inputs["physics"]
    if physics["initial_condition"]!="C=0 everywhere" or physics["boundary_condition"]!="C=0 on all four edges":
        raise ValueError("Only declared homogeneous IC/BC allowed.")
    return inputs,SparseTrainingData(inputs["positions"],inputs["times"],inputs["measurements"],**physics["transport"])


def information_case(root, identifier):
    """Oracle sensitivity/geometry for description, never passed into training."""
    install_guard(root)
    config=yaml.safe_load((root/"protocol.yaml").read_text())
    output=root/"information"/identifier
    if (output/"results.json").exists():
        return identifier
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    inputs,data=load_sparse(root,identifier)
    truth=json.loads(case_path(root,identifier,"truth/source.json").read_text())
    theta=np.array([truth[k] for k in ("x_s","y_s","Q")])
    response=unit_operator(data.positions.numpy(),data.times.numpy(),config["physics"],config["inference_solver"],config["source_family"]["sigma"])
    jac,steps=jacobian_stability(theta,response,config["scaling"],config["sensitivity"])
    row=dict(scenario_id=identifier,sensor_rms=float(data.measurements.square().mean().sqrt()),
        **information_metrics(jac,config["scaling"]["parameter_scales"]),
        finite_difference_stable=all(r["stable"] for r in steps),
        **geometry(theta[:2],data.positions.numpy(),config["physics"],truth["sigma"],.8))
    np.savez_compressed(output/"jacobian.npz",scaled_jacobian=jac)
    write_json(output/"steps.json",steps);write_json(output/"results.json",row)
    return identifier


def pool_jobs(fn,jobs,workers,label):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fn,*job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            print(f"{label} COMPLETE {i}/{len(jobs)}: {future.result()}",flush=True)


def information(root):
    config,_,_=verify(root);manifest(root)
    if (root/"information_complete.json").exists():
        raise FileExistsError("Information strata already frozen.")
    pool_jobs(information_case,[(root,k) for k in IDS],config["runtime"]["workers"],"INFORMATION")
    rows=[json.loads((root/"information"/k/"results.json").read_text()) for k in IDS]
    threshold=float(np.median([r["sensor_rms"] for r in rows]))
    for row in rows:
        row["information_group"]="lower" if row["sensor_rms"]<=threshold else "higher"
    write_csv(root/"observability.csv",rows)
    write_json(root/"information_complete.json",dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        median_sensor_rms=threshold,split_before_PINN_fitting=True,observability_hash=sha256(root/"observability.csv")))


def require_information(root):
    record=json.loads((root/"information_complete.json").read_text())
    if sha256(root/"observability.csv")!=record["observability_hash"]:
        raise ValueError("Frozen information split changed.")


def load_candidate(path,candidate):
    if candidate=="J0":
        return load_fit(path)
    saved=torch.load(path,map_location="cpu",weights_only=True)
    recipe=saved["recipe"];s=recipe["training"]
    model=ConstrainedConcentrationMLP(.8,s["width"],s["depth"],recipe["formulation"]["constraint"])
    source=ProfiledGaussianSource(s["initial_x"],s["initial_y"],s["initial_Q"],recipe["sigma"])
    model.load_state_dict(saved["model_state_dict"]);source.load_state_dict(saved["source_state_dict"])
    model.eval()
    return model,source


def independent_field_Q(model,source,data,diagnostics):
    """Post-fit optimum for fixed neural field/location on diagnostic points.

    This measures PDE consistency of Q, not a source-truth target, and never
    updates the saved source. Uses the already specified evaluation point set.
    Compare separately with the observation-conditioned optimum at this same
    location; a difference indicates field/observation inconsistency.
    """
    points,_,_=diagnostic_points(diagnostics,data.times[-1].item())
    residuals,gaussians=[],[]
    for p in points.split(512):
        p=p.requires_grad_()
        residuals.append(autograd_pde_residual(model(p),p,u=data.u,v=data.v,D=data.D,S=0.).detach())
        gaussians.append(torch.exp(-((p[:,:1]-source.x_s)**2+(p[:,1:2]-source.y_s)**2)/(2*source.sigma**2)).detach())
    return float(profile_Q(torch.cat(residuals),torch.cat(gaussians)))


def fit_one(root,candidate,identifier,seed):
    if identifier not in IDS or candidate not in ("J0","J1","J2","J3"):
        raise ValueError("Undeclared candidate/case forbidden.")
    state=install_guard(root);require_information(root)
    config=yaml.safe_load((root/"protocol.yaml").read_text())
    if seed!=42 and (identifier not in config["optimization"]["sensitivity_ids"] or seed not in config["optimization"]["additional_seeds"]):
        raise ValueError("Undeclared repeated-seed run.")
    run=root/"runs"/candidate/identifier/f"seed_{seed}"
    if (run/"metrics.json").exists():
        return f"{candidate}/{identifier}/{seed} (preserved)"
    run.mkdir(parents=True,exist_ok=False)
    recipe=recipe_for(root,"B_revised",seed)
    known=next(r for r in json.loads((root/"input_manifest.json").read_text()) if r["scenario_id"]==identifier)
    (run/"config.yaml").write_text(yaml.safe_dump(dict(recipe=recipe,candidate=config["candidates"][candidate],
        profiling=config["profiling"],input_hashes=known,protocol_hash=sha256(root/"protocol.yaml")),sort_keys=False))
    attempt=dict(training_terminated=False,computational_success=False,completed_updates=0,failure="")
    start=time.perf_counter();state["fitting_case"]=identifier
    hybrid_diagnostics=None
    try:
        case=case_path(root,identifier)
        if sha256(case/"inputs/observations.npz")!=known["observations_hash"] or sha256(case/"inputs/physics.json")!=known["physics_hash"]:
            raise ValueError("Sparse inputs changed.")
        inputs,data=load_sparse(root,identifier)
        torch.set_num_threads(1)
        if candidate=="J3":
            parent=root/"runs/J0"/identifier/f"seed_{seed}"
            parent_attempt=json.loads((parent/"attempt.json").read_text())
            if not parent_attempt["computational_success"]:
                raise FloatingPointError("Prerequisite J0 failed; no substitute location allowed.")
            location=json.loads((parent/"source_estimate.json").read_text())
            estimate,profile=observation_profile(data,{k:location[k] for k in ("x_s","y_s")},config["inference_solver"],recipe["sigma"])
            native,x,y,hybrid_diagnostics=numerical_field(data,estimate,config["inference_solver"],recipe["sigma"],config["evaluation_times"])
            np.savez_compressed(run/"forward_checkpoint.npz",fields=native,x=x,y=y,times=config["evaluation_times"])
            write_json(run/"observation_profile.json",profile)
            attempt["prerequisite_seconds"]=parent_attempt["training_seconds"]
        else:
            audit=GradientPathway(data,config["gradient_interval"])
            settings=TrainingSettings.from_mapping(recipe["training"])
            logger=logging.getLogger("revision7");logger.setLevel(logging.WARNING)
            previous_Q=settings.initial_Q
            with (run/"training_history.jsonl").open("x") as stream:
                def record(values):
                    nonlocal previous_Q
                    row=dict(values);row["effective_Q_update"]=row["source/Q"]-previous_Q
                    previous_Q=row["source/Q"]
                    stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush()
                    attempt["completed_updates"]=row["epoch"]
                if candidate=="J0":
                    model,source,optimizer,_=fit_candidate(data,settings,recipe["sigma"],seed,recipe["formulation"],logger,record,audit)
                    optimizer_state=optimizer.state_dict()
                else:
                    model,source,optimizer_state,_=fit_profiled(data,settings,recipe["sigma"],seed,recipe["formulation"],
                        config["candidates"][candidate],config["profiling"],logger,record,audit)
            torch.save(dict(model_state_dict=model.state_dict(),source_state_dict=source.state_dict(),
                optimizer_state_dict=optimizer_state,recipe=recipe,candidate=candidate),run/"model.pt")
            write_csv(run/"gradient_pathway.csv",audit.rows)
            estimate=source.estimates()
        write_json(run/"source_estimate.json",estimate)
        attempt["computational_success"]=True
    except FloatingPointError as exc:
        attempt["failure"]=f"Failed fit retained without retry: {exc}"
    except Exception as exc:
        attempt["failure"]=f"Unexpected {type(exc).__name__}: {exc}"
        write_json(run/"review_required.json",dict(error=attempt["failure"]))
        raise
    finally:
        state["fitting_case"]=None
        attempt.update(training_terminated=True,training_seconds=time.perf_counter()-start)
        attempt["runtime_seconds"]=attempt["training_seconds"]+attempt.get("prerequisite_seconds",0.)
        write_json(run/"attempt.json",attempt)
    # Fitting has irreversibly terminated before source/reference truth opens.
    truth=json.loads(case_path(root,identifier,"truth/source.json").read_text())
    generation=json.loads(case_path(root,identifier,"generation.json").read_text())
    row=dict(candidate=candidate,scenario_id=identifier,optimization_seed=seed,primary=seed==42,**attempt,
        x_true=truth["x_s"],y_true=truth["y_s"],Q_true=truth["Q"],observations_hash=known["observations_hash"],
        reference_target_met=generation["reference_target_met"],reference_error_estimate=generation["max_reference_error_estimate"],
        recovery_success=False,evaluation_success=False)
    keys=("x_pred","y_pred","Q_pred","signed_Q_error","localization_error","relative_strength_error","relative_l2","rmse","mae",
          "negative_fraction","minimum_concentration","pde_rmse","pde_mae","bc_rmse","bc_maxabs","ic_rmse","ic_maxabs","sensor_mse")
    row.update(dict.fromkeys(keys))
    if attempt["computational_success"]:
        row.update(x_pred=estimate["x_s"],y_pred=estimate["y_s"],Q_pred=estimate["Q"],
            signed_Q_error=estimate["Q"]-truth["Q"],**parameter_metrics(estimate,truth))
        if candidate!="J3":
            model,source=load_candidate(run/"model.pt",candidate)
            row.update(physical_diagnostics(model,source,config["physics"],.8,config["diagnostics"]),pde_diagnostic_method="independent_neural_autograd")
            with torch.no_grad():
                row["sensor_mse"]=float((model(inputs["features"].float()).double()-inputs["targets"]).square().mean())
            # Observation-conditioned optimum at the *estimated* final location:
            # a post-fit diagnostic, never fed back into neural fitting.
            conditional,profile=observation_profile(data,{k:estimate[k] for k in ("x_s","y_s")},config["inference_solver"],recipe["sigma"])
            write_json(run/"conditional_Q_diagnostic.json",profile | dict(conditional_Q=conditional["Q"],used_for_training=False))
            row["conditional_Q_at_estimated_location"]=conditional["Q"]
            row["PDE_Q_at_fixed_field"]=independent_field_Q(model,source,data,config["diagnostics"])
            row["field_observation_Q_gap"]=row["PDE_Q_at_fixed_field"]-conditional["Q"]
        else:
            row.update(hybrid_diagnostics,sensor_mse=profile["sensor_mse"])
        reference_path=root/generation["reference_path"]
        if sha256(reference_path)!=generation["reference_data_hash"]:
            raise ValueError("Reference changed before evaluation.")
        with np.load(reference_path,allow_pickle=False) as stored:
            reference,x,y,times=[stored[k] for k in ("fields","x","y","field_times")]
            if candidate=="J3":
                predicted=F.interpolate(torch.tensor(native)[:,None],size=reference.shape[-2:],mode="bilinear",align_corners=True)[:,0].numpy()
            else:
                predicted=predict_snapshots(model,x,y,times,config["diagnostics"]["prediction_batch"])
            row.update(concentration_metrics(predicted,reference),negative_fraction=float(np.mean(predicted<0)),minimum_concentration=float(predicted.min()))
            np.savez_compressed(run/"predictions.npz",fields=predicted[:,::4,::4],reference=reference[:,::4,::4],
                                x=x[::4],y=y[::4],times=times)
        row["recovery_success"]=recovery_decision(True,row,config["recovery"])
        row["evaluation_success"]=True
        if candidate!="J3":
            training_figures(run,config["plot_dpi"])
    write_json(run/"metrics.json",row)
    return f"{candidate}/{identifier}/{seed}"


def execution_lock(root):
    path=root/"fitting_execution"
    if not path.exists():
        path.mkdir();snapshot(path)
    else:
        previous=json.loads((path/"provenance.json").read_text())
        if any(sha256(ROOT/name)!=digest for name,digest in previous["source_hashes"].items()):
            raise ValueError("Fitting execution code changed; do not silently resume.")


def train(root,candidate):
    config,_,_=verify(root);manifest(root);require_information(root);execution_lock(root)
    if (root/f"{candidate}_results.csv").exists():
        raise FileExistsError("Completed candidate preserved; no rerun.")
    if candidate!="J0" and not (root/"J0_results.csv").exists():
        raise RuntimeError("Complete frozen J0 baseline before interventions.")
    if candidate!="J0" and not (root/"J0_coupling_diagnostic.json").exists():
        raise RuntimeError("Quantify J0 coupling before intervention fitting.")
    pool_jobs(fit_one,[(root,candidate,k,42) for k in IDS],config["runtime"]["workers"],candidate)
    rows=[json.loads((root/"runs"/candidate/k/"seed_42/metrics.json").read_text()) for k in IDS]
    write_csv(root/f"{candidate}_results.csv",rows)
    if candidate=="J0":
        trajectories=[]
        for row in rows:
            if row["computational_success"]:
                gradients=pd.read_csv(root/"runs/J0"/row["scenario_id"]/"seed_42/gradient_pathway.csv")
                trajectories.append(dict(scenario_id=row["scenario_id"],PDE_Q_at_fixed_field=row["PDE_Q_at_fixed_field"],
                    field_observation_Q_gap=row["field_observation_Q_gap"],**trajectory_summary(gradients,row["conditional_Q_at_estimated_location"])))
        write_json(root/"J0_coupling_diagnostic.json",dict(completed_before_interventions=True,
            derivation_hash=sha256(root/"derivation.md"),trajectories=trajectories,
            observation="Direct data-to-source partial gradients are zero; trajectory drift is descriptive, not a unique causal attribution."))


def sensitivity(root):
    config,_,_=verify(root);require_information(root);execution_lock(root)
    if (root/"seed_sensitivity.csv").exists():
        raise FileExistsError("Completed sensitivity study preserved.")
    for candidate in config["optimization"]["sensitivity_candidates"]:
        if not (root/f"{candidate}_results.csv").exists():
            raise RuntimeError("All primary candidates must complete first.")
        pool_jobs(fit_one,[(root,candidate,k,seed) for k in config["optimization"]["sensitivity_ids"]
            for seed in config["optimization"]["additional_seeds"]],config["runtime"]["workers"],f"{candidate} SENSITIVITY")
    rows=[json.loads((root/"runs"/candidate/k/f"seed_{seed}/metrics.json").read_text())
        for candidate in config["optimization"]["sensitivity_candidates"] for k in config["optimization"]["sensitivity_ids"]
        for seed in [42,*config["optimization"]["additional_seeds"]]]
    write_csv(root/"seed_sensitivity.csv",rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("information","J0","J1","J2","J3","sensitivity"),required=True)
    args=parser.parse_args();root=ROOT/"results/inverse_formulation_revision"
    if args.stage=="information":
        information(root)
    elif args.stage=="sensitivity":
        sensitivity(root)
    else:
        train(root,args.stage)
