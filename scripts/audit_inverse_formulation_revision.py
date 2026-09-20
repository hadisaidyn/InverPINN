"""Read-only post-fit audit of all Revision-7 artifacts, never model updates.

This additional integrity layer is independent of the training writer. It
recomputes concentration/source metrics, checks truth against the frozen plan,
and verifies observation identity, exact recipes and optimizer step counts.
Only reporting/plotting code may differ from the archived fitting execution.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from inverpinn.data.development_v3 import IDS, development_plan, manifest, verify, verify_consumed
from inverpinn.data.single_source_benchmark import object_hash
from inverpinn.data.synthetic_reference import ROOT, load_model_inputs, sha256, write_json
from inverpinn.evaluation.single_source_benchmark import load_evaluation_truth, physical_diagnostics, predict_snapshots
from inverpinn.evaluation.observability import forward
from inverpinn.training.hybrid_source import numerical_field, observation_profile
from inverpinn.training.single_source import SparseTrainingData
from run_inverse_formulation_revision import load_candidate, independent_field_Q


def verify_truth_against_plan(truth, generation, planned_source):
    """Truth JSON cannot change independently of a locked generation plan."""
    if object_hash(truth)!=generation["source_hash"] or object_hash(truth)!=object_hash(planned_source):
        raise ValueError("Source truth differs from its frozen plan/manifest.")


def checked_difference(actual, expected, name):
    if actual is None or expected is None or not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError(f"Missing/nonfinite audited value: {name}")
    difference=abs(float(actual)-float(expected))
    if not math.isclose(float(actual),float(expected),rel_tol=1e-11,abs_tol=1e-12):
        raise ValueError(f"Recomputed {name} differs: {actual} vs {expected}")
    return difference


def audit_run(root,candidate,identifier,seed):
    torch.set_num_threads(1)
    config=yaml.safe_load((root/"protocol.yaml").read_text())
    frozen=yaml.safe_load((root/"B_revised_frozen.yaml").read_text())
    run=root/"runs"/candidate/identifier/f"seed_{seed}"
    row=json.loads((run/"metrics.json").read_text())
    attempt,generation,truth,reference_path=load_evaluation_truth(root,identifier,run)
    planned=next(r for r in json.loads((root/"scenario_plan.json").read_text()) if r["scenario_id"]==identifier)
    verify_truth_against_plan(truth,generation,planned["source"])
    recipe=dict(training=frozen["inference"]["training"],formulation=frozen["inference"]["formulation"],sigma=.08,seed=seed)
    saved_config=yaml.safe_load((run/"config.yaml").read_text())
    if saved_config["recipe"]!=recipe or saved_config["candidate"]!=config["candidates"][candidate] or saved_config["profiling"]!=config["profiling"]:
        raise ValueError("Run recipe differs from the preregistered candidate.")
    if row["optimization_seed"]!=seed or row["primary"]!=(seed==42) or row["scenario_id"]!=identifier:
        raise ValueError("Seed/primary identity changed.")
    if row["observations_hash"]!=generation["observations_hash"] or sha256(root/"datasets"/identifier/"inputs/observations.npz")!=generation["observations_hash"]:
        raise ValueError("Observation mismatch.")
    if not attempt["computational_success"]:
        if row["recovery_success"] or row["evaluation_success"]:
            raise ValueError("A failed fit was promoted to a success.")
        return dict(candidate=candidate,scenario_id=identifier,seed=seed,failed_fit_retained=True,maximum_metric_difference=0.)
    estimate=json.loads((run/"source_estimate.json").read_text())
    if set(estimate)!={"x_s","y_s","Q"} or not all(math.isfinite(v) for v in estimate.values()) or not (0<=estimate["x_s"]<=1 and 0<=estimate["y_s"]<=1 and estimate["Q"]>0):
        raise ValueError("Invalid physical source estimate.")
    inputs=load_model_inputs(root/"datasets"/identifier)
    data=SparseTrainingData(inputs["positions"],inputs["times"],inputs["measurements"],**config["physics"])
    differences=[]
    for key in ("x","y","Q"):
        truth_key={"x":"x_s","y":"y_s","Q":"Q"}[key]
        differences.append(checked_difference(row[f"{key}_true"],truth[truth_key],f"{key} truth"))
        differences.append(checked_difference(row[f"{key}_pred"],estimate[truth_key],f"{key} estimate"))
    metrics=dict(localization_error=math.hypot(estimate["x_s"]-truth["x_s"],estimate["y_s"]-truth["y_s"]),
        relative_strength_error=abs(estimate["Q"]-truth["Q"])/abs(truth["Q"]),signed_Q_error=estimate["Q"]-truth["Q"])
    coupling={}
    if candidate!="J3":
        saved=torch.load(run/"model.pt",map_location="cpu",weights_only=True)
        if saved["recipe"]!=recipe or saved["candidate"]!=candidate or attempt["completed_updates"]!=12000:
            raise ValueError("Checkpoint recipe/update count changed.")
        state=saved["optimizer_state_dict"]
        groups={"joint":state} if candidate=="J0" else state
        expected_steps={"field":9000,"source":3000} if candidate=="J2" else {"joint":12000}
        if groups.keys()!=expected_steps.keys():
            raise ValueError("Optimizer block names changed.")
        for block,optimizer in groups.items():
            if not optimizer["state"] or {int(v["step"]) for v in optimizer["state"].values()}!={expected_steps[block]}:
                raise ValueError("Incorrect optimizer block update count.")
            for group in optimizer["param_groups"]:
                if group["lr"]!=.001 or tuple(group["betas"])!=(.9,.999) or group["eps"]!=1e-8 or group["weight_decay"]!=0 or group["amsgrad"] or group["maximize"]:
                    raise ValueError("Optimizer hyperparameters changed.")
        model,source=load_candidate(run/"model.pt",candidate)
        for key,value in source.estimates().items():
            differences.append(checked_difference(value,estimate[key],f"checkpoint {key}"))
        if candidate in ("J1","J2") and "raw_Q" in dict(source.named_parameters()):
            raise ValueError("Profiled Q was an independent optimizer parameter.")
        physical=physical_diagnostics(model,source,config["physics"],.8,config["diagnostics"])
        # Post-fit explanatory check, not a selection endpoint. It compares
        # the neural readings with the independent numerical readings from
        # the SAME inferred source. The gap includes 201-grid discretization.
        h=forward([estimate["x_s"],estimate["y_s"],1.],data.positions.numpy(),data.times.numpy(),config["physics"],config["inference_solver"],.08)
        observations=data.measurements.numpy().reshape(-1)
        with torch.no_grad():
            neural=model(inputs["features"].float()).double().numpy().reshape(-1)
        q_observed=float(h@observations/(h@h))
        metrics["conditional_Q_at_estimated_location"]=q_observed
        metrics["PDE_Q_at_fixed_field"]=independent_field_Q(model,source,data,config["diagnostics"])
        metrics["field_observation_Q_gap"]=metrics["PDE_Q_at_fixed_field"]-q_observed
        metrics["sensor_mse"]=float(np.mean((neural-observations)**2))
        numerical=estimate["Q"]*h
        coupling=dict(neural_sensor_rmse=float(np.sqrt(metrics["sensor_mse"])),
            forward_from_neural_source_sensor_rmse=float(np.sqrt(np.mean((numerical-observations)**2))),
            neural_forward_sensor_gap_relative_l2=float(np.linalg.norm(neural-numerical)/np.linalg.norm(observations)),
            diagnostic_label="postfit_201_grid_consistency_includes_discretization_not_selection")
    else:
        parent=json.loads((root/"runs/J0"/identifier/f"seed_{seed}/source_estimate.json").read_text())
        if any(estimate[k]!=parent[k] for k in ("x_s","y_s")):
            raise ValueError("Hybrid changed PINN location.")
        recomputed,_=observation_profile(data,{k:parent[k] for k in ("x_s","y_s")},config["inference_solver"],.08)
        differences.append(checked_difference(estimate["Q"],recomputed["Q"],"observation-only hybrid Q"))
        native,_,_,physical=numerical_field(data,estimate,config["inference_solver"],.08,config["evaluation_times"])
    with np.load(reference_path,allow_pickle=False) as ref:
        if not np.array_equal(ref["measurements"],inputs["measurements"].numpy()) or not np.array_equal(ref["sensor_positions"],inputs["positions"].numpy()):
            raise ValueError("Inputs differ from the zero-noise fixed-sensor reference.")
        truth_field=ref["fields"]
        predicted=(F.interpolate(torch.tensor(native)[:,None],size=truth_field.shape[-2:],mode="bilinear",align_corners=True)[:,0].numpy()
            if candidate=="J3" else predict_snapshots(model,ref["x"],ref["y"],ref["field_times"],8192))
        error=predicted-truth_field
        # Direct independent reductions, not the training writer's metric helper.
        metrics.update(relative_l2=float(np.sqrt(np.sum(error**2)/np.sum(truth_field**2))),
            rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(abs(error))),
            negative_fraction=float(np.mean(predicted<0)),minimum_concentration=float(predicted.min()))
        with np.load(run/"predictions.npz",allow_pickle=False) as preview:
            if not np.array_equal(preview["fields"],predicted[:,::4,::4]) or not np.array_equal(preview["reference"],truth_field[:,::4,::4]):
                raise ValueError("Saved plot preview differs from its checkpoint/reference.")
    for key in ("pde_rmse","pde_mae","bc_rmse","bc_maxabs","ic_rmse","ic_maxabs"):
        metrics[key]=physical[key]
    for key,value in metrics.items():
        differences.append(checked_difference(value,row[key],key))
    expected_recovery=metrics["localization_error"]<=.08 and metrics["relative_strength_error"]<=.2
    if row["recovery_success"]!=expected_recovery:
        raise ValueError("Recovery threshold changed.")
    return dict(candidate=candidate,scenario_id=identifier,seed=seed,failed_fit_retained=False,
        maximum_metric_difference=max(differences),checkpoint_hash=sha256(run/("forward_checkpoint.npz" if candidate=="J3" else "model.pt")),**coupling)


def audit(root):
    if (root/"integrity_audit.json").exists():
        raise FileExistsError("Completed integrity audit is immutable.")
    config,plan,_=verify(root);manifest(root)
    if development_plan(config)!=plan:
        raise ValueError("Sampling plan cannot be reproduced.")
    code=json.loads((root/"fitting_execution/provenance.json").read_text())["source_hashes"]
    changed=[p for p,h in code.items() if sha256(ROOT/p)!=h]
    allowed={"scripts/report_inverse_formulation_revision.py","src/inverpinn/visualization/inverse_formulation.py"}
    if set(changed)-allowed:
        raise ValueError(f"Fitting/selection implementation changed: {changed}")
    for item in plan:
        case=root/"datasets"/item["scenario_id"]
        source=json.loads((case/"truth/source.json").read_text())
        generation=json.loads((case/"generation.json").read_text())
        verify_truth_against_plan(source,generation,item["source"])
        for grid in (161,321,641):
            ref_config=yaml.safe_load((case/f"truth/grid_{grid}/config.yaml").read_text())
            if ref_config["source"]!=source or ref_config["physics"]!=config["physics"]:
                raise ValueError("Reference configuration differs from source truth/physics.")
    jobs=[(root,name,k,seed) for name in ("J0","J1","J2","J3") for k in IDS
        for seed in ([42,142,242] if k in config["optimization"]["sensitivity_ids"] else [42])]
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(audit_run,*job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            results.append(future.result())
            if i%8==0:
                print(f"AUDITED {i}/{len(jobs)}",flush=True)
    # Published tables must reproduce the preserved individual JSON rows.
    paths=[*(root/f"{n}_results.csv" for n in ("J0","J1","J2","J3")),root/"seed_sensitivity.csv"]
    for path in paths:
        for row in pd.read_csv(path).to_dict("records"):
            raw=json.loads((root/"runs"/row["candidate"]/row["scenario_id"]/f"seed_{row['optimization_seed']}"/"metrics.json").read_text())
            for key in ("observations_hash","primary","recovery_success","computational_success"):
                if row[key]!=raw[key]:
                    raise ValueError("Published table differs from a preserved run.")
            if raw["computational_success"]:
                for key in ("x_pred","y_pred","Q_pred","localization_error","relative_strength_error","relative_l2","rmse","mae","pde_rmse"):
                    checked_difference(row[key],raw[key],f"published {key}")
    result=dict(audited_runs=len(results),failed_runs_retained=sum(r["failed_fit_retained"] for r in results),
        maximum_metric_difference=max(r["maximum_metric_difference"] for r in results),
        historical_files_unchanged=verify_consumed(root),model_and_selection_code_unchanged=True,
        reporting_only_code_changes=changed,table_hashes={p.name:sha256(p) for p in paths},runs=results)
    write_json(root/"integrity_audit.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="runs"},indent=2),flush=True)


if __name__=="__main__":
    audit(ROOT/"results/inverse_formulation_revision")
