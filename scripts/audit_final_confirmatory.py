"""Independent final-run audit; recompute metrics, never refit or update J1."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml
from inverpinn.data.final_confirmatory import IDS,METHODS,verify,manifest,old_inventory
from inverpinn.data.synthetic_reference import ROOT,sha256,write_json,load_model_inputs
from inverpinn.evaluation.single_source_benchmark import load_evaluation_truth,physical_diagnostics,predict_snapshots
from inverpinn.evaluation.final_confirmatory import validate_rows
from inverpinn.physics.finite_difference import solve_advection_diffusion
from run_inverse_formulation_revision import load_candidate


def audit_run(root,method,identifier):
    torch.set_num_threads(1);run=root/"runs"/method/identifier
    config=yaml.safe_load((root/"protocol.yaml").read_text());j1=yaml.safe_load((root/"J1_frozen.yaml").read_text())
    row=json.loads((run/"metrics.json").read_text());attempt,gen,truth,reference=load_evaluation_truth(root,identifier,run)
    saved_config=yaml.safe_load((run/"config.yaml").read_text())
    if saved_config["J1"]!=j1 or saved_config["protocol_hash"]!=sha256(root/"protocol.yaml") or row["optimization_seed"]!=42:
        raise ValueError("Run config/seed changed.")
    if not attempt["computational_success"]:
        if row["recovery_success"] or row["evaluation_success"]:raise ValueError("Failure promoted to success.")
        return dict(method=method,scenario_id=identifier,failed_retained=True,maximum_metric_difference=0.)
    estimate=json.loads((run/"source_estimate.json").read_text());differences=[]
    def compare(key,value):
        if not math.isfinite(value) or row[key] is None or not math.isclose(row[key],value,rel_tol=1e-10,abs_tol=1e-12):raise ValueError(f"Recomputed {key} differs: {identifier}/{method}")
        differences.append(abs(row[key]-value))
    for key,sourcekey in (("x","x_s"),("y","y_s"),("Q","Q")):
        compare(f"{key}_true",truth[sourcekey]);compare(f"{key}_pred",estimate[sourcekey])
    compare("localization_error",math.hypot(estimate["x_s"]-truth["x_s"],estimate["y_s"]-truth["y_s"]))
    compare("relative_strength_error",abs(estimate["Q"]-truth["Q"])/truth["Q"])
    compare("signed_Q_error",estimate["Q"]-truth["Q"]);compare("signed_relative_Q_error",(estimate["Q"]-truth["Q"])/truth["Q"])
    inputs=load_model_inputs(root/"datasets"/identifier)
    if row["observations_hash"]!=gen["observations_hash"]:raise ValueError("Observation identity changed.")
    if method!="classical":
        saved=torch.load(run/"model.pt",weights_only=True,map_location="cpu")
        if saved["recipe"]!=j1["recipe"] or attempt["completed_updates"]!=12000:raise ValueError("Checkpoint recipe/budget changed.")
        state=saved["optimizer_state_dict"];state=state["joint"] if method=="J1" else state
        if {int(v["step"]) for v in state["state"].values()}!={12000}:raise ValueError("Adam update count changed.")
        for group in state["param_groups"]:
            if group["lr"]!=.001 or tuple(group["betas"])!=(.9,.999) or group["eps"]!=1e-8 or group["weight_decay"]!=0 or group["amsgrad"]:raise ValueError("Adam configuration changed.")
        model,source=load_candidate(run/"model.pt","J1" if method=="J1" else "J0")
        if source.estimates()!=estimate:raise ValueError("Source checkpoint/estimate mismatch.")
        if method=="J1" and "raw_Q" in dict(source.named_parameters()):raise ValueError("J1 Q was an Adam parameter.")
        physical=physical_diagnostics(model,source,config["physics"],.8,config["diagnostics"])
        for key in ("pde_rmse","pde_mae","bc_rmse","bc_maxabs","ic_rmse","ic_maxabs"):compare(key,physical[key])
    else:
        best=json.loads((run/"optimizer_result.json").read_text());starts=json.loads((run/"starts.json").read_text())
        winner=min(range(len(starts)),key=lambda i:(starts[i]["sensor_mse"],i))
        if len(starts)!=3 or winner!=best["selected_start"] or any(best[k]!=starts[winner][k] for k in ("x_pred","y_pred","Q_pred","sensor_mse")):
            raise ValueError("Classical start selection changed.")
        for key,sourcekey in (("x_pred","x_s"),("y_pred","y_s"),("Q_pred","Q")):
            if best[key]!=estimate[sourcekey]:raise ValueError("Classical source differs from chosen start.")
    with np.load(reference) as ref:
        if not np.array_equal(ref["measurements"],inputs["measurements"].numpy()) or not np.array_equal(ref["sensor_positions"],inputs["positions"].numpy()):raise ValueError("Zero-noise inputs changed.")
        if method=="classical":
            with np.load(run/"forward_checkpoint.npz") as checkpoint:native=checkpoint["fields"]
            independent=solve_advection_diffusion(**config["inference_solver"],**config["physics"],sources=[estimate|dict(sigma=.08)],
                output_mode="selected_times",output_times=config["evaluation_times"],sensor_positions=inputs["positions"],measurement_times=inputs["times"].tolist())
            if not np.array_equal(native,independent["fields"].numpy()):raise ValueError("Classical forward checkpoint differs from recomputation.")
            observed_error=independent["measurements"].numpy()-inputs["measurements"].numpy()
            if not math.isclose(float(np.mean(observed_error**2)),best["sensor_mse"],rel_tol=1e-10,abs_tol=1e-14):raise ValueError("Classical source does not reproduce selected observation mismatch.")
            predicted=F.interpolate(torch.tensor(native)[:,None],size=ref["fields"].shape[-2:],mode="bilinear",align_corners=True)[:,0].numpy()
        else:predicted=predict_snapshots(model,ref["x"],ref["y"],ref["field_times"],8192)
        error=predicted-ref["fields"]
        compare("relative_l2",float(np.sqrt(np.sum(error**2)/np.sum(ref["fields"]**2))))
        compare("rmse",float(np.sqrt(np.mean(error**2))));compare("mae",float(np.mean(abs(error))))
        compare("negative_fraction",float(np.mean(predicted<0)));compare("minimum_concentration",float(predicted.min()))
        with np.load(run/"predictions.npz") as preview:
            if not np.array_equal(preview["fields"],predicted[:,::4,::4]) or not np.array_equal(preview["reference"],ref["fields"][:,::4,::4]):raise ValueError("Plot preview changed.")
    if row["recovery_success"]!=(row["localization_error"]<=.08 and row["relative_strength_error"]<=.2):raise ValueError("Recovery criterion changed.")
    return dict(method=method,scenario_id=identifier,failed_retained=False,maximum_metric_difference=max(differences),
        checkpoint_hash=sha256(run/("forward_checkpoint.npz" if method=="classical" else "model.pt")))


def audit(root):
    config,plan,_=verify(root);manifest(root)
    if (root/"integrity_audit.json").exists():raise FileExistsError("Completed audit preserved.")
    execution=json.loads((root/"fitting_execution/provenance.json").read_text())
    if any(sha256(ROOT/k)!=v for k,v in execution["source_hashes"].items()):raise ValueError("Execution code changed after fitting began.")
    historical=json.loads((root/"consumed_hashes.json").read_text())
    if old_inventory()!=historical:raise ValueError("Consumed artifacts changed.")
    for item in plan:
        for candidate in config["reference"]["candidates"]:
            saved=yaml.safe_load((root/"datasets"/item["scenario_id"]/f"truth/grid_{candidate['nx']}/config.yaml").read_text())
            if saved["source"]!=item["source"] or saved["physics"]!=config["physics"]:raise ValueError("Reference physics/source differs from frozen plan.")
    frames=[pd.read_csv(root/f"{m}_raw.csv") for m in METHODS];frame=pd.concat(frames,ignore_index=True);validate_rows(frame)
    for row in frame.to_dict("records"):
        saved=json.loads((root/"runs"/row["method"]/row["scenario_id"]/"metrics.json").read_text())
        for key,value in saved.items():
            if value is None:continue
            if isinstance(value,float):
                if not math.isclose(row[key],value,rel_tol=1e-12,abs_tol=1e-14):raise ValueError("Raw CSV differs from saved metric.")
            elif row[key]!=value and not (value=="" and pd.isna(row[key])):raise ValueError("Raw CSV identity differs.")
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(audit_run,root,m,k) for m in METHODS for k in IDS]
        for i,f in enumerate(as_completed(jobs),1):
            rows.append(f.result())
            if i%10==0:print(f"AUDIT {i}/90",flush=True)
    write_json(root/"integrity_audit.json",dict(audited_runs=90,historical_files_unchanged=len(historical),execution_code_unchanged=True,
        maximum_metric_difference=max(r["maximum_metric_difference"] for r in rows),failed_runs=sum(r["failed_retained"] for r in rows),
        result_hashes={f"{m}_raw.csv":sha256(root/f"{m}_raw.csv") for m in METHODS},runs=rows))


if __name__=="__main__":audit(ROOT/"results/final_confirmatory")
