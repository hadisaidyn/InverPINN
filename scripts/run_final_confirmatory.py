"""Execute exactly frozen J1, B_revised and classical inverse, without rescue."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import logging
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml
from inverpinn.data.final_confirmatory import IDS,METHODS,verify,manifest,install_guard
from inverpinn.data.single_source_benchmark import write_csv
from inverpinn.data.synthetic_reference import ROOT,sha256,write_json,load_model_inputs
from inverpinn.evaluation.observability import forward,classical_inverse,unit_operator,jacobian_stability,information_metrics,geometry
from inverpinn.evaluation.single_source_benchmark import load_evaluation_truth,parameter_metrics,physical_diagnostics,predict_snapshots,recovery_decision
from inverpinn.evaluation.metrics import concentration_metrics
from inverpinn.physics.finite_difference import solve_advection_diffusion
from inverpinn.training.single_source import SparseTrainingData,TrainingSettings
from inverpinn.training.inverse_formulation import fit_profiled,GradientPathway
from inverpinn.training.development_candidates import fit_candidate
from inverpinn.visualization.single_source_benchmark import training_figures
from run_inverse_formulation_revision import load_candidate
from run_model_diagnosis import snapshot


def pool(function,jobs,workers,label):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(function,*args) for args in jobs]
        for i,future in enumerate(as_completed(futures),1):
            print(f"{label} {i}/{len(jobs)}: {future.result()}",flush=True)


def bank_point(root,index,location):
    install_guard(root);torch.set_num_threads(1)
    config=yaml.safe_load((root/"protocol.yaml").read_text())
    path=root/"maps"/f"point_{index:03d}.npz"
    if path.exists():raise FileExistsError("Partial classical bank needs review, not replacement.")
    positions=np.array(json.loads((root/"sensor_geometry.json").read_text()))
    schedule=config["measurement_times"];times=np.linspace(schedule["start"],schedule["stop"],schedule["count"])
    start=time.perf_counter()
    h=forward([*location,1.],positions,times,config["physics"],config["inference_solver"],.08)
    np.savez_compressed(path,location=location,readings=h,seconds=time.perf_counter()-start)
    return index


def bank(root):
    config,_,_=verify(root)
    output=root/"maps";output.mkdir(exist_ok=False)
    axis=np.linspace(*config["observability_map"]["position_range"],config["observability_map"]["grid_count"])
    locations=np.array([(x,y) for y in axis for x in axis]);start=time.perf_counter()
    pool(bank_point,[(root,i,p) for i,p in enumerate(locations)],config["runtime"]["workers"],"UNIT BANK")
    values=[];seconds=[]
    for i,location in enumerate(locations):
        with np.load(output/f"point_{i:03d}.npz") as stored:
            if not np.array_equal(stored["location"],location):raise ValueError("Classical bank mapping changed.")
            values.append(stored["readings"]);seconds.append(float(stored["seconds"]))
    np.savez_compressed(output/"unit_responses.npz",locations=locations,unit_readings=values)
    write_json(output/"bank_lock.json",dict(sha256=sha256(output/"unit_responses.npz"),wall_seconds=time.perf_counter()-start,
        summed_forward_seconds=sum(seconds),amortized_forward_seconds_per_case=sum(seconds)/30,truth_used=False))


def information_case(root,identifier):
    install_guard(root);torch.set_num_threads(1)
    config=yaml.safe_load((root/"protocol.yaml").read_text());case=root/"datasets"/identifier
    output=root/"information"/identifier;output.mkdir(parents=True,exist_ok=False)
    inputs=load_model_inputs(case);truth=json.loads((case/"truth/source.json").read_text())
    theta=np.array([truth[k] for k in ("x_s","y_s","Q")])
    response=unit_operator(inputs["positions"].numpy(),inputs["times"].numpy(),config["physics"],config["inference_solver"],.08)
    jac,steps=jacobian_stability(theta,response,config["scaling"],config["sensitivity"])
    row=dict(scenario_id=identifier,sensor_rms=float(inputs["measurements"].square().mean().sqrt()),
        **information_metrics(jac,config["scaling"]["parameter_scales"]),finite_difference_stable=all(r["stable"] for r in steps),
        **geometry(theta[:2],inputs["positions"].numpy(),config["physics"],.08,.8))
    np.savez_compressed(output/"jacobian.npz",scaled_jacobian=jac)
    write_json(output/"steps.json",steps);write_json(output/"results.json",row)
    return identifier


def information(root):
    config,_,_=verify(root);manifest(root)
    if (root/"information_complete.json").exists():raise FileExistsError("Information split frozen.")
    pool(information_case,[(root,k) for k in IDS],config["runtime"]["workers"],"INFORMATION")
    rows=[json.loads((root/"information"/k/"results.json").read_text()) for k in IDS]
    threshold=float(np.median([r["sensor_rms"] for r in rows]))
    for row in rows:row["information_group"]="lower" if row["sensor_rms"]<=threshold else "higher"
    write_csv(root/"observability.csv",rows)
    write_json(root/"information_complete.json",dict(threshold=threshold,observability_hash=sha256(root/"observability.csv"),
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),before_fitting=True))


def fit_sparse(method,inputs,j1,config,run,attempt):
    """Sparse-only fitting adapter. Never receives a path to source truth."""
    data=SparseTrainingData(inputs["positions"],inputs["times"],inputs["measurements"],**inputs["physics"]["transport"])
    recipe=j1["recipe"];settings=TrainingSettings.from_mapping(recipe["training"])
    logger=logging.getLogger("revision8");logger.setLevel(logging.WARNING)
    audit=GradientPathway(data,j1["gradient_interval"]);previous_Q=settings.initial_Q
    with (run/"training_history.jsonl").open("x") as stream:
        def record(values):
            nonlocal previous_Q
            row=dict(values);row["effective_Q_update"]=row["source/Q"]-previous_Q;previous_Q=row["source/Q"]
            stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush();attempt["completed_updates"]=row["epoch"]
        if method=="J1":
            model,source,state,_=fit_profiled(data,settings,recipe["sigma"],recipe["seed"],recipe["formulation"],j1["candidate"],j1["profiling"],logger,record,audit)
        elif method=="B_revised":
            model,source,optimizer,_=fit_candidate(data,settings,recipe["sigma"],recipe["seed"],recipe["formulation"],logger,record,audit)
            state=optimizer.state_dict()
        else:raise ValueError("Undeclared neural method.")
    torch.save(dict(model_state_dict=model.state_dict(),source_state_dict=source.state_dict(),optimizer_state_dict=state,
        recipe=recipe,candidate=method),run/"model.pt")
    write_csv(run/"gradient_pathway.csv",audit.rows)
    return source.estimates()


def evaluate(root,method,identifier,config):
    run=root/"runs"/method/identifier
    if (run/"metrics.json").exists():raise FileExistsError("Evaluation already saved.")
    attempt,generation,truth,reference_path=load_evaluation_truth(root,identifier,run)
    row=dict(method=method,scenario_id=identifier,optimization_seed=42,**attempt,x_true=truth["x_s"],y_true=truth["y_s"],Q_true=truth["Q"],
        observations_hash=generation["observations_hash"],reference_data_hash=generation["reference_data_hash"],
        reference_target_met=generation["reference_target_met"],reference_error_estimate=generation["max_reference_error_estimate"],
        evaluation_success=False,recovery_success=False)
    row.update(dict.fromkeys(("x_pred","y_pred","Q_pred","localization_error","relative_strength_error","signed_Q_error","signed_relative_Q_error",
        "relative_l2","rmse","mae","negative_fraction","minimum_concentration","pde_rmse","pde_mae","bc_rmse","bc_maxabs","ic_rmse","ic_maxabs")))
    if attempt["computational_success"]:
        estimate=json.loads((run/"source_estimate.json").read_text())
        row.update(x_pred=estimate["x_s"],y_pred=estimate["y_s"],Q_pred=estimate["Q"],signed_Q_error=estimate["Q"]-truth["Q"],
            signed_relative_Q_error=(estimate["Q"]-truth["Q"])/truth["Q"],**parameter_metrics(estimate,truth))
        if method!="classical":
            model,source=load_candidate(run/"model.pt","J1" if method=="J1" else "J0")
            row.update(physical_diagnostics(model,source,config["physics"],.8,config["diagnostics"]))
        with np.load(reference_path) as ref:
            if method=="classical":
                numerical=solve_advection_diffusion(**config["inference_solver"],**config["physics"],sources=[estimate|dict(sigma=.08)],
                    output_mode="selected_times",output_times=config["evaluation_times"])
                np.savez_compressed(run/"forward_checkpoint.npz",fields=numerical["fields"].numpy(),x=numerical["x"].numpy(),y=numerical["y"].numpy(),times=config["evaluation_times"])
                prediction=F.interpolate(numerical["fields"][:,None],size=ref["fields"].shape[-2:],mode="bilinear",align_corners=True)[:,0].numpy()
            else:prediction=predict_snapshots(model,ref["x"],ref["y"],ref["field_times"],config["diagnostics"]["prediction_batch"])
            row.update(concentration_metrics(prediction,ref["fields"]),negative_fraction=float(np.mean(prediction<0)),minimum_concentration=float(prediction.min()))
            np.savez_compressed(run/"predictions.npz",fields=prediction[:,::4,::4],reference=ref["fields"][:,::4,::4],x=ref["x"][::4],y=ref["y"][::4],times=ref["field_times"])
        row["evaluation_success"]=True;row["recovery_success"]=recovery_decision(True,row,config["recovery"])
    if method!="classical" and (run/"training_history.jsonl").exists():training_figures(run,config["plot_dpi"])
    write_json(run/"metrics.json",row)
    return row


def fit_one(root,method,identifier):
    if method not in METHODS or identifier not in IDS:raise ValueError("Undeclared primary run.")
    state=install_guard(root);torch.set_num_threads(1)
    config=yaml.safe_load((root/"protocol.yaml").read_text());j1=yaml.safe_load((root/"J1_frozen.yaml").read_text())
    run=root/"runs"/method/identifier
    if (run/"metrics.json").exists():return f"{method}/{identifier} preserved"
    run.mkdir(parents=True,exist_ok=False)
    known=next(r for r in json.loads((root/"input_manifest.json").read_text()) if r["scenario_id"]==identifier)
    (run/"config.yaml").write_text(yaml.safe_dump(dict(method=method,J1=j1,input_hashes=known,protocol_hash=sha256(root/"protocol.yaml")),sort_keys=False))
    attempt=dict(training_terminated=False,computational_success=False,completed_updates=0,failure="")
    start=time.perf_counter();state.update(fitting_case=identifier,method=method)
    try:
        case=root/"datasets"/identifier
        if sha256(case/"inputs/observations.npz")!=known["observations_hash"] or sha256(case/"inputs/physics.json")!=known["physics_hash"]:raise ValueError("Sparse inputs changed.")
        inputs=load_model_inputs(case)
        if method=="classical":
            with np.load(root/"maps/unit_responses.npz") as bank:
                best,starts=classical_inverse(inputs["measurements"].numpy(),inputs["positions"].numpy(),inputs["times"].numpy(),
                    inputs["physics"]["transport"],config["inference_solver"],.08,bank["locations"],bank["unit_readings"],config["classical"],config["scaling"],config["sensitivity"])
            write_json(run/"starts.json",starts);write_json(run/"optimizer_result.json",best)
            estimate=dict(x_s=best["x_pred"],y_s=best["y_pred"],Q=best["Q_pred"])
        else:estimate=fit_sparse(method,inputs,j1,config,run,attempt)
        write_json(run/"source_estimate.json",estimate);attempt["computational_success"]=True
    except FloatingPointError as exc:
        attempt["failure"]=f"Numerical failure retained without retry: {exc}"
    except Exception as exc:
        attempt["failure"]=f"Unexpected {type(exc).__name__}: {exc}"
        write_json(run/"review_required.json",dict(error=attempt["failure"]))
        raise
    finally:
        state.update(fitting_case=None,method=None)
        attempt.update(training_terminated=True,runtime_seconds=time.perf_counter()-start,terminated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(run/"attempt.json",attempt)
    evaluate(root,method,identifier,config)
    return f"{method}/{identifier}"


def execution_lock(root):
    path=root/"fitting_execution"
    if not path.exists():path.mkdir();snapshot(path)
    else:
        hashes=json.loads((path/"provenance.json").read_text())["source_hashes"]
        if any(sha256(ROOT/k)!=v for k,v in hashes.items()):raise ValueError("Code changed after fitting began; preserve and review.")


def train(root,method):
    config,_,_=verify(root);manifest(root)
    info=json.loads((root/"information_complete.json").read_text())
    if sha256(root/"observability.csv")!=info["observability_hash"]:raise ValueError("Information split changed.")
    bank_lock=json.loads((root/"maps/bank_lock.json").read_text())
    if sha256(root/"maps/unit_responses.npz")!=bank_lock["sha256"]:raise ValueError("Classical bank changed.")
    execution_lock(root)
    if (root/f"{method}_raw.csv").exists():raise FileExistsError("Primary results already complete.")
    pool(fit_one,[(root,method,k) for k in IDS],config["runtime"]["workers"],method)
    write_csv(root/f"{method}_raw.csv",[json.loads((root/"runs"/method/k/"metrics.json").read_text()) for k in IDS])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage",choices=("bank","information",*METHODS),required=True)
    args=parser.parse_args();root=ROOT/"results/final_confirmatory"
    if args.stage=="bank":bank(root)
    elif args.stage=="information":information(root)
    else:train(root,args.stage)
