"""Run unchanged B_revised once per diagnostic_v2 case, after information freeze."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from inverpinn.data.diagnostic_v2 import (IDS, diagnostic_path, install_guard, manifest,
    require_information_complete, verify)
from inverpinn.data.single_source_benchmark import write_csv
from inverpinn.data.synthetic_reference import ROOT, load_model_inputs, sha256, write_json
from inverpinn.evaluation.metrics import concentration_metrics
from inverpinn.evaluation.observability import network_envelope
from inverpinn.evaluation.single_source_benchmark import parameter_metrics,physical_diagnostics,predict_snapshots,recovery_decision
from inverpinn.training.development_candidates import fit_candidate
from inverpinn.training.single_source import SparseTrainingData,TrainingSettings
from inverpinn.visualization.single_source_benchmark import training_figures
from run_fresh_blind_benchmark import load_fit,recipe_for


def train_case(root,identifier):
    if identifier not in IDS:
        raise ValueError("Consumed scenarios cannot enter diagnostic fitting.")
    state=install_guard(root)
    require_information_complete(root)
    config=yaml.safe_load((root/"protocol.yaml").read_text())
    output=root/"runs/pinn"/identifier
    if (output/"metrics.json").exists():
        return identifier
    output.mkdir(parents=True,exist_ok=False)
    seed=config["optimization"]["primary_seed"]
    if seed!=42:
        raise ValueError("Frozen optimization seed must be 42.")
    recipe=recipe_for(root,"B_revised",seed)
    known=next(r for r in json.loads((root/"input_manifest.json").read_text()) if r["scenario_id"]==identifier)
    (output/"config.yaml").write_text(yaml.safe_dump(dict(recipe=recipe,input_hashes=known),sort_keys=False))
    case=diagnostic_path(root,identifier)
    attempt=dict(training_terminated=False,computational_success=False,failure="",completed_updates=0)
    start=time.perf_counter();state["fitting_case"]=identifier
    try:
        if sha256(case/"inputs/observations.npz")!=known["observations_hash"] or sha256(case/"inputs/physics.json")!=known["physics_hash"]:
            raise ValueError("Frozen input-only manifest mismatch.")
        inputs=load_model_inputs(case)
        if inputs["physics"]["transport"]!=config["physics"]:
            raise ValueError("Known transport changed.")
        data=SparseTrainingData(inputs["positions"],inputs["times"],inputs["measurements"],**config["physics"])
        settings=TrainingSettings.from_mapping(recipe["training"])
        logger=logging.getLogger("diagnostic_v2_PINN");logger.setLevel(logging.WARNING)
        with (output/"training_history.jsonl").open("x") as stream:
            def record(values):
                stream.write(json.dumps(values,allow_nan=False)+"\n");stream.flush()
                attempt["completed_updates"]=values["epoch"]
            model,source,optimizer,_=fit_candidate(data,settings,recipe["sigma"],seed,recipe["formulation"],logger,record)
        torch.save(dict(model_state_dict=model.state_dict(),source_state_dict=source.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),recipe=recipe),output/"model.pt")
        write_json(output/"source_estimate.json",source.estimates())
        attempt["computational_success"]=True
    except FloatingPointError as exc:
        attempt["failure"]=f"Nonfinite fit retained without retry: {exc}"
    except Exception as exc:
        attempt["failure"]=f"Unexpected error: {type(exc).__name__}: {exc}"
        write_json(output/"review_required.json",dict(error=attempt["failure"]))
        raise
    finally:
        state["fitting_case"]=None
        attempt.update(training_terminated=True,training_seconds=time.perf_counter()-start)
        write_json(output/"attempt.json",attempt)
    # Source/dense truth is first opened after the persisted terminal barrier.
    truth=json.loads((case/"truth/source.json").read_text())
    generation=json.loads((case/"generation.json").read_text())
    row=dict(scenario_id=identifier,optimization_seed=seed,**attempt,x_true=truth["x_s"],y_true=truth["y_s"],Q_true=truth["Q"],
        observations_hash=known["observations_hash"],reference_data_hash=generation["reference_data_hash"],
        reference_target_met=generation["reference_target_met"],reference_error_estimate=generation["max_reference_error_estimate"],
        recovery_success=False,evaluation_success=False)
    if attempt["computational_success"]:
        torch.set_num_threads(config["runtime"]["torch_threads"])
        model,source=load_fit(output/"model.pt",config["measurement_times"]["stop"])
        estimate=source.estimates()
        row.update(x_pred=estimate["x_s"],y_pred=estimate["y_s"],Q_pred=estimate["Q"],
            signed_Q_error=estimate["Q"]-truth["Q"],**parameter_metrics(estimate,truth))
        with torch.no_grad():
            predicted_sensor=model(inputs["features"].float()).double().numpy().reshape(-1)
        observed=inputs["measurements"].numpy().reshape(-1)
        row["sensor_mse"]=float(np.mean((predicted_sensor-observed)**2))
        np.savez_compressed(output/"sensor_predictions.npz",predicted=predicted_sensor,observed=observed)
        with np.load(case/"truth/grid_641/checkpoint.npz",allow_pickle=False) as stored:
            reference,x,y,times=[stored[k] for k in ("fields","x","y","field_times")]
            predicted=predict_snapshots(model,x,y,times,config["diagnostics"]["prediction_batch"])
            row.update(concentration_metrics(predicted,reference),negative_fraction=float(np.mean(predicted<0)),minimum_concentration=float(predicted.min()))
            np.savez_compressed(output/"predictions.npz",fields=predicted,x=x,y=y,field_times=times)
            n=config["envelope"]["network_probe_grid"]
            mask_fields=F.interpolate(torch.tensor(reference)[:,None],size=(n,n),mode="bilinear",align_corners=True)[:,0].numpy()
        network,maps=network_envelope(model,mask_fields,np.linspace(0,1,n),times,config["measurement_times"]["stop"],
            config["envelope"]["active_fraction_of_snapshot_max"])
        write_csv(output/"envelope_network.csv",[dict(scenario_id=identifier,**r) for r in network])
        if identifier==config["envelope"]["latent_map_scenario"]:
            np.savez_compressed(output/"envelope_network_maps.npz",axis=np.linspace(0,1,n),times=times,
                **{k:np.stack([m[k] for m in maps]) for k in maps[0]})
        row.update(physical_diagnostics(model,source,config["physics"],config["measurement_times"]["stop"],config["diagnostics"]))
        row["recovery_success"]=recovery_decision(True,row,config["recovery"])
        row["evaluation_success"]=True
        training_figures(output,config["plot_dpi"])
    else:
        for key in ("x_pred","y_pred","Q_pred","signed_Q_error","localization_error","relative_strength_error","relative_l2","rmse","mae","sensor_mse",
                    "negative_fraction","minimum_concentration","pde_rmse","pde_mae","bc_rmse","bc_maxabs","ic_rmse","ic_maxabs"):
            row[key]=None
    write_json(output/"metrics.json",row)
    return identifier


def run(root):
    config,_,_=verify(root);manifest(root)
    require_information_complete(root)
    if (root/"pinn_complete.json").exists():
        raise FileExistsError("Completed diagnostic PINN runs preserved; no rerun.")
    from run_model_diagnosis import snapshot
    execution=root/"pinn_execution"
    if execution.exists():
        previous=json.loads((execution/"provenance.json").read_text())
        if any(sha256(ROOT/p)!=h for p,h in previous["source_hashes"].items()):
            raise ValueError("PINN-stage execution code changed; partial runs retained.")
    else:
        execution.mkdir();snapshot(execution)
    with ProcessPoolExecutor(max_workers=config["runtime"]["workers"]) as pool:
        futures=[pool.submit(train_case,root,k) for k in IDS]
        for i,future in enumerate(as_completed(futures),1):
            identifier=future.result()
            print(f"FROZEN PINN COMPLETE {i}/30: {identifier}",flush=True)
    rows=[json.loads((root/"runs/pinn"/k/"metrics.json").read_text()) for k in IDS]
    write_csv(root/"pinn_results.csv",rows)
    network=[]
    import pandas as pd
    for identifier in IDS:
        path=root/"runs/pinn"/identifier/"envelope_network.csv"
        if path.exists():
            network.extend(pd.read_csv(path).to_dict("records"))
    if network:
        write_csv(root/"envelope_network.csv",network)
    verify(root);manifest(root);require_information_complete(root)
    write_json(root/"pinn_complete.json",dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        total=30,completed=sum(r["computational_success"] for r in rows),optimization_seed=42,
        raw_hash=sha256(root/"pinn_results.csv")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/observability_diagnosis.yaml")
    args=parser.parse_args()
    settings=yaml.safe_load((ROOT/args.config).read_text())
    run(ROOT/settings["output_dir"])
