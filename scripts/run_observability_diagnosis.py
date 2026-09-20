"""Complete forward/information diagnostics before any diagnostic-v2 PINN fit."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from inverpinn.data.diagnostic_v2 import (IDS, diagnostic_path, install_guard, manifest,
    require_information_complete, verify, verify_consumed)
from inverpinn.data.single_source_benchmark import write_csv
from inverpinn.data.synthetic_reference import ROOT, load_model_inputs, sha256, write_json
from inverpinn.evaluation.metrics import concentration_metrics
from inverpinn.evaluation.observability import (amplitude_mismatch, classical_inverse, conditional_Q,
    envelope_latent, forward, geometry, information_metrics, jacobian_stability, scaled_jacobian, unit_operator)
from inverpinn.evaluation.single_source_benchmark import distribution, parameter_metrics, recovery_decision, wilson_interval
from inverpinn.physics.finite_difference import solve_advection_diffusion


def solver_for(config, grid):
    return next(s for s in config["inference_grids"] if s["nx"]==grid)


def map_point(root, index, location):
    """Truth-independent candidate forward sensitivity: no scenario files."""
    install_guard(root)
    config=yaml.safe_load((root/"protocol.yaml").read_text())
    directory=root/"maps/points"/f"point_{index:03d}"
    if (directory/"metrics.json").exists():
        return json.loads((directory/"metrics.json").read_text())
    directory.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config["runtime"]["torch_threads"])
    positions=np.array(json.loads((root/"sensor_geometry.json").read_text()))
    schedule=config["measurement_times"]
    times=np.linspace(schedule["start"],schedule["stop"],schedule["count"])
    solver=solver_for(config,config["classical"]["grid"])
    response=unit_operator(positions,times,config["physics"],solver,config["source_family"]["sigma"])
    theta=[*location,config["observability_map"]["representative_Q"]]
    start=time.perf_counter()
    h,jac=scaled_jacobian(theta,response,config["scaling"]["parameter_scales"],
        config["scaling"]["concentration_scale"],config["sensitivity"]["selected_step"])
    row=dict(index=index,x=location[0],y=location[1],Q=theta[2],
        sensor_rms=float(np.sqrt(np.mean((theta[2]*h)**2))),runtime_seconds=time.perf_counter()-start,
        **information_metrics(jac,config["scaling"]["parameter_scales"]))
    np.savez_compressed(directory/"checkpoint.npz",unit_readings=h,scaled_jacobian=jac,location=location)
    write_json(directory/"metrics.json",row)
    return row


def oracle_case(root, identifier):
    """Explicit oracle location/true-parameter diagnostics, never inverse fitting."""
    install_guard(root)
    config=yaml.safe_load((root/"protocol.yaml").read_text())
    output=root/"diagnostics"/identifier
    if (output/"results.json").exists():
        return identifier
    output.mkdir(parents=True,exist_ok=False)
    (output/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    torch.set_num_threads(config["runtime"]["torch_threads"])
    case=diagnostic_path(root,identifier)
    inputs=load_model_inputs(case)
    truth=json.loads((case/"truth/source.json").read_text())
    positions,times=inputs["positions"].numpy(),inputs["times"].numpy()
    observed=inputs["measurements"].numpy().reshape(-1)
    theta=np.array([truth[k] for k in ("x_s","y_s","Q")])
    vectors=dict(reference=observed,positions=positions,times=times)
    rows=[]
    for solver in config["inference_grids"]:
        start=time.perf_counter()
        h=forward([*theta[:2],1.],positions,times,config["physics"],solver,truth["sigma"])
        direct=forward(theta,positions,times,config["physics"],solver,truth["sigma"])
        estimate=conditional_Q(h,observed,config["observability_map"]["representative_Q"],config["scaling"]["concentration_scale"])
        row=dict(scenario_id=identifier,grid=solver["nx"],dt=solver["dt"],steps=solver["steps"],
            x_true=theta[0],y_true=theta[1],Q_true=theta[2],**estimate,
            signed_Q_error=estimate["Q_analytic"]-theta[2],
            absolute_Q_error=abs(estimate["Q_analytic"]-theta[2]),
            relative_Q_error=abs(estimate["Q_analytic"]-theta[2])/theta[2],
            signed_relative_Q_error=(estimate["Q_analytic"]-theta[2])/theta[2],
            linearity_relative_l2=float(np.linalg.norm(direct-theta[2]*h)/np.linalg.norm(direct)),
            runtime_seconds=time.perf_counter()-start,**amplitude_mismatch(direct,observed))
        rows.append(row)
        vectors[f"unit_{solver['nx']}"]=h;vectors[f"inference_true_{solver['nx']}"]=direct
    np.savez_compressed(output/"sensor_vectors.npz",**vectors)
    response=unit_operator(positions,times,config["physics"],solver_for(config,config["classical"]["grid"]),truth["sigma"])
    jac,steps=jacobian_stability(theta,response,config["scaling"],config["sensitivity"])
    np.savez_compressed(output/"jacobian.npz",scaled_jacobian=jac,information=jac.T@jac)
    jac_row=dict(scenario_id=identifier,**information_metrics(jac,config["scaling"]["parameter_scales"]),
        finite_difference_stable=all(s["stable"] for s in steps))
    geo=dict(scenario_id=identifier,x_true=theta[0],y_true=theta[1],Q_true=theta[2],
        sensor_rms=float(np.sqrt(np.mean(observed**2))),maximum_sensor_signal=float(observed.max()),
        **geometry(theta[:2],positions,config["physics"],truth["sigma"],config["geometry"]["plume_horizon"]))
    reference=case/"truth/grid_641/checkpoint.npz"
    with np.load(reference,allow_pickle=False) as arrays:
        envelope,maps=envelope_latent(arrays["fields"],arrays["x"],arrays["y"],arrays["field_times"],
            config["measurement_times"]["stop"],config["envelope"]["active_fraction_of_snapshot_max"])
        if identifier==config["envelope"]["latent_map_scenario"]:
            np.savez_compressed(output/"envelope_maps.npz",x=arrays["x"],y=arrays["y"],times=arrays["field_times"],
                **{k:np.stack([m[k] for m in maps]) for k in maps[0]})
    write_json(output/"results.json",dict(resolutions=rows,jacobian=jac_row,geometry=geo,
        step_stability=[dict(scenario_id=identifier,**r) for r in steps],
        envelope=[dict(scenario_id=identifier,x_true=theta[0],y_true=theta[1],**r) for r in envelope]))
    return identifier


def classical_case(root,identifier):
    """Input-only joint fit followed by terminal-barrier source/field evaluation."""
    state=install_guard(root)
    config=yaml.safe_load((root/"protocol.yaml").read_text())
    output=root/"runs/classical"/identifier
    if (output/"metrics.json").exists():
        return identifier
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config["runtime"]["torch_threads"])
    known=next(r for r in json.loads((root/"input_manifest.json").read_text()) if r["scenario_id"]==identifier)
    with np.load(root/"maps/unit_responses.npz",allow_pickle=False) as arrays:
        candidates,responses=arrays["locations"],arrays["unit_readings"]
    settings={k:config[k] for k in ("physics","source_family","classical","scaling","sensitivity","inference_grids")}
    (output/"config.yaml").write_text(yaml.safe_dump(settings|dict(input_hashes=known)))
    case=diagnostic_path(root,identifier)
    attempt=dict(training_terminated=False,computational_success=False,failure="")
    start=time.perf_counter();state["fitting_case"]=identifier
    try:
        if sha256(case/"inputs/observations.npz")!=known["observations_hash"] or sha256(case/"inputs/physics.json")!=known["physics_hash"]:
            raise ValueError("Sparse-only input hash changed.")
        inputs=load_model_inputs(case)
        best,starts=classical_inverse(inputs["measurements"].numpy(),inputs["positions"].numpy(),inputs["times"].numpy(),
            inputs["physics"]["transport"],solver_for(config,config["classical"]["grid"]),config["source_family"]["sigma"],
            candidates,responses,config["classical"],config["scaling"],config["sensitivity"])
        write_json(output/"starts.json",starts)
        write_json(output/"estimate.json",best)
        attempt["computational_success"]=True
    except (FloatingPointError,ValueError,RuntimeError) as exc:
        attempt["failure"]=f"Classical fit failure retained: {type(exc).__name__}: {exc}"
    finally:
        state["fitting_case"]=None
        attempt.update(training_terminated=True,runtime_seconds=time.perf_counter()-start)
        write_json(output/"attempt.json",attempt)
    # No source/evaluation truth above the persisted terminal barrier.
    truth=json.loads((case/"truth/source.json").read_text())
    row=dict(scenario_id=identifier,**attempt,x_true=truth["x_s"],y_true=truth["y_s"],Q_true=truth["Q"],
             observations_hash=known["observations_hash"],recovery_success=False)
    if attempt["computational_success"]:
        estimate=dict(x_s=best["x_pred"],y_s=best["y_pred"],Q=best["Q_pred"])
        row.update(best,**parameter_metrics(estimate,truth),signed_Q_error=estimate["Q"]-truth["Q"])
        row["selected_start_seconds"]=best["runtime_seconds"]
        row["runtime_seconds"]=attempt["runtime_seconds"]
        solution=solve_advection_diffusion(**solver_for(config,config["classical"]["grid"]),**config["physics"],
            sources=[estimate|dict(sigma=config["source_family"]["sigma"])],output_mode="selected_times",output_times=config["evaluation_times"])
        with np.load(case/"truth/grid_641/checkpoint.npz",allow_pickle=False) as reference:
            prediction=F.interpolate(solution["fields"][:,None],size=reference["fields"].shape[-2:],mode="bilinear",align_corners=True)[:,0].numpy()
            row.update(concentration_metrics(prediction,reference["fields"]))
            np.savez_compressed(output/"predictions.npz",fields=prediction,x=reference["x"],y=reference["y"],field_times=reference["field_times"],native_fields=solution["fields"].numpy())
        row["recovery_success"]=recovery_decision(True,row,config["recovery"])
    else:
        for key in ("x_pred","y_pred","Q_pred","localization_error","relative_strength_error","relative_l2","rmse","mae","sensor_mse","signed_Q_error"):
            row[key]=None
    write_json(output/"metrics.json",row)
    return identifier


def correlations(frame,outcomes,features):
    """Descriptive Spearman coefficients, explicit missing/constant values."""
    rows=[]
    for outcome in outcomes:
        for feature in features:
            valid=frame[[outcome,feature]].dropna()
            value=valid[outcome].corr(valid[feature],method="spearman") if len(valid)>1 and all(valid[c].nunique()>1 for c in valid) else np.nan
            rows.append(dict(outcome=outcome,feature=feature,count=len(valid),spearman=float(value) if np.isfinite(value) else None))
    return rows


def summarize(frame,columns):
    return {key:distribution(frame[key].tolist(),len(frame)) for key in columns}


def pool_jobs(function,jobs,workers,label):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(function,*job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if label!="MAP" or i%21==0:
                print(f"{label} COMPLETE {i}/{len(jobs)}",flush=True)


def information(root):
    config,_,_=verify(root)
    manifest(root)
    if (root/"information_complete.json").exists():
        require_information_complete(root)
        print("Information stage already complete and unchanged.")
        return
    verify_consumed(root)
    from run_model_diagnosis import snapshot
    execution=root/"information_execution"
    if execution.exists():
        previous=json.loads((execution/"provenance.json").read_text())
        if any(sha256(ROOT/p)!=h for p,h in previous["source_hashes"].items()):
            raise ValueError("Information-stage code changed; preserve partial run for review.")
    else:
        execution.mkdir();snapshot(execution)
    install_guard(root)
    workers=config["runtime"]["workers"]
    spec=config["observability_map"]
    axis=np.linspace(*spec["position_range"],spec["grid_count"])
    locations=np.array([(x,y) for y in axis for x in axis])
    pool_jobs(map_point,[(root,i,p.tolist()) for i,p in enumerate(locations)],workers,"MAP")
    map_rows=[json.loads((root/"maps/points"/f"point_{i:03d}/metrics.json").read_text()) for i in range(len(locations))]
    unit=[]
    for i in range(len(locations)):
        with np.load(root/"maps/points"/f"point_{i:03d}/checkpoint.npz",allow_pickle=False) as arrays:
            unit.append(arrays["unit_readings"])
    if not (root/"maps/unit_responses.npz").exists():
        np.savez_compressed(root/"maps/unit_responses.npz",locations=locations,unit_readings=np.stack(unit),axis=axis)
        write_csv(root/"map_metrics.csv",map_rows)
    pool_jobs(oracle_case,[(root,k) for k in IDS],workers,"ORACLE/JACOBIAN")
    pool_jobs(classical_case,[(root,k) for k in IDS],workers,"CLASSICAL")
    pieces=[json.loads((root/"diagnostics"/k/"results.json").read_text()) for k in IDS]
    tables={"resolution_bias_results":[r for piece in pieces for r in piece["resolutions"]],
        "jacobian_metrics":[p["jacobian"] for p in pieces],"geometry_metrics":[p["geometry"] for p in pieces],
        "jacobian_step_stability":[r for p in pieces for r in p["step_stability"]],
        "envelope_reference":[r for p in pieces for r in p["envelope"]],
        "classical_inverse_results":[json.loads((root/"runs/classical"/k/"metrics.json").read_text()) for k in IDS]}
    tables["conditional_Q_results"]=[r for r in tables["resolution_bias_results"] if r["grid"]==config["classical"]["grid"]]
    for name,rows in tables.items():
        if (root/f"{name}.csv").exists():
            raise FileExistsError("Partial aggregate outputs preserved; no silent overwrite.")
        write_csv(root/f"{name}.csv",rows)
    frames={k:pd.DataFrame(v) for k,v in tables.items()}
    combined=frames["classical_inverse_results"].merge(frames["jacobian_metrics"],on="scenario_id",validate="one_to_one").merge(
        frames["geometry_metrics"].drop(columns=["x_true","y_true","Q_true"]),on="scenario_id",validate="one_to_one")
    features=["sensor_rms","maximum_sensor_signal","x_true","y_true","Q_true","nearest_sensor_distance", "boundary_distance",
        "downwind_sensor_count","nearest_downwind_distance","min_downwind_crosswind_distance","reachable_downwind_count","plume_corridor_count"]
    corr=correlations(combined,["localization_error","relative_strength_error","jacobian_condition","sigma_min","Q_effective_information"],features)
    write_csv(root/"prospective_signal_correlations.csv",corr)
    resolution={str(g):summarize(group,["signed_Q_error","absolute_Q_error","signed_relative_Q_error","relative_Q_error","amplitude_ratio",
        "relative_observation_l2","signed_observation_bias","runtime_seconds","analytic_numerical_difference","linearity_relative_l2"])
        for g,group in frames["resolution_bias_results"].groupby("grid")}
    classical=frames["classical_inverse_results"]
    success=int(classical.recovery_success.sum())
    maps=pd.DataFrame(map_rows)
    result=dict(resolution=resolution,classical=dict(recovered=success,total=30,
        recovery_Wilson95=wilson_interval(success,30,config["recovery"]["confidence_level"]),
        metrics=summarize(classical,["localization_error","relative_strength_error","relative_l2","sensor_mse","runtime_seconds"])),
        jacobian=summarize(frames["jacobian_metrics"],["sigma_min","jacobian_condition","Q_sensitivity_norm","Q_effective_information"]),
        unstable_J_cases=frames["jacobian_metrics"].loc[~frames["jacobian_metrics"].finite_difference_stable,"scenario_id"].tolist(),
        relative_coverage_thresholds=dict(sigma_min=float(maps.sigma_min.quantile(spec["weak_smin_quantile"])),
            Q_effective_information=float(maps.Q_effective_information.quantile(spec["weak_Q_information_quantile"]))),
        correlations=corr,local_information_not_posterior=True)
    from inverpinn.visualization.observability import information_figures
    information_figures(root,config,maps,frames,combined)
    write_json(root/"information_summary.json",result)
    verify(root);manifest(root)
    files=[p for p in root.rglob("*") if p.is_file() and not p.is_relative_to(root/"datasets")]
    write_json(root/"information_complete.json",dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scenario_count=30,grid_points=len(locations),hashes={str(p.relative_to(root)):sha256(p) for p in sorted(files)}))
    print("INFORMATION STAGE COMPLETE; frozen PINN may now run.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/observability_diagnosis.yaml")
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/args.config).read_text())
    information(ROOT/config["output_dir"])
