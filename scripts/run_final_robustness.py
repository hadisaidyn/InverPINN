"""Conditional frozen-J1 stress tests. Refuse execution unless confirmation passes.

This is not model selection. All perturbations are specified in the original
confirmation protocol; one shared control is reused across three displays.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
import yaml
from inverpinn.data.final_confirmatory import verify,manifest,previous_sources,install_guard
from inverpinn.data.final_robustness import require_confirmation,nested_sensors,noisy_observations,wind_conditions
from inverpinn.data.single_source_benchmark import object_hash,generate_scenario,write_csv
from inverpinn.data.synthetic_reference import ROOT,sha256,write_json,provenance,load_model_inputs
from inverpinn.evaluation.single_source_benchmark import load_evaluation_truth,parameter_metrics,predict_snapshots,physical_diagnostics,recovery_decision,wilson_interval,distribution
from inverpinn.evaluation.metrics import concentration_metrics
from run_final_confirmatory import fit_sparse,execution_lock,pool
from run_inverse_formulation_revision import load_candidate
from inverpinn.visualization.single_source_benchmark import training_figures,save_figure

CONDITIONS=("control","sensors_5","sensors_10","noise_5","noise_10","noise_20","speed_minus10","speed_plus10","direction_minus10","direction_plus10")


def initialize(final,root):
    """Only a verified pass can generate even the first robustness source."""
    parent=require_confirmation(final);_,previous_final,parent_lock=verify(final)
    if root.exists():raise FileExistsError("Robustness run already exists.")
    prior,seeds,triples=previous_sources();prior+=previous_final
    spec=parent["conditional_robustness"];seed=spec["scenario_seed"]
    if seed in seeds or seed==parent["scenario_sampling"]["seed"]:raise ValueError("Consumed robustness seed.")
    root.mkdir(parents=True)
    config=dict(parent,experiment="final_robustness_v1",output_dir="results/final_robustness",
        scenario_sampling=dict(seed=seed,count=20,method="IID_uniform"))
    (root/"protocol.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    for name in ("protocol.md","sensor_geometry.json","J1_frozen.yaml","B_revised_frozen.yaml"):
        shutil.copyfile(final/name,root/name)
    choices=nested_sensors(np.array(json.loads((root/"sensor_geometry.json").read_text())))
    write_json(root/"perturbations.json",dict(farthest_point_order=choices,materialized_indices={n:sorted(v) for n,v in choices.items()},
        wind=wind_conditions(),conditions=CONDITIONS,noise=spec["noise_model"],noise_seed=spec["noise_seed"],fixed_before_results=True))
    pre=parent_lock|dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),protocol_hash=object_hash(config),
        confirmation_decision_hash=sha256(final/"decision.json"),perturbation_hash=sha256(root/"perturbations.json"))
    pre["code_hashes"]=dict(parent_lock["code_hashes"],**{"scripts/run_final_robustness.py":sha256(ROOT/"scripts/run_final_robustness.py")})
    write_json(root/"preregistration.json",pre)
    rng=np.random.default_rng(seed);values=rng.uniform([.25,.25,.8],[.75,.75,1.4],size=(20,3));plan=[]
    for i,(theta,child) in enumerate(zip(values,np.random.SeedSequence(seed).spawn(20)),1):
        source=dict(zip(("x_s","y_s","Q"),map(float,theta)))
        if object_hash(source) in triples:raise ValueError("Consumed source collision; preserve without redraw.")
        source["sigma"]=.08
        plan.append(dict(scenario_id=f"robustness_v1_{i:03d}",split="robustness_v1",source=source,source_hash=object_hash(source),
            generation_seed=int(child.generate_state(1,dtype=np.uint64)[0])%(2**63-1)))
    for key in ("scenario_id","source_hash","generation_seed"):
        if len({r[key] for r in plan})!=20 or {r[key] for r in plan}&{r[key] for r in prior}:raise ValueError("Robustness overlap; no replacement.")
    write_json(root/"scenario_plan.json",plan);write_json(root/"protocol_lock.json",pre|dict(plan_hash=object_hash(plan)))
    write_json(root/"independence_check.json",dict(prior_cases=len(prior),new_cases=20,overlap=0,seed=seed))
    provenance(root)


def generate(root):
    config,plan,_=verify(root)
    with ProcessPoolExecutor(max_workers=2,initializer=install_guard,initargs=(root,)) as executor:
        futures=[]
        for row in plan:
            case=root/"datasets"/row["scenario_id"]
            if case.exists():raise FileExistsError("Existing robustness reference preserved.")
            futures.append(executor.submit(generate_scenario,root,row))
        for i,f in enumerate(as_completed(futures),1):
            result=f.result();print(f"ROBUSTNESS REFERENCE {i}/20 target={result['reference_target_met']}",flush=True)
    manifest(root)


def perturb(inputs,condition,spec,indices,index):
    """Outcome-independent sparse data and inference-wind transformation."""
    if condition not in CONDITIONS:raise ValueError("Undeclared robustness condition.")
    data={k:inputs[k].clone() for k in ("positions","times","measurements")}
    data["physics"]=dict(transport=dict(inputs["physics"]["transport"]))
    if condition.startswith("sensors_"):
        selected=torch.tensor(indices[str(int(condition.split("_")[1]))],dtype=torch.long)
        data["positions"]=data["positions"][selected];data["measurements"]=data["measurements"][:,selected]
    elif condition.startswith("noise_"):
        fraction=int(condition.split("_")[1])/100
        data["measurements"]=torch.tensor(noisy_observations(data["measurements"].numpy(),fraction,spec["noise_seed"],index))
    elif condition in wind_conditions():data["physics"]["transport"].update(wind_conditions()[condition])
    return data


def fit_one(root,condition,identifier,index):
    state=install_guard(root);torch.set_num_threads(1)
    config=yaml.safe_load((root/"protocol.yaml").read_text());j1=yaml.safe_load((root/"J1_frozen.yaml").read_text())
    perturbations=json.loads((root/"perturbations.json").read_text());run=root/"runs"/condition/identifier
    if run.exists():raise FileExistsError("Existing robustness fit preserved; no retry.")
    run.mkdir(parents=True)
    inputs=perturb(load_model_inputs(root/"datasets"/identifier),condition,config["conditional_robustness"],perturbations["materialized_indices"],index)
    np.savez_compressed(run/"observations.npz",**{k:inputs[k].numpy() for k in ("positions","times","measurements")})
    (run/"config.yaml").write_text(yaml.safe_dump(dict(J1=j1,condition=condition,inference_physics=inputs["physics"]["transport"],
        observation_hash=sha256(run/"observations.npz"),protocol_hash=sha256(root/"protocol.yaml")),sort_keys=False))
    attempt=dict(training_terminated=False,computational_success=False,completed_updates=0,failure="")
    start=time.perf_counter();state.update(fitting_case=identifier,method=condition)
    try:
        estimate=fit_sparse("J1",inputs,j1,config,run,attempt)
        write_json(run/"source_estimate.json",estimate);attempt["computational_success"]=True
    except FloatingPointError as exc:attempt["failure"]=f"Numerical failure retained without retry: {exc}"
    except Exception as exc:
        attempt["failure"]=f"Unexpected {type(exc).__name__}: {exc}";write_json(run/"review_required.json",dict(error=attempt["failure"]));raise
    finally:
        state.update(fitting_case=None,method=None);attempt.update(training_terminated=True,runtime_seconds=time.perf_counter()-start)
        write_json(run/"attempt.json",attempt)
    _,generation,truth,reference=load_evaluation_truth(root,identifier,run)
    row=dict(condition=condition,scenario_id=identifier,optimization_seed=42,**attempt,x_true=truth["x_s"],y_true=truth["y_s"],Q_true=truth["Q"],
        observation_hash=sha256(run/"observations.npz"),reference_data_hash=generation["reference_data_hash"],reference_target_met=generation["reference_target_met"],
        evaluation_success=False,recovery_success=False)
    row.update(dict.fromkeys(("x_pred","y_pred","Q_pred","localization_error","relative_strength_error","relative_l2","rmse","mae","pde_rmse","negative_fraction","bc_maxabs","ic_maxabs")))
    if attempt["computational_success"]:
        row.update(x_pred=estimate["x_s"],y_pred=estimate["y_s"],Q_pred=estimate["Q"],**parameter_metrics(estimate,truth))
        model,source=load_candidate(run/"model.pt","J1")
        # Diagnose against the generating PDE; inference wind may be wrong.
        row.update(physical_diagnostics(model,source,config["physics"],.8,config["diagnostics"]),pde_basis="true_generating_physics")
        with np.load(reference) as ref:
            predicted=predict_snapshots(model,ref["x"],ref["y"],ref["field_times"],8192)
            row.update(concentration_metrics(predicted,ref["fields"]),negative_fraction=float(np.mean(predicted<0)))
            np.savez_compressed(run/"predictions.npz",fields=predicted[:,::4,::4],reference=ref["fields"][:,::4,::4],times=ref["field_times"])
        row["evaluation_success"]=True;row["recovery_success"]=recovery_decision(True,row,config["recovery"])
    training_figures(run,config["plot_dpi"]);write_json(run/"metrics.json",row)
    return f"{condition}/{identifier}"


def train(root):
    config,plan,lock=verify(root);manifest(root)
    if sha256(root/"perturbations.json")!=lock["perturbation_hash"]:raise ValueError("Perturbation plan changed.")
    execution_lock(root)
    for condition in CONDITIONS:
        if (root/f"{condition}_raw.csv").exists():raise FileExistsError("Existing robustness results preserved.")
        pool(fit_one,[(root,condition,p["scenario_id"],i) for i,p in enumerate(plan)],2,condition)
        write_csv(root/f"{condition}_raw.csv",[json.loads((root/"runs"/condition/p["scenario_id"]/"metrics.json").read_text()) for p in plan])


def report(root):
    import matplotlib.pyplot as plt
    if (root/"summary.json").exists():raise FileExistsError("Robustness summary already exists.")
    config,plan,_=verify(root);frame=pd.concat([pd.read_csv(root/f"{c}_raw.csv") for c in CONDITIONS],ignore_index=True)
    summaries={}
    for c in CONDITIONS:
        rows=frame[frame.condition==c]
        if len(rows)!=20 or set(rows.scenario_id)!={p["scenario_id"] for p in plan}:raise ValueError("Missing robustness cases.")
        successes=int(rows.recovery_success.sum())
        summaries[c]=dict(count=20,recovered=successes,recovery_rate=successes/20,Wilson95=wilson_interval(successes,20,.95),
            metrics={k:distribution(rows[k],20) for k in ("localization_error","relative_strength_error","relative_l2","rmse","mae")})
    views=dict(sensor_count=(["sensors_5","sensors_10","control"],["5","10","20"]),noise=(["control","noise_5","noise_10","noise_20"],["0%","5%","10%","20%"]),
        wind_error=(["control","speed_minus10","speed_plus10","direction_minus10","direction_plus10"],["correct","speed −10%","speed +10%","direction −10°","direction +10°"]))
    table=[]
    for name,(conditions,labels) in views.items():
        x=np.arange(len(conditions));rates=np.array([summaries[c]["recovery_rate"] for c in conditions]);ci=np.array([summaries[c]["Wilson95"] for c in conditions])
        fig,ax=plt.subplots(figsize=(7,4),layout="constrained");ax.errorbar(x,rates,yerr=np.stack([rates-ci[:,0],ci[:,1]-rates]),fmt="o-",capsize=4)
        ax.set(xticks=x,xticklabels=labels,ylim=(0,1.05),ylabel="Joint recovery fraction (95% Wilson)",xlabel=name.replace("_"," "))
        save_figure(fig,root/f"recovery_vs_{name}",config["plot_dpi"])
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
        for ax,key,label in zip(axes,["localization_error","relative_strength_error","relative_l2"],["Localization","Relative Q error","Concentration L2"]):
            groups=[frame.loc[frame.condition==c,key].dropna().to_numpy() for c in conditions]
            ax.boxplot(groups,tick_labels=labels,showfliers=False)
            for i,g in enumerate(groups,1):ax.scatter(i+np.linspace(-.12,.12,len(g)),g,s=12,alpha=.65)
            ax.set(ylabel=label,yscale="log");ax.tick_params(axis="x",rotation=20)
        save_figure(fig,root/f"error_vs_{name}",config["plot_dpi"])
        for c,label in zip(conditions,labels):
            s=summaries[c];table.append(dict(study=name,level=label,condition=c,recovered=s["recovered"],count=20,
                **{f"{k}_median":s["metrics"][k]["median"] for k in ("localization_error","relative_strength_error","relative_l2")}))
    write_csv(root/"all_raw.csv",frame.astype(object).where(pd.notna(frame),None).to_dict("records"));write_csv(root/"summary.csv",table)
    write_json(root/"summary.json",dict(conditions=summaries,unique_fits=200,reused_control=True,model_hash=sha256(root/"J1_frozen.yaml"),model_tuned=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage",choices=("initialize","generate","train","report"),required=True)
    args=parser.parse_args();final=ROOT/"results/final_confirmatory";root=ROOT/"results/final_robustness"
    require_confirmation(final)
    if args.stage=="initialize":initialize(final,root)
    elif args.stage=="generate":generate(root)
    elif args.stage=="train":train(root)
    else:report(root)
