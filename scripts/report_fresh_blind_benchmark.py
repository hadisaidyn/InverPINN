"""Complete revision-5 confirmatory report, preserving every attempted run."""

import argparse
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import yaml

from inverpinn.data.fresh_blind import install_guard, jobs, manifest, verify
from inverpinn.data.single_source_benchmark import object_hash, write_csv
from inverpinn.data.synthetic_reference import ROOT, load_model_inputs, sha256, write_json
from inverpinn.evaluation.fresh_blind import (METRICS, exploratory_correlations, interpretation, paired_analysis,
    primary_rows, seed_sensitivity, summarize_model)
from inverpinn.visualization.fresh_blind import figures, reconstruction
from run_fresh_blind_benchmark import run_path


def records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def report(root):
    install_guard(root)
    config, _, lock = verify(root)
    manifest_rows = manifest(root)
    if (root/"summary.json").exists() or (root/"report_started.json").exists():
        raise FileExistsError("Existing completed/partial report preserved; no overwrite.")
    if not (root/"fits_complete.json").exists() or not (root/"identifiability/summary.json").exists():
        raise RuntimeError("All registered fits and both identifiability diagnostics must finish first.")
    rows = pd.DataFrame([json.loads((run_path(root,*job)/"metrics.json").read_text()) for job in jobs(config)])
    write_json(root/"report_started.json", dict(started_at_utc=datetime.now(timezone.utc).isoformat()))
    baseline, revised = (primary_rows(rows,config,m) for m in ("B0","B_revised"))
    paired, comparison = paired_analysis(baseline,revised,config["analysis"])
    seeds, sensitivity = seed_sensitivity(rows,config)
    for name,frame in (("B0_raw",baseline),("B_revised_raw",revised),("all_runs",rows),
                       ("paired_results",paired),("seed_sensitivity",seeds)):
        frame.to_csv(root/f"{name}.csv",index=False)
    write_csv(root/"seed_sensitivity_summary.csv",sensitivity)
    summary = {name:summarize_model(frame,config) for name,frame in (("B0",baseline),("B_revised",revised))}
    summary.update(paired=comparison, seed_sensitivity=sensitivity,
                   classification=interpretation(summary["B0"],summary["B_revised"],config),
                   bootstrap=dict(method="Paired percentile resampling of scenario indices; conditional on fixed setup",
                       seed=config["analysis"]["bootstrap_seed"],replicates=config["analysis"]["bootstrap_samples"]))
    features=[]
    sensor_positions=np.array(json.loads((root/"sensor_geometry.json").read_text()))
    for row in revised.itertuples():
        inputs=load_model_inputs(root/"datasets"/row.scenario_id)
        features.append(dict(scenario_id=row.scenario_id,
            nearest_sensor_distance=float(np.linalg.norm(sensor_positions-[row.x_true,row.y_true],axis=1).min()),
            boundary_distance=float(min(row.x_true,row.y_true,1-row.x_true,1-row.y_true)),
            observed_sensor_rms=float(inputs["measurements"].square().mean().sqrt())))
    feature_frame=pd.DataFrame(features)
    feature_frame.to_csv(root/"exploratory_features.csv",index=False)
    associations=exploratory_correlations(revised,feature_frame)
    write_csv(root/"exploratory_correlations.csv",associations)
    summary["exploratory_correlations"]=associations
    quality=[]
    for row in manifest_rows:
        reference_table=pd.read_csv(root/"datasets"/row["scenario_id"]/"truth/reference_benchmark.csv")
        subset=reference_table[(reference_table.grid_size==641)&reference_table.scope.isin(["selected_field","sensor_trajectory"])]
        for entry in records(subset):
            quality.append(dict(scenario_id=row["scenario_id"],scope=entry["scope"],time=entry["time"],
                Richardson_estimate=entry["Richardson_relative_L2_estimate"],order=entry["Richardson_order"],
                target_met=entry["Richardson_relative_L2_estimate"] is not None and entry["Richardson_relative_L2_estimate"] <= config["reference"]["target_relative_L2"]))
    write_csv(root/"reference_quality.csv",quality)
    summary["reference_quality"]=dict(certified_bound=False, target=config["reference"]["target_relative_L2"],
        flagged_scenarios=[r["scenario_id"] for r in manifest_rows if not r["reference_target_met"]],
        all_assessed_components=quality)
    ranked=revised.sort_values(["localization_error","scenario_id"],na_position="last")
    selected={"best_case":records(ranked.iloc[[0]])[0],"median_case":records(ranked.iloc[[(len(ranked)-1)//2]])[0],
              "worst_case":records(ranked.iloc[[-1]])[0]}
    summary["mechanical_selections"]={k:v["scenario_id"] for k,v in selected.items()}
    summary["worst_three_localization_ranked"]=records(ranked.iloc[-3:].iloc[::-1])
    summary["identifiability"]=json.loads((root/"identifiability/summary.json").read_text())
    base=["models/mlp.py","physics/advection_diffusion.py","physics/autodiff.py","physics/trainable_source.py",
          "training/forward_pinn.py","training/losses.py","training/single_source.py"]
    extra=["models/constrained_concentration.py","physics/diagnostic_source.py","physics/nondimensional.py","training/development_candidates.py"]
    dependencies={m:{"src/inverpinn/"+name:lock["code_hashes"]["src/inverpinn/"+name] for name in names}
                  for m,names in (("B0",base),("B_revised",base+extra))}
    write_json(root/"model_code_hashes.json",dependencies)
    summary["provenance"]=dict(frozen_commit=config["frozen_commit"],model_config_hashes=lock["model_config_hashes"],
        model_code_bundle_hashes={m:object_hash(files) for m,files in dependencies.items()},
        solver_hash=lock["reference_solver_hash"],protocol_hash=lock["protocol_hash"],
        manifest_sha256=sha256(root/"scenario_manifest.csv"),sensor_layout_hash=lock["sensor_layout_hash"],
        old_data_or_outcomes_opened=False, models_changed=False)
    summary["limitations"]=["Thirty sources, one fixed layout/transport/noise setting; no real-world inference",
        "Additional optimizer seeds never replace primary seed 42; physical source initialization unchanged",
        "Richardson estimates are conditional and target misses retained",
        "Discrete coarse-solver compatibility landscapes are not posterior probabilities",
        "Exploratory correlations are associations, not causes; source-strength/geometry confounding remains"]
    flat=[dict(model=model,metric=metric,**stats) for model in ("B0","B_revised") for metric,stats in summary[model]["metrics"].items()]
    write_csv(root/"summary.csv",flat)
    figures(root,baseline,revised,paired,seeds,config)
    for name,row in selected.items():
        reconstruction(root,row,root/name,config)
    verify(root); manifest(root)
    write_json(root/"summary.json",summary)
    from run_model_diagnosis import snapshot
    snapshot_dir=root/"report_execution"
    snapshot_dir.mkdir()
    snapshot(snapshot_dir)
    hashes={str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in ("artifact_hashes.json","report_complete.json")}
    write_json(root/"artifact_hashes.json",hashes)
    write_json(root/"report_complete.json",dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_manifest_sha256=sha256(root/"artifact_hashes.json"),summary_sha256=sha256(root/"summary.json"),
        classification=summary["classification"]))
    print(json.dumps({k:summary[k] for k in ("classification","B0","B_revised","paired")},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/fresh_blind_benchmark.yaml")
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/args.config).read_text())
    report(ROOT/config["output_dir"])
