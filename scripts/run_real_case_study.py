"""Exploratory winter case study: measured readings != inferred source intensity."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from inverpinn.data.coordinates import PROJECTED_CRS, to_wgs84, to_projected
from inverpinn.data.episode import split_sensors, select_winter_episode, episode_inputs
from inverpinn.evaluation.metrics import concentration_metrics
from inverpinn.training.case_study import train_case_study
from inverpinn.training.forward_pinn import require_finite


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load_inputs(config):
    """Verify paired processed inputs/provenance, never relabel a known fixture as real."""
    paths={key:Path(config[key]).resolve() for key in ("observations","meteorology","provenance")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Verified processed case-study input missing: {path}. No episode or real result can be claimed.")
    manifest=json.loads(paths["provenance"].read_text())
    for key,name in (("observations","modeling.nc"),("meteorology","meteorology.nc")):
        if sha256(paths[key]) != manifest["outputs_sha256"][name]:
            raise ValueError(f"Provenance hash mismatch for {key}.")
    if config["data_kind"] not in ("real","synthetic_test"):
        raise ValueError("data_kind must be real or synthetic_test.")
    if config["data_kind"] == "real":
        if config["permission_confirmed"] is not True:
            raise ValueError("Real data requires explicit permission/provenance confirmation.")
        description=str(manifest.get("source_label",""))+" "+str(manifest.get("usage_permission",""))
        if not manifest.get("source_label") or not manifest.get("usage_permission") or any(
                word in description.lower() for word in ("synthetic","fixture","demo","replace","unverified")):
            raise ValueError("Synthetic/placeholder/unverified provenance cannot support a real case study.")
    with xr.open_dataset(paths["observations"]) as ds:
        obs=ds.load()
    with xr.open_dataset(paths["meteorology"]) as ds:
        met=ds.load()
    for ds in (obs,met):
        if ds.attrs.get("crs") != "EPSG:32643" or ds.attrs.get("time_basis") != "UTC":
            raise ValueError("Require processed EPSG:32643 coordinates and UTC time.")
        if ds.attrs.get("purpose") != "retrospective_reanalysis":
            raise ValueError("Only documented retrospective datasets are supported.")
    if obs.attrs.get("pm25_timestamp_meaning") != "instantaneous" or obs.pm25.attrs.get("units") != "ug m-3":
        raise ValueError("Require instantaneous PM2.5 in ug m-3.")
    for key,unit in (("u_x","m s-1"),("v_y","m s-1"),("t2m","K")):
        if met[key].attrs.get("units") != unit:
            raise ValueError(f"Unexpected {key} units.")
    if obs.time.attrs.get("timezone") != "UTC" or met.time.attrs.get("timezone") != "UTC":
        raise ValueError("Explicit UTC timestamp metadata required.")
    mt=pd.DatetimeIndex(met.time.values)
    if mt.hasnans or mt.has_duplicates or not mt.is_monotonic_increasing or not mt.equals(mt.floor("h")):
        raise ValueError("Meteorology must use unique increasing exact hourly UTC timestamps.")
    if not np.isin(obs.ready_for_model.values,[0,1]).all():
        raise ValueError("ready_for_model must be an explicit binary QC mask.")
    ready=obs.ready_for_model.values==1
    for name in ("x","y"):
        if obs[name].attrs.get("units") != "m":
            raise ValueError("Processed observation x/y coordinates must be in metres.")
    if ready.any():
        px,py=to_projected(obs.longitude.values[ready],obs.latitude.values[ready])
        if not (np.allclose(px,obs.x.values[ready],atol=1e-3,rtol=0) and np.allclose(py,obs.y.values[ready],atol=1e-3,rtol=0)):
            raise ValueError("Observation projected/geographic coordinates disagree.")
    return obs,met,manifest,{key:str(path) for key,path in paths.items()}


def run(config):
    """Fixed-budget multiseed inverse reconstruction, with no emitter attribution."""
    seeds=config["training_seeds"]
    if len(seeds)<2 or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError("At least two distinct nonnegative training seeds are required.")
    if type(config["map_size"]) is not int or config["map_size"]<3:
        raise ValueError("map_size must be integer >=3.")
    training=config["training"]
    for key in ("steps","width","depth","data_batch","collocation_batch","initial_batch","boundary_per_edge"):
        if type(training[key]) is not int or training[key]<1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("learning_rate","time_scale_seconds","concentration_scale_ug_m3"):
        if not np.isfinite(training[key]) or training[key]<=0:
            raise ValueError(f"{key} must be positive and finite.")
    if not np.isfinite(training["diffusion_m2_s"]) or training["diffusion_m2_s"]<0:
        raise ValueError("diffusion_m2_s must be finite and nonnegative.")
    if set(training["loss_weights"]) != {"data","pde","initial","boundary"} or any(
            not np.isfinite(v) or v<=0 for v in training["loss_weights"].values()):
        raise ValueError("Case-study maps require four positive finite loss weights; ablations belong in the separate runner.")
    output=Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    obs,met,manifest,paths=load_inputs(config)
    training_ids,heldout_ids=split_sensors(obs,seed=config["split_seed"],holdout_fraction=config["holdout_fraction"])
    episode,candidates=select_winter_episode(obs,met,training_ids,**config["episode"])
    frame,winds,domain,diagnostics=episode_inputs(obs,met,episode,training_ids,heldout_ids)
    output.mkdir(parents=True)
    resolved=config | paths | dict(output_dir=str(output),episode_selected=episode,domain=domain,
        training_sensors=training_ids,heldout_sensors=heldout_ids,diagnostics=diagnostics,
        torch_version=str(torch.__version__),numpy_version=np.__version__,input_sha256={k:sha256(v) for k,v in paths.items()})
    (output/"config.yaml").write_text(yaml.safe_dump(resolved))
    (output/"input_provenance.json").write_text(json.dumps(manifest,indent=2)+"\n")
    code_paths=[Path(__file__),*Path(__file__).resolve().parents[1].joinpath("src/inverpinn").glob("*/case_study.py"),
                Path(__file__).resolve().parents[1]/"src/inverpinn/data/episode.py"]
    (output/"code_sha256.json").write_text(json.dumps({str(p):sha256(p) for p in code_paths},indent=2)+"\n")
    (output/"episode.json").write_text(json.dumps(episode,indent=2)+"\n")
    candidates.to_csv(output/"episode_candidates.csv",index=False)
    frame.to_csv(output/"episode_observations.csv",index=False)
    pd.DataFrame(winds,columns=["regional_u_x_m_s","regional_v_y_m_s"]).assign(hour=np.arange(len(winds))).to_csv(output/"episode_meteorology.csv",index=False)
    try:
        ts,cs=config["training"]["time_scale_seconds"],config["training"]["concentration_scale_ug_m3"]
        grid=np.linspace(0,1,config["map_size"])
        gy,gx=np.meshgrid(grid,grid,indexing="ij")
        x=domain["x0"]+domain["length_m"]*grid
        y=domain["y0"]+domain["length_m"]*grid
        lon,lat=to_wgs84(domain["x0"]+gx*domain["length_m"],domain["y0"]+gy*domain["length_m"])
        maps,metrics,source_rows,prediction_rows=[],[],[],[]
        for seed in seeds:
            folder=output/f"seed_{seed}"
            folder.mkdir()
            with (folder/"training_history.jsonl").open("w") as stream:
                def save(record):
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                    stream.flush()
                model,source,optimizer,history=train_case_study(frame[frame.split=="train"],winds,domain,
                    diagnostics["initial_boundary_background_ug_m3"],config["training"],seed=seed,on_step=save)
            torch.save(dict(model_state_dict=model.state_dict(),source_state_dict=source.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),config=resolved,seed=seed,epoch=config["training"]["steps"]),folder/"model.pt")
            evaluation=torch.tensor(np.column_stack([(frame.x-domain["x0"])/domain["length_m"],
                (frame.y-domain["y0"])/domain["length_m"],frame.time_seconds/ts]),dtype=torch.float32)
            with torch.no_grad():
                prediction=model(evaluation).squeeze(1)*cs
                intensity=source(torch.tensor(gx,dtype=torch.float32),torch.tensor(gy,dtype=torch.float32))*cs/ts
            require_finite(prediction,"case-study sensor evaluation")
            require_finite(intensity,"case-study intensity map")
            maps.append(intensity.numpy())
            for split in ("train","heldout"):
                mask=frame.split.eq(split).to_numpy()
                metrics.append(dict(seed=seed,split=split,n_readings=int(mask.sum()),
                    **concentration_metrics(prediction.numpy()[mask],frame.pm25.to_numpy()[mask]),
                    negative_prediction_fraction=float((prediction.numpy()[mask]<0).mean())))
            predicted=frame.copy()
            predicted["prediction_ug_m3"]=prediction.numpy()
            predicted["seed"]=seed
            prediction_rows.append(predicted)
            for i,estimate in enumerate(source.estimates()["sources"]):
                px,py=domain["x0"]+estimate["x_s"]*domain["length_m"],domain["y0"]+estimate["y_s"]*domain["length_m"]
                plon,plat=to_wgs84(px,py)
                source_rows.append(dict(seed=seed,component=i,x=px,y=py,longitude=float(plon),latitude=float(plat),
                    peak_ug_m3_s=estimate["Q"]*cs/ts,sigma_m=config["training"]["sources"][i]["sigma_m"]))
            fig,ax=plt.subplots(layout="constrained")
            for key in ("data","pde","initial","boundary","total"):
                ax.semilogy([row["epoch"] for row in history],[max(row[f"loss/{key}"],1e-30) for row in history],label=key)
            ax.set(xlabel="Epoch",ylabel="Scaled MSE / weighted total",title=f"Case-study fit: seed {seed}")
            ax.legend()
            fig.savefig(folder/"training_losses.png",dpi=150)
            plt.close(fig)
            print(f"Finished seed {seed}; source map is a model estimate, not an emitter inventory.",flush=True)
        maps=np.stack(maps)
        product=xr.Dataset(dict(source_intensity=(("seed","y","x"),maps),
            source_mean=(("y","x"),maps.mean(axis=0)),source_std=(("y","x"),maps.std(axis=0,ddof=1))),
            coords=dict(seed=seeds,x=x,y=y,longitude=(("y","x"),lon),latitude=(("y","x"),lat)))
        for name in ("source_intensity","source_mean","source_std"):
            product[name].attrs=dict(units="ug m-3 s-1",grid_mapping="crs")
        product["crs"]=xr.DataArray(0,attrs=PROJECTED_CRS.to_cf())
        product.x.attrs["units"]=product.y.attrs["units"]="m"
        product.longitude.attrs["units"]="degrees_east"
        product.latitude.attrs["units"]="degrees_north"
        product.attrs=dict(crs="EPSG:32643",data_kind=config["data_kind"],episode_start=episode["start_utc"],episode_end_exclusive=episode["end_utc"],
            validation_status="exploratory_not_validated_for_emitter_attribution",
            interpretation="Exploratory transport-equivalent source intensity; not calibrated likelihood, mass emissions or emitter attribution",
            uncertainty="Sample SD across optimizer seeds only; excludes measurement and structural uncertainty")
        product.to_netcdf(output/"source_intensity.nc")
        pd.DataFrame(metrics).to_csv(output/"sensor_metrics.csv",index=False)
        pd.DataFrame(source_rows).to_csv(output/"estimated_components.csv",index=False)
        pd.concat(prediction_rows).to_csv(output/"sensor_predictions.csv",index=False)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
        for ax,values,title in zip(axes,(maps.mean(axis=0),maps.std(axis=0,ddof=1)),("Estimated effective source intensity","Optimizer-seed SD (not confidence)")):
            mesh=ax.pcolormesh(x,y,values,shading="auto",cmap="inferno",vmin=0)
            for split,marker,color in (("train","o","cyan"),("heldout","^","lime")):
                points=frame[frame.split==split].drop_duplicates(["sensor_id","x","y"])
                ax.scatter(points.x,points.y,s=15,marker=marker,c=color,label=f"{split} sensors")
            ax.set(title=title,xlabel="UTM 43N easting (m)",ylabel="UTM 43N northing (m)",aspect="equal")
            ax.legend(fontsize=8)
            fig.colorbar(mesh,ax=ax,label="µg m⁻³ s⁻¹")
        label="SYNTHETIC METHOD CHECK — NOT REAL OBSERVATIONS" if config["data_kind"]=="synthetic_test" else "EXPLORATORY MODEL ESTIMATE — NOT EMITTER ATTRIBUTION"
        fig.suptitle(label)
        fig.savefig(output/"source_intensity.png",dpi=170)
        plt.close(fig)
        scores=pd.DataFrame(metrics)
        facts=dict(input_qc_ready_rows=int((obs.ready_for_model.values==1).sum()),input_total_rows=obs.sizes["observation"],
            selected_rows=len(frame),training_sensors=training_ids,heldout_sensors=heldout_ids,
            observed_pm25_min=float(frame.pm25.min()),observed_pm25_max=float(frame.pm25.max()),
            observed_pm25_mean=float(frame.pm25.mean()))
        (output/"reported_observation_summary.json").write_text(json.dumps(facts,indent=2)+"\n")
        lines=["# Exploratory winter pollution case study", "", label, "", "## Measured facts", "",
            "No real measured facts: this run is a synthetic method check." if config["data_kind"]=="synthetic_test" else
            "The following describes reported sensor measurements, not independently verified source activity.",
            f"Episode: {episode['start_utc']} to {episode['end_utc']} (end exclusive).",
            f"Selection: {episode['selection']}; {episode['candidate_count']} admissible windows. Winter: {episode['winter_definition']}.",
            f"Selected readings: {len(frame)}; reported PM2.5 range {frame.pm25.min():.3f}–{frame.pm25.max():.3f} µg/m³.",
            "Candidate scores use training sensors only. Held-out stations never select the episode or fit model/priors.",
            "Input manifest/hashes and QC exclusions are recorded. No emission-source ground truth is supplied.",
            "", "## Model estimates", "", "The map is positive, time-constant, transport-equivalent forcing in µg m⁻³ s⁻¹.",
            "It is not a calibrated source probability, mass emission flux, or a measured concentration map.",
            "", "| Split | Mean sensor MAE | Mean sensor RMSE | Mean sensor relative L2 |", "|---|---:|---:|---:|"]
        for split in ("train","heldout"):
            group=scores[scores.split==split]
            relative="undefined" if group.relative_l2.isna().any() else f"{group.relative_l2.mean():.5f}"
            lines.append(f"| {split} | {group.mae.mean():.5f} | {group.rmse.mean():.5f} | {relative} |")
        lines += ["", "Errors compare sensor concentrations only; dense reconstruction/localization accuracy is unknown.",
            "All seeds/final checkpoints are retained. Map SD reflects optimizer-seed sensitivity only, not a confidence interval.",
            "", "## Interpretation", "", "Brighter regions are where this specified transport model assigns greater effective positive forcing.",
            "They may reflect local emissions, transported/background pollution, meteorological or sensor error, or model misspecification.",
            "They do not establish facilities, ownership, undocumented activity, illegality, or regulatory violations.",
            "Any follow-up attribution requires independent measurements, emissions records and domain-expert validation.",
            "", "## Assumptions and uncertainties", "",
            "- Retrospective reanalysis, not operational forecasting or causal source attribution.",
            "- Regional hourly wind is spatially uniform; sub-grid mountain/urban flow is unresolved. No missing wind hours are filled.",
            "- Fixed effective diffusion, source count and Gaussian widths; source forcing is constant over the episode.",
            "- Initial concentration and all-edge boundary concentration share a constant training-derived background, an empirical prior, not a measured spatial field.",
            "- 2-D planar transport omits vertical mixing, terrain, deposition and chemistry. Temperature and BLH are diagnostics, not fitted transport terms.",
            "- No conversion to mass emissions: concentration forcing has no independently validated mixing-depth/flux interpretation.",
            "- Limited sensor coverage/calibration and episode-selection bias remain. A high-pollution selected window is not representative of all winter conditions.",
            "- Nonunique inverse solutions and Gaussian/IC/BC/diffusion priors can dominate the map; seed spread does not capture this full uncertainty.",
            "- Held-out-station errors assess within-episode spatial reconstruction, not future forecasting or emitter localization.",
            "- Facility attribution and calibrated likelihood claims are explicitly unsupported.",
            "", "![Exploratory source intensity and optimizer variation](source_intensity.png)"]
        (output/"report.md").write_text("\n".join(lines)+"\n")
        return scores
    except Exception as exc:
        (output/"failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/real_case_study.yaml"))
    args=parser.parse_args()
    print(run(yaml.safe_load(args.config.read_text())).to_string(index=False))
