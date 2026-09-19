"""Prepare one reproducible PM2.5/ERA5 modeling NetCDF from authorized local inputs."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
import pandas as pd
import pyproj
import xarray as xr
import yaml

from inverpinn.data.era5 import ingest_era5, cache_raw
from inverpinn.data.modeling import merge_observations


def run(config):
    """Cache immutable raw copies, normalize, merge, and save data/QC/provenance.

    Output directory must be new and disjoint from the raw cache. No random
    draws, fitted scaling, row dropping, network requests or model training.
    The NetCDF plus config/provenance is the preprocessing checkpoint.
    """
    output, raw = Path(config["output_dir"]).resolve(), Path(config["raw_cache_dir"]).resolve()
    if output == raw or output in raw.parents or raw in output.parents:
        raise ValueError("Raw cache and processed output must be separate directory trees.")
    if output.exists():
        raise FileExistsError(output)
    # Refuse to label retrospective ERA5 as real-time data before accessing files.
    if config["purpose"] != "retrospective_reanalysis":
        raise ValueError("ERA5 pipeline supports retrospective_reanalysis only.")
    output.mkdir(parents=True)
    resolved = config | dict(output_dir=str(output), raw_cache_dir=str(raw),
        era5_files=[str(Path(p).resolve()) for p in config["era5_files"]],
        sensor_csv=str(Path(config["sensor_csv"]).resolve()))
    (output / "config.yaml").write_text(yaml.safe_dump(resolved))
    try:
        met, era5_sources = ingest_era5(config["era5_files"], bbox=config["bbox"],
            start=config["start"], end=config["end"], raw_cache_dir=raw / "era5")
        pm_source = cache_raw(config["sensor_csv"], raw / "pm25")
        observations = pd.read_csv(pm_source["cached"], dtype={"sensor_id":str})
        model, qc = merge_observations(observations, met, sensor_timezone=config["sensor_timezone"],
            max_age_seconds=config["max_age_seconds"], require_blh=config["require_blh"],
            purpose=config["purpose"], pm25_units=config["pm25_units"], timestamp_meaning=config["timestamp_meaning"])
        met.to_netcdf(output / "meteorology.nc", engine="netcdf4")
        model.to_netcdf(output / "modeling.nc", engine="netcdf4")
        (output / "qc_summary.json").write_text(json.dumps(qc, indent=2, allow_nan=False)+"\n")
        try:
            tz_version = importlib.metadata.version("tzdata")
        except importlib.metadata.PackageNotFoundError:
            tz_version = "system IANA database; OS="+platform.platform()
        provenance = dict(inputs=dict(era5=era5_sources, pm25=pm_source),
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                          xarray=xr.__version__, pyproj=pyproj.__version__, proj=pyproj.proj_version_str,
                          netCDF4=netCDF4.__version__, tzdata=tz_version),
            deterministic="no stochastic preprocessing", units={k:v.attrs["units"] for k,v in model.variables.items() if "units" in v.attrs},
            source_label=config["source_label"], usage_permission=config["usage_permission"])
        provenance["outputs_sha256"] = {}
        provenance["code_sha256"] = {}
        for path in [Path(__file__), *Path(__file__).resolve().parents[1].joinpath("src/inverpinn/data").glob("*.py")]:
            with path.open("rb") as stream:
                provenance["code_sha256"][path.name] = hashlib.file_digest(stream,"sha256").hexdigest()
        for name in ("meteorology.nc", "modeling.nc"):
            with (output / name).open("rb") as stream:
                provenance["outputs_sha256"][name] = hashlib.file_digest(stream,"sha256").hexdigest()
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
        fig, ax = plt.subplots(figsize=(7,5), layout="constrained")
        finite = np.isfinite(model.x) & np.isfinite(model.y)
        for ready, color, label in ((1,"tab:blue","Ready"),(0,"tab:red","Flagged")):
            mask = finite & (model.ready_for_model == ready)
            ax.scatter(model.x.values[mask], model.y.values[mask], c=color, s=25, label=label, alpha=0.6)
        ax.set(xlabel="UTM 43N easting (m)", ylabel="UTM 43N northing (m)", aspect="equal", title="Observation coordinates and QC")
        ax.legend()
        fig.savefig(output / "qc_positions.png", dpi=160)
        plt.close(fig)
        lines = ["# Modeling dataset quality control", "", f"Rows: {qc['rows']}; ready: {qc['ready_for_model']}; flagged: {qc['not_ready']}.",
            "No rows dropped. Flags overlap. Missing values were not filled.", "", "| Flag | Rows |", "|---|---:|"]
        lines += [f"| {key} | {value} |" for key,value in qc["flag_counts"].items()]
        lines += ["", "UTC time; WGS84 longitude/latitude retained; EPSG:32643 metres for modeling.",
            "Past-valid-hour matching with recorded age; strict bilinear spatial interpolation.",
            "This is retrospective ERA5, not a dataset certified for real-time forecasts.",
            "No training or model weights: modeling.nc, configuration and provenance form the preprocessing checkpoint.",
            "", "![Observation QC](qc_positions.png)"]
        (output / "qc_report.md").write_text("\n".join(lines)+"\n")
        return qc
    except Exception as exc:
        (output / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/almaty.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(args.config.read_text())), indent=2))
