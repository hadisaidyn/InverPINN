"""Diagnose thermal inversions from prepared vertical profiles, never surface proxies."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import yaml

from inverpinn.data.inversions import diagnose_inversions, synthetic_profiles


def digest(path):
    """Hash an existing input or artifact for reproducibility."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(config):
    """Save config, coverage, layers, episodes, plot, checkpoint and provenance.

    No training or randomness is involved. Output must be new. Unavailable or
    invalid input writes a failure report, not a misleading zero-event table.
    """
    config = dict(config)
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    config["output_dir"] = str(output)
    if config.get("profiles_file"):
        config["profiles_file"] = str(Path(config["profiles_file"]).resolve())
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    try:
        if config.get("data_kind") not in ("real", "synthetic_test"):
            raise ValueError("data_kind must be real or synthetic_test.")
        if any(config.get(k) is None for k in ("start", "end", "height_bounds_m", "cadence_seconds")):
            raise ValueError("Study dates, cadence and height domain must be specified; template dates are not a confirmed study period.")
        if config["data_kind"] == "synthetic_test":
            if config.get("profiles_file"):
                raise ValueError("Synthetic method check uses manufactured profiles, not an input file.")
            ds = synthetic_profiles()
            provenance = dict(input="manufactured analytical fixture", input_sha256=None)
        else:
            path = Path(config["profiles_file"])
            if not path.is_file():
                raise FileNotFoundError(f"Vertical-profile input unavailable: {path}. t2m/u10/v10/blh cannot establish a thermal inversion.")
            if not config.get("usage_permission_confirmed") or not config.get("source_description"):
                raise ValueError("Real data require confirmed usage permission and source_description.")
            with xr.open_dataset(path) as opened:
                ds = opened.load()
            if ds.attrs.get("data_kind") != "real":
                raise ValueError("Synthetic profiles cannot be labelled a real study.")
            provenance = dict(input=str(path), input_sha256=digest(path),
                              source_description=config["source_description"])
        layers, coverage, events = diagnose_inversions(ds, **{k:config[k] for k in
            ("start", "end", "cadence_seconds", "height_bounds_m", "local_timezone")})
        tag = "SYNTHETIC METHOD CHECK - NOT OBSERVATIONS" if config["data_kind"] == "synthetic_test" else "REPORTED PROFILE INVERSION CANDIDATES"
        # Keep the data-kind label inside every CSV, not only in its filename.
        for name, table in (("inversion_layers", layers), ("profile_coverage", coverage), ("inversion_events", events)):
            table.insert(0, "data_kind", config["data_kind"])
            table.to_csv(output / f"{name}.csv", index=False)
        ds.to_netcdf(output / "profiles_checkpoint.nc")
        summary = dict(status="assessed", data_kind=config["data_kind"],
            study_start=config["start"], study_end=config["end"],
            profile_status_counts={str(k):int(v) for k,v in coverage.status.value_counts().items()},
            candidate_layers=len(layers), candidate_events=len(events),
            partial_profiles=int(coverage.partial_profile.sum()),
            caveat="Resolved candidates, not confidence probabilities. No detection does not establish physical absence.")
        (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        for site, group in coverage.groupby("site", sort=False):
            times = pd.to_datetime(group.time_utc, utc=True)
            ax.plot(times, group.strongest_layer_K, marker="o", label=site)
            missing = group.status == "insufficient_data"
            ax.scatter(times[missing], np.zeros(missing.sum()), marker="x", color="red")
        ax.set(title=tag+"\nRed crosses: unassessable samples", xlabel="UTC", ylabel="Strongest resolved layer temperature rise (K)")
        ax.legend()
        fig.autofmt_xdate()
        fig.savefig(output / "inversion_diagnostic.png", dpi=160)
        plt.close(fig)
        provenance.update(data_kind=config["data_kind"], deterministic="No random draws; no fitted thresholds",
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, xarray=xr.__version__),
            code_sha256={str(p):digest(p) for p in (Path(__file__).resolve(),
                Path(__file__).resolve().parents[1]/"src/inverpinn/data/inversions.py")})
        provenance["output_sha256"] = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
        report = [f"# {tag}", "", f"Candidate events: {len(events)}; candidate layers: {len(layers)}.",
            "", "## Facts about the input", "", ds.attrs["source_description"],
            f"Requested interval: {config['start']} to {config['end']} (inclusive).",
            f"Investigated heights: {config['height_bounds_m']} m AGL.",
            f"Profile status counts: {summary['profile_status_counts']}.",
            "", "## Diagnostic estimates", "", "Definition: adjacent actual-temperature gradient > 0 K/m.",
            "No minimum-strength/depth/duration filter; missing values are not interpolated.",
            "See inversion_events.csv, inversion_layers.csv and profile_coverage.csv.",
            "", "## Interpretation and uncertainty", "",
            "Candidates are limited by vertical/time sampling and temperature errors, not confirmed continuous events.",
            "Elapsed sample span is not exact event duration. Runs can involve different layers.",
            "No inversion probabilities, emission attribution or causal PM2.5 claims are made.",
            "Scientific definition and references: docs/thermal_inversions.md in the repository.",
            "No neural network is fitted; profiles_checkpoint.nc is the diagnostic input checkpoint."]
        (output / "report.md").write_text("\n".join(report)+"\n")
        return summary
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        (output / "summary.json").write_text(json.dumps(dict(status="not_assessed", reason=message,
            candidate_events=None, caveat="Not assessed is not zero inversion events."), indent=2)+"\n")
        (output / "report.md").write_text("# Inversion study NOT ASSESSED\n\n"+message+
            "\n\nNo valid study-period event table is available. This does not imply zero events.\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/inversions.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(args.config.read_text())), indent=2))
