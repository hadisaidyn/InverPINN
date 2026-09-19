"""Evaluate fixed-setting interpolation baselines on identical sparse readings."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml

from inverpinn.models.interpolation import idw_interpolate, rbf_interpolate, ordinary_kriging
from inverpinn.evaluation.metrics import concentration_metrics


def evaluate(config):
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    dataset = Path(config["dataset"]).resolve()
    # Only noisy sensor observations and coordinates are exposed to interpolators.
    with np.load(dataset, allow_pickle=False) as data:
        n = config["n_sensors"]
        positions, readings = data[f"sensor_positions_{n}"].copy(), data[f"measurements_{n}"].copy()
        x, y, t = (data[key].copy() for key in ("x", "y", "t"))
    if len(readings) != len(t):
        raise ValueError("Observation times must match dataset times.")
    yy, xx = np.meshgrid(y, x, indexing="ij")
    queries = np.column_stack([xx.ravel(), yy.ravel()])
    flat = {
        "IDW": idw_interpolate(positions, readings, queries, power=config["idw_power"]),
        "RBF": rbf_interpolate(positions, readings, queries, smoothing=config["rbf_smoothing"]),
        "ordinary_kriging": ordinary_kriging(positions, readings, queries, range_length=config["kriging_range"], nugget=config["kriging_nugget"]),
    }
    # Truth is loaded only after every baseline prediction has been computed.
    with np.load(dataset, allow_pickle=False) as data:
        truth = data["C"].copy()
    predictions = {name: values.reshape(len(t), len(y), len(x)) for name, values in flat.items()}
    rows = []
    for name, predicted in predictions.items():
        for scope, a, b in (("space_time", predicted, truth), ("final_time", predicted[-1], truth[-1])):
            rows.append(dict(method=name, scope=scope, n_sensors=n, **concentration_metrics(a,b)))
    output.mkdir(parents=True)
    config = config | dict(dataset=str(dataset), output_dir=str(output),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        observations_sha256=hashlib.sha256(readings.tobytes()).hexdigest(),
        positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(), scipy_version=scipy.__version__,
        numpy_version=np.__version__, tuning="none; fixed settings, no dense-truth model selection")
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    # Observations+settings form a reproducible interpolation checkpoint.
    np.savez_compressed(output / "interpolation_checkpoint.npz", positions=positions, measurements=readings, t=t)
    np.savez_compressed(output / "predictions.npz", **predictions, x=x, y=y, t=t)
    fields = {"Numerical reference": truth[-1]} | {key: value[-1] for key,value in predictions.items()}
    lo, hi = min(value.min() for value in fields.values()), max(value.max() for value in fields.values())
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), layout="constrained")
    for ax, (name, field) in zip(axes, fields.items()):
        mesh = ax.pcolormesh(x, y, field, shading="auto", cmap="inferno", vmin=lo, vmax=max(hi,lo+1e-12))
        ax.scatter(positions[:,0], positions[:,1], s=10, c="cyan", edgecolors="black", linewidths=0.3)
        ax.set(xlabel="x", ylabel="y", aspect="equal", title=name)
    fig.colorbar(mesh, ax=list(axes), label="Concentration")
    fig.savefig(output / "baseline_predictions.png", dpi=180)
    plt.close(fig)
    lines = ["# Spatial interpolation baselines", "", f"All methods use exactly the same {n} sensor positions and noisy observations.",
        "Every observed time is interpolated independently; there is no temporal or PDE information.", "",
        "| Method | Space-time relative L2 | MAE | RMSE |", "|---|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "space_time":
            relative = "undefined" if row["relative_l2"] is None else f"{100*row['relative_l2']:.3f}%"
            lines.append(f"| {row['method']} | {relative} | {row['mae']:.6f} | {row['rmse']:.6f} |")
    lines += ["", f"Preset IDW power={config['idw_power']}; thin-plate RBF smoothing={config['rbf_smoothing']}, degree=1.",
        f"Ordinary kriging uses unit-sill exponential covariance, range={config['kriging_range']}, nugget={config['kriging_nugget']}.",
        "No settings are optimized using dense truth. No additional boundary pseudo-observations or clipping are used.",
        "These are simple fixed baselines, not claims about optimally tuned interpolation or calibrated kriging uncertainty.",
        "Full-grid errors include observed locations and extrapolation outside the sensor convex hull. This is one numerical scenario, not a held-out experiment.",
        "A PINN also receives physical constraints; identical observations do not imply identical information or compute budgets.",
        "", "![Final reconstructions](baseline_predictions.png)"]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/spatial_baselines.yaml"))
    args = parser.parse_args()
    print(evaluate(yaml.safe_load(args.config.read_text())).to_string(index=False))


if __name__ == "__main__":
    main()
