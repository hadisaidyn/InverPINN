"""Fit a single source using SciPy and the numerical PDE solver."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import yaml

from inverpinn.physics.inverse_source import fit_source
from inverpinn.evaluation.metrics import concentration_metrics, matched_source_metrics


def run(config):
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    dataset = Path(config["dataset"]).resolve()
    with np.load(dataset, allow_pickle=False) as data:
        n = config["n_sensors"]
        positions = data[f"sensor_positions_{n}"].copy()
        observations = data[f"measurements_{n}"].copy()
        x, y, t = (data[key].copy() for key in ("x", "y", "t"))
        transport = {key: float(data[key]) for key in ("u", "v", "D")}
    if (len(t) < 2 or t[0] != 0 or not np.isfinite(t).all() or t[1] <= 0
            or not np.allclose(t, np.arange(len(t))*t[1], rtol=1e-10, atol=1e-12)
            or not np.allclose(x, np.linspace(0, 1, len(x)))
            or not np.allclose(y, np.linspace(0, 1, len(y)))):
        raise ValueError("Require uniform unit-square grid and uniform times starting at zero.")
    solver = dict(nx=len(x), ny=len(y), dt=float(t[1]), steps=len(t)-1, **transport)
    config = config | dict(dataset=str(dataset), output_dir=str(output), solver=solver,
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        observations_sha256=hashlib.sha256(observations.tobytes()).hexdigest(),
        positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
        scipy_version=scipy.__version__, numpy_version=np.__version__, torch_version=str(torch.__version__),
        optimizer="least_squares/trf/2-point", dtype="float64", device="cpu",
        boundary="zero Dirichlet on every edge", initial_condition="zero",
        randomness="none in optimization; stored dataset observations used unchanged")
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    try:
        started = time.perf_counter()
        result, prediction, readings, history = fit_source(positions, observations,
            solver=solver, sigma=config["sigma"], initial=config["initial"],
            max_nfev=config["max_nfev"], tolerance=config["tolerance"])
        elapsed = time.perf_counter()-started
        pd.DataFrame(history).to_csv(output / "optimization_history.csv", index=False)
        np.savez_compressed(output / "checkpoint.npz", parameters=result.x,
                            positions=positions, observations=observations, predicted_measurements=readings)
        np.savez_compressed(output / "predictions.npz", C_predicted=prediction, x=x, y=y, t=t)
        # Truth is deliberately unavailable to the fitting routine.
        with np.load(dataset, allow_pickle=False) as data:
            truth = data["C"].copy()
            true_positions, true_Q = data["source_positions"].copy(), data["source_Q"].copy()
        metrics = dict(source_estimate=dict(zip(("x_s", "y_s", "Q"), result.x.tolist())),
            success=bool(result.success), status=int(result.status), message=str(result.message),
            nfev=int(result.nfev), objective_calls=len(history), elapsed_seconds=elapsed,
            optimality=float(result.optimality), sensor_mse=float(np.mean((readings-observations)**2)),
            concentration=concentration_metrics(prediction, truth))
        if len(true_Q) == 1:
            metrics["matched_sources"] = matched_source_metrics(result.x[None, :2], result.x[2:], true_positions, true_Q)
        else:
            metrics["source_evaluation_note"] = "Single-source hypothesis; localization not scored against multiple sources."
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for ax, field, title in zip(axes[:2], (truth[-1], prediction[-1]), ("Numerical reference", "Classical inverse")):
            mesh = ax.pcolormesh(x, y, field, shading="auto", cmap="inferno", vmin=0,
                                vmax=max(truth.max(), prediction.max(), 1e-12))
            ax.scatter(*positions.T, s=8, c="white", label="Sensors")
            ax.scatter(*true_positions.T, marker="o", facecolors="none", edgecolors="lime", label="True")
            ax.scatter(*result.x[:2], marker="x", c="cyan", label="Estimated")
            ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            ax.legend()
            fig.colorbar(mesh, ax=ax)
        axes[2].semilogy([r["evaluation"] for r in history], [max(r["mse"], 1e-30) for r in history])
        axes[2].set(xlabel="Objective call (includes Jacobian probes)", ylabel="Sensor MSE")
        fig.savefig(output / "classical_inverse.png", dpi=180)
        plt.close(fig)
        (output / "report.md").write_text("# Classical PDE-constrained inverse baseline\n\n"
            + f"Fit uses exactly {n} stored sensors and all their noisy time samples.\n\n"
            + "```json\n"+json.dumps(metrics, indent=2)+"\n```\n\n"
            + "Known wind, diffusion, width, zero initial condition and zero edge concentrations. "
            "One fixed initial guess; no truth-based tuning. Optimizer convergence does not guarantee a global optimum. "
            "Generation and inversion use the same discretization (an inverse crime): this tests implementation "
            "and is an optimistic baseline, not evidence of real-data performance. The PINN uses a continuous PDE "
            "residual instead; identical observations do not imply identical numerical error or compute.\n\n"
            "![Fit and optimization](classical_inverse.png)\n")
        if not result.success:
            raise RuntimeError(f"Inverse optimization did not converge; artifacts saved: {result.message}")
        return metrics
    except Exception as exc:
        (output / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classical_inverse.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(args.config.read_text())), indent=2))
