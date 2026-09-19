"""Jointly estimate concentration and K Gaussian sources from sparse data."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from inverpinn.physics.trainable_source import TrainableGaussianSource, TrainableGaussianMixture
from inverpinn.evaluation.metrics import matched_source_metrics
from inverpinn.training.forward_pinn import train_forward_pinn, require_finite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/inverse_pinn.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    dataset = Path(config["dataset"]).resolve()
    # Do not load C, source_positions or source_Q before optimization.
    with np.load(dataset, allow_pickle=False) as data:
        n = config["n_sensors"]
        positions = torch.from_numpy(data[f"sensor_positions_{n}"].copy())
        times = torch.from_numpy(data["t"].copy())
        observations = torch.from_numpy(data[f"measurements_{n}"].copy())
        x, y = data["x"].copy(), data["y"].copy()
        known = {key: float(data[key]) for key in ("u", "v", "D")}
    if not (x[0] == y[0] == 0 and x[-1] == y[-1] == 1):
        raise ValueError("Inverse model requires a unit-square domain.")
    initial = config["source_initial"]
    source = TrainableGaussianMixture(initial) if isinstance(initial, list) else TrainableGaussianSource(**initial)
    output.mkdir(parents=True)
    config.update(dataset=str(dataset), output_dir=str(output), known_transport=known,
                  dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  torch_version=str(torch.__version__), dtype="float32", device="cpu")
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    logger = logging.getLogger("inverpinn.inverse")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "training.log")
    logger.addHandler(handler)
    try:
        with (output / "training_history.jsonl").open("w") as stream:
            def save_record(record):
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
            model, optimizer, history = train_forward_pinn(positions, times, observations, known, config,
                logger, on_step=save_record, source_model=source)
        torch.save({"model_state_dict": model.state_dict(), "source_state_dict": source.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "epoch": config["steps"], "config": config}, output / "model.pt")
        estimates = source.estimates()
        estimated_sources = estimates.get("sources", [estimates])
        (output / "source_estimate.json").write_text(json.dumps(estimates, indent=2, allow_nan=False)+"\n")
        # Evaluation-only truth; it does not choose checkpoints or hyperparameters.
        with np.load(dataset, allow_pickle=False) as data:
            truth = data["C"].copy()
            true_positions, true_Q = data["source_positions"].copy(), data["source_Q"].copy()
        prediction = np.empty_like(truth)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        model.eval()
        with torch.no_grad():
            for i, time in enumerate(times):
                points = torch.tensor(np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, time.item())], 1), dtype=torch.float32)
                values = model(points)
                require_finite(values, f"evaluation t index {i}")
                prediction[i] = values.numpy().reshape(len(y), len(x))
        require_finite(torch.from_numpy(truth), "reference concentration")
        error = prediction-truth
        norm = np.linalg.norm(truth.ravel())
        metrics = {"source_estimate": estimates, "relative_l2": float(np.linalg.norm(error.ravel())/norm) if norm else None,
                   "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error**2).mean()))}
        metrics["matched_sources"] = matched_source_metrics(
            [[s["x_s"], s["y_s"]] for s in estimated_sources], [s["Q"] for s in estimated_sources], true_positions, true_Q)
        if len(true_Q) == 1 and "sources" not in estimates:
            metrics.update(true_source={"x_s": float(true_positions[0, 0]), "y_s": float(true_positions[0, 1]), "Q": float(true_Q[0])},
                           source_location_error=float(np.linalg.norm(np.array([estimates["x_s"], estimates["y_s"]])-true_positions[0])),
                           source_Q_absolute_error=abs(estimates["Q"]-float(true_Q[0])))
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
        np.savez_compressed(output / "predictions.npz", C_predicted=prediction, t=times.numpy(), x=x, y=y)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, name in zip(axes, ("x_s", "y_s", "Q")):
            for i in range(len(estimated_sources)):
                key = f"source/{i}/{name}" if "sources" in estimates else f"source/{name}"
                ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=f"Source {i+1}")
            ax.set(xlabel="Epoch", ylabel=name, title=f"Estimated {name}")
            ax.legend()
        fig.savefig(output / "source_parameters.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(layout="constrained")
        for name in ("data", "pde", "initial", "boundary", "total"):
            ax.semilogy([r["epoch"] for r in history], [max(r[f"loss/{name}"], 1e-30) for r in history], label=name)
        ax.set(xlabel="Epoch", ylabel="Raw MSE / weighted total", title="Inverse PINN losses")
        ax.legend()
        fig.savefig(output / "training_losses.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        mesh = ax.pcolormesh(x, y, prediction[-1], shading="auto", cmap="inferno")
        ax.scatter([s["x_s"] for s in estimated_sources], [s["y_s"] for s in estimated_sources], marker="x", c="cyan", label="Estimated sources")
        ax.scatter(true_positions[:,0], true_positions[:,1], facecolors="none", edgecolors="lime", label="True sources")
        ax.set(xlabel="x", ylabel="y", aspect="equal", title="Inverse PINN final concentration")
        ax.legend()
        fig.colorbar(mesh, ax=ax, label="Concentration")
        fig.savefig(output / "pinn_prediction.png", dpi=180)
        plt.close(fig)
        print(json.dumps(metrics, indent=2))
    except Exception as exc:
        (output / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
