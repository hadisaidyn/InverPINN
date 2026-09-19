"""Train an MLP on sparse observations alone, then evaluate numerical truth."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import yaml

from inverpinn.models.mlp import ConcentrationMLP


def train_observations(positions, times, measurements, lower, upper, config):
    """Fit sensor MSE only; this function has no access to full-field truth.

    Every time/sensor pair contributes a coordinate and observed concentration.
    Targets are divided by their RMS computed exclusively from observations.
    Adam minimizes mean squared error of these scaled observations; returned
    history is converted back to physical concentration-squared units.
    """
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive.")
    nt, ns = measurements.shape
    points = torch.cat([positions.repeat(nt, 1), times.repeat_interleave(ns)[:, None]], dim=1).float()
    targets = measurements.reshape(-1, 1).float()
    scale = targets.square().mean().sqrt().clamp_min(1e-8)
    model = ConcentrationMLP(lower, upper, config["width"], config["depth"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"] + 1)
    history = []
    for epoch in range(config["epochs"]):
        indices = torch.randperm(len(points), generator=generator)[:config["batch_size"]]
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(points[indices]), targets[indices] / scale)
        loss.backward()
        optimizer.step()
        # Full sensor loss after each update makes the curve comparable over time.
        with torch.no_grad():
            mse = nn.functional.mse_loss(model(points) * scale, targets).item()
        history.append(mse)
        if (epoch + 1) % 300 == 0:
            print(f"Update {epoch+1}: sensor MSE={mse:.6g}", flush=True)
    return model, scale.item(), history


def relative_l2(prediction, reference):
    """Return ||prediction-reference||₂/||reference||₂; undefined for zero truth."""
    denominator = np.linalg.norm(reference.ravel())
    return float(np.linalg.norm((prediction-reference).ravel()) / denominator) if denominator > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_baseline.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new output_dir.")
    path = Path(config["dataset"]).resolve()
    with np.load(path, allow_pickle=False) as data:
        n = config["n_sensors"]
        positions = torch.from_numpy(data[f"sensor_positions_{n}"].copy())
        observations = torch.from_numpy(data[f"measurements_{n}"].copy())
        x, y, t = (data[k].copy() for k in ("x", "y", "t"))
    lower, upper = [float(x[0]), float(y[0]), float(t[0])], [float(x[-1]), float(y[-1]), float(t[-1])]
    model, scale, history = train_observations(positions, torch.from_numpy(t), observations, lower, upper, config)
    # Load dense reference ONLY after training. It never enters the optimizer.
    with np.load(path, allow_pickle=False) as data:
        truth = data["C"].copy()
    yy, xx = np.meshgrid(y, x, indexing="ij")
    predicted = np.empty(truth.shape, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i, time in enumerate(t):
            points = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, time)], axis=1)
            predicted[i] = (model(torch.from_numpy(points).float()) * scale).numpy().reshape(len(y), len(x))
    metrics = {"relative_l2": relative_l2(predicted, truth), "final_relative_l2": relative_l2(predicted[-1], truth[-1]),
               "sensor_mse": history[-1], "evaluation": "entire grid and all times; training sensor locations included",
               "loss": "sensor MSE only; no PDE, boundary, or initial-condition loss"}
    output.mkdir(parents=True)
    config.update(dataset=str(path), output_dir=str(output), lower=lower, upper=upper, target_scale=scale,
                  dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), torch_version=str(torch.__version__),
                  device="cpu", dtype="float32", optimizer="Adam", epoch_definition="one randomly sampled minibatch update")
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"model_state_dict": model.state_dict(), "target_scale": scale, "config": config}, output / "model.pt")
    np.savez_compressed(output / "predictions.npz", C_predicted=predicted, x=x, y=y, t=t, sensor_mse=np.array(history))
    fig, ax = plt.subplots(layout="constrained")
    ax.semilogy(np.arange(1, len(history)+1), np.maximum(history, 1e-30))
    ax.set(xlabel="Optimizer update", ylabel="Sensor MSE", title="Data-only MLP training")
    fig.savefig(output / "training_curve.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    vmin, vmax = min(0., float(predicted[-1].min())), max(float(predicted[-1].max()), float(truth[-1].max()), 1e-12)
    for ax, values, title in zip(axes, [truth[-1], predicted[-1]], ["Numerical reference", "Data-only MLP"]):
        mesh = ax.pcolormesh(x, y, values, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.scatter(positions[:, 0], positions[:, 1], s=12, c="cyan", edgecolors="black", linewidths=0.4)
        ax.set(xlabel="x", ylabel="y", aspect="equal", title=title)
    fig.colorbar(mesh, ax=list(axes), label="Concentration C")
    fig.suptitle(f"Final concentration at t={t[-1]:.3f}")
    fig.savefig(output / "predicted_concentration.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
