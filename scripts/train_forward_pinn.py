"""Train and evaluate a forward PINN with known pollution sources."""

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

from inverpinn.training.forward_pinn import train_forward_pinn, require_finite


def reconstruction_metrics(prediction, reference):
    """Uniform-grid errors; relative L2 is undefined (null) for zero reference."""
    require_finite(torch.as_tensor(prediction), "evaluation predictions")
    require_finite(torch.as_tensor(reference), "evaluation reference")
    if prediction.shape != reference.shape:
        raise ValueError("Prediction/reference shapes must match.")
    error = prediction.astype(np.float64)-reference
    norm = np.linalg.norm(reference.ravel())
    return {"relative_l2": float(np.linalg.norm(error.ravel())/norm) if norm > 0 else None,
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pinn.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    dataset = Path(config["dataset"]).resolve()
    metadata = json.loads(Path(config["metadata"]).read_text())
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != metadata["dataset_sha256"]:
        raise ValueError("Dataset checksum does not match metadata.")
    with np.load(dataset, allow_pickle=False) as data:
        n = config["n_sensors"]
        positions = torch.from_numpy(data[f"sensor_positions_{n}"].copy())
        times = torch.from_numpy(data["t"].copy())
        observations = torch.from_numpy(data[f"measurements_{n}"].copy())
        x, y = data["x"].copy(), data["y"].copy()
        known = {k: float(data[k]) for k in ("u", "v", "D")}
        known["sources"] = [dict(x_s=float(p[0]), y_s=float(p[1]), Q=float(q), sigma=float(s))
            for p, q, s in zip(data["source_positions"], data["source_Q"], data["source_sigma"])]
    if not (x[0] == y[0] == 0 and x[-1] == y[-1] == 1):
        raise ValueError("Forward PINN requires the unit-square domain.")
    output.mkdir(parents=True)
    config.update(known_parameters=known, dataset=str(dataset), dataset_sha256=digest,
                  output_dir=str(output), torch_version=str(torch.__version__), dtype="float32", device="cpu")
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    logger = logging.getLogger("inverpinn.forward_pinn")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "training.log")
    logger.addHandler(handler)
    try:
        with (output / "training_history.jsonl").open("w") as stream:
            def save_record(record):
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
            model, optimizer, history = train_forward_pinn(positions, times, observations, known, config, logger, save_record)
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "config": config, "step": config["steps"]}, output / "model.pt")
        # Numerical truth is used only after optimization, for reconstruction metrics.
        with np.load(dataset, allow_pickle=False) as data:
            truth = data["C"].copy()
        prediction = np.empty_like(truth)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        model.eval()
        with torch.no_grad():
            for i, time in enumerate(times):
                points = torch.tensor(np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, time.item())], 1), dtype=torch.float32)
                values = model(points)
                require_finite(values, f"evaluation time index {i}")
                prediction[i] = values.numpy().reshape(len(y), len(x))
        metrics = {"space_time": reconstruction_metrics(prediction, truth),
                   "final_time": reconstruction_metrics(prediction[-1], truth[-1]),
                   "scope": "Full numerical grid, including sensor locations; not an independent simulation"}
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
        np.savez_compressed(output / "predictions.npz", C_predicted=prediction, x=x, y=y, t=times.numpy())
        lo, hi = min(0., prediction[-1].min()), max(truth[-1].max(), prediction[-1].max(), 1e-12)
        for name, field, title in (("ground_truth", truth[-1], "Numerical ground truth"),
                ("pinn_prediction", prediction[-1], "Forward PINN prediction"),
                ("absolute_error", np.abs(prediction[-1]-truth[-1]), "Absolute reconstruction error")):
            fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
            mesh = ax.pcolormesh(x, y, field, shading="auto", cmap="inferno",
                                 vmin=0 if name == "absolute_error" else lo,
                                 vmax=max(field.max(), 1e-12) if name == "absolute_error" else hi)
            ax.set(xlabel="x", ylabel="y", aspect="equal", title=f"{title}, t={times[-1]:.3f}")
            fig.colorbar(mesh, ax=ax, label="Concentration" if name != "absolute_error" else "Absolute error")
            fig.savefig(output / f"{name}.png", dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(layout="constrained")
        for name in ("data", "pde", "initial", "boundary", "total"):
            ax.semilogy([row["step"] for row in history], [max(row[f"loss/{name}"], 1e-30) for row in history], label=name)
        ax.set(xlabel="Optimizer step", ylabel="Raw MSE / weighted total", title="Forward PINN losses (resampled batches)")
        ax.legend()
        fig.savefig(output / "training_losses.png", dpi=180)
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
