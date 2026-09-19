"""Generate sparse observations from an existing forward-simulation checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from inverpinn.data.sensors import sample_sensors


def main() -> None:
    """Save each sparse dataset, overlay, metrics, and provenance configuration.

    A local seed gives repeatable placement and independent measurement noise.
    The common seed produces nested positions across sensor counts, allowing
    comparisons with the same first sensors. Noise draws are reproducible per
    dataset but are not shared across different counts. Full-field numerical
    truth stays in the forward checkpoint; sparse datasets contain only sensor
    coordinates, times, and noisy observations.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sensors.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    checkpoint = Path(config["forward_checkpoint"]).resolve()
    truth = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dataset_dir, output = Path(config["dataset_dir"]).resolve(), Path(config["output_dir"]).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    config.update(forward_checkpoint=str(checkpoint), dataset_dir=str(dataset_dir), output_dir=str(output))
    config["forward_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config["forward_config"] = truth["config"]
    config["interpolation"] = "bilinear; fixed positions; all stored times"
    config["noise_model"] = "independent additive Gaussian; absolute std; no clipping"
    counts = config["sensor_counts"]
    overview, panels = plt.subplots(1, len(counts), figsize=(5*len(counts), 4.5), squeeze=False, layout="constrained")
    for count, panel in zip(counts, panels[0]):
        positions, observations = sample_sensors(truth["C"], truth["x"], truth["y"],
            n_sensors=count, noise_std=config["noise_std"], seed=config["seed"])
        _, clean = sample_sensors(truth["C"], truth["x"], truth["y"], positions=positions)
        resolved = config | {"n_sensors": count, "positions": positions.tolist()}
        torch.save({"positions": positions, "t": truth["t"], "measurements": observations,
                    "config": resolved}, dataset_dir / f"sensors_{count}.pt")
        pd.DataFrame({"sensor_id": torch.arange(count).repeat(len(truth["t"])).numpy(),
                      "t": truth["t"].repeat_interleave(count).numpy(),
                      "x": positions[:, 0].repeat(len(truth["t"])).numpy(),
                      "y": positions[:, 1].repeat(len(truth["t"])).numpy(),
                      "concentration": observations.flatten().numpy()}).to_csv(dataset_dir / f"sensors_{count}.csv", index=False)
        (output / f"sensors_{count}_config.yaml").write_text(yaml.safe_dump(resolved))
        error = observations - clean
        metrics = {"n_sensors": count, "n_times": len(truth["t"]), "n_observations": observations.numel(),
                   "noise_rmse": error.square().mean().sqrt().item(), "noise_mean": error.mean().item()}
        (output / f"sensors_{count}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        for target in (ax, panel):
            mesh = target.pcolormesh(truth["x"].numpy(), truth["y"].numpy(), truth["C"][-1].numpy(),
                                    shading="auto", cmap="inferno", vmin=0, vmax=max(truth["C"][-1].max().item(), 1e-12))
            target.scatter(positions[:, 0], positions[:, 1], s=28, facecolors="cyan", edgecolors="black", linewidths=0.6)
            target.set(xlabel="x", ylabel="y", aspect="equal", title=f"{count} fixed sensors")
            if target is ax:
                individual_mesh = mesh
        fig.colorbar(individual_mesh, ax=ax, label="Final concentration C")
        fig.savefig(output / f"sensors_{count}.png", dpi=180)
        plt.close(fig)
    overview.colorbar(mesh, ax=list(panels[0]), label="Final concentration C", shrink=0.8)
    overview.suptitle(f"Sparse sensor locations at t = {truth['t'][-1]:.3f}")
    overview.savefig(output / "sensor_positions.png", dpi=180)
    plt.close(overview)
    print(f"Saved sensor datasets to {dataset_dir}; plots and metadata to {output}")


if __name__ == "__main__":
    main()
