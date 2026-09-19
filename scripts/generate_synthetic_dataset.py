"""Generate a self-contained synthetic simulation and sparse observations.

Run from the repository root: python scripts/generate_synthetic_dataset.py
Use --config for another experiment and --output-dir for a separate run.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from inverpinn.physics.finite_difference import solve_advection_diffusion, stability_number
from inverpinn.data.sensors import sample_sensors
from inverpinn.physics.sources import gaussian_source


def generate_dataset(config: dict) -> Path:
    """Save numerical truth and observations without requiring prior artifacts.

    Truth means the finite-difference solution for known emission parameters,
    not an exact analytic PDE solution. C has dimensions [time,y,x]. Sensor
    readings use bilinear spatial interpolation at all saved times, followed
    by independent Gaussian noise in concentration units. The Gaussian Q is
    peak source intensity, not integrated emissions. All outputs use float64.

    Local seeded sampling does not change global RNG state. Equal configs in
    the same software environment reproduce equal arrays. Output directories
    must not exist, preventing accidental replacement of earlier experiments.
    The NPZ serves as the simulation checkpoint; no model is trained.
    """
    # Copy configuration so resolving paths does not change the caller's data.
    config = json.loads(json.dumps(config))
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose --output-dir for a new run.")
    counts = config["sensor_counts"]
    if (not isinstance(counts, list) or not counts
            or any(type(n) is not int or n < 1 for n in counts) or len(set(counts)) != len(counts)):
        raise ValueError("sensor_counts must be a nonempty list of distinct positive integers.")
    settings = config["solver"]
    c = solve_advection_diffusion(**settings)
    x = torch.linspace(0, 1, settings["nx"], dtype=torch.float64)
    y = torch.linspace(0, 1, settings["ny"], dtype=torch.float64)
    t = torch.arange(settings["steps"] + 1, dtype=torch.float64) * settings["dt"]
    sources = settings.get("sources", [])
    arrays = {"C": c.numpy(), "x": x.numpy(), "y": y.numpy(), "t": t.numpy(),
              "source_positions": np.array([[s["x_s"], s["y_s"]] for s in sources], dtype=np.float64).reshape(-1, 2),
              "source_Q": np.array([s["Q"] for s in sources], dtype=np.float64),
              "source_sigma": np.array([s["sigma"] for s in sources], dtype=np.float64),
              "u": np.array(settings["u"], dtype=np.float64),
              "v": np.array(settings["v"], dtype=np.float64),
              "D": np.array(settings["D"], dtype=np.float64),
              "seed": np.array(config["seed"], dtype=np.int64),
              "noise_std": np.array(config["noise_std"], dtype=np.float64)}
    dimensions = {"C": ["time", "y", "x"], "x": ["x"], "y": ["y"], "t": ["time"],
                  "source_positions": ["source", "xy"], "source_Q": ["source"], "source_sigma": ["source"],
                  **{k: [] for k in ("u", "v", "D", "seed", "noise_std")}}
    metrics = {"concentration_min": c.min().item(), "concentration_max": c.max().item(),
               "stability_number": stability_number(dx=1/(len(x)-1), dy=1/(len(y)-1),
                   **{k: settings[k] for k in ("dt", "u", "v", "D")}), "sensors": {}}
    for n in counts:
        positions, readings = sample_sensors(c, x, y, n_sensors=n, seed=config["seed"], noise_std=config["noise_std"])
        _, clean = sample_sensors(c, x, y, positions=positions)
        arrays[f"sensor_positions_{n}"] = positions.numpy()
        arrays[f"measurements_{n}"] = readings.numpy()
        arrays[f"clean_measurements_{n}"] = clean.numpy()
        dimensions[f"sensor_positions_{n}"] = [f"sensor_{n}", "xy"]
        dimensions[f"measurements_{n}"] = ["time", f"sensor_{n}"]
        dimensions[f"clean_measurements_{n}"] = ["time", f"sensor_{n}"]
        metrics["sensors"][str(n)] = {"noise_rmse": (readings-clean).square().mean().sqrt().item(),
                                      "observations": readings.numel()}
    output.mkdir(parents=True, exist_ok=False)
    config["output_dir"] = str(output)
    np.savez_compressed(output / "dataset.npz", **arrays)
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metadata = {
        "schema_version": 1, "config": config, "array_dimensions": dimensions,
        "coordinate_order": "xy means [x,y]; C is [time,y,x]",
        "truth": "Numerical finite-difference reference, not an analytic PDE solution",
        "initial_condition": "C=0 everywhere", "boundary_condition": "C=0 on all four edges",
        "scheme": "forward Euler, first-order upwind advection, centered diffusion",
        "source_equation": "S=sum Q*exp(-((x-x_s)^2+(y-y_s)^2)/(2*sigma^2))",
        "units": {"x,y,sigma": "domain length", "t": "simulation time", "C": "concentration",
                  "u,v": "length/time", "D": "length^2/time", "Q": "concentration/time",
                  "measurements,noise_std": "concentration"},
        "sampling": "fixed uniform random positions, bilinear interpolation, all times",
        "noise": "independent Gaussian with absolute std; no clipping",
        "placement_seed": config["seed"], "noise_seed": config["seed"]+1,
        "sensor_comparison": "nested positions; noise is reproducible per count but not shared across counts",
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__},
        "platform": platform.platform(),
        "dataset_sha256": hashlib.sha256((output / "dataset.npz").read_bytes()).hexdigest(),
        "code_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in (
            ("pipeline", __file__),
            ("solver", inspect.getfile(solve_advection_diffusion)),
            ("source", inspect.getfile(gaussian_source)),
            ("sampling", inspect.getfile(sample_sensors)))},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, axes = plt.subplots(1, len(counts), figsize=(5*len(counts), 4.5), squeeze=False, layout="constrained")
    for ax, n in zip(axes[0], counts):
        mesh = ax.pcolormesh(x.numpy(), y.numpy(), c[-1].numpy(), shading="auto", cmap="inferno",
                            vmin=0, vmax=max(c[-1].max().item(), 1e-12))
        p = arrays[f"sensor_positions_{n}"]
        ax.scatter(p[:, 0], p[:, 1], s=25, c="cyan", edgecolors="black", linewidths=0.5)
        ax.set(xlabel="x", ylabel="y", aspect="equal", title=f"{n} sensors")
    fig.colorbar(mesh, ax=list(axes[0]), label="Final concentration C", shrink=0.8)
    fig.suptitle(f"Synthetic simulation, t = {t[-1]:.3f}, seed = {config['seed']}")
    fig.savefig(output / "sensor_positions.png", dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    """Load a configuration and generate every artifact in one invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    print(f"Saved synthetic dataset to {generate_dataset(config)}")


if __name__ == "__main__":
    main()
