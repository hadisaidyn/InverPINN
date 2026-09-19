"""Plot a reproducible Gaussian emission field from a YAML configuration.

Run from the repository root after installing the package:
    python scripts/visualize_source.py --config configs/synthetic.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from inverpinn.physics.sources import gaussian_source


def main() -> None:
    """Save a heatmap, field checkpoint, metrics, and exact resolved config.

    The plotted quantity is S(x,y), an emission rate. Solving for transported
    concentration C(x,y,t) is a separate experiment. No trained model exists;
    the checkpoint stores the evaluated grid and its source parameters.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.manual_seed(config["experiment"]["seed"])
    grid = config["grid"]
    x = torch.linspace(grid["x_min"], grid["x_max"], grid["points"], dtype=torch.float64)
    y = torch.linspace(grid["y_min"], grid["y_max"], grid["points"], dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    source = config["source"]
    intensity = gaussian_source(xx, yy, **source)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    heatmap = ax.pcolormesh(x.numpy(), y.numpy(), intensity.numpy(), shading="auto", cmap="inferno", vmin=0, vmax=source["Q"])
    ax.plot(source["x_s"], source["y_s"], "+", color="cyan", markersize=10, label="Source center")
    ax.set(xlabel="x (domain units)", ylabel="y (domain units)", aspect="equal",
           title=f"Gaussian emission source\nQ = {source['Q']:g}, σ = {source['sigma']:g}")
    ax.legend(loc="upper right")
    fig.colorbar(heatmap, ax=ax, label="Source intensity S (concentration / time)")
    fig.savefig(output / "synthetic_source.png", dpi=180)
    plt.close(fig)

    config["output_dir"] = str(output.resolve())
    (output / "synthetic_source_config.yaml").write_text(yaml.safe_dump(config))
    metrics = {"sampled_max": intensity.max().item(), "sampled_min": intensity.min().item(),
               "analytic_peak": source["Q"], "grid_points": intensity.numel()}
    (output / "synthetic_source_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"x": x, "y": y, "source_intensity": intensity, "config": config}, output / "synthetic_source.pt")
    print(f"Saved Gaussian source artifacts to {output.resolve()}")


if __name__ == "__main__":
    main()
