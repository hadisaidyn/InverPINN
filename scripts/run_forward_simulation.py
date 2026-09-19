"""Run the configured forward solver; use --animate to also save a GIF."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import yaml

from inverpinn.physics.finite_difference import solve_advection_diffusion, stability_number


def main() -> None:
    """Save C[t,y,x], coordinates, metrics, resolved config, and a final plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/forward.yaml"))
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config["animation"] = bool(args.animate or config.get("animation", False))
    torch.manual_seed(config["experiment"]["seed"])
    settings = config["solver"]
    concentration = solve_advection_diffusion(**settings)
    x = torch.linspace(0, 1, settings["nx"], dtype=torch.float64)
    y = torch.linspace(0, 1, settings["ny"], dtype=torch.float64)
    t = torch.arange(settings["steps"] + 1, dtype=torch.float64) * settings["dt"]
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output)
    config["boundary_condition"] = "zero Dirichlet on all four edges"
    config["initial_condition"] = "zero everywhere"
    config["dtype"] = "float64"
    cfl = stability_number(dx=1/(settings["nx"]-1), dy=1/(settings["ny"]-1),
                           **{k: settings[k] for k in ("dt", "u", "v", "D")})
    metrics = {"stability_number": cfl, "final_time": t[-1].item(),
               "minimum": concentration.min().item(), "maximum": concentration.max().item(),
               "final_maximum": concentration[-1].max().item()}
    (output / "forward_simulation_config.yaml").write_text(yaml.safe_dump(config))
    (output / "forward_simulation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"C": concentration, "x": x, "y": y, "t": t, "config": config},
               output / "forward_simulation.pt")

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    plot = ax.pcolormesh(x.numpy(), y.numpy(), concentration[-1].numpy(), shading="auto",
                         cmap="inferno", vmin=0, vmax=max(metrics["maximum"], 1e-12))
    ax.set(xlabel="x (domain units)", ylabel="y (domain units)", aspect="equal")
    ax.set_title(f"Concentration at t = {t[-1]:.3f}\nu = {settings['u']}, v = {settings['v']}, D = {settings['D']}")
    fig.colorbar(plot, ax=ax, label="Concentration C")
    fig.savefig(output / "forward_simulation_final.png", dpi=180)
    if config["animation"]:
        frames = sorted(set(range(0, len(t), max(1, len(t)//80))) | {len(t)-1})

        def update(frame: int) -> tuple:
            plot.set_array(concentration[frame].numpy())
            ax.set_title(f"Concentration at t = {t[frame]:.3f}")
            return (plot,)

        animation = FuncAnimation(fig, update, frames=frames, interval=80)
        animation.save(output / "forward_simulation.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"Saved forward simulation to {output}; stability number = {cfl:g}")


if __name__ == "__main__":
    main()
