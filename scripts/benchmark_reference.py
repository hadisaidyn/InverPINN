"""Benchmark reference resolution/storage only; no source recovery or training."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from inverpinn.data.sensors import sample_sensors
from inverpinn.data.synthetic_reference import ROOT, run_refinement, write_json
from inverpinn.physics.finite_difference import solve_advection_diffusion


def storage_equivalence():
    """Measure explicit differences against a small full-history regression."""
    options = dict(nx=21, ny=17, dt=0.005, steps=20, u=0.35, v=-0.15, D=0.005,
                   sources=[dict(x_s=0.3, y_s=0.7, Q=1.0, sigma=0.08)])
    full = solve_advection_diffusion(**options)
    positions = torch.tensor([[0.37, 0.43], [0.61, 0.72], [0, 0.4]], dtype=torch.float64)
    indices = [0, 3, 10, 20]
    times = [i*options["dt"] for i in indices]
    selected = solve_advection_diffusion(**options, output_mode="selected_times", output_times=times,
        sensor_positions=positions, measurement_times=times)
    sensors = solve_advection_diffusion(**options, output_mode="sensor_only", sensor_positions=positions,
        measurement_times=times)
    final = solve_advection_diffusion(**options, output_mode="final_only")
    _, expected = sample_sensors(full[indices], selected["x"], selected["y"], positions=positions)
    result = dict(final_max_absolute_difference=float((final-full[-1]).abs().max()),
        selected_max_absolute_difference=float((selected["fields"]-full[indices]).abs().max()),
        sensor_max_absolute_difference=float((sensors["measurements"]-expected).abs().max()),
        selected_sensor_max_absolute_difference=float((selected["measurements"]-expected).abs().max()))
    if any(v != 0 for v in result.values()):
        raise AssertionError(f"Native-time storage regression: {result}")
    return result | dict(solver=options, times=times, positions=positions.tolist(),
                         interpretation="Native outputs are bitwise identical; off-step interpolation has separate automated tests.")


def plot_benchmark(output, dpi):
    """Show estimates without relabeling inter-grid differences as exact errors."""
    frame = pd.read_csv(output/"reference_benchmark.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for t, group in frame[frame.scope == "selected_field"].groupby("time"):
        ax.semilogy(group.grid_size, group.Richardson_relative_L2_estimate*100, "o-", label=f"Field t={t:g}")
    sensors = frame[frame.scope == "sensor_trajectory"]
    ax.semilogy(sensors.grid_size, sensors.Richardson_relative_L2_estimate*100, "s--", label="Pooled sensor trajectory")
    config = yaml.safe_load((output/"config.yaml").read_text())
    ax.axhline(100*config["target_relative_L2"], color="black", linestyle=":", label="Desired target (not certified)")
    ax.set(xlabel="Grid nodes per axis", ylabel="Richardson relative L2 estimate (%)",
           xticks=sensors.grid_size.tolist(), title="Reference-resolution study")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output/"reference_convergence.png", dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/reference_benchmark.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(args.output_dir or ROOT/config["output_dir"]).resolve()
    equivalence = storage_equivalence()
    summary, _ = run_refinement(config, output)
    write_json(output/"storage_equivalence.json", equivalence)
    plot_benchmark(output, config["plot_dpi"])
    print(f"Cheapest candidate meeting estimated target: {summary['selected_grid']}; certified bound: False", flush=True)


if __name__ == "__main__":
    main()
