"""Controlled wind-magnitude and direction experiments with paired seeds."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from inverpinn.physics.finite_difference import solve_advection_diffusion, stability_number
from inverpinn.evaluation.sensor_sweep import ERROR_COLUMNS, summarize_results
from run_sensor_sweep import run_sweep


def wind_conditions(config):
    """One-factor-at-a-time design, deduplicating the shared reference wind.

    Speed sweep holds direction fixed; direction sweep holds nonzero speed
    fixed. Direction is angle counterclockwise from +x, pointing toward flow:
    u=speed*cos(theta), v=speed*sin(theta). For calm wind, direction is undefined
    and stored as None. Reference must appear in both input lists.
    """
    speeds, angles = config["speeds"], config["directions_deg"]
    speed, angle = config["reference_speed"], config["reference_direction_deg"]
    if (not speeds or any(not math.isfinite(s) or s < 0 for s in speeds)
            or len(set(speeds)) != len(speeds) or speed not in speeds or speed <= 0):
        raise ValueError("Distinct nonnegative speeds must include a positive reference_speed.")
    if (not angles or any(not math.isfinite(a) or not 0 <= a < 360 for a in angles)
            or len(set(angles)) != len(angles) or angle not in angles):
        raise ValueError("Distinct directions in [0,360) must include reference_direction_deg.")
    rows = {}
    for magnitude, direction in [(s, angle) for s in speeds] + [(speed, a) for a in angles]:
        key = (magnitude, direction if magnitude else None)
        theta = math.radians(direction)
        u, v = magnitude*math.cos(theta), magnitude*math.sin(theta)
        # Remove roundoff-only components at cardinal angles.
        u, v = (0. if abs(value) < 1e-14 else value for value in (u, v))
        rows[key] = dict(condition=f"speed_{magnitude:g}_angle_{direction:g}" if magnitude else "calm",
            wind_speed=magnitude, wind_direction_deg=key[1], u=u, v=v,
            magnitude_comparison=direction == angle,
            direction_comparison=magnitude == speed,
            is_reference=magnitude == speed and direction == angle)
    return list(rows.values())


def summarize_wind(raw, conditions, seeds):
    """Report sample SD and paired differences from the reference wind.

    All seeds must be present exactly once per condition. The paired difference
    subtracts reference localization error for the SAME seed, helping separate
    wind effects from differences in placement and initialization.
    """
    if len(raw) != len(conditions)*len(seeds) or set(zip(raw.condition, raw.seed)) != {
        (c["condition"], s) for c in conditions for s in seeds}:
        raise ValueError("Wind results must contain each condition/seed pair exactly once.")
    reference = next(c["condition"] for c in conditions if c["is_reference"])
    base = raw[raw.condition == reference].set_index("seed").localization_error
    rows = []
    for condition in conditions:
        group = raw[raw.condition == condition["condition"]]
        counts = group.n_sensors.unique().tolist()
        if len(counts) != 1 or raw.n_sensors.nunique() != 1:
            raise ValueError("Sensor count must stay fixed across wind conditions.")
        summary = summarize_results(group, counts, seeds).iloc[0].to_dict()
        paired = group.set_index("seed").localization_error-base
        summary.update(condition, paired_localization_delta_mean=paired.mean(), paired_localization_delta_std=paired.std(ddof=1))
        rows.append(summary)
    return pd.DataFrame(rows)


def run_wind_sweep(config):
    """Generate a new forward field per wind; repeat otherwise fixed inverse fits.

    Source, D, domain, grid, dt, time horizon and boundaries are copied from one
    forward configuration. A combined CFL check is performed for every wind
    before work begins: unstable settings fail, never silently change dt.
    All seeds use fixed sensor positions and absolute Gaussian noise draws
    across winds. u and v are known to the inverse solver, not inferred.
    """
    config = dict(config)
    conditions = wind_conditions(config)
    solver = yaml.safe_load(Path(config["forward_config"]).read_text())["solver"]
    base_inverse = yaml.safe_load(Path(config["inverse_config"]).read_text())
    if len(solver["sources"]) != 1:
        raise ValueError("Wind sweep assumes exactly one Gaussian source.")
    if solver["sources"][0]["sigma"] != base_inverse["source_initial"]["sigma"]:
        raise ValueError("Known inverse width must match the forward source width.")
    for c in conditions:
        c["stability_number"] = stability_number(dx=1/(solver["nx"]-1), dy=1/(solver["ny"]-1),
            dt=solver["dt"], u=c["u"], v=c["v"], D=solver["D"])
        if c["stability_number"] > 1:
            raise ValueError(f"Unstable wind {c['condition']}; choose a common stable dt for all conditions.")
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Wind sweep already exists: {output}")
    output.mkdir(parents=True)
    (output / "reference_fields").mkdir()
    config.update(output_dir=str(output), base_forward_solver=solver, base_inverse_config=base_inverse,
                  conditions=conditions, direction_convention="flow toward, counterclockwise from +x",
                  noise_definition="fixed absolute Gaussian standard deviation; paired draws across winds",
                  std_ddof=1)
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    pd.DataFrame(conditions).to_csv(output / "conditions.csv", index=False)
    torch.set_num_threads(1)
    x = np.linspace(0, 1, solver["nx"])
    y = np.linspace(0, 1, solver["ny"])
    t = np.arange(solver["steps"]+1)*solver["dt"]
    rows = []
    for condition in conditions:
        settings = solver | {"u": condition["u"], "v": condition["v"]}
        C = solve_advection_diffusion(**settings).numpy()
        reference = output / "reference_fields" / f"{condition['condition']}.npz"
        sources = settings["sources"]
        np.savez_compressed(reference, C=C, x=x, y=y, t=t, u=settings["u"], v=settings["v"], D=settings["D"],
            source_positions=np.array([[s["x_s"],s["y_s"]] for s in sources]),
            source_Q=np.array([s["Q"] for s in sources]), source_sigma=np.array([s["sigma"] for s in sources]))
        reference.with_suffix(".yaml").write_text(yaml.safe_dump(dict(solver=settings,
            sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), boundary="zero Dirichlet, all edges", initial="zero")))
        directory = output / condition["condition"]
        child = {key: config[key] for key in ("seeds", "noise_std", "inverse_config", "workers")}
        child.update(sensor_counts=[config["n_sensors"]], reference_dataset=str(reference), output_dir=str(directory))
        try:
            run_sweep(child)
        except Exception as exc:
            if (directory / "raw_results.csv").exists():
                failed = pd.read_csv(directory / "raw_results.csv").assign(**condition)
                rows.append(failed)
            if rows:
                pd.concat(rows, ignore_index=True).to_csv(output / "raw_results.csv", index=False)
            (output / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            raise
        rows.append(pd.read_csv(directory / "raw_results.csv").assign(**condition))
        raw = pd.concat(rows, ignore_index=True)
        raw.to_csv(output / "raw_results.csv", index=False)
        print(f"Completed wind {condition['condition']} ({len(rows)}/{len(conditions)} conditions)", flush=True)
    reference_id = next(c["condition"] for c in conditions if c["is_reference"])
    base_errors = raw[raw.condition == reference_id].set_index("seed").localization_error
    raw["localization_delta_vs_reference"] = raw.localization_error-raw.seed.map(base_errors)
    raw.to_csv(output / "raw_results.csv", index=False)
    summary = summarize_wind(raw, conditions, config["seeds"])
    summary.to_csv(output / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, flag, coordinate, label in ((axes[0], "magnitude_comparison", "wind_speed", "Wind speed (domain length/time)"),
                                      (axes[1], "direction_comparison", "wind_direction_deg", "Flow direction (degrees from +x)")):
        subset = summary[summary[flag]].sort_values(coordinate)
        ax.errorbar(subset[coordinate], subset.localization_error_mean, yerr=subset.localization_error_std,
                    marker="o", capsize=5, label="Mean ± sample SD")
        for seed in config["seeds"]:
            points = raw[(raw.seed == seed) & raw[flag]].sort_values(coordinate)
            ax.plot(points[coordinate], points.localization_error, ".", alpha=0.4, color="gray")
        ax.set(xlabel=label, ylabel="Localization error (domain units)", xticks=subset[coordinate])
        ax.legend()
    axes[0].set_title(f"Magnitude sweep: direction {config['reference_direction_deg']}°")
    axes[1].set_title(f"Direction sweep: speed {config['reference_speed']}")
    fig.savefig(output / "localization_error_vs_wind.png", dpi=180)
    plt.close(fig)
    lines = ["# Controlled wind experiments", "", f"Seeds {config['seeds']}; {config['n_sensors']} sensors; {base_inverse['steps']} Adam steps per run.",
        "", "| Condition | u | v | Localization mean ± SD | Paired change from reference mean ± SD |",
        "|---|---:|---:|---:|---:|"]
    for _, row in summary.iterrows():
        lines.append(f"| {row.condition} | {row.u:g} | {row.v:g} | {row.localization_error_mean:.6f} ± {row.localization_error_std:.6f} | {row.paired_localization_delta_mean:.6f} ± {row.paired_localization_delta_std:.6f} |")
    lines += ["", f"Reference: {reference_id}. Negative paired changes mean better localization than the reference for the same seed.",
        "Sample SD uses ddof=1 and is not a confidence interval. All runs are retained, without truth-based checkpoint selection.",
        "Source coordinates/strength/width, diffusion, grid, dt, initial/boundary conditions, time horizon, sensor count and optimizer settings stay fixed.",
        "For each seed, sensor positions, absolute Gaussian noise draws, initialization and collocation draws stay fixed across winds.",
        "The signal-to-noise ratio may change because concentration changes while absolute noise remains fixed.",
        "u,v are known to the inverse PINN. Angles describe flow toward a direction, not meteorological wind-from bearings; calm has no direction.",
        "Finite-domain boundaries and direction-dependent upwind numerical diffusion also affect these comparisons. This is one controlled synthetic source scenario.",
        "", "![Localization accuracy versus wind](localization_error_vs_wind.png)"]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/wind_sweep.yaml"))
    args = parser.parse_args()
    print(run_wind_sweep(yaml.safe_load(args.config.read_text())).to_string(index=False))


if __name__ == "__main__":
    main()
