"""Run Gaussian-noise robustness experiments with paired independent seeds."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from run_sensor_sweep import run_sweep
from inverpinn.evaluation.sensor_sweep import ERROR_COLUMNS, summarize_results


def summarize_noise(raw, levels, seeds):
    """Require complete runs and compute mean ± sample SD across seeds per level."""
    if len(raw) != len(levels)*len(seeds) or set(zip(raw.noise_percent, raw.seed)) != {(p,s) for p in levels for s in seeds}:
        raise ValueError("Need exactly one result per noise level and seed.")
    rows = []
    for level in levels:
        group = raw[raw.noise_percent == level]
        counts = group.n_sensors.unique().tolist()
        if len(counts) != 1 or raw.n_sensors.nunique() != 1:
            raise ValueError("Noise experiments must keep sensor count fixed.")
        result = summarize_results(group, counts, seeds).iloc[0].to_dict()
        result["noise_percent"] = level
        rows.append(result)
    return pd.DataFrame(rows)


def run_noise_sweep(config):
    """Vary only noise amplitude within a seed; retain every final-step result.

    Placement, standard-normal noise draws, initial weights and collocation
    sequences are identical across noise levels for a given seed. Different
    seeds independently change these factors. There is no clipping of noisy
    observations, no tuning, and no outlier removal. Width, transport and
    optimizer settings remain fixed. SD describes variability, not a CI.
    """
    levels, seeds = config["noise_percentages"], config["seeds"]
    if not levels or len(set(levels)) != len(levels) or any(not math.isfinite(p) or p < 0 for p in levels):
        raise ValueError("Noise levels must be distinct finite nonnegative percentages.")
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Noise sweep already exists: {output}")
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config | {
        "noise_definition": "Gaussian std = noise_percent/100 * RMS(clean readings over time and sensors)",
        "error_bars": "sample standard deviation across seeds; ddof=1; not confidence intervals"}))
    rows = []
    for level in levels:
        directory = output / f"noise_{level:g}pct"
        child = {key: config[key] for key in ("reference_dataset", "inverse_config", "workers", "seeds")}
        child.update(sensor_counts=[config["n_sensors"]], noise_percent=level, output_dir=str(directory))
        try:
            run_sweep(child)
        except Exception as exc:
            partial = directory / "raw_results.csv"
            if partial.exists():
                rows.append(pd.read_csv(partial))
            if rows:
                pd.concat(rows).to_csv(output / "raw_results.csv", index=False)
            (output / "failure.txt").write_text(f"Noise {level}% failed: {type(exc).__name__}: {exc}\n")
            raise
        rows.append(pd.read_csv(directory / "raw_results.csv"))
        raw = pd.concat(rows, ignore_index=True).sort_values(["noise_percent", "seed"])
        raw.to_csv(output / "raw_results.csv", index=False)
        print(f"Finished noise level {level}% ({len(rows)}/{len(levels)} levels)", flush=True)
    summary = summarize_noise(raw, levels, seeds)
    summary.to_csv(output / "summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for seed in seeds:
        points = raw[raw.seed == seed].sort_values("noise_percent")
        ax.plot(points.noise_percent, points.localization_error, ".-", color="gray", alpha=0.4,
                linewidth=0.8, label="Individual seeds" if seed == seeds[0] else None)
    ax.errorbar(summary.noise_percent, summary.localization_error_mean, yerr=summary.localization_error_std,
                marker="o", capsize=5, linewidth=2, label="Mean ± sample SD (5 seeds)")
    ax.set(xlabel="Measurement noise (% of clean sensor RMS)", ylabel="Source localization error (domain units)",
           title=f"Inverse PINN noise robustness — {config['n_sensors']} sensors", xticks=levels)
    ax.legend()
    fig.savefig(output / "localization_error_vs_noise.png", dpi=180)
    plt.close(fig)
    lines = ["# Measurement-noise robustness", "", f"{config['n_sensors']} fixed sensors per run; seeds {seeds}.",
        "Gaussian standard deviation = noise percentage / 100 × RMS of all noise-free sensor readings.",
        "Noise is additive, has constant standard deviation within a run, and is not clipped at zero.", "",
        "| Noise | Localization error (mean ± sample SD) |", "|---|---:|"]
    for _, row in summary.iterrows():
        lines.append(f"| {row.noise_percent:g}% | {row.localization_error_mean:.6f} ± {row.localization_error_std:.6f} |")
    lines += ["", "All five seeds are included per level. Error bars are sample standard deviations (ddof=1), not confidence intervals.",
        "The same seed uses identical sensor positions, base noise draws, initialization and collocation sequence across noise levels.",
        "Different seeds change all these factors. Consequently bars include sensor-placement and optimization variability, not only measurement noise.",
        "A zero-noise run can still have substantial inverse error. Non-monotonic means do not by themselves establish a benefit from adding noise.",
        "All runs use the same configured final-step estimator; the reference is one numerical pollution scenario, with known width/wind/diffusion.",
        "", "![Localization versus measurement noise](localization_error_vs_noise.png)"]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/noise_sweep.yaml"))
    args = parser.parse_args()
    print(run_noise_sweep(yaml.safe_load(args.config.read_text())).to_string(index=False))


if __name__ == "__main__":
    main()
