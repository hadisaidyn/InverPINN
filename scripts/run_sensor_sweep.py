"""Run independent seeded inverse PINNs and report sensor-count sensitivity."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from inverpinn.data.sensors import sample_sensors, relative_noise_std
from inverpinn.evaluation.metrics import source_localization_error, relative_source_strength_error
from inverpinn.evaluation.sensor_sweep import ERROR_COLUMNS, summarize_results


def run_sweep(config: dict) -> pd.DataFrame:
    """Generate one dataset per seed and train once per (sensor count, seed).

    The deterministic numerical field is shared across runs. Each seed freshly
    generates sensor positions, Gaussian noise, MLP initialization and training
    collocation samples. Counts are paired within each seed, with nested sensor
    locations; noise draws are not shared across counts. Independent subprocesses
    prevent global PyTorch random state from leaking between concurrent runs.
    Every run uses the exact same base inverse settings and final-step estimator.
    No truth-based checkpoint selection, retries, or outlier rejection occurs.
    """
    config = copy.deepcopy(config)
    counts, seeds = config["sensor_counts"], config["seeds"]
    if (not counts or any(type(n) is not int or n < 1 for n in counts)
            or len(set(counts)) != len(counts)):
        raise ValueError("sensor_counts must contain distinct positive integers.")
    if (len(seeds) != 5 or any(type(s) is not int or not 0 <= s < 2**63-1 for s in seeds)
            or len(set(seeds)) != 5):
        raise ValueError("Exactly five distinct valid random seeds are required.")
    if type(config["workers"]) is not int or config["workers"] < 1:
        raise ValueError("workers must be a positive integer.")
    if "noise_percent" in config and ("noise_std" in config or len(counts) != 1):
        raise ValueError("Relative noise requires exactly one sensor count and no noise_std setting.")
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"Sweep output already exists: {output}. Choose a new output_dir.")
    reference = Path(config["reference_dataset"]).resolve()
    base = yaml.safe_load(Path(config["inverse_config"]).read_text())
    with np.load(reference, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ("C", "x", "y", "t", "u", "v", "D",
                                                   "source_positions", "source_Q", "source_sigma")}
    if arrays["source_positions"].shape != (1, 2) or arrays["source_Q"].shape != (1,):
        raise ValueError("The inverse sweep currently assumes exactly one source.")
    if arrays["source_Q"][0] <= 0 or np.linalg.norm(arrays["C"].ravel()) == 0:
        raise ValueError("Relative errors require positive reference Q and nonzero concentration.")
    output.mkdir(parents=True)
    (output / "datasets").mkdir()
    (output / "configs").mkdir()
    (output / "runs").mkdir()
    config.update(output_dir=str(output), reference_dataset=str(reference), base_inverse_config=base,
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  std_ddof=1, torch_version=str(torch.__version__),
                  design="five independent seeds; paired sensor counts; fixed numerical reference")
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    torch.set_num_threads(1)
    jobs = []
    noise_by_seed = {}
    for seed in seeds:
        sampled = dict(arrays)
        if "noise_percent" in config:
            _, clean = sample_sensors(torch.from_numpy(arrays["C"]), torch.from_numpy(arrays["x"]),
                torch.from_numpy(arrays["y"]), n_sensors=counts[0], noise_std=0., seed=seed)
            std = relative_noise_std(clean, config["noise_percent"])
            sampled["noise_percent"] = np.array(config["noise_percent"])
            sampled["clean_sensor_rms"] = np.array(clean.square().mean().sqrt().item())
        else:
            std = config["noise_std"]
        noise_by_seed[seed] = std
        sampled.update(seed=np.array(seed), noise_std=np.array(std))
        for count in counts:
            p, measurements = sample_sensors(torch.from_numpy(arrays["C"]), torch.from_numpy(arrays["x"]),
                torch.from_numpy(arrays["y"]), n_sensors=count, noise_std=std, seed=seed)
            sampled[f"sensor_positions_{count}"] = p.numpy()
            sampled[f"measurements_{count}"] = measurements.numpy()
        dataset = output / "datasets" / f"seed_{seed}.npz"
        np.savez_compressed(dataset, **sampled)
        for count in counts:
            run_config = copy.deepcopy(base)
            run_config["experiment"]["seed"] = seed
            run_dir = output / "runs" / f"sensors_{count}_seed_{seed}"
            run_config.update(dataset=str(dataset), output_dir=str(run_dir), n_sensors=count)
            config_path = output / "configs" / f"sensors_{count}_seed_{seed}.yaml"
            config_path.write_text(yaml.safe_dump(run_config))
            jobs.append((count, seed, run_dir, config_path))
    config["absolute_noise_std_by_seed"] = noise_by_seed
    config["noise_definition"] = "percent/100 times RMS of clean sensor readings" if "noise_percent" in config else "absolute concentration std"
    (output / "config.yaml").write_text(yaml.safe_dump(config))

    def run_one(job):
        count, seed, run_dir, config_path = job
        row = {"n_sensors": count, "seed": seed, "status": "failed", "run_dir": str(run_dir),
               "steps": base["steps"], "noise_std": noise_by_seed[seed], "failure": ""}
        if "noise_percent" in config:
            row["noise_percent"] = config["noise_percent"]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        script = Path(__file__).resolve().with_name("train_inverse_pinn.py")
        log_path = output / "runs" / f"sensors_{count}_seed_{seed}_console.log"
        try:
            with log_path.open("w") as stream:
                subprocess.run([sys.executable, str(script), "--config", str(config_path)],
                    stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            estimate = metrics["source_estimate"]
            row.update(localization_error=source_localization_error([estimate["x_s"], estimate["y_s"]], arrays["source_positions"][0]),
                relative_strength_error=relative_source_strength_error(estimate["Q"], arrays["source_Q"][0]),
                concentration_relative_l2=metrics["relative_l2"],
                estimated_x_s=estimate["x_s"], estimated_y_s=estimate["y_s"], estimated_Q=estimate["Q"],
                true_x_s=float(arrays["source_positions"][0, 0]), true_y_s=float(arrays["source_positions"][0, 1]),
                true_Q=float(arrays["source_Q"][0]))
            if any(row[key] is None or not np.isfinite(row[key]) for key in ERROR_COLUMNS):
                raise ValueError("Undefined/nonfinite evaluation metric")
            row["status"] = "success"
        except Exception as exc:
            row["failure"] = f"{type(exc).__name__}: {exc}; see {log_path}"
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            raw = pd.DataFrame(rows).sort_values(["n_sensors", "seed"])
            # Save each completed result so a later failure never loses earlier runs.
            temporary = output / "raw_results.tmp.csv"
            raw.to_csv(temporary, index=False)
            temporary.replace(output / "raw_results.csv")
            print(f"Completed {len(rows)}/{len(jobs)}: N={row['n_sensors']} seed={row['seed']} {row['status']}", flush=True)
    if not raw["status"].eq("success").all():
        raise RuntimeError("Sweep contains failed runs; raw_results.csv and run logs retain all results. No complete summary produced.")
    summary = summarize_results(raw, counts, seeds)
    summary.to_csv(output / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    titles = ("Source localization", "Relative source-strength error", "Concentration relative L2")
    for ax, key, title in zip(axes, ERROR_COLUMNS, titles):
        scale = 1 if key == "localization_error" else 100
        ax.errorbar(summary.n_sensors, summary[f"{key}_mean"]*scale,
                    yerr=summary[f"{key}_std"]*scale, marker="o", capsize=5, label="Mean ± sample SD")
        for seed in seeds:
            points = raw[raw.seed == seed].sort_values("n_sensors")
            ax.plot(points.n_sensors, points[key]*scale, ".", alpha=0.4, color="gray")
        ax.set(xlabel="Sensor count", ylabel="Domain units" if scale == 1 else "Error (%)", title=title, xticks=counts)
        ax.legend(fontsize=8)
    fig.suptitle("Inverse PINN: five independent seeds per sensor count")
    fig.savefig(output / "error_vs_sensor_count.png", dpi=180)
    plt.close(fig)
    lines = ["# Sensor-count experiment", "", f"Seeds: {seeds}. Each run: {base['steps']} Adam updates; noise std by seed={noise_by_seed}.", "",
             "Mean ± sample standard deviation (ddof=1); all five runs included per count.", "",
             "| Sensors | Localization (domain units) | Strength error (%) | Concentration L2 (%) |",
             "|---|---:|---:|---:|"]
    for _, row in summary.iterrows():
        cells = [f"{row[f'{key}_mean']*(1 if key == 'localization_error' else 100):.4f} ± {row[f'{key}_std']*(1 if key == 'localization_error' else 100):.4f}" for key in ERROR_COLUMNS]
        lines.append(f"| {int(row.n_sensors)} | " + " | ".join(cells) + " |")
    lines += ["", "Each seed independently changes sensor placement/noise, initialization and collocation sampling.",
              "Counts are paired within seeds. Width, wind, diffusion, source initialization and all optimizer settings are fixed.",
              "Concentration L2 covers the full space-time grid, including sensor locations. Reference is one numerical simulation.",
              "No outliers are excluded; error bars are not confidence intervals and do not imply monotonic improvement.",
              "", "![Error versus sensor count](error_vs_sensor_count.png)"]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sensor_sweep.yaml"))
    args = parser.parse_args()
    summary = run_sweep(yaml.safe_load(args.config.read_text()))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
