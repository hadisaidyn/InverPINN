"""Run loss ablations with paired observations, initialization and sample streams."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import yaml

from inverpinn.evaluation.ablation import variants, summarize


def run(config):
    base = yaml.safe_load(Path(config["inverse_config"]).read_text())
    conditions = variants(base, config["pde_weights"])
    seeds = config["seeds"]
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not 0 <= s < 2**63-1 for s in seeds)):
        raise ValueError("Require at least two distinct valid seeds for sample SD.")
    if type(config["workers"]) is not int or config["workers"] < 1:
        raise ValueError("workers must be positive integer.")
    datasets = [Path(p).resolve() for p in config["datasets"]]
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Require distinct datasets.")
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    metadata = []
    for i, path in enumerate(datasets):
        with np.load(path, allow_pickle=False) as data:
            positions = data[f"sensor_positions_{base['n_sensors']}"]
            readings = data[f"measurements_{base['n_sensors']}"]
        metadata.append(dict(dataset_id=i, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
            observations_sha256=hashlib.sha256(readings.tobytes()).hexdigest()))
    output.mkdir(parents=True)
    (output / "configs").mkdir()
    (output / "runs").mkdir()
    (output / "config.yaml").write_text(yaml.safe_dump(config | dict(base_config=base, datasets_resolved=metadata)))
    jobs = []
    for item in metadata:
        for seed in seeds:
            for condition, template in conditions.items():
                name = f"dataset_{item['dataset_id']}_{condition}_seed_{seed}"
                run_dir = output / "runs" / name
                settings = template | dict(dataset=item["path"], output_dir=str(run_dir),
                    experiment=template["experiment"] | dict(seed=seed))
                path = output / "configs" / f"{name}.yaml"
                path.write_text(yaml.safe_dump(settings))
                jobs.append((item, seed, condition, path, run_dir))

    def execute(job):
        item, seed, condition, path, run_dir = job
        row = dict(dataset_id=item["dataset_id"], seed=seed, condition=condition,
            dataset_sha256=item["sha256"], status="failed", run_dir=str(run_dir),
            source_data_coupling=condition != "B_no_pde")
        row.update({f"weight_{k}": v for k,v in conditions[condition]["loss_weights"].items()})
        try:
            with (run_dir.parent / f"{run_dir.name}.log").open("w") as stream:
                subprocess.run([sys.executable, str(Path(__file__).with_name("train_inverse_pinn.py")),
                    "--config", str(path)], stdout=stream, stderr=subprocess.STDOUT, check=True,
                    env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
            metrics = json.loads((run_dir / "metrics.json").read_text())
            row.update(localization_error=metrics["matched_sources"]["mean_localization_error"],
                concentration_relative_l2=metrics["relative_l2"])
            if any(row[k] is None or not np.isfinite(row[k]) for k in ("localization_error", "concentration_relative_l2")):
                raise ValueError("Undefined/nonfinite metrics")
            row["status"] = "success"
        except Exception as exc:
            row["failure"] = f"{type(exc).__name__}: {exc}"
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for future in as_completed([pool.submit(execute, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            raw = pd.DataFrame(rows).sort_values(["dataset_id", "condition", "seed"])
            raw.to_csv(output / "raw_results.csv", index=False)
            print(f"Completed {len(rows)}/{len(jobs)}: {row['condition']} seed={row['seed']} {row['status']}", flush=True)
    summary = summarize(raw, list(range(len(datasets))), seeds, list(conditions))
    summary.to_csv(output / "summary.csv", index=False)
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4*len(datasets)), squeeze=False, layout="constrained")
    lines = ["# Loss ablation experiment", "", f"Training seeds: {seeds}; {base['steps']} updates per run.",
        "Identical stored observations, initialization and random sample stream within each paired seed.",
        "Only one loss weight changes; no truth-based tuning or checkpoint selection.", "",
        "| Dataset | Condition | Localization mean ± SD | Concentration L2 (%) mean ± SD |",
        "|---|---|---:|---:|"]
    for i in range(len(datasets)):
        group = summary[summary.dataset_id == i]
        for ax, metric, scale, label in zip(axes[i], ("localization_error", "concentration_relative_l2"), (1,100),
                ("Localization (domain units)", "Concentration relative L2 (%)")):
            ax.errorbar(range(len(group)), group[f"{metric}_mean"]*scale,
                yerr=group[f"{metric}_std"]*scale, fmt="o", capsize=4)
            ax.set(xticks=range(len(group)), xticklabels=group.condition, ylabel=label, title=f"Dataset {i}")
            ax.tick_params(axis="x", rotation=25)
        for _, r in group.iterrows():
            lines.append(f"| {i} | {r.condition} | {r.localization_error_mean:.5f} ± {r.localization_error_std:.5f} | "
                f"{100*r.concentration_relative_l2_mean:.3f} ± {100*r.concentration_relative_l2_std:.3f} |")
    fig.savefig(output / "ablation_comparison.png", dpi=180)
    plt.close(fig)
    lines += ["", "B has no source-data coupling: its localization is the INITIAL-GUESS error, not a learned estimate.",
        "B retains explicit IC/BC losses; it is not the earlier purely data-only baseline.",
        "Removing IC loss does not remove t=0 sensor readings; removing BC loss does not remove near-edge sensors.",
        "Zero-weight terms remain computed for diagnostics and paired sampling, so this is not a speed ablation.",
        "Error bars are sample SD across training seeds on fixed observations, not confidence intervals or dataset variability.",
        "Summary CSV also reports paired error changes from A. All runs are retained; failures prevent a complete summary.",
        "PDE weights depend on units/scaling. This sweep is a sensitivity study, not held-out hyperparameter selection.",
        "", "![Ablation comparison](ablation_comparison.png)"]
    (output / "report.md").write_text("\n".join(lines)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ablation.yaml"))
    args = parser.parse_args()
    print(run(yaml.safe_load(args.config.read_text())).to_string(index=False))
