"""Generate at most three single-source infrastructure pilots; never train."""

import argparse
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from inverpinn.data.synthetic_reference import ROOT, SCHEME, UNITS, load_model_inputs, provenance, run_refinement, sha256, write_json


def generate_pilot(config, output):
    """Separate noisy model inputs from all source/evaluation/reference truth.

    Each scenario repeats the three-grid check rather than inheriting a 1%
    accuracy claim from a different source location. Failure to meet the target
    is recorded explicitly: finest-grid outputs remain diagnostic pilots,
    never qualified benchmark references. Noise uses a local seed+1 generator,
    absolute concentration units and no clipping (including initial readings).
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot: {output}")
    config = json.loads(json.dumps(config))
    scenarios = config["scenarios"]
    if not 1 <= len(scenarios) <= 3:
        raise ValueError("This milestone permits only 1-3 pilot scenarios.")
    names = [s["scenario_id"] for s in scenarios]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", n) for n in names):
        raise ValueError("Scenario IDs must be unique safe directory names.")
    seeds = [s["seed"] for s in scenarios]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Pilot scenarios require independent distinct seeds.")
    noise = config["noise"]
    if noise["kind"] != "gaussian_absolute" or not math.isfinite(noise["std"]) or noise["std"] < 0:
        raise ValueError("Use finite nonnegative absolute Gaussian noise std.")
    base = yaml.safe_load((ROOT/config["benchmark_config"]).read_text())
    resolved = config | dict(output_dir=str(output), reference_configuration=base)
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.yaml").write_text(yaml.safe_dump(resolved))
    run_provenance = provenance(output)
    summaries = []
    for scenario in scenarios:
        directory = output/scenario["scenario_id"]
        directory.mkdir()
        settings = base | dict(seed=scenario["seed"], source=scenario["source"])
        accuracy, frozen = run_refinement(settings, directory/"truth")
        # Store the chosen reference exactly once, in truth/grid_N/checkpoint.npz.
        # The other checkpoints are required convergence evidence, not duplicates.
        chosen = accuracy["selected_grid"] or max(c["nx"] for c in settings["candidates"])
        reference = directory/"truth"/f"grid_{chosen}"/"checkpoint.npz"
        with np.load(reference, allow_pickle=False) as stored:
            clean = torch.from_numpy(stored["measurements"].copy())
            positions = stored["sensor_positions"].copy()
            times = stored["measurement_times"].copy()
            final_field = stored["fields"][-1].copy()
            x, y = stored["x"].copy(), stored["y"].copy()
        draws = torch.randn(clean.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(scenario["seed"]+1))
        noisy = clean+noise["std"]*draws
        inputs = directory/"inputs"
        inputs.mkdir()
        np.savez_compressed(inputs/"observations.npz", positions=positions, times=times, measurements=noisy.numpy())
        input_physics = dict(transport=settings["physics"], domain=settings["domain"],
            initial_condition="C=0 everywhere", boundary_condition="C=0 on all four edges",
            units={k:v for k,v in UNITS.items() if k not in ("Q", "sigma")})
        write_json(inputs/"physics.json", input_physics)
        write_json(directory/"truth"/"source.json", scenario["source"] | dict(Q_definition="Gaussian peak concentration/time rate, not integrated emissions"))
        (directory/"config.yaml").write_text(yaml.safe_dump(scenario | dict(noise=noise, reference_configuration=frozen)))
        generation = next(c for c in accuracy["candidates"] if c["grid_size"] == chosen)
        metadata = dict(schema_version=2, scenario_id=scenario["scenario_id"], random_seed=scenario["seed"],
            placement_seed=scenario["seed"], noise_seed=scenario["seed"]+1, noise=noise,
            domain=settings["domain"], physics=settings["physics"], units=UNITS,
            source_truth_file="truth/source.json", reference_checkpoint=str(reference.relative_to(directory)),
            reference_solver=generation, reference_accuracy=accuracy,
            solver_type=SCHEME, provenance=dict(git_commit=run_provenance["git_commit"],
                code_provenance_file="truth/provenance.json", source_archive="truth/source_snapshot.zip"),
            generated_at_utc=json.loads((directory/"truth"/"provenance.json").read_text())["generated_at_utc"],
            sensor_coordinates=positions.tolist(), measurement_times=times.tolist(),
            evaluation_times=settings["evaluation_times"], final_field_index=len(settings["evaluation_times"])-1,
            initial_condition=input_physics["initial_condition"], boundary_condition=input_physics["boundary_condition"],
            dimensions=dict(fields=["evaluation_time", "y", "x"], measurements=["measurement_time", "sensor"], positions=["sensor", "xy"]),
            sampling="Bilinear in space; linear between adjacent Euler states in time; no snapping/extrapolation",
            inputs_sha256={p.name:sha256(p) for p in inputs.iterdir()}, reference_sha256=sha256(reference),
            purpose="Infrastructure pilot, not a recovery benchmark", benchmark_release_ready=False,
            estimated_target_met=accuracy["selected_grid"] is not None)
        write_json(directory/"metadata.json", metadata)
        loaded = load_model_inputs(directory)
        metrics = dict(sensor_count=len(positions), measurement_count=len(times),
            noisy_sensor_values=noisy.numel(), features=list(loaded["features"].shape),
            noise_rmse=float((noisy-clean).square().mean().sqrt()),
            estimated_target_met=metadata["estimated_target_met"], chosen_reference_grid=chosen,
            max_assessed_Richardson_relative_L2_estimate=accuracy["per_grid"][str(chosen)]["max_assessed_Richardson_relative_L2_estimate"],
            runtime_seconds=generation["runtime_seconds"], peak_process_rss_bytes=generation["peak_process_rss_bytes"],
            retained_field_bytes=generation["retained_field_bytes"], retained_measurement_bytes=generation["retained_measurement_bytes"])
        write_json(directory/"metrics.json", metrics)
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        mesh = ax.pcolormesh(x, y, final_field, shading="auto", cmap="inferno")
        ax.scatter(*positions.T, s=15, c="cyan", label="Sensors")
        ax.scatter(scenario["source"]["x_s"], scenario["source"]["y_s"], marker="x", c="lime", label="True source (evaluation only)")
        ax.set(xlabel="x (synthetic length)", ylabel="y (synthetic length)", aspect="equal",
               title=f"{scenario['scenario_id']}: infrastructure pilot only")
        ax.legend(fontsize=8)
        fig.colorbar(mesh, ax=ax, label="C (synthetic concentration)")
        fig.savefig(directory/"pilot.png", dpi=base["plot_dpi"])
        plt.close(fig)
        summaries.append(dict(scenario_id=scenario["scenario_id"], seed=scenario["seed"], **metrics,
                              disk_bytes=sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())))
        print(f"Pilot {scenario['scenario_id']}: grid {chosen}, estimated target met={metadata['estimated_target_met']}", flush=True)
    write_json(output/"pilot_summary.json", dict(purpose="Three-scenario infrastructure pilot, never training", scenarios=summaries))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/reference_pilot.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    generate_pilot(config, args.output_dir or ROOT/config["output_dir"])


if __name__ == "__main__":
    main()
