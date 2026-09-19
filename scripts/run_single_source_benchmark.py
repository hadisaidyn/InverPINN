"""Freeze and run the revision-3 single-source benchmark; never tune on results."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from inverpinn.data.single_source_benchmark import (freeze_manifest, freeze_protocol,
    generate_scenario, load_frozen, write_csv)
from inverpinn.data.synthetic_reference import ROOT, inference_observations, load_model_inputs, sha256, write_json
from inverpinn.evaluation.single_source_benchmark import evaluate_run, load_evaluation_truth, summarize
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings, fit_sparse_source
from inverpinn.visualization.single_source_benchmark import (aggregate_figures, landscape_figure,
    reconstruction, training_figures)


def train_attempt(input_directory, output, recipe, input_hashes):
    """Loader boundary: sparse files -> strict tensors -> optimizer -> checkpoint.

    Recipe contains ONLY training settings, known sigma and optimization seed;
    neither scenario metadata nor evaluation configuration reaches fitting.
    A terminal attempt record is written on both success and Python exceptions.
    """
    if set(recipe) != {"training", "sigma", "optimization_seed"}:
        raise ValueError("Training recipe must not contain evaluation or source-truth fields.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.yaml").write_text(yaml.safe_dump(recipe | {"input_hashes": input_hashes}))
    logger = logging.getLogger(f"single_source.{output.name}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output/"training.log")
    logger.addHandler(handler)
    started = time.perf_counter()
    attempt = dict(training_terminated=False, computational_success=False, failure="", completed_updates=0)
    try:
        for name, digest in input_hashes.items():
            if sha256(Path(input_directory)/"inputs"/name) != digest:
                raise ValueError("Sparse input hash differs from the frozen manifest.")
        inputs = load_model_inputs(input_directory)
        physics = inputs["physics"]
        if physics["domain"] != {"x": [0., 1.], "y": [0., 1.]} or physics["initial_condition"] != "C=0 everywhere" or physics["boundary_condition"] != "C=0 on all four edges":
            raise ValueError("This benchmark requires the documented unit square and zero IC/BC.")
        data = SparseTrainingData(inputs["positions"], inputs["times"], inputs["measurements"], **physics["transport"])
        settings = TrainingSettings.from_mapping(recipe["training"])
        with (output/"training_history.jsonl").open("w") as stream:
            def record(values):
                stream.write(json.dumps(values, allow_nan=False)+"\n")
                stream.flush()
                attempt["completed_updates"] = values["epoch"]
            model, source, optimizer, _ = fit_sparse_source(data, settings, recipe["sigma"],
                recipe["optimization_seed"], logger, record)
        estimate = source.estimates()
        if not (all(np.isfinite(v) for v in estimate.values()) and
                0 <= estimate["x_s"] <= 1 and 0 <= estimate["y_s"] <= 1 and estimate["Q"] > 0):
            raise FloatingPointError("Invalid estimated source parameters.")
        torch.save(dict(model_state_dict=model.state_dict(), source_state_dict=source.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), epoch=attempt["completed_updates"], recipe=recipe), output/"model.pt")
        write_json(output/"source_estimate.json", estimate)
        attempt["computational_success"] = True
    except Exception as exc:
        attempt["failure"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {output}: {attempt['failure']}", flush=True)
    finally:
        logger.removeHandler(handler)
        handler.close()
        attempt.update(training_terminated=True, training_seconds=time.perf_counter()-started)
        write_json(output/"attempt.json", attempt)
    return attempt


def train_and_evaluate(root, scenario_id, seed):
    """No truth-bearing file is opened until train_attempt returns terminally."""
    root = Path(root)
    # The protocol holds sampling rules, but no sampled source values. Do not
    # load scenario_plan.json, source.json, generation.json or dense fields here.
    config = yaml.safe_load((root/"protocol.yaml").read_text())
    with (root/"scenario_manifest.csv").open() as stream:
        manifest = next(r for r in csv.DictReader(stream) if r["scenario_id"] == scenario_id)
    run = root/"runs"/scenario_id/f"seed_{seed}"
    recipe = dict(training=config["training"], sigma=config["source_family"]["sigma"], optimization_seed=seed)
    attempt = train_attempt(root/"datasets"/scenario_id, run, recipe,
        {"observations.npz": manifest["observations_hash"], "physics.json": manifest["physics_hash"]})
    # Explicit temporal barrier; all truth evaluation begins below this line.
    row = evaluate_run(root, scenario_id, seed, config)
    if (run/"training_history.jsonl").exists():
        training_figures(run, config["plot_dpi"])
    reconstruction(root, row, run, config["plot_dpi"])
    print(f"FINISHED {scenario_id} seed={seed}: computational={attempt['computational_success']}, "
          f"recovery={row['recovery_success']}, localization={row['localization_error']}", flush=True)
    return row


def run_references(root, resume=False):
    """Two independent CPU jobs at most; preserve completed references."""
    root = Path(root)
    config, plan, _ = load_frozen(root)
    pending = []
    for scenario in plan:
        directory = root/"datasets"/scenario["scenario_id"]
        if resume and (directory/"generation.json").exists():
            continue
        if directory.exists():
            raise FileExistsError(f"Incomplete/existing reference is preserved; not silently rerun: {directory}")
        pending.append(scenario)
    with ProcessPoolExecutor(max_workers=config["runtime"]["workers"]) as pool:
        tasks = {pool.submit(generate_scenario, root, s): s["scenario_id"] for s in pending}
        for task in as_completed(tasks):
            row = task.result()
            print(f"REFERENCE READY {row['scenario_id']}: estimated error={row['max_reference_error_estimate']}", flush=True)
    return freeze_manifest(root)


def planned_runs(config, plan, split="all"):
    """Fixed primary runs followed by the predeclared additional seeds."""
    optimization = config["optimization"]
    jobs = []
    for scenario in plan:
        if split != "all" and scenario["split"] != split:
            continue
        jobs.append((scenario["scenario_id"], optimization["primary_seed"]))
        if scenario["scenario_id"] in optimization["additional_seed_scenarios"]:
            jobs.extend((scenario["scenario_id"], seed) for seed in optimization["additional_seeds"])
    return jobs


def run_training(root, resume=False, split="all"):
    """Failures count as terminal attempts; explicit resume never retries them."""
    root = Path(root)
    config, plan, _ = load_frozen(root)
    freeze_manifest(root)
    lock = json.loads((root/"manifest_lock.json").read_text())
    if sha256(root/"scenario_manifest.csv") != lock["manifest_csv_hash"]:
        raise ValueError("Manifest CSV changed after freeze.")
    pending = []
    for scenario_id, seed in planned_runs(config, plan, split):
        output = root/"runs"/scenario_id/f"seed_{seed}"
        if resume and (output/"attempt.json").exists():
            if not (output/"metrics.json").exists():
                evaluate_run(root, scenario_id, seed, config)
            continue
        if output.exists():
            # A process killed outside Python's exception handling is a failed
            # attempt, not permission to rerun a seed until it gives a nice fit.
            if not resume:
                raise FileExistsError(f"Existing run: {output}")
            write_json(output/"attempt.json", dict(training_terminated=True, computational_success=False,
                failure="Interrupted attempt retained; not retried", completed_updates=None, training_seconds=None))
            evaluate_run(root, scenario_id, seed, config)
            continue
        pending.append((scenario_id, seed))
    # Stage barrier: development fits complete before any held-out fit begins.
    for stage in ("development", "heldout"):
        with ProcessPoolExecutor(max_workers=config["runtime"]["workers"]) as pool:
            tasks = [pool.submit(train_and_evaluate, root, s, seed) for s, seed in pending if s.startswith(stage+"_")]
            for task in as_completed(tasks):
                task.result()


def run_landscape(root):
    """Evaluate candidate location compatibility at fixed true Q, after fitting.

    The 201² forward solver is deliberately distinct from the 641² reference.
    A mismatch floor remains even at truth. No inverse optimizer is run here.
    """
    root = Path(root)
    config, _, _ = load_frozen(root)
    output = root/"identifiability"
    output.mkdir(exist_ok=False)
    spec = config["identifiability"]
    scenario_id, seed = spec["scenario_id"], config["optimization"]["primary_seed"]
    run = root/"runs"/scenario_id/f"seed_{seed}"
    _, _, truth, _ = load_evaluation_truth(root, scenario_id, run)
    estimate_path = run/"source_estimate.json"
    predicted = json.loads(estimate_path.read_text()) if estimate_path.exists() else None
    inputs = load_model_inputs(root/"datasets"/scenario_id)
    inference = yaml.safe_load((ROOT/config["reference"]["inference_config"]).read_text())
    reference = next(c for c in config["reference"]["candidates"] if c["nx"] == config["reference"]["selected_grid"])
    (output/"config.yaml").write_text(yaml.safe_dump(spec | dict(inference=inference, reference=reference)))
    torch.set_num_threads(1)
    axis = np.linspace(*spec["position_range"], spec["grid_count"])
    def mismatch(x, y):
        candidate = dict(x_s=float(x), y_s=float(y), Q=truth["Q"], sigma=truth["sigma"])
        readings = inference_observations(candidate, inputs, inference, reference)
        return float((readings-inputs["measurements"]).square().mean())
    mse = np.empty((len(axis), len(axis)))
    rows = []
    for j, y in enumerate(axis):
        for i, x in enumerate(axis):
            mse[j, i] = mismatch(x, y)
            rows.append(dict(x=float(x), y=float(y), fixed_Q=truth["Q"], sensor_mse=mse[j, i]))
        print(f"LANDSCAPE row {j+1}/{len(axis)}", flush=True)
    true_mse = mismatch(truth["x_s"], truth["y_s"])
    predicted_mse = mismatch(predicted["x_s"], predicted["y_s"]) if predicted else None
    j, i = np.unravel_index(np.argmin(mse), mse.shape)
    xx, yy = np.meshgrid(axis, axis)
    distances = np.hypot(xx-truth["x_s"], yy-truth["y_s"])
    summary = dict(scenario_id=scenario_id, true_source=truth, PINN_source=predicted,
        true_location_mse=true_mse, PINN_location_mse_at_true_Q=predicted_mse,
        grid_minimum_mse=float(mse[j, i]), grid_minimum_location=[float(axis[i]), float(axis[j])],
        grid_minimum_localization_error=float(distances[j, i]),
        minimum_mse_outside_one_width=float(mse[distances > truth["sigma"]].min()),
        minimum_mse_outside_two_widths=float(mse[distances > 2*truth["sigma"]].min()),
        probability_map=False, limitation="One conditional fixed-true-Q, finite-grid diagnostic; not joint identifiability or a posterior.")
    write_csv(output/"landscape.csv", rows)
    np.savez_compressed(output/"checkpoint.npz", axis=axis, sensor_mse=mse)
    write_json(output/"metrics.json", summary)
    landscape_figure(root, axis, mse, truth, predicted, config["plot_dpi"])
    return summary


def report(root):
    """Require ALL scheduled rows; report primary held-out and restarts separately."""
    root = Path(root)
    config, plan, _ = load_frozen(root)
    if (root/"summary.json").exists():
        raise FileExistsError("Final report exists; it will not be overwritten.")
    rows = [json.loads((root/"runs"/s/f"seed_{seed}"/"metrics.json").read_text())
            for s, seed in planned_runs(config, plan)]
    summary = summarize(rows, config)
    write_csv(root/"raw_results.csv", rows)
    flattened = [dict(split=split, metric=metric, **values)
        for split in ("development", "heldout") for metric, values in summary[split]["metrics"].items()]
    write_csv(root/"summary.csv", flattened)
    aggregate_figures(root, rows, config)
    write_json(root/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/single_source_benchmark.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=("all", "freeze", "generate", "train", "landscape", "report"), default="all")
    parser.add_argument("--split", choices=("all", "development", "heldout"), default="all")
    parser.add_argument("--resume", action="store_true", help="Validate freeze; skip terminal attempts, never retry failures.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = (args.output_dir or ROOT/config["output_dir"]).resolve()
    if args.stage in ("all", "freeze"):
        if args.resume:
            frozen, _, _ = load_frozen(root)
            if config != frozen:
                raise ValueError("Requested config differs from frozen protocol.")
        else:
            freeze_protocol(config, root)
    if args.stage in ("all", "generate"):
        run_references(root, args.resume)
    if args.stage in ("all", "train"):
        run_training(root, args.resume, args.split)
    if args.stage in ("all", "landscape"):
        if not (args.resume and (root/"identifiability"/"metrics.json").exists()):
            run_landscape(root)
    if args.stage in ("all", "report"):
        report(root)


if __name__ == "__main__":
    main()
