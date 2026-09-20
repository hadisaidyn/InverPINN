"""Run only the frozen revision-5 primary fits and predeclared seed substudy.

Generation is a separate stage. No source truth enters the frozen fitting APIs.
Missing/failed runs remain terminal; no best epoch, retry or model rescue exists.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from inverpinn.data.fresh_blind import FRESH_IDS, fresh_path, install_guard, jobs, manifest, verify
from inverpinn.data.synthetic_reference import ROOT, load_model_inputs, sha256, write_json
from inverpinn.evaluation.metrics import concentration_metrics
from inverpinn.evaluation.single_source_benchmark import parameter_metrics, physical_diagnostics, predict_snapshots, recovery_decision
from inverpinn.models.mlp import ConcentrationMLP
from inverpinn.models.constrained_concentration import ConstrainedConcentrationMLP
from inverpinn.physics.trainable_source import TrainableGaussianSource
from inverpinn.physics.diagnostic_source import DiagnosticGaussianSource
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings, fit_sparse_source
from inverpinn.training.development_candidates import fit_candidate
from inverpinn.visualization.single_source_benchmark import training_figures


def run_path(root, model, identifier, seed):
    if model not in ("B0", "B_revised") or identifier not in FRESH_IDS:
        raise ValueError("Only frozen models and fresh IDs allowed.")
    return Path(root)/"runs"/model/identifier/f"seed_{seed}"


def recipe_for(root, model, seed):
    """Use frozen file content, not a retyped or tuned training configuration."""
    config = yaml.safe_load((Path(root)/"protocol.yaml").read_text())
    path = Path(root)/f"{model}_frozen.yaml"
    if sha256(path) != config["models"][model]["sha256"]:
        raise ValueError("Frozen model copy changed.")
    frozen = yaml.safe_load(path.read_text())
    if model == "B0":
        training, formulation = frozen["training"], "baseline"
    else:
        training, formulation = frozen["inference"]["training"], frozen["inference"]["formulation"]
    return dict(training=training, formulation=formulation, sigma=config["source_family"]["sigma"], seed=seed)


def load_fit(path, final_time=.8):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    recipe = saved["recipe"]
    training, form = recipe["training"], recipe["formulation"]
    args = [training[k] for k in ("initial_x", "initial_y", "initial_Q")]+[recipe["sigma"]]
    if form == "baseline":
        model = ConcentrationMLP([0.,0.,0.], [1.,1.,final_time], training["width"], training["depth"])
        source = TrainableGaussianSource(*args)
    else:
        model = ConstrainedConcentrationMLP(final_time, training["width"], training["depth"], form["constraint"])
        source = DiagnosticGaussianSource(*args, parameterization=form["Q_parameterization"])
    model.load_state_dict(saved["model_state_dict"])
    source.load_state_dict(saved["source_state_dict"])
    model.eval()
    return model, source


def evaluate(root, model_name, identifier, seed, config):
    """Truth access only after a persisted terminal attempt; never feed it back."""
    run = run_path(root, model_name, identifier, seed)
    attempt = json.loads((run/"attempt.json").read_text())
    if attempt["training_terminated"] is not True:
        raise RuntimeError("Evaluation requires terminal training.")
    if (run/"metrics.json").exists():
        raise FileExistsError("Existing evaluation is immutable.")
    case = fresh_path(root, identifier)
    generation = json.loads((case/"generation.json").read_text())
    truth = json.loads((case/"truth/source.json").read_text())
    from inverpinn.data.single_source_benchmark import object_hash
    if object_hash(truth) != generation["source_hash"]:
        raise ValueError("Fresh truth hash changed.")
    reference = (root/generation["reference_path"]).resolve()
    if not reference.is_relative_to(case) or sha256(reference) != generation["reference_data_hash"]:
        raise ValueError("Reference path/hash changed.")
    keys = ("x_pred", "y_pred", "Q_pred", "localization_error", "relative_strength_error", "relative_l2", "rmse", "mae",
            "negative_fraction", "minimum_concentration", "pde_rmse", "pde_mae", "pde_maxabs", "bc_rmse", "bc_mae", "bc_maxabs",
            "ic_rmse", "ic_mae", "ic_maxabs", "signed_Q_error", "signed_relative_Q_error")
    row = dict.fromkeys(keys)
    row.update(model=model_name, scenario_id=identifier, optimization_seed=seed,
        primary=seed == config["optimization"]["primary_seed"],
        computational_success=attempt["computational_success"], evaluation_success=False, recovery_success=False,
        failure=attempt["failure"], completed_updates=attempt["completed_updates"], training_seconds=attempt["training_seconds"],
        x_true=truth["x_s"], y_true=truth["y_s"], Q_true=truth["Q"],
        observations_hash=generation["observations_hash"], reference_data_hash=generation["reference_data_hash"],
        reference_target_met=generation["reference_target_met"], reference_error_estimate=generation["max_reference_error_estimate"])
    if attempt["computational_success"]:
        torch.set_num_threads(1)
        model, source = load_fit(run/"model.pt", config["measurement_times"]["stop"])
        estimate = source.estimates()
        row.update(x_pred=estimate["x_s"], y_pred=estimate["y_s"], Q_pred=estimate["Q"], **parameter_metrics(estimate, truth))
        row["signed_Q_error"] = estimate["Q"]-truth["Q"]
        row["signed_relative_Q_error"] = row["signed_Q_error"]/truth["Q"]
        with np.load(reference, allow_pickle=False) as stored:
            fields, x, y, times = [stored[k] for k in ("fields", "x", "y", "field_times")]
            predicted = predict_snapshots(model, x, y, times, config["diagnostics"]["prediction_batch"])
            row.update(concentration_metrics(predicted, fields), negative_fraction=float(np.mean(predicted < 0)),
                       minimum_concentration=float(predicted.min()))
            np.savez_compressed(run/"predictions.npz", fields=predicted, x=x, y=y, field_times=times)
            write_json(run/"snapshot_metrics.json", [dict(time=float(t), **concentration_metrics(p, f)) for t,p,f in zip(times,predicted,fields)])
        row.update(physical_diagnostics(model, source, config["physics"], config["measurement_times"]["stop"], config["diagnostics"]))
        row["evaluation_success"] = True
        row["recovery_success"] = recovery_decision(True, row, config["recovery"])
    write_json(run/"metrics.json", row)
    return row


def train_one(root, model_name, identifier, seed):
    root = Path(root)
    state = install_guard(root)
    config = yaml.safe_load((root/"protocol.yaml").read_text())
    if (model_name, identifier, seed) not in jobs(config):
        raise ValueError("Run was not preregistered.")
    run = run_path(root, model_name, identifier, seed)
    run.mkdir(parents=True, exist_ok=False)
    recipe = recipe_for(root, model_name, seed)
    manifest_lock = json.loads((root/"manifest_lock.json").read_text())
    if sha256(root/"input_manifest.json") != manifest_lock["inputs_hash"]:
        raise ValueError("Input-only manifest changed.")
    hashes = next(r for r in json.loads((root/"input_manifest.json").read_text()) if r["scenario_id"] == identifier)
    (run/"config.yaml").write_text(yaml.safe_dump(dict(recipe=recipe, input_hashes=hashes), sort_keys=False))
    case = fresh_path(root, identifier)
    attempt = dict(training_terminated=False, computational_success=False, failure="", completed_updates=0)
    start = time.perf_counter()
    state["fitting_case"] = identifier
    try:
        if sha256(case/"inputs/observations.npz") != hashes["observations_hash"] or sha256(case/"inputs/physics.json") != hashes["physics_hash"]:
            raise ValueError("Sparse input differs from immutable manifest.")
        inputs = load_model_inputs(case)
        physics = inputs["physics"]
        if (physics["transport"] != config["physics"] or physics["domain"] != config["domain"]
                or physics["initial_condition"] != "C=0 everywhere" or physics["boundary_condition"] != "C=0 on all four edges"):
            raise ValueError("Controlled physics changed.")
        data = SparseTrainingData(inputs["positions"], inputs["times"], inputs["measurements"], **physics["transport"])
        settings = TrainingSettings.from_mapping(recipe["training"])
        logger = logging.getLogger("fresh_blind_fit")
        logger.setLevel(logging.WARNING)
        with (run/"training_history.jsonl").open("x") as stream:
            def record(values):
                stream.write(json.dumps(values, allow_nan=False)+"\n")
                stream.flush()
                attempt["completed_updates"] = values["epoch"]
            if model_name == "B0":
                model, source, optimizer, _ = fit_sparse_source(data, settings, recipe["sigma"], seed, logger, record)
            else:
                model, source, optimizer, _ = fit_candidate(data, settings, recipe["sigma"], seed, recipe["formulation"], logger, record)
        torch.save(dict(model_state_dict=model.state_dict(), source_state_dict=source.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), recipe=recipe), run/"model.pt")
        write_json(run/"source_estimate.json", source.estimates())
        attempt["computational_success"] = True
    except FloatingPointError as exc:
        attempt["failure"] = f"Numerical optimization failure retained: {exc}"
    except Exception as exc:
        attempt["failure"] = f"Unexpected error, benchmark review required: {type(exc).__name__}: {exc}"
        write_json(run/"invalidation_review_required.json", dict(error=attempt["failure"]))
        raise
    finally:
        state["fitting_case"] = None
        attempt.update(training_terminated=True, training_seconds=time.perf_counter()-start)
        write_json(run/"attempt.json", attempt)
    try:
        row = evaluate(root, model_name, identifier, seed, config)
        training_figures(run, config["plot_dpi"])
    except Exception as exc:
        write_json(run/"invalidation_review_required.json", dict(stage="evaluation_or_plotting",
            error=f"{type(exc).__name__}: {exc}", training_checkpoint_preserved=True))
        raise
    # Progress deliberately omits source truth, estimates and recovery metrics.
    print(f"FIT COMPLETE {model_name} {identifier} seed={seed}; computational={row['computational_success']}", flush=True)
    return model_name, identifier, seed


def run_all(root):
    from run_model_diagnosis import snapshot
    root = Path(root)
    install_guard(root)
    config, _, _ = verify(root)
    rows = manifest(root)
    if any(r["max_reference_error_estimate"] is None for r in rows):
        raise RuntimeError("Reference validity review required.")
    archive = root/"fit_execution"
    if not archive.exists():
        archive.mkdir()
        snapshot(archive)
    execution = json.loads((archive/"provenance.json").read_text())
    for name, digest in execution["source_hashes"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Execution code changed after fit freeze: {name}")
    pending = []
    for model, identifier, seed in jobs(config):
        run = run_path(root, model, identifier, seed)
        if (run/"invalidation_review_required.json").exists():
            raise RuntimeError(f"Unresolved benchmark bug: {run}")
        if (run/"metrics.json").exists():
            continue
        if run.exists():
            if not (run/"attempt.json").exists():
                write_json(run/"attempt.json", dict(training_terminated=True, computational_success=False,
                    failure="Interrupted attempt retained; no retry", completed_updates=None, training_seconds=None))
            evaluate(root, model, identifier, seed, config)
            continue
        pending.append((model, identifier, seed))
    with ProcessPoolExecutor(max_workers=config["runtime"]["workers"]) as pool:
        futures = [pool.submit(train_one, root, *job) for job in pending]
        try:
            for future in as_completed(futures):
                future.result()
        except Exception:
            for future in futures:
                future.cancel()  # leave unstarted jobs unstarted; never retry a failed fit
            raise
    verify(root)
    manifest(root)
    write_json(root/"fits_complete.json", dict(scheduled=80, primary=60, additional_seed_runs=20,
        execution_archive_sha256=sha256(archive/"source_snapshot.zip")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fresh_blind_benchmark.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/args.config).read_text())
    run_all(ROOT/config["output_dir"])
