"""Revision 4: guarded development-only diagnosis and controlled experiments.

No benchmark-wide plan, manifest, report, glob or held-out loader is used.
Every run saves a recipe, checkpoint, losses, gradients, metrics and figures.
Dense metrics use all four 641² snapshots; saved plot previews use every fourth
node only. Re-evaluation can recover full predictions from the checkpoint.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml

from inverpinn.data.development_only import (DEVELOPMENT_IDS, development_hashes,
    development_path, development_truth, sparse_inputs, install_development_read_guard)
from inverpinn.data.synthetic_reference import ROOT, sha256, write_json
from inverpinn.evaluation.metrics import concentration_metrics
from inverpinn.evaluation.single_source_benchmark import parameter_metrics, physical_diagnostics, predict_snapshots
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.training.objective_diagnostics import ObjectiveAudit
from inverpinn.training.single_source import SparseTrainingData, TrainingSettings
from inverpinn.training.development_candidates import fit_audited_baseline, fit_candidate, loss_factors
from inverpinn.visualization.model_diagnosis import audit_figures, residual_figure, run_figures


def snapshot(directory):
    """Archive executable code and package versions alongside the git revision."""
    files = sorted(list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").glob("*.py")))
    with zipfile.ZipFile(directory/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    write_json(directory/"provenance.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        working_tree=git("status", "--short"), source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in files},
        archive_sha256=sha256(directory/"source_snapshot.zip")))
    write_json(directory/"environment.json", dict(python=platform.python_version(), platform=platform.platform(),
        packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}))


def initialize(config):
    root, output = ROOT/config["benchmark_root"], ROOT/config["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    (output/"baseline_protocol.yaml").write_text(yaml.safe_dump(config))
    snapshot(output)
    write_json(output/"development_data_hashes.json", development_hashes(root))


def residual_maps(model, source, physics, config, run):
    axis = np.linspace(0, 1, config["residual_map_grid"])
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    maps, rows = [], []
    estimate = source.estimates()
    near = (xx-estimate["x_s"])**2+(yy-estimate["y_s"])**2 <= (2*config["sigma"])**2
    for t in config["residual_map_times"]:
        points = torch.tensor(np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, t))), dtype=torch.float32)
        pieces = []
        for p in points.split(config["diagnostics"]["prediction_batch"]):
            p = p.clone().requires_grad_()
            pieces.append(autograd_pde_residual(model(p), p, **physics, S=source(p[:, :1], p[:, 1:2])).detach().numpy())
        residual = np.concatenate(pieces).reshape(xx.shape)
        maps.append(residual)
        for region, mask in (("near_estimate_2sigma", near), ("away", ~near)):
            rows.append(dict(time=t, region=region, count=int(mask.sum()),
                             rmse=float(np.sqrt(np.mean(residual[mask]**2)))))
    pd.DataFrame(rows).to_csv(run/"residual_regions.csv", index=False)
    np.savez_compressed(run/"residual_maps.npz", residual=np.stack(maps), x=axis, y=axis, times=config["residual_map_times"])
    residual_figure(run, estimate, maps, config["residual_map_times"], config["plot_dpi"])


def run_one(config, identifier, candidate):
    """Sparse-only fitting followed by a terminal barrier and development truth."""
    root, output = ROOT/config["benchmark_root"], ROOT/config["output_dir"]
    development_path(root, identifier)  # guard before opening any case files
    run = output/"runs"/candidate["name"]/identifier
    run.mkdir(parents=True, exist_ok=False)
    inputs = sparse_inputs(root, identifier)
    physics = inputs["physics"]
    if (physics["domain"] != {"x":[0., 1.], "y":[0., 1.]} or
        physics["initial_condition"] != "C=0 everywhere" or physics["boundary_condition"] != "C=0 on all four edges"):
        raise ValueError("Only the controlled unit-square homogeneous-IC/BC problem is permitted.")
    data = SparseTrainingData(inputs["positions"], inputs["times"], inputs["measurements"], **physics["transport"])
    recipe = dict(training=candidate["training"], sigma=config["sigma"], seed=config["optimization_seed"],
                  formulation=candidate.get("formulation", "baseline"))
    (run/"config.yaml").write_text(yaml.safe_dump(recipe))
    settings = TrainingSettings.from_mapping(recipe["training"])
    audit = ObjectiveAudit(settings.steps, config["gradient_interval"])
    history = []
    logger = logging.getLogger("diagnosis.silent")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    attempt = dict(training_terminated=False, computational_success=False, failure="")
    start = time.perf_counter()
    try:
        if recipe["formulation"] == "baseline":
            model, source, optimizer, _ = fit_audited_baseline(data, settings, recipe["sigma"], recipe["seed"],
                logger, history.append, on_objective=audit)
        else:
            model, source, optimizer, _ = fit_candidate(data, settings, recipe["sigma"], recipe["seed"],
                recipe["formulation"], logger, history.append, on_objective=audit)
        torch.save(dict(model_state_dict=model.state_dict(), source_state_dict=source.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), recipe=recipe), run/"model.pt")
        attempt["computational_success"] = True
    except Exception as exc:
        attempt["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        attempt.update(training_terminated=True, completed_updates=len(history), training_seconds=time.perf_counter()-start)
        write_json(run/"attempt.json", attempt)
        pd.DataFrame(history).to_csv(run/"loss_components.csv", index=False)
        pd.DataFrame(audit.gradients).to_csv(run/"gradient_norms.csv", index=False)
        pd.DataFrame(audit.coverage).to_csv(run/"collocation_coverage.csv", index=False)
    row = dict(candidate=candidate["name"], scenario_id=identifier, **attempt)
    if not attempt["computational_success"]:
        write_json(run/"metrics.json", row)
        raise RuntimeError(f"Failed run retained at {run}: {attempt['failure']}")
    # No truth-bearing file has been passed to, or read by, the fit function.
    truth, reference, manifest = development_truth(root, identifier, run)
    estimate = source.estimates()
    write_json(run/"source_estimate.json", estimate)
    with np.load(reference, allow_pickle=False) as stored:
        fields, x, y, times = [stored[k] for k in ("fields", "x", "y", "field_times")]
        prediction = predict_snapshots(model, x, y, times, config["diagnostics"]["prediction_batch"])
        row.update(concentration_metrics(prediction, fields))
        row.update(negative_fraction=float(np.mean(prediction < 0)), minimum_concentration=float(prediction.min()))
        preview = dict(fields=prediction[:, ::4, ::4].astype(np.float32), truth=fields[:, ::4, ::4],
                       x=x[::4], y=y[::4], times=times)
        np.savez_compressed(run/"prediction_preview.npz", **preview)
    with torch.no_grad():
        sensor_mse = (model(inputs["features"].float())-inputs["targets"].float()).square().mean().item()
    row.update(parameter_metrics(estimate, truth))
    row.update(x_true=truth["x_s"], y_true=truth["y_s"], Q_true=truth["Q"],
               x_pred=estimate["x_s"], y_pred=estimate["y_s"], Q_pred=estimate["Q"], sensor_mse=sensor_mse,
               signed_relative_Q_error=(estimate["Q"]-truth["Q"])/truth["Q"],
               reference_target_met=manifest["reference_target_met"])
    row.update(physical_diagnostics(model, source, physics["transport"], data.times[-1].item(), config["diagnostics"]))
    factors = dict(data=1., pde=1., initial=1., boundary=1.) if recipe["formulation"] == "baseline" else loss_factors(recipe["formulation"])
    row["objective"] = sum(recipe["training"]["loss_weights"][k]*factors[k]*v for k,v in
        dict(data=sensor_mse, pde=row["pde_rmse"]**2, initial=row["ic_rmse"]**2, boundary=row["bc_rmse"]**2).items())
    row["recovery_success"] = row["localization_error"] <= .08 and row["relative_strength_error"] <= .2
    write_json(run/"metrics.json", row)
    run_figures(run, history, audit.gradients, preview, config["plot_dpi"])
    if identifier in config["residual_map_cases"]:
        residual_maps(model, source, physics["transport"], config, run)
    print(f"DONE {candidate['name']} {identifier}: loc={row['localization_error']:.5g}, Qerr={row['relative_strength_error']:.5g}, L2={row['relative_l2']:.5g}", flush=True)
    return row


def run_candidates(config, candidates, resume):
    output = ROOT/config["output_dir"]
    jobs = []
    for candidate in candidates:
        for identifier in DEVELOPMENT_IDS:
            path = output/"runs"/candidate["name"]/identifier
            if path.exists():
                if resume and (path/"metrics.json").exists():
                    continue
                raise FileExistsError(f"Existing attempt preserved (not silently restarted): {path}")
            jobs.append((identifier, candidate))
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=install_development_read_guard,
                             initargs=(ROOT/config["benchmark_root"],)) as pool:
        futures = [pool.submit(run_one, config, identifier, candidate) for identifier, candidate in jobs]
        for future in as_completed(futures):
            future.result()


def baseline_audit(config):
    output = ROOT/config["output_dir"]
    histories, gradients, rows = [], [], []
    for identifier in DEVELOPMENT_IDS:
        run = output/"runs"/"B0"/identifier
        rows.append(json.loads((run/"metrics.json").read_text()))
        histories.append(pd.read_csv(run/"loss_components.csv").assign(scenario_id=identifier))
        gradients.append(pd.read_csv(run/"gradient_norms.csv").assign(scenario_id=identifier))
    frame, gradients = pd.concat(histories), pd.concat(gradients)
    frame.to_csv(output/"loss_components.csv", index=False)
    gradients.to_csv(output/"gradient_norms.csv", index=False)
    pd.DataFrame(rows).to_csv(output/"baseline_results.csv", index=False)
    audit_figures(frame, gradients, output, config["plot_dpi"])
    write_json(output/"baseline_complete.json", dict(completed_at_utc=datetime.now(timezone.utc).isoformat(), count=len(rows)))


def resolve_candidates(config, plan):
    """Resolve an explicit one-factor ladder, never reading scenario truth."""
    import copy
    resolved = {"B0":dict(name="B0", training=copy.deepcopy(config["training"]),
        formulation=dict(constraint="unconstrained", Q_parameterization="softplus", scales=None))}
    allowed = {"name", "parent", "change", "scales", "constraint", "Q_parameterization",
               "initial_Q", "initial_x", "initial_y", "steps"}
    for entry in plan["candidates"]:
        if not set(entry) <= allowed or entry["name"] in resolved:
            raise ValueError("Unexpected or duplicate candidate specification.")
        candidate = copy.deepcopy(resolved[entry["parent"]])
        candidate.update(name=entry["name"], parent=entry["parent"], change=entry["change"])
        for key in ("initial_Q", "initial_x", "initial_y", "steps"):
            if key in entry:
                candidate["training"][key] = entry[key]
        for key in ("constraint", "Q_parameterization"):
            if key in entry:
                candidate["formulation"][key] = entry[key]
        if entry.get("scales"):
            candidate["formulation"]["scales"] = copy.deepcopy(plan["scales"])
        resolved[entry["name"]] = candidate
    return list(resolved.values())[1:]


def freeze_candidates(config):
    output = ROOT/config["output_dir"]
    if not (output/"baseline_complete.json").exists():
        raise RuntimeError("Complete baseline audit BEFORE candidate changes.")
    plan = yaml.safe_load((ROOT/"configs/model_diagnosis_candidates.yaml").read_text())
    frozen = output/"candidate_protocol"
    if frozen.exists():
        if yaml.safe_load((frozen/"plan.yaml").read_text()) != plan:
            raise ValueError("Candidate plan changed after freeze.")
        # Plot/report/access-guard additions may evolve, but scientific fitting
        # code must not be mixed silently within a frozen experiment.
        provenance = json.loads((frozen/"provenance.json").read_text())
        scientific = ("src/inverpinn/training/", "src/inverpinn/physics/", "src/inverpinn/models/")
        for name, digest in provenance["source_hashes"].items():
            if name.startswith(scientific) and sha256(ROOT/name) != digest:
                raise ValueError(f"Scientific code changed after candidate freeze: {name}")
        return yaml.safe_load((frozen/"resolved.yaml").read_text())
    frozen.mkdir()
    candidates = resolve_candidates(config, plan)
    (frozen/"plan.yaml").write_text(yaml.safe_dump(plan))
    (frozen/"resolved.yaml").write_text(yaml.safe_dump(candidates))
    for filename in ("model_diagnosis_selection.md", "model_diagnosis_initial_audit.md", "nondimensionalization.md"):
        (frozen/filename).write_text((ROOT/"docs"/filename).read_text())
    snapshot(frozen)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_diagnosis.yaml")
    parser.add_argument("--stage", choices=["baseline", "candidates"], default="baseline")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/args.config).read_text())
    install_development_read_guard(ROOT/config["benchmark_root"])
    output = ROOT/config["output_dir"]
    if not output.exists():
        initialize(config)
    elif not args.resume:
        raise FileExistsError("Output exists. --resume permits only not-yet-started runs.")
    if yaml.safe_load((output/"baseline_protocol.yaml").read_text()) != config:
        raise ValueError("Configuration changed after baseline freeze.")
    if args.stage == "baseline":
        run_candidates(config, [dict(name="B0", training=config["training"])], args.resume)
        baseline_audit(config)
    else:
        run_candidates(config, freeze_candidates(config), args.resume)
    if development_hashes(ROOT/config["benchmark_root"]) != json.loads((output/"development_data_hashes.json").read_text()):
        raise RuntimeError("Development reference data changed.")


if __name__ == "__main__":
    main()
