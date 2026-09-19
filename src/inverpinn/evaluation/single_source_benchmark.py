"""Post-training recovery metrics and independent physical diagnostics.

Truth is opened only after a persisted terminal training-attempt record. Dense
errors equally weight four specified snapshots; these are not a continuous-time
integral. Negative predictions remain untouched throughout evaluation.
"""

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from inverpinn.data.synthetic_reference import sha256, write_json
from inverpinn.evaluation.metrics import concentration_metrics, relative_source_strength_error, source_localization_error
from inverpinn.models.mlp import ConcentrationMLP
from inverpinn.physics.autodiff import autograd_pde_residual
from inverpinn.physics.trainable_source import TrainableGaussianSource
from inverpinn.training.forward_pinn import require_finite


def residual_statistics(values):
    """RMSE, MAE and maximum absolute violation relative to zero."""
    require_finite(values, "independent diagnostic")
    values = values.detach().double()
    return dict(rmse=values.square().mean().sqrt().item(),
                mae=values.abs().mean().item(), maxabs=values.abs().max().item())


def diagnostic_points(settings, final_time):
    """Fresh deterministic uniform draws, using no training-generator state."""
    generator = torch.Generator().manual_seed(settings["seed"])
    upper = torch.tensor([1., 1., final_time])
    interior = torch.rand((settings["interior_count"], 3), generator=generator)*upper
    initial = torch.rand((settings["initial_count"], 3), generator=generator)*upper
    initial[:, 2] = 0
    edges = []
    for axis, value in ((0, 0.), (0, 1.), (1, 0.), (1, 1.)):
        edge = torch.rand((settings["boundary_per_edge"], 3), generator=generator)*upper
        edge[:, axis] = value
        edges.append(edge)
    return interior, initial, torch.cat(edges)


def physical_diagnostics(model, source, physics, final_time, settings):
    """Independently measure r, C(x,y,0), and C on all four clean edges.

    r has concentration/time units; IC/BC violations have concentration units.
    This finite sample is a diagnostic, not a bound over the continuous domain.
    """
    interior, initial, boundary = diagnostic_points(settings, final_time)
    residuals = []
    for points in interior.split(settings["prediction_batch"]):
        points = points.clone().requires_grad_()
        residuals.append(autograd_pde_residual(model(points), points, **physics,
                         S=source(points[:, :1], points[:, 1:2])).detach())
    with torch.no_grad():
        ic, bc = model(initial), model(boundary)
    return {f"{prefix}_{name}": value
            for prefix, values in (("pde", torch.cat(residuals)), ("ic", ic), ("bc", bc))
            for name, value in residual_statistics(values).items()}


def predict_snapshots(model, x, y, times, batch):
    """Evaluate on a dense grid only after fitting, without clamping C."""
    yy, xx = np.meshgrid(y, x, indexing="ij")
    prediction = np.empty((len(times), len(y), len(x)), dtype=np.float64)
    with torch.no_grad():
        for i, time in enumerate(times):
            points = torch.tensor(np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, time))), dtype=torch.float32)
            values = []
            for chunk in points.split(batch):
                result = model(chunk)
                require_finite(result, "dense evaluation prediction")
                values.append(result[:, 0])
            prediction[i] = torch.cat(values).numpy().reshape(len(y), len(x))
    return prediction


def parameter_metrics(estimate, truth):
    """Euclidean location and absolute relative peak-strength error."""
    return dict(localization_error=source_localization_error(
        [estimate["x_s"], estimate["y_s"]], [truth["x_s"], truth["y_s"]]),
        relative_strength_error=relative_source_strength_error(estimate["Q"], truth["Q"]))


def recovery_decision(computational_success, metrics, criterion):
    """Secondary parameter recovery is not implied by a finite optimizer loss."""
    return bool(computational_success and metrics.get("localization_error") is not None
        and metrics.get("relative_strength_error") is not None
        and metrics["localization_error"] <= criterion["secondary_localization_max"]
        and metrics["relative_strength_error"] <= criterion["secondary_relative_strength_max"])


def load_evaluation_truth(root, scenario_id, run):
    """Refuse truth access before training has terminated (success OR failure)."""
    run = Path(run)
    attempt = json.loads((run/"attempt.json").read_text())  # gate before any truth I/O
    if attempt.get("training_terminated") is not True:
        raise RuntimeError("Evaluation requires a terminated training attempt.")
    directory = Path(root)/"datasets"/scenario_id
    manifest = json.loads((directory/"generation.json").read_text())
    checkpoint = Path(root)/manifest["reference_path"]
    if sha256(checkpoint) != manifest["reference_data_hash"]:
        raise ValueError("Reference-data hash changed after manifest freeze.")
    truth = json.loads((directory/"truth"/"source.json").read_text())
    from inverpinn.data.single_source_benchmark import object_hash
    if object_hash(truth) != manifest["source_hash"]:
        raise ValueError("Source-truth hash changed after manifest freeze.")
    return attempt, manifest, truth, checkpoint


def evaluate_run(root, scenario_id, seed, config):
    """Keep every attempted fit, even when no usable model was returned."""
    root = Path(root)
    run = root/"runs"/scenario_id/f"seed_{seed}"
    attempt, manifest, truth, reference = load_evaluation_truth(root, scenario_id, run)
    row = dict(scenario_id=scenario_id, split=manifest["split"], optimization_seed=seed,
        computational_success=attempt["computational_success"], evaluation_success=False,
        recovery_success=False, failure=attempt.get("failure", ""),
        x_true=truth["x_s"], y_true=truth["y_s"], Q_true=truth["Q"],
        x_pred=None, y_pred=None, Q_pred=None,
        localization_error=None, relative_strength_error=None,
        relative_l2=None, rmse=None, mae=None, negative_fraction=None, minimum_concentration=None,
        pde_rmse=None, pde_mae=None, ic_rmse=None, ic_mae=None, ic_maxabs=None,
        bc_rmse=None, bc_mae=None, bc_maxabs=None,
        reference_target_met=manifest["reference_target_met"],
        reference_error_estimate=manifest["max_reference_error_estimate"],
        training_seconds=attempt["training_seconds"])
    if attempt["computational_success"]:
        try:
            checkpoint = torch.load(run/"model.pt", map_location="cpu", weights_only=True)
            model = ConcentrationMLP([0., 0., 0.], [1., 1., config["measurement_times"]["stop"]],
                                     config["training"]["width"], config["training"]["depth"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            initial = config["training"]
            source = TrainableGaussianSource(initial["initial_x"], initial["initial_y"],
                                             initial["initial_Q"], config["source_family"]["sigma"])
            source.load_state_dict(checkpoint["source_state_dict"])
            estimate = source.estimates()
            row.update(x_pred=estimate["x_s"], y_pred=estimate["y_s"], Q_pred=estimate["Q"],
                       **parameter_metrics(estimate, truth))
            row["recovery_success"] = recovery_decision(True, row, config["recovery"])
            with np.load(reference, allow_pickle=False) as stored:
                fields, x, y, times = [stored[k] for k in ("fields", "x", "y", "field_times")]
                prediction = predict_snapshots(model, x, y, times, config["diagnostics"]["prediction_batch"])
                row.update(concentration_metrics(prediction, fields))
                write_json(run/"snapshot_metrics.json", [dict(time=float(t), **concentration_metrics(p, c))
                                                         for t, p, c in zip(times, prediction, fields)])
                np.savez_compressed(run/"predictions.npz", fields=prediction, x=x, y=y, field_times=times)
            row.update(negative_fraction=float(np.mean(prediction < 0)), minimum_concentration=float(prediction.min()))
            row.update(physical_diagnostics(model, source, config["physics"],
                                           config["measurement_times"]["stop"], config["diagnostics"]))
            row["evaluation_success"] = True
        except Exception as exc:
            row["failure"] = f"Evaluation {type(exc).__name__}: {exc}"
            # Parameter recovery already measured may remain available, but an
            # incomplete evaluation cannot be advertised as complete success.
            write_json(run/"evaluation_failure.json", dict(error=row["failure"]))
    write_json(run/"metrics.json", row)
    return row


def distribution(values, total):
    """Finite-case descriptive statistics with explicit missing denominator."""
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    result = dict(count=len(values), missing=total-len(values))
    if not len(values):
        return result | {k: None for k in ("mean", "median", "std", "q25", "q75", "q90", "max")}
    return result | dict(mean=float(values.mean()), median=float(np.median(values)),
        std=float(values.std(ddof=1)) if len(values) > 1 else None,
        q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
        q90=float(np.quantile(values, .90)), max=float(values.max()))


def wilson_interval(recovered, total, confidence):
    """Wilson score interval for a binomial proportion; includes failed runs."""
    if total < 1 or not 0 <= recovered <= total:
        raise ValueError("Require 0 <= recovered <= total and total > 0.")
    z = NormalDist().inv_cdf((1+confidence)/2)
    p, denominator = recovered/total, 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., center-half), min(1., center+half)]


def summarize(rows, config):
    """Primary-seed scenario statistics, with restarts explicitly separated."""
    metrics = ("localization_error", "relative_strength_error", "relative_l2", "rmse", "mae",
               "negative_fraction", "minimum_concentration", "pde_rmse", "pde_mae", "bc_rmse", "bc_mae", "ic_rmse", "ic_mae")
    result = {}
    for split in ("development", "heldout"):
        subset = [r for r in rows if r["split"] == split and r["optimization_seed"] == config["optimization"]["primary_seed"]]
        if not subset:
            continue
        recovered = sum(r["recovery_success"] for r in subset)
        result[split] = dict(total=len(subset), computationally_successful=sum(r["computational_success"] for r in subset),
            evaluated=sum(r["evaluation_success"] for r in subset), recovered=recovered,
            recovery_rate=recovered/len(subset), recovery_wilson_interval=wilson_interval(recovered, len(subset), config["recovery"]["confidence_level"]),
            reference_target_misses=sum(not r["reference_target_met"] for r in subset),
            metrics={k: distribution([r.get(k) for r in subset], len(subset)) for k in metrics})
    sensitivity = []
    for scenario_id in config["optimization"]["additional_seed_scenarios"]:
        subset = [r for r in rows if r["scenario_id"] == scenario_id]
        if subset:
            sensitivity.append(dict(scenario_id=scenario_id, seeds=[r["optimization_seed"] for r in subset],
                recoveries=sum(r["recovery_success"] for r in subset),
                computationally_successful=sum(r["computational_success"] for r in subset),
                **{k: distribution([r[k] for r in subset], len(subset))
                   for k in ("localization_error", "relative_strength_error", "relative_l2")}))
    result["optimizer_sensitivity"] = sensitivity
    result["all_runs"] = dict(total=len(rows), computationally_successful=sum(r["computational_success"] for r in rows),
        evaluated=sum(r["evaluation_success"] for r in rows), recovered=sum(r["recovery_success"] for r in rows))
    return result
