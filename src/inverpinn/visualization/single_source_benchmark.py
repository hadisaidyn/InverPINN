"""Unclipped, mechanically selected benchmark figures (PNG and vector PDF)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def save_figure(fig, output, dpi):
    """Save the same labeled figure as a raster and publication-scale vector."""
    output = Path(output)
    fig.savefig(output.with_suffix(".png"), dpi=dpi)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def training_figures(run, dpi):
    """Plot all completed updates, including partial histories of failed fits."""
    run = Path(run)
    records = [json.loads(line) for line in (run/"training_history.jsonl").read_text().splitlines()]
    if not records:
        return
    epochs = [r["epoch"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name in ("data", "pde", "initial", "boundary", "total"):
        axes[0].plot(epochs, [r[f"loss/{name}"] for r in records], label=name)
    axes[0].set(yscale="log", xlabel="Optimizer update", ylabel="Raw MSE / weighted total")
    axes[0].legend(fontsize=8)
    for name in ("x_s", "y_s", "Q"):
        axes[1].plot(epochs, [r[f"source/{name}"] for r in records], label=name)
    axes[1].set(xlabel="Optimizer update", ylabel="Estimated parameter (synthetic units)")
    axes[1].legend()
    save_figure(fig, run/"training_losses_and_parameters", dpi)


def reconstruction(root, row, directory, dpi):
    """Show t=0.8 truth/prediction/error on identical maps; retain negative C."""
    root, directory = Path(root), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not row["evaluation_success"]:
        (directory/"failure.txt").write_text(row["failure"] or "Evaluation unavailable; no replacement example selected.\n")
        return
    dataset = root/"datasets"/row["scenario_id"]
    reference = json.loads((dataset/"generation.json").read_text())["reference_path"]
    with np.load(root/reference) as stored:
        truth, x, y = stored["fields"][-1], stored["x"], stored["y"]
    with np.load(root/"runs"/row["scenario_id"]/f"seed_{row['optimization_seed']}"/"predictions.npz") as stored:
        predicted = stored["fields"][-1]
    positions = np.asarray(json.loads((root/"sensor_geometry.json").read_text()))
    error = np.abs(predicted-truth)
    limits = dict(vmin=min(truth.min(), predicted.min()), vmax=max(truth.max(), predicted.max()))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, field, title in zip(axes, (truth, predicted, error), ("Reference C", "PINN C (unclipped)", "Absolute error")):
        mesh = ax.pcolormesh(x, y, field, shading="auto", cmap="inferno", **({} if title == "Absolute error" else limits))
        ax.scatter(*positions.T, s=9, c="cyan", label="Sensors")
        ax.scatter(row["x_true"], row["y_true"], s=70, marker="o", facecolors="none", edgecolors="lime", label="True source")
        ax.scatter(row["x_pred"], row["y_pred"], s=70, marker="x", c="white", label="Estimated source")
        ax.set(xlabel="x (synthetic length)", ylabel="y (synthetic length)", aspect="equal", title=title)
        fig.colorbar(mesh, ax=ax, label="Synthetic concentration")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"{row['scenario_id']}, seed {row['optimization_seed']}: localization error {row['localization_error']:.4f}")
    save_figure(fig, directory/"reconstruction", dpi)


def aggregate_figures(root, rows, config):
    """Plot every primary held-out case, without choosing favorable restarts."""
    root = Path(root)
    dpi = config["plot_dpi"]
    primary = [r for r in rows if r["split"] == "heldout" and r["optimization_seed"] == config["optimization"]["primary_seed"]]
    finite = [r for r in primary if r["localization_error"] is not None]
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    positions = np.asarray(json.loads((root/"sensor_geometry.json").read_text()))
    ax.scatter(*positions.T, marker="^", s=18, c="gray", label="Fixed sensors")
    ax.scatter([r["x_true"] for r in primary], [r["y_true"] for r in primary], facecolors="none", edgecolors="#0072B2", label="True source")
    ax.scatter([r["x_pred"] for r in finite], [r["y_pred"] for r in finite], marker="x", c="#D55E00", label="PINN source")
    for row in finite:
        ax.plot([row["x_true"], row["x_pred"]], [row["y_true"], row["y_pred"]], color="0.45", alpha=.6, lw=.7)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="x (synthetic length)", ylabel="y (synthetic length)",
           title=f"All {len(primary)} held-out scenarios; {len(primary)-len(finite)} missing predictions")
    ax.legend(fontsize=8)
    save_figure(fig, root/"true_vs_predicted_sources", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    errors = sorted(r["localization_error"] for r in finite)
    axes[0].hist(errors, bins=10, color="#0072B2", edgecolor="white")
    axes[0].set(xlabel="Localization error (synthetic length)", ylabel="Scenario count")
    axes[1].step(errors, np.arange(1, len(errors)+1)/len(primary), where="post", color="#0072B2")
    axes[1].set(xlabel="Localization error (synthetic length)", ylabel="Fraction of ALL held-out scenarios", ylim=(0, 1.02))
    for ax in axes:
        ax.axvline(config["recovery"]["secondary_localization_max"], ls="--", color="#D55E00", label="One source width")
        ax.legend(fontsize=8)
    save_figure(fig, root/"localization_error_distribution", dpi)

    fig, ax = plt.subplots(figsize=(5.5, 5), layout="constrained")
    ax.scatter([r["Q_true"] for r in finite], [r["Q_pred"] for r in finite], c="#0072B2")
    bounds = [0, max([r["Q_true"] for r in primary]+[r["Q_pred"] for r in finite])*1.05]
    ax.plot(bounds, bounds, color="gray", ls="--", label="Identity")
    ax.set(xlabel="True Q (concentration/time)", ylabel="Estimated Q (concentration/time)", title="Held-out primary seed")
    ax.legend()
    save_figure(fig, root/"source_strength_recovery", dpi)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, metric, label in zip(axes, ("pde_rmse", "bc_rmse", "negative_fraction"),
                                 ("PDE residual RMSE (C/time)", "BC RMSE (C)", "Fraction C < 0")):
        ax.bar(np.arange(len(primary)), [r[metric] if r[metric] is not None else np.nan for r in primary], color="#0072B2")
        ax.set(ylabel=label)
    axes[-1].set(xticks=np.arange(len(primary)), xticklabels=[r["scenario_id"].split("_")[-1] for r in primary], xlabel="Held-out scenario ID")
    save_figure(fig, root/"physical_diagnostics", dpi)

    # Missing localization sorts to worst, rather than quietly disappearing.
    ranked = sorted(primary, key=lambda r: (float("inf") if r["localization_error"] is None else r["localization_error"], r["scenario_id"]))
    selected = dict(best_case=ranked[0], median_case=ranked[(len(ranked)-1)//2], worst_case=ranked[-1])
    for name, row in selected.items():
        directory = root/name
        directory.mkdir(exist_ok=False)
        (directory/"selection.json").write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
        reconstruction(root, row, directory, dpi)


def landscape_figure(root, axis, mse, truth, predicted, dpi):
    """Log-color observation mismatch: explicitly NOT a probability map."""
    root = Path(root)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    positive = mse[mse > 0]
    norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.size and positive.max() > positive.min() else None
    mesh = ax.pcolormesh(axis, axis, mse, shading="auto", cmap="viridis", norm=norm)
    ax.scatter(truth["x_s"], truth["y_s"], marker="o", s=90, facecolors="none", edgecolors="red", label="True source")
    if predicted is not None:
        ax.scatter(predicted["x_s"], predicted["y_s"], marker="x", s=90, c="white", label="PINN source")
    ax.set(xlabel="Candidate xs", ylabel="Candidate ys", aspect="equal", title="Location identifiability; true Q fixed\nObservation MSE, not posterior probability")
    ax.legend(fontsize=8)
    fig.colorbar(mesh, ax=ax, label="Sensor MSE (concentration²; log color)")
    save_figure(fig, root/"source_location_loss_landscape", dpi)
